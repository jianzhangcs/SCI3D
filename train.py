import torch
import scipy.io as sio
import numpy as np
import os
import shutil
from dataloaders_distributed import train_dali_loader
from argparse import ArgumentParser
from network.HQS_temporal_3DConv_RFMb_HSA import HQSNet
from sci_utilities import batch_PSNR, rgb2ycbcr, normalize_augment
import tensorboardX
import time
import torch.distributed as dist
from apex.parallel import DistributedDataParallel as DDP
from apex import amp, optimizers, parallel
import torch.backends.cudnn as cudnn
from learnig_rate_schedulers import WarmupStepLR

# from thop import profile, clever_format
cudnn.benchmark = True

parser = ArgumentParser(description='Learned ADMM')

parser.add_argument('--warmup_steps', type=int, default=5, help='epoch number of learnig rate warmup')
parser.add_argument('--start_epoch', type=int, default=0, help='epoch number of start training')
parser.add_argument('--end_epoch', type=int, default=300, help='epoch number of end training')
parser.add_argument('--layer_num', type=int, default=10, help='phase number of HQS-Net')
parser.add_argument('--learning_rate', type=float, default=1e-4, help='learning rate')
parser.add_argument('--batch_size', type=int, default=1, help='batch size')
parser.add_argument('--patch_size', type=int, default=128, help='training patch size')
parser.add_argument('--n_channels', type=int, default=1, help='1 for gray, 3 for color')
parser.add_argument("--cs_ratio", type=int, default=8, help="Compress 8 frames to 1 frame")
parser.add_argument("--max_number_patches", "--m", type=int, default=25600,
                    help="Number of training sequence comprised of 8 frames per epoch")

parser.add_argument('--save_interval', type=int, default=1, help='save model on test set every x epochs')
parser.add_argument("--test_interval", type=int, default=1, help='evaluate model on test set every x epochs')
parser.add_argument('--gpu_list', type=str, default='0', help='gpu index')
parser.add_argument("--sRGB_path", type=str,
                    default='/home/wzy/Desktop/wzy/cs/SCI2020-ae86zhizhi-distributed_fast_temporal_amp/DAVIS-train-mp4',
                    help='path to video dataset')
parser.add_argument('--test_path', type=str,
                    default='/home/wzy/Desktop/wzy/cs/SCI2020-ae86zhizhi-distributed_fast_temporal_amp/code/simulation_dataset',
                    help='path to testset')
parser.add_argument('--model_dir', type=str, default='/home/wzy/Desktop/wzy/cs/SCI2020-ae86zhizhi-distributed_fast_temporal_amp/code/model', help='trained or pre-trained model directory')
parser.add_argument('--algo_name', type=str, default='Learned_ADMM_temporal_3DConv_RFMb_HSA', help='log directory')

parser.add_argument('--opt-level', type=str, default='O1')
parser.add_argument("--local_rank", default=0, type=int)
args = parser.parse_args()

start_epoch = args.start_epoch
end_epoch = args.end_epoch
learning_rate = args.learning_rate
layer_num = args.layer_num
gpu_list = args.gpu_list
batch_size = args.batch_size
best_psnr = 0
# os.environ["CUDA_VISIBLE_DEVICES"] = gpu_list
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# torch.cuda.set_device(1)

args.gpu = args.local_rank
torch.cuda.set_device(args.gpu)
torch.distributed.init_process_group(backend='nccl', init_method='env://')
args.world_size = torch.distributed.get_world_size()
args.total_batch_size = args.world_size * args.batch_size

# scale learning_rate according to total_batch_size
args.learning_rate = args.learning_rate * float(args.batch_size * args.world_size) / 1.
learning_rate = args.learning_rate


def main():
    # Load dataset and mask
    # transform = transforms.Compose([
    #     transforms.RandomCrop(args.patch_size),
    #     transforms.ToTensor()
    # ])
    # Training_labels = mydata.MultiFramesDataset(args, transform)
    # dataset = Imgdataset(args.sRGB_path)
    # loader_train = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)

    loader_train = train_dali_loader(batch_size=args.batch_size,
                                     file_root=args.sRGB_path,
                                     sequence_length=args.cs_ratio,
                                     crop_size=args.patch_size,
                                     epoch_size=args.max_number_patches // args.world_size,
                                     num_shards=args.world_size,  #
                                     device_id=args.local_rank,  #
                                     shard_id=args.local_rank,  #
                                     random_shuffle=True
                                     )
    num_minibatches = int(args.max_number_patches // args.batch_size // args.world_size)
    example_data = sio.loadmat(os.path.join(args.test_path, 'kobe.mat'))
    mask = example_data['mask'] / 1.0

    # masks used for
    mask = torch.from_numpy(mask).type(torch.FloatTensor).cuda()
    mask = mask.permute(2, 0, 1)
    if args.n_channels == 3:
        cs_ratio, w, h = mask.shape
        mask = mask[:, None].expand(cs_ratio, 3, w, h)  # torch.Size([8, 3, 256, 256])
    elif args.n_channels == 1:
        mask = mask[:, None]
    mask_train = mask.type(torch.FloatTensor).cuda()

    # Data loader
    # if (platform.system() == "Windows"):
    #     rand_loader = DataLoader(dataset=Training_labels, batch_size=batch_size, num_workers=0,
    #                              shuffle=True)
    # else:
    #     rand_loader = DataLoader(dataset=Training_labels, batch_size=batch_size, num_workers=4,
    #                              shuffle=False)

    # model defination
    model = HQSNet(args)
    # model = nn.DataParallel(model)
    model = model.cuda()

    # loss and optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = WarmupStepLR(warmup_steps=args.warmup_steps, optimizer=optimizer, step_size=10, gamma=0.9)

    model, optimizer = amp.initialize(model, optimizer, opt_level=args.opt_level)

    model = DDP(model, delay_allreduce=True)

    global model_dir
    # load model and make model saving/log dir
    model_dir = "%s/Video_SCI_%s_layer_%d" % (
        args.model_dir, args.algo_name, layer_num)

    if not os.path.exists(model_dir) and args.local_rank in [-1, 0]:
        os.makedirs(model_dir)

    if start_epoch > 0:
        # Use a local scope to avoid dangling references
        def resume():
            checkpoint_filename = '%s/net_%d.pth' % (model_dir, start_epoch)
            if os.path.isfile(checkpoint_filename):
                print("=> loading checkpoint '{}'".format(checkpoint_filename))
                checkpoint = torch.load(checkpoint_filename,
                                        map_location=lambda storage, loc: storage.cuda(args.local_rank))
                best_psnr = checkpoint['best_psnr']
                model.load_state_dict(checkpoint['state_dict'])
                optimizer.load_state_dict(checkpoint['optimizer'])
                amp.load_state_dict(checkpoint['amp'])
                print("=> loaded checkpoint '{}' (epoch {})"
                      .format(checkpoint_filename, checkpoint['epoch']))
            else:
                print("=> no checkpoint found at '{}'".format(checkpoint_filename))

        resume()

    # tensorboardX
    writer = tensorboardX.SummaryWriter(log_dir=model_dir)

    # Training
    iter = 0
    # save_dict = {
    #     'HQS-Net': model.state_dict()
    # }

    show_test = True
    for epoch_i in range(start_epoch + 1, end_epoch + 1):
        if args.local_rank in [-1, 0]:
            print("current learning rate: %f" % scheduler.get_last_lr()[0])
            time_start = time.time()
        for i, data in enumerate(loader_train, 0):

            # print(data[0]['data'].shape)
            iter += 1

            model.zero_grad()
            optimizer.zero_grad()

            # img_train, meas = data[0], data[1]
            # img_train = img_train[:,:, None, :, :]
            # meas = meas[:, None, None, :, :]

            d0, d1, d2, d3, d4 = data[0]['data'].shape
            img_train = normalize_augment(data[0]['data']).view(d0, d1, d2, d3, d4)
            # img_train = data #torch.Size([16, 8, 3, 64, 64])

            c = img_train.shape[2]
            # input rgb image output y chanel
            if args.n_channels == 1 and c == 3:
                img_train = rgb2ycbcr(img_train)
            elif args.n_channels == 3 and c == 1:
                img_train = img_train.repeat(1, 1, 3, 1, 1)

            xx = np.random.randint(257 - args.patch_size)
            yy = np.random.randint(257 - args.patch_size)
            input_mask = mask_train[:, :, xx:xx + args.patch_size,
                         yy:yy + args.patch_size]  # torch.Size([8, 3, 64, 64])

            img_train = img_train.type(torch.FloatTensor).cuda()

            meas = torch.sum(img_train * input_mask, dim=1, keepdim=True)  # torch.Size([16, 1, 3, 64, 64])

            out_train = model(meas, input_mask)

            # macs, params = profile(model, inputs=(meas, input_mask, ))
            # macs, params = clever_format([macs, params], "%.3f")
            #
            # print("FLOPs: %.3f", macs)
            # print("params: %.3f", params)

            loss = torch.mean(torch.pow(out_train - img_train, 2))
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
            # with amp.scale_loss(loss, optimizer) as scaled_loss:
            #     scaled_loss.backward()

            optimizer.step()

            # results
            # model.eval()
            out_train = torch.clamp(out_train, 0., 1.)
            psnr_train = batch_PSNR(out_train, img_train, 1.)
            psnr_train = torch.as_tensor(float(psnr_train)).cuda()
            loss = reduce_tensor(loss)
            psnr_train = reduce_tensor(psnr_train)
            if args.local_rank in [-1, 0]:
                writer.add_scalar('psnr', psnr_train.item(), iter)
                writer.add_scalar('loss', loss.item(), iter)
                print("%s [epoch %d][%d/%d] loss: %.4f PSNR_train: %.2f dB" %
                      (time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()), epoch_i, i, num_minibatches, loss.item(),
                       psnr_train.item()))
        if args.local_rank in [-1, 0]:
            time_end = time.time()
        if args.local_rank in [-1, 0]:
            print('time cost', time_end - time_start)

        scheduler.step()

        if args.local_rank in [-1, 0] and epoch_i % args.test_interval == 0:
            # evaluation
            model.eval()
            name_list = os.listdir(args.test_path)
            # name_list = ['kobe', 'traffic', 'runner', 'drop', 'crash', 'aerial']
            psnr_all = np.zeros((1, len(name_list)))

            k = 0
            test_time = 0
            for cur_name in name_list:
                Training_data = sio.loadmat(os.path.join(args.test_path, cur_name))
                mask = Training_data['mask'] / 1.0
                orig = Training_data['orig'] / 255.

                mask = torch.from_numpy(mask).type(torch.FloatTensor).cuda()
                orig_tensor = torch.from_numpy(orig).type(torch.FloatTensor).cuda()

                orig_tensor = orig_tensor.unsqueeze(0).permute(0, 3, 1, 2)  # torch.Size([1, 32, 256, 256])
                d0, d1, d2, d3 = orig_tensor.shape
                orig_tensor = orig_tensor[:, :, None].expand(d0, d1, args.n_channels, d2, d3)
                d0_ = d1 // 8
                orig_tensor = orig_tensor.contiguous().view(d0_, 8, args.n_channels, d2,
                                                            d3)  # torch.Size([4, 8, 3, 256, 256])

                mask = mask.permute(2, 0, 1)
                cs_ratio, w, h = mask.shape
                mask = mask[:, None].expand(cs_ratio, args.n_channels, w, h)  # torch.Size([8, 3, 256, 256])
                mask = mask.type(torch.FloatTensor).cuda()

                if show_test:
                    test = orig_tensor[0, :]  # 8*3*256*256
                    writer.add_images(cur_name, test)

                with torch.no_grad():
                    meas_test = torch.sum(orig_tensor * mask, dim=1, keepdim=True)
                    start = time.time()
                    out_test = model(meas_test, mask)
                    end = time.time()
                    test_time += (end - start)
                    out_test = torch.clamp(out_test, 0, 1)
                    psnr_test = batch_PSNR(out_test, orig_tensor, 1.)
                    psnr_all[0, k] = psnr_test
                    k = k + 1
                    print("name %s, epoch %6d, every test result: %.4f" % (
                        cur_name, epoch_i, psnr_test))
                    writer.add_scalar('%s' % cur_name, psnr_test, epoch_i)
                    # torch.Size([4, 8, 3, 256, 256])
                    out = out_test[0, :]

                    writer.add_images(cur_name + '_reconstructed', out, epoch_i)
            print('test time %.4f' % (test_time / 6))
            writer.add_scalar("avg", np.mean(psnr_all), epoch_i)
            print("avg %.4f" % np.mean(psnr_all))
            show_test = False
            model.train()

            if args.local_rank in [-1, 0] and epoch_i % args.save_interval == 0:
                global best_psnr
                # if not os.path.exists(args.save_path):
                #     os.mkdir(args.save_path)
                # dir = os.path.join(args.save_path, 'layer_num%d' % args.num_of_layers)
                # if not os.path.exists(dir):
                #     os.mkdir(dir)
                is_best = np.mean(psnr_all) > best_psnr
                best_psnr = max(np.mean(psnr_all), best_psnr)
                save_checkpoint({
                    'epoch': epoch_i,
                    'state_dict': model.state_dict(),
                    'best_psnr': best_psnr,
                    'optimizer': optimizer.state_dict(),
                    'amp': amp.state_dict(),
                }, is_best, filename=os.path.join(model_dir, 'net_%d.pth' % epoch_i))
                print('best test psnr till now %.4f' % best_psnr)
                print('checkpoint with %d iterations has been saved.' % epoch_i)
                print()
                print()


def save_checkpoint(state, is_best, filename='checkpoint.pth'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, os.path.join(model_dir, 'model_best.pth'))


def reduce_tensor(tensor):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt /= args.world_size
    return rt


if __name__ == "__main__":
    main()
