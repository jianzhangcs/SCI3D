import os
import time
import math
import torch
# import torchvision
import imageio
import pytorch_ssim
import numpy as np
import scipy.io as sio
from argparse import ArgumentParser
from sci_utilities import batch_PSNR
from thop import profile, clever_format
import torchvision.transforms.functional as F
from network.HQS_temporal_3DConv_RFMb_HSA import HQSNet


parser = ArgumentParser(description='Learned ADMM')

parser.add_argument('--gpu_list', type=str, default='0', help='gpu index')
parser.add_argument('--test_dir', type=str, default='./simulation_dataset', help='path to test set')
parser.add_argument('--model_dir', type=str, default='/media/wzy/data/result/3DConv/3DCN_RFMb_HSA', help='trained or pre-trained model directory')
parser.add_argument('--result', type=str, default='/home/wzy/Desktop/wzy/cs/SCI2020-ae86zhizhi-distributed_fast_temporal_amp/code/results/simulation', help='results for reconstructed frames')

parser.add_argument('--layer_num', type=int, default=10, help='phase number of HQS-Net')
parser.add_argument('--n_channels', type=int, default=1, help='1 for gray, 3 for color')
parser.add_argument('--temporal_patch', type=int, default=3,
                    help='the number of frames where temporal step takes as input')
parser.add_argument('--seed', type=int, default=1,
                    help='random seed')

args = parser.parse_args()

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_list
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)

def test_simulation():
    if not os.path.exists(args.result):
        os.makedirs(args.result)

    test_list = os.listdir(args.test_dir)
    psnr_all = torch.zeros(len(test_list))
    ssim_all = torch.zeros(len(test_list))
    total_time = 0
    start_epoch = 200

    model = HQSNet(args)
    model.load_state_dict({k.replace('module.', ''): v for k, v in
                           torch.load('%s/net_%d.pth' % (args.model_dir, start_epoch))['state_dict'].items()})
    model.to(device)
    # model = torch.load('%s/net_%d.pth' % (args.model_dir, start_epoch))
    # model.to(device)

    total_num = 0
    for i in range(len(test_list)):

        Training_data = sio.loadmat(os.path.join(args.test_dir, test_list[i]))

        mask = Training_data['mask'] / 1.0
        orig = Training_data['orig'] / 255.

        mask = torch.from_numpy(mask).type(torch.FloatTensor).to(device) #torch.Size([256, 256, 8])
        orig_tensor = torch.from_numpy(orig).type(torch.FloatTensor).to(device) #torch.Size([256, 256, 40])

        orig_tensor = orig_tensor.unsqueeze(0).permute(0, 3, 1, 2)  # torch.Size([1, 32, 256, 256])
        d0, d1, d2, d3 = orig_tensor.shape
        orig_tensor = orig_tensor[:, :, None].expand(d0, d1, 1, d2, d3)
        d0_ = d1 // 8
        orig_tensor = orig_tensor.contiguous().view(d0_, 8, 1, d2, d3)  # torch.Size([5, 8, 1, 256, 256])

        mask = mask.permute(2, 0, 1)
        cs_ratio, w, h = mask.shape
        mask = mask[:, None].expand(cs_ratio, 1, w, h)  # torch.Size([8, 3, 256, 256])
        mask = mask.type(torch.FloatTensor).to(device)


        with torch.no_grad():
            meas_test = torch.sum(orig_tensor * mask, dim=1, keepdim=True) #torch.Size([5, 1, 1, 256, 256])
            # if i==0:
            #     macs, params = profile(model, inputs=(meas_test[0, ...][None], mask, ))
            #     macs, params = clever_format([macs, params], "%.3f")
            total_num += meas_test.shape[0]
            torch.cuda.synchronize()
            start = time.time()
            out_test = model(meas_test, mask)
            torch.cuda.synchronize()
            end = time.time()
            total_time += (end-start)
            out_test = torch.clamp(out_test, 0, 1)
            psnr_test = batch_PSNR(out_test, orig_tensor, 1.)
            out_test = out_test.view(-1, 1, 256, 256)
            orig_tensor = orig_tensor.view(-1, 1, 256, 256)
            ssim_test = pytorch_ssim.ssim(out_test.cpu(), orig_tensor.cpu())
            psnr_all[i] = psnr_test
            ssim_all[i] = ssim_test


            sio.savemat(os.path.join(args.result, test_list[i]), {'recon': out_test.cpu().numpy()})
        print('%s psnr %.4f ssim %.4f' % (test_list[i], psnr_test, ssim_test))
    print('average psnr %.4f average ssim %.4f averagt time %.4f'%(torch.mean(psnr_all), torch.mean(ssim_all), total_time/total_num))


if __name__ == "__main__":
    test_simulation()
