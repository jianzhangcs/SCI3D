import math
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch.nn.init as init
from random import choices
from skimage.measure.simple_metrics import compare_psnr


def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
    elif classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
    elif classname.find('BatchNorm') != -1:
        # nn.init.uniform(m.weight.data, 1.0, 0.02)
        m.weight.data.normal_(mean=0, std=math.sqrt(2. / 9. / 64.)).clamp_(-0.025, 0.025)
        nn.init.constant_(m.bias.data, 0.0)


def batch_PSNR(img, imclean, data_range):
    Img = img.data.cpu().numpy().astype(np.float32)
    Iclean = imclean.data.cpu().numpy().astype(np.float32)
    PSNR = 0
    for i in range(Img.shape[0]):
        PSNR += compare_psnr(Iclean[i, :, :, :], Img[i, :, :, :], data_range=data_range)
    return (PSNR / Img.shape[0])


def data_augmentation(image, mode):
    out = np.transpose(image, (1, 2, 0))
    if mode == 0:
        # original
        out = out
    elif mode == 1:
        # flip up and down
        out = np.flipud(out)
    elif mode == 2:
        # rotate counterwise 90 degree
        out = np.rot90(out)
    elif mode == 3:
        # rotate 90 degree and flip up and down
        out = np.rot90(out)
        out = np.flipud(out)
    elif mode == 4:
        # rotate 180 degree
        out = np.rot90(out, k=2)
    elif mode == 5:
        # rotate 180 degree and flip
        out = np.rot90(out, k=2)
        out = np.flipud(out)
    elif mode == 6:
        # rotate 270 degree
        out = np.rot90(out, k=3)
    elif mode == 7:
        # rotate 270 degree and flip
        out = np.rot90(out, k=3)
        out = np.flipud(out)
    return np.transpose(out, (2, 0, 1))


def A_operator(z, Phi):
    y = torch.sum(Phi * z, 1, keepdim=True)
    return y


def At_operator(z, Phi):
    y = z * Phi
    return y

def shift_back(inputs, step):
    # torch.Size([1, 28, 1, 128, 155])
    d0, d1, d2, d3, d4 = inputs.shape
    for i in range(d1):
        inputs[:, i, :, :, :] = torch.roll(inputs[:, i, :, :, :], (-1)*step*i, dims=1)
    output = inputs[:, :, :, :, 0:d4-step*(d1-1)]
    return output

def shift(inputs, step):
    d0, d1, d2, d3, d4 = inputs.shape
    output = torch.zeros(d0, d1, d2, d3, d4+(d1-1)*step).to(inputs.device)
    for i in range(d1):
        output[:, i, :, :, i*step:i*step+d4] = inputs[:, i, :, :, :]
    return output

def initialize_weights(net_l, scale=1):
    if not isinstance(net_l, list):
        net_l = [net_l]
    for net in net_l:
        for m in net.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, a=0, mode='fan_in')
                m.weight.data *= scale
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                init.kaiming_normal_(m.weight, a=0, mode='fan_in')
                m.weight.data *= scale
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias.data, 0.0)


def pack_gbrg_raw(raw):
    # pack GBRG Bayer raw to 4 channels
    black_level = 240
    white_level = 2 ** 12 - 1
    im = raw.astype(np.float32)  # (1080, 1920)
    im = np.maximum(im - black_level, 0) / (white_level - black_level)

    im = np.expand_dims(im, axis=2)  # (1080, 1920, 1)
    img_shape = im.shape
    H = img_shape[0]
    W = img_shape[1]

    out = np.concatenate((im[1:H:2, 0:W:2, :],  # B
                          im[1:H:2, 1:W:2, :],  # G
                          im[0:H:2, 1:W:2, :],  # R
                          im[0:H:2, 0:W:2, :]), axis=2)  # (540, 960, 4) #G
    return out


def variable_to_cv2_image(invar, conv_rgb_to_bgr=True):
    r"""Converts a torch.autograd.Variable to an OpenCV image

    Args:
        invar: a torch.autograd.Variable
        conv_rgb_to_bgr: boolean. If True, convert output image from RGB to BGR color space
    Returns:
        a HxWxC uint8 image
    """
    assert torch.max(invar) <= 1.0

    size4 = len(invar.size()) == 4
    if size4:
        nchannels = invar.size()[1]
    else:
        nchannels = invar.size()[0]

    if nchannels == 1:
        if size4:
            res = invar.data.cpu().numpy()[0, 0, :]
        else:
            res = invar.data.cpu().numpy()[0, :]
        res = (res * 255.).clip(0, 255).astype(np.uint8)
    elif nchannels == 3:
        if size4:
            res = invar.data.cpu().numpy()[0]
        else:
            res = invar.data.cpu().numpy()
        res = res.transpose(1, 2, 0)
        res = (res * 255.).clip(0, 255).astype(np.uint8)
        if conv_rgb_to_bgr:
            res = cv2.cvtColor(res, cv2.COLOR_RGB2BGR)
    else:
        raise Exception('Number of color channels not supported')
    return res


def normalize(data):
    r"""Normalizes a unit8 image to a float32 image in the range [0, 1]

    Args:
        data: a unint8 numpy array to normalize from [0, 255] to [0, 1]
    """
    return np.float32(data / 255.)


def toggle_grad(model, requires_grad):
    for p in model.parameters():
        p.requires_grad_(requires_grad)


def compute_loss(d_out, target):
    targets = d_out.new_full(size=d_out.size(), fill_value=target)
    loss = F.binary_cross_entropy_with_logits(d_out, targets)

    return loss


def rgb2ycbcr(rgb):
    img_r = rgb[:, :, 0, :, :]
    img_g = rgb[:, :, 1, :, :]
    img_b = rgb[:, :, 2, :, :]
    arr = 0.256789 * img_r + 0.504129 * img_g + 0.097906 * img_b + 16 / 255.  # torch.Size([8, 1080, 1920])
    # arr[:, 0, :, :] = 0.256789 * img_r + 0.504129 * img_g + 0.097906 * img_b + 16/255.
    # arr[:, 1, :, :] = -0.148223 * img_r - 0.290992 * img_g + 0.439215 * img_b + 128/255.
    # arr[:, 2, :, :] = 0.439215 * img_r - 0.367789 * img_g - 0.071426 * img_b + 128/255.
    return arr[:, :, None]


def normalize_augment(datain):
    '''Normalizes and augments an input patch of dim [N, num_frames, C. H, W] in [0., 255.] to \
        [N, num_frames*C. H, W] in  [0., 1.]. It also returns the central frame of the temporal \
        patch as a ground truth.
    '''

    def transform(sample):
        # define transformations
        do_nothing = lambda x: x
        do_nothing.__name__ = 'do_nothing'
        flipud = lambda x: torch.flip(x, dims=[2])
        flipud.__name__ = 'flipup'
        rot90 = lambda x: torch.rot90(x, k=1, dims=[2, 3])
        rot90.__name__ = 'rot90'
        rot90_flipud = lambda x: torch.flip(torch.rot90(x, k=1, dims=[2, 3]), dims=[2])
        rot90_flipud.__name__ = 'rot90_flipud'
        rot180 = lambda x: torch.rot90(x, k=2, dims=[2, 3])
        rot180.__name__ = 'rot180'
        rot180_flipud = lambda x: torch.flip(torch.rot90(x, k=2, dims=[2, 3]), dims=[2])
        rot180_flipud.__name__ = 'rot180_flipud'
        rot270 = lambda x: torch.rot90(x, k=3, dims=[2, 3])
        rot270.__name__ = 'rot270'
        rot270_flipud = lambda x: torch.flip(torch.rot90(x, k=3, dims=[2, 3]), dims=[2])
        rot270_flipud.__name__ = 'rot270_flipud'
        add_csnt = lambda x: x + torch.normal(mean=torch.zeros(x.size()[0], 1, 1, 1), \
                                              std=(5 / 255.)).expand_as(x).to(x.device)
        add_csnt.__name__ = 'add_csnt'

        # define transformations and their frequency, then pick one.
        aug_list = [do_nothing, flipud, rot90, rot90_flipud, \
                    rot180, rot180_flipud, rot270, rot270_flipud, add_csnt]
        w_aug = [32, 12, 12, 12, 12, 12, 12, 12, 12]  # one fourth chances to do_nothing
        transf = choices(aug_list, w_aug)

        # transform all images in array
        return transf[0](sample)

    img_train = datain
    # convert to [N, num_frames*C. H, W] in  [0., 1.] from [N, num_frames, C. H, W] in [0., 255.]
    img_train = img_train.view(img_train.size()[0], -1, img_train.size()[-2], img_train.size()[-1]) / 255.

    # augment
    img_train = transform(img_train)


    return img_train
