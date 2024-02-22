import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import sys
import cv2
from collections import OrderedDict

def YCrCb2rgb(imgs_y, imgs_cr, imgs_cb):#0~1之间
    r = 1.164*(imgs_y - 16/255.0) + 1.596*(imgs_cr - 128/255.0)
    g = 1.164*(imgs_y - 16/255.0) - 0.392*(imgs_cb - 128/255.0) - 0.813*(imgs_cr - 128/255.0)
    b = 1.164*(imgs_y - 16/255.0) + 2.017*(imgs_cb - 128/255.0)
    rgb_imgs = torch.cat((r,g,b), dim=1)
    return rgb_imgs

def rgb2YCrCb(imgs_r, imgs_g, imgs_b):#0~1之间
    y  = 0.257 * imgs_r + 0.564 * imgs_g + 0.098 * imgs_b + 16/255.0
    Cr = 0.439 * imgs_r - 0.368 * imgs_g - 0.071 * imgs_b + 128/255.0
    Cb = -0.148* imgs_r - 0.291 * imgs_g + 0.439 * imgs_b + 128/255.0
    y_imgs = torch.cat((y, Cr, Cb), dim=1)
    return y_imgs

# tool functions
def get_img_seq(img_seq_dir):
    img_seq = []
    for root, _, fnames in sorted(os.walk(img_seq_dir)):
        for fname in sorted(fnames):
            if any(fname.endswith(ext) for ext in '.png'):
                img_name = os.path.join(root, fname)
                img_seq.append(cv2.imread(img_name))
    return img_seq


def features_grad(features):
    kernel = [[1 / 8, 1 / 8, 1 / 8], [1 / 8, -1, 1 / 8], [1 / 8, 1 / 8, 1 / 8]]
    kernel = torch.FloatTensor(kernel).unsqueeze(0).unsqueeze(0)
    kernel = kernel.cuda()
    _, c, _, _ = features.shape
    c = int(c)
    for i in range(c):
        feat_grad = F.conv2d(features[:, i:i + 1, :, :], kernel, stride=1, padding=1)
        if i == 0:
            feat_grads = feat_grad
        else:
            feat_grads = torch.cat((feat_grads, feat_grad), dim=1)
    return feat_grads


# network functions
def pad(pad_type, padding):
    pad_type = pad_type.lower()
    if padding == 0:
        return None


def get_valid_padding(kernel_size, dilation):
    kernel_size = kernel_size + (kernel_size - 1) * (dilation - 1)
    padding = (kernel_size - 1) // 2
    return padding


def activation(act_type='prelu', slope=0.2, n_prelu=1):
    act_type = act_type.lower()
    if act_type == 'prelu':
        layer = nn.PReLU(num_parameters=n_prelu, init=slope)
    elif act_type == 'lrelu':
        layer = nn.LeakyReLU(negative_slope=slope, inplace=True)
    else:
        raise NotImplementedError('[ERROR] Activation layer [%s] is not implemented!' % act_type)
    return layer


def norm(n_feature, norm_type='bn'):
    norm_type = norm_type.lower()
    if norm_type == 'bn':
        layer = nn.BatchNorm2d(n_feature)
    else:
        raise NotImplementedError('[ERROR] %s.sequential() does not support OrderedDict' % norm_type)
    return layer


def sequential(*args):
    if len(args) == 1:
        if isinstance(args[0], OrderedDict):
            raise NotImplementedError('[ERROR] %s.sequential() does not support OrderedDict' % sys.modules[__name__])
        else:
            return args[0]
    modules = []
    for module in args:
        if isinstance(module, nn.Sequential):
            for submodule in module:
                modules.append(submodule)
        elif isinstance(module, nn.Module):
            modules.append(module)
    return nn.Sequential(*modules)


def ConvBlock(in_channels, out_channels, kernel_size, stride=1, dilation=1, bias=True, valid_padding=True, padding=0,
              act_type='prelu', norm_type='bn', pad_type='zero'):
    if valid_padding:
        padding = get_valid_padding(kernel_size, dilation)
    else:
        pass
    p = pad(pad_type, padding) if pad_type and pad_type != 'zero' else None
    conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, dilation=dilation,
                     bias=bias)

    act = activation(act_type) if act_type else None
    n = norm(out_channels, norm_type) if norm_type else None
    return sequential(p, conv, n, act)