import math
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import torch

from Attention_module import *


def init_weights(modules):
    pass


class MeanShift(nn.Module):
    def __init__(self, mean_rgb, sub):
        super(MeanShift, self).__init__()
        sign = -1 if sub else 1
        r = mean_rgb[0] * sign
        g = mean_rgb[1] * sign
        b = mean_rgb[2] * sign
        self.shifter = nn.Conv2d(3, 3, 1, 1, 0)
        self.shifter.weight.data = torch.eye(3).view(3, 3, 1, 1)
        self.shifter.bias.data = torch.Tensor([r, g, b])
        for params in self.shifter.parameters():
            params.requires_grad = False

    def forward(self, x):
        x = self.shifter(x)
        return x


class UpsampleBlock(nn.Module):
    def __init__(self, n_channels, scale, multi_scale, wn, group=1):
        super(UpsampleBlock, self).__init__()

        if multi_scale:
            self.up2 = _UpsampleBlock(n_channels, scale=2, wn=wn, group=group)
            self.up3 = _UpsampleBlock(n_channels, scale=3, wn=wn, group=group)
            self.up4 = _UpsampleBlock(n_channels, scale=4, wn=wn, group=group)
        else:
            self.up = _UpsampleBlock(n_channels, scale=scale, wn=wn, group=group)

        self.multi_scale = multi_scale

    def forward(self, x, scale):
        if self.multi_scale:
            if scale == 2:
                return self.up2(x)
            elif scale == 3:
                return self.up3(x)
            elif scale == 4:
                return self.up4(x)
        else:
            return self.up(x)


class _UpsampleBlock(nn.Module):
    def __init__(self, n_channels, scale, wn, group=1):
        super(_UpsampleBlock, self).__init__()

        modules = []

        if scale == 2 or scale == 4 or scale == 8:
            for _ in range(int(math.log(scale, 2))):
                modules += [nn.PReLU(),wn(nn.Conv2d(n_channels, 4 * n_channels, 3, 1, 1, groups=group))]
                modules += [nn.PixelShuffle(2)]

        elif scale == 3:
            modules += [nn.PReLU(),wn(nn.Conv2d(n_channels, 9 * n_channels, 3, 1, 1, groups=group))]
            modules += [nn.PixelShuffle(3)]

        elif scale == 5:
            modules += [wn(nn.Conv2d(n_channels, 25 * n_channels, 3, 1, 1, groups=group)), nn.ReLU(inplace=True)]
            modules += [nn.PixelShuffle(5)]

        self.body = nn.Sequential(*modules)
        init_weights(self.modules)
    def forward(self, x):
        out = self.body(x)
        return out
       


class BasicConv2d(nn.Module): 

    def __init__(self, wn, in_planes, out_planes, kernel_size, stride, padding=0):
        super(BasicConv2d, self).__init__()
        self.conv = wn(nn.Conv2d(in_planes, out_planes,
                                 kernel_size=kernel_size, stride=stride,
                                 padding=padding, bias=True))

        self.LR = nn.PReLU()
        init_weights(self.modules)

    def forward(self, x):
        x = self.LR(x)
        x = self.conv(x)
        return x

# 原始代码
class RFEM(nn.Module):
    def __init__(self,
                 wn, in_channels, out_channels):
        super(RFEM, self).__init__()
        self.DiVA = DiVA_attention()
        self.conv_1_3 = wn(nn.Conv2d(in_channels,out_channels,(1,3),padding=(0,3 // 2),bias=True))
        self.conv_3_1 = wn(nn.Conv2d(in_channels,out_channels,(3,1),padding=(3 // 2,0),bias=True))
        self.con_3_3 = wn(nn.Conv2d(in_channels,out_channels,3,padding=3 // 2,bias=True))
        self.confusion = wn(nn.Conv2d(out_channels*3,out_channels*2,1,padding=1 // 2))
        self.confusion1 = wn(nn.Conv2d(out_channels*3,out_channels,1,padding=1 // 2))
        self.conv1 = wn(nn.Conv2d(out_channels*2,out_channels,3,padding=3 // 2))
        self.bn = nn.BatchNorm2d(out_channels)
        self.prelu = nn.PReLU()
    def forward(self, x):
        input_x = self.prelu(x)
        out_x = self.con_3_3(input_x)
        out_x_v = self.conv_1_3(input_x)
        out_x_h = self.conv_3_1(input_x)
        oam_out = torch.cat([out_x,out_x_h,out_x_v],1)
        out_DiVA = self.DiVA(oam_out)
        out_Multf = self.confusion1(self.prelu(oam_out))
        cat_out = self.confusion(self.prelu(oam_out)) # 通道数为64*2
        out = self.bn(self.conv1(self.prelu(cat_out)))+x
        return out, out_DiVA, out_Multf

class RFEM_2(nn.Module):
    def __init__(self,
                 wn, in_channels, out_channels):
        super(RFEM_2, self).__init__()
        self.DiVA = DiVA_attention()
        self.conv_1_3 = wn(nn.Conv2d(in_channels,out_channels,(1,3),padding=(0,3 // 2),bias=True))
        self.conv_3_1 = wn(nn.Conv2d(in_channels,out_channels,(3,1),padding=(3 // 2,0),bias=True))
        self.con_3_3 = wn(nn.Conv2d(in_channels,out_channels,3,padding=3 // 2,bias=True))
        self.confusion = wn(nn.Conv2d(out_channels*3,out_channels*2,1,padding=1 // 2))
        self.confusion1 = wn(nn.Conv2d(out_channels*3,out_channels,1,padding=1 // 2))
        self.conv1 = wn(nn.Conv2d(out_channels*2,out_channels,3,padding=3 // 2))
        self.bn = nn.BatchNorm2d(out_channels)
        self.prelu = nn.PReLU()
    def forward(self, x):
        input_x = self.prelu(x)
        out_x = self.con_3_3(input_x)
        out_x_v = self.conv_1_3(input_x)
        out_x_h = self.conv_3_1(input_x)
        oam_out = torch.cat([out_x,out_x_h,out_x_v],1)
        out_DiVA = self.DiVA(oam_out)
        out_Multf = self.confusion1(self.prelu(oam_out))
        cat_out = self.confusion(self.prelu(oam_out))
        out = self.bn(self.conv1(self.prelu(cat_out)))+x
        return out, out_DiVA, out_Multf

class RFEM_3(nn.Module):
    def __init__(self,
                 wn, in_channels, out_channels):
        super(RFEM_3, self).__init__()
        self.DiVA = DiVA_attention()
        self.conv_1_3 = wn(nn.Conv2d(in_channels,out_channels,(1,3),padding=(0,3 // 2),bias=True))
        self.conv_3_1 = wn(nn.Conv2d(in_channels,out_channels,(3,1),padding=(3 // 2,0),bias=True))
        self.con_3_3 = wn(nn.Conv2d(in_channels,out_channels,3,padding=3 // 2,bias=True))
        self.confusion = wn(nn.Conv2d(out_channels*3,out_channels*2,1,padding=1 // 2))
        self.confusion1 = wn(nn.Conv2d(out_channels*3,out_channels,1,padding=1 // 2))
        self.conv1 = wn(nn.Conv2d(out_channels*2,out_channels,3,padding=3 // 2))
        self.bn = nn.BatchNorm2d(64)
        self.prelu = nn.PReLU()
        
    def forward(self, x):
        input_x = self.prelu(x)
        out_x = self.con_3_3(input_x)
        out_x_v = self.conv_1_3(input_x)
        out_x_h = self.conv_3_1(input_x)
        oam_out = torch.cat([out_x,out_x_h,out_x_v],1)
        out_DiVA = self.DiVA(oam_out)
        out_Multf = self.confusion1(self.prelu(oam_out))
        cat_out = self.confusion(self.prelu(oam_out)) # 通道数为64*2
        out = self.bn(self.conv1(self.prelu(cat_out)))+x
        return out, out_DiVA, out_Multf