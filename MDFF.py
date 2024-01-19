import ops as ops
import torch
import torch.nn as nn
import torch.nn.functional as F
from Attention_module import *

class RFEG(nn.Module):
    def __init__(self,
                 in_channels, out_channels, wn,
                 group=1):
        super(RFEG, self).__init__()
        self.rb1 = ops.RFEM(wn, in_channels, out_channels)
        self.rb2 = ops.RFEM(wn, in_channels, out_channels)
        self.rb3 = ops.RFEM(wn, in_channels, out_channels)
        self.reduction_1 = ops.BasicConv2d(wn, out_channels * 3, out_channels, 1, 1, 0)
        self.reduction_2 = ops.BasicConv2d(wn, out_channels * 9, out_channels, 1, 1, 0)
        self.reduction_3 = ops.BasicConv2d(wn, out_channels * 3, out_channels, 1, 1, 0)
    def forward(self, x):
        c0 = o0 = x
        b1, A_1, c1 = self.rb1(o0)
        b2, A_2, c2 = self.rb2(b1)
        b3, A_3, c3 = self.rb3(b2)
        Mult_Feature_bank = self.reduction_1(torch.cat([c1, c2, c3], 1))
        Attention_bank = self.reduction_2(torch.cat([A_1, A_2, A_3], 1))
        High_Feature_bank = self.reduction_3(torch.cat([b1, b2, b3], 1))
        out_mult = Mult_Feature_bank+ Attention_bank
        out = out_mult+High_Feature_bank
        return out, out_mult

class RFEG_2(nn.Module):
    def __init__(self,
                 in_channels, out_channels, wn,
                 group=1):
        super(RFEG_2, self).__init__()
        self.rb1 = ops.RFEM_2(wn, in_channels, out_channels)
        self.rb2 = ops.RFEM_2(wn, in_channels, out_channels)
        self.rb3 = ops.RFEM_2(wn, in_channels, out_channels)
        self.reduction_1 = ops.BasicConv2d(wn, out_channels * 3, out_channels, 1, 1, 0)
        self.reduction_2 = ops.BasicConv2d(wn, out_channels * 9, out_channels, 1, 1, 0)
        self.reduction_3 = ops.BasicConv2d(wn, out_channels * 3, out_channels, 1, 1, 0)
    def forward(self, x):
        c0 = o0 = x
        b1, A_1, c1 = self.rb1(o0)
        b2, A_2, c2 = self.rb2(b1)
        b3, A_3, c3 = self.rb3(b2)
        Mult_Feature_bank = self.reduction_1(torch.cat([c1, c2, c3], 1))
        Attention_bank = self.reduction_2(torch.cat([A_1, A_2, A_3], 1))
        High_Feature_bank = self.reduction_3(torch.cat([b1, b2, b3], 1))
        out_mult = Mult_Feature_bank+ Attention_bank
        out = out_mult+High_Feature_bank
        return out, out_mult

class RFEG_3(nn.Module):
    def __init__(self,
                 in_channels, out_channels, wn,
                 group=1):
        super(RFEG_3, self).__init__()
        self.rb1 = ops.RFEM_3(wn, in_channels, out_channels)
        self.rb2 = ops.RFEM_3(wn, in_channels, out_channels)
        self.rb3 = ops.RFEM_3(wn, in_channels, out_channels)
        self.reduction_1 = ops.BasicConv2d(wn, out_channels * 3, out_channels, 1, 1, 0)
        self.reduction_2 = ops.BasicConv2d(wn, out_channels * 9, out_channels, 1, 1, 0)
        self.reduction_3 = ops.BasicConv2d(wn, out_channels * 3, out_channels, 1, 1, 0)
    def forward(self, x):
        c0 = o0 = x
        b1, A_1, c1 = self.rb1(o0)
        b2, A_2, c2 = self.rb2(b1)
        b3, A_3, c3 = self.rb3(b2)
        Mult_Feature_bank = self.reduction_1(torch.cat([c1, c2, c3], 1))
        Attention_bank = self.reduction_2(torch.cat([A_1, A_2, A_3], 1))
        High_Feature_bank = self.reduction_3(torch.cat([b1, b2, b3], 1))
        out_mult = Mult_Feature_bank+ Attention_bank
        out = out_mult+High_Feature_bank
        return out, out_mult

class Net(nn.Module):
    def __init__(self, **kwargs):
        super(Net, self).__init__()
        wn = lambda x: torch.nn.utils.weight_norm(x)
        scale = kwargs.get("scale")
        group = kwargs.get("group", 4)
        n_feats = 64
        self.sub_mean = ops.MeanShift((0.4488, 0.4371, 0.4040), sub=True)
        self.add_mean = ops.MeanShift((0.4488, 0.4371, 0.4040), sub=False)
        self.entry_1 = nn.Conv2d(3, n_feats, 3, 1, 1)
        self.b1 = RFEG(n_feats, n_feats, wn=wn)
        self.b2 = RFEG_2(n_feats, n_feats, wn=wn)
        self.b3 = RFEG_3(n_feats, n_feats, wn=wn)
        self.reduction_1 = ops.BasicConv2d(wn, n_feats * 3, n_feats, 1, 1, 0)
        self.reduction_2 = ops.BasicConv2d(wn, n_feats * 3, n_feats, 1, 1, 0)
        self.upsample = ops.UpsampleBlock(n_feats, scale=scale, multi_scale=False, wn=wn, group=group)
        self.exit1 = wn(nn.Conv2d(n_feats, 3, 3, 1, 1))
        self.linear = nn.PReLU(init=1.0)
    def forward(self, x, scale):
        x = self.sub_mean(x)
        res = x
        z = self.entry_1(self.linear(x))
        c0 = o0 = z
        b1, A_1 = self.b1(o0)
        b2, A_2 = self.b2(b1)
        b3, A_3 = self.b3(b2)
        Feature_bank = self.reduction_1(torch.cat([b1, b2, b3], 1))
        Mult_Feature_bank = self.reduction_2(torch.cat([A_1, A_2, A_3], 1))
        out = Feature_bank + Mult_Feature_bank
        out = self.upsample(out, scale=scale)
        out = self.exit1(self.linear(out))
        skip = F.interpolate(res, size=(z.size(-2) * scale, z.size(-1) * scale), mode='bicubic', align_corners=False)
        out = skip + out
        out = self.add_mean(out)
        return out