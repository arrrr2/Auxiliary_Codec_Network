
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from fvcore.nn import FlopCountAnalysis, parameter_count_table


def default_conv(in_channels, out_channels, kernel_size, bias=True):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size//2), bias=bias)

class ResBlock(nn.Module):
    def __init__(
        self, conv, n_feats, kernel_size,
        bias=True, bn=False, act=nn.ReLU(True), res_scale=1):

        super(ResBlock, self).__init__()
        m = []
        for i in range(2):
            m.append(conv(n_feats, n_feats, kernel_size, bias=bias))
            if bn:
                m.append(nn.BatchNorm2d(n_feats))
            if i == 0:
                m.append(act)

        self.body = nn.Sequential(*m)
        self.res_scale = res_scale

    def forward(self, x):
        res = self.body(x).mul(self.res_scale)
        res += x

        return res

class Upsampler(nn.Sequential):
    def __init__(self, conv, scale, n_feats, bn=False, act=False, bias=True):

        m = []
        if (scale & (scale - 1)) == 0:    
            for _ in range(int(math.log(scale, 2))):
                m.append(conv(n_feats, 4 * n_feats, 3, bias))
                m.append(nn.PixelShuffle(2))
                if bn:
                    m.append(nn.BatchNorm2d(n_feats))
                if act == 'relu':
                    m.append(nn.ReLU(True))
                elif act == 'prelu':
                    m.append(nn.PReLU(n_feats))

        elif scale == 3:
            m.append(conv(n_feats, 9 * n_feats, 3, bias))
            m.append(nn.PixelShuffle(3))
            if bn:
                m.append(nn.BatchNorm2d(n_feats))
            if act == 'relu':
                m.append(nn.ReLU(True))
            elif act == 'prelu':
                m.append(nn.PReLU(n_feats))
        else:
            raise NotImplementedError

        super(Upsampler, self).__init__(*m)

class edsr(nn.Module):
    def __init__(self, scale=2, n_colors=3, n_resblocks=16, n_feats=64, kernel_size=3, res_scale=1, conv=default_conv):
                # edsr-baseline: r16,f64,x2
        super(edsr, self).__init__()
        kernel_size = 3
        act = nn.ReLU(True)


        m_head = [conv(n_colors, n_feats, kernel_size)]

        m_body = [
            ResBlock(
                conv, n_feats, kernel_size, act=act, res_scale=res_scale
            ) for _ in range(n_resblocks)
        ]
        m_body.append( conv(n_feats, n_feats, kernel_size) )

        self.head = nn.Sequential(*m_head)
        self.body = nn.Sequential(*m_body)

        
        self.out_dim = n_colors
        if scale == 1:
            m_tail = [
                conv(n_feats, n_colors, kernel_size)
            ]
        else:
            m_tail = [
                Upsampler(conv, scale, n_feats, act=False),
                conv(n_feats, n_colors, kernel_size)
            ]
        self.tail = nn.Sequential(*m_tail)

    def forward(self, x):
        x = self.head(x)

        res = self.body(x)
        res += x

        x = self.tail(res)
        return x


if __name__=='__main__':
    tester = torch.rand([16, 3, 128, 128])
    net = edsr()
    net.train()

    result = net(tester)
    print(result.shape)
    print(parameter_count_table(net))