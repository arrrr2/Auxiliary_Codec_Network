import torch
import torch.nn as nn
import torch.nn.functional as FF
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
    
class tad(nn.Module):
    def __init__(self, scale=2, c_channels=3,  n_feats=64, resblocks=3, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.scale = scale

        self.head = nn.Sequential(
            nn.Conv2d(c_channels, n_feats, 3, 1, 1), 
            nn.Conv2d(n_feats, n_feats, 3, 1, 1), 
            nn.PixelUnshuffle(scale),
            nn.Conv2d(n_feats * (scale ** 2), n_feats, 3, 1, 1), 
        )

        self.body = nn.Sequential(
            nn.Sequential(*[ResBlock(default_conv, n_feats, 3) for i in range(resblocks)]),
            nn.Conv2d(n_feats, n_feats, 3, 1, 1)
        )

        self.tail = nn.Sequential(
            nn.Conv2d(n_feats, c_channels, 3, 1, 1)
        )


    def forward(self, x):
        lr = FF.interpolate(
            x, scale_factor=1/self.scale, mode='bicubic', antialias=True)
        x = self.head(x)
        y = self.body(x)
        x = y + x
        x = self.tail(x)
        x = x + lr
        return x


if __name__=='__main__':
    tester = torch.rand([16, 3, 128, 128])
    net = tad()
    net.train()

    result = net(tester)
    print(result.shape)
    print(parameter_count_table(net, max_depth=5))