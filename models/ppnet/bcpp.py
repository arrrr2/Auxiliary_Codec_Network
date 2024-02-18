import torch
import torch.nn as nn
import torch.nn.functional as FF
from fvcore.nn import FlopCountAnalysis, parameter_count_table

class bicubicpp(nn.Module):
    def __init__(self, scale=2, ch=32, ch_in=3,
                 relu=nn.LeakyReLU(), padding_mode='reflect', bias=True):
        super(bicubicpp, self).__init__()
        self.ch_in = ch_in
        self.conv0 = nn.Conv2d(ch_in, ch, kernel_size=3, stride=2, padding=1, padding_mode=padding_mode)
        self.conv1 = nn.Conv2d(ch, ch, kernel_size=3, padding=1, padding_mode=padding_mode)
        self.conv2 = nn.Conv2d(ch, ch, kernel_size=3, padding=1, padding_mode=padding_mode)
        self.conv_out = nn.Conv2d(ch, (2*scale)**2 * ch_in, kernel_size=3, padding=1, padding_mode=padding_mode)
        self.Depth2Space = nn.PixelShuffle(2*scale)
        self.act = relu
        self.padding_mode = padding_mode
        self.mask_layer = None


    def forward(self, x):
        x0 = self.conv0(x)
        x0 = self.act(x0)
        x1 = self.conv1(x0)
        x1 = self.act(x1)
        x2 = self.conv2(x1)
        x2 = self.act(x2) + x0
        y = self.conv_out(x2)
        y = self.Depth2Space(y)
        return y
    
