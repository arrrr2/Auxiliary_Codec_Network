import torch
import torch.nn as nn
import torch.nn.functional as FF
from fvcore.nn import FlopCountAnalysis, parameter_count_table



class jpeg(nn.Module):
    def __init__(self, color_channels=3, m=10, block_size=8, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.color_channels = color_channels 
        self.m = m
        self.block_size = block_size
        channels = color_channels * (block_size ** 2)

        self.head = nn.Sequential(
            nn.PixelUnshuffle(block_size)
        )
        self.body = self.repblocks(m, color_channels * (block_size ** 2))
        self.tail = nn.Sequential(
            nn.Conv2d(channels, channels, 1, 1),
            nn.AdaptiveAvgPool2d((1,1)),
            nn.Conv2d(channels, channels, 1, 1),
            nn.Conv2d(channels, 1, 1, 1),
        )


    def forward(self, x):
        x = self.head(x)
        x = self.body(x)
        x = self.tail(x)
        # x = torch.mean(x, dim=1, keepdim=True)
        while 1 in x.shape and len(x.shape) > 1:
            x = torch.squeeze(x, 1)
        return x

    def repblocks(self, m:int=10, channels=192):
        rep = []
        c = channels
        for _ in range(m):
            rep.append(nn.Conv2d(c, c, 1, 1))
            rep.append(nn.BatchNorm2d(c))
            rep.append(nn.LeakyReLU(inplace=True))
        return nn.Sequential(*rep)
    

if __name__=='__main__':
    tester = torch.rand([16, 3, 128, 128])
    net = jpeg()
    net.train()

    result = net(tester)
    print(result)
    print(parameter_count_table(net))