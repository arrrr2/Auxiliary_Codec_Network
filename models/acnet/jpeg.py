import torch
import torch.nn as nn
import torch.nn.functional as FF
from fvcore.nn import FlopCountAnalysis, parameter_count_table

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
        self.activation = nn.LeakyReLU(0.1)

    def forward(self, x):
        x = self.conv(x)
        x = self.activation(x)
        return x

class jpeg(nn.Module):
    def __init__(self, color_channels=3, num_levels=4):
        super(jpeg, self).__init__()
        base_channels = color_channels * 64
        self.pixel_unshuffle = nn.PixelUnshuffle(8)
        self.encoder_blocks = nn.ModuleList()
        self.decoder_blocks = nn.ModuleList()
        self.skip_connections = nn.ModuleList()
        self.start_conv = nn.Conv2d(base_channels, base_channels, kernel_size=1, stride=1, padding=0)
        num_levels -= 1
        for _ in range(num_levels):
            self.encoder_blocks.append(ConvBlock(base_channels, base_channels))
            self.decoder_blocks.append(ConvBlock(base_channels, base_channels))
            self.skip_connections.append(nn.Identity())
        self.final_conv = nn.Conv2d(base_channels, base_channels, kernel_size=1, stride=1, padding=0)
        self.pixel_shuffle = nn.PixelShuffle(8)

    def forward(self, x):
        x = self.pixel_unshuffle(x)
        skip_outputs = []
        for encoder in self.encoder_blocks:
            x = encoder(x)
            skip_outputs.append(x)
        x = self.start_conv(x)
        for i, decoder in enumerate(self.decoder_blocks):
            skip_input = skip_outputs[-(i+1)]
            skip_output = self.skip_connections[i](skip_input)
            x = decoder(x + skip_output)  
        x = self.final_conv(x)
        x = self.pixel_shuffle(x)
        return x




if __name__=='__main__':
    tester = torch.rand([16, 3, 128, 128])
    net = jpeg()
    zero = torch.zeros([16, 3, 128, 128])

    cri = nn.MSELoss()
    opt = torch.optim.Adam(net.parameters())

    result = net(tester)

    loss = cri(result, zero)
    # result.register_hook(lambda grad: print(grad))
    # loss.backward()
    # for name, params in net.named_parameters():	
        # print('-->name:', name, '-->grad_requirs:',params.requires_grad, \
            # ' -->grad_shape:',params.grad.shape)
    # print(result.shape)
    # print(parameter_count_table(net))

    # from torchviz import make_dot
    # net_img = make_dot(result)
    # import graphviz
    # net_img.render("net", format = 'png')
    
    from torchviz import make_dot
    import graphviz
    net_img = make_dot(result)
    net_img.render("net", format = 'png')
