from .resnet import *
from .resnest import *
import torch.nn as nn
import torch
from .multilevelNet import multilevelNet
from .HRfineNet import HRfineNet

__all__ = ['MLPE50', 'MLPE101',]

class MLPE(nn.Module):
    def __init__(self, resnet, output_shape, num_class, pretrained=True):
        super(MLPE, self).__init__()
        channel_settings = [2048, 1024, 512, 256]
        # channel_settings = [1024, 512, 256, 128]
        self.resnet = resnet
        self.multilevel_net = multilevelNet(channel_settings, output_shape, num_class)
        self.HRfine_net = HRfineNet(channel_settings[-1], output_shape, num_class)

    def forward(self, x):
        res_out = self.resnet(x)
        _fms, _outs = self.global_net(res_out)
        out = self.HRfine_net(_fms)
        return _outs, out

def MLPE50(out_size,num_class,pretrained=True):
    res50 = resnest50(pretrained=pretrained)
    model = MLPE(res50, output_shape=out_size, num_class=num_class, pretrained=pretrained)

    return model


def MLPE101(out_size,num_class,pretrained=True):
    res101 = resnest101(pretrained=pretrained)
    model = MLPE(res101, output_shape=out_size, num_class=num_class, pretrained=pretrained)

    return model


