import torch
import torch.nn as nn
import numpy as np 

from complex import CPLX






class ComplexConv2D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding=0, dilation=1,
                 groups=1, bias=True, padding_mode='zeros'):
        super().__init__()

        self.rConv2d = nn.Conv2d(in_channels, out_channels, kernel_size, 
                                stride=stride, padding=padding, 
                                dilation=dilation, groups=groups, 
                                bias=bias)

        self.iConv2d = nn.Conv2d(in_channels, out_channels, kernel_size, 
                                stride=stride, padding=padding, 
                                dilation=dilation, groups=groups, 
                                bias=bias)
    
    def forward(self, x):
        r = self.rConv2d(x.r) - self.iConv2d(x.i)
        i = self.rConv2d(x.i) + self.iConv2d(x.r)

        out = CPLX(r, i)

        return out


class ComplexConvTranspose2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding=0, dilation=1,
                 groups=1, bias=True, padding_mode='zeros'):
        super().__init__()

        self.rConvTranspose2d = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, 
                                stride=stride, padding=padding, 
                                dilation=dilation, groups=groups, 
                                bias=bias)