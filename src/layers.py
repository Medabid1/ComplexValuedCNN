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

        self.iConvTranspose2d = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, 
                                stride=stride, padding=padding, 
                                dilation=dilation, groups=groups, 
                                bias=bias)

    def forward(self, x):
        r = self.rConvTranspose2d(x.r) - self.iConvTranspose2d(x.i)
        i = self.rConvTranspose2d(x.i) + self.iConvTranspose2d(x.r)

        out = CPLX(r, i)

        return out


class ComplexLinear(nn.Module):
    def __init__(self, in_features, out_features, bias=True  ):
        super().__init__()

        self.rLinear = nn.Linear(in_features, out_features, bias=bias)

        self.iLinear = nn.Linear(in_features, out_features, bias=bias)
    
    def forward(self, x):
        r = self.rLinear(x.r) - self.iLinear(x.i)
        i = self.rLinear(x.i) + self.iLinear(x.r)

        out = CPLX(r, i)

        return out

class ComplexAvgPool2d(nn.Module):
    def __init__(self, kernel_size, stride=None, padding=0, 
                ceil_mode=False, count_include_pad=True):

        self.rAvg = nn.AvgPool2d(kernel_size, stride=stride, padding=padding, 
                                ceil_mode=ceil_mode, count_include_pad=count_include_pad )
        
        self.iAvg = nn.AvgPool2d(kernel_size, stride=stride, padding=padding, 
                                ceil_mode=ceil_mode, count_include_pad=count_include_pad)

    def forward(self, x):
        return CPLX(self.rAvg(x.r), self.iAvg(x.i))