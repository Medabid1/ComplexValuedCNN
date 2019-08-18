import torch
import torch.nn as nn 
import numpy as np 

from src.complex import CPLX





class zReLU(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x):
        rmask = torch.ge(x.r, torch.zeors_like(x.r)).float()
        imask = torch.ge(x.i, torch.zeors_like(x.i)).float()
        mask = rmask * imask
        out = CPLX(x.r * mask, x.i * mask)
        return out

class CReLU(nn.Module):
    def __init__(self):
        super().__init__()
        self.rRelu = nn.ReLU()
        self.iRelu = nn.ReLU()

    def forward(self, x):
        return CPLX(self.rRelu(x.r), self.iRelu(x.i))

