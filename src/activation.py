import torch
import troch.nn as nn 
import numpy as np 

from complex import CPLX





class CReLU(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x):
        rmask = torch.ge(x.r, torch.zeors_like(x.r)).float()
        imask = torch.ge(x.i, torch.zeors_like(x.i)).float()

        mask = rmask * imask

        out = CPLX(x.r * mask, x.i * mask)

        return out 