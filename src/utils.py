import torch 
from src.complex import CPLX




def concat(axis=-1, *args):

    x_r = torch.concat([j.r for j in args], axis=axis)
    x_i = torch.concat([j.i for j in args], axis=axis)

    return CPLX(x_r, x_i)