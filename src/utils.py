import torch 
from src.complex import CPLX




def concat(axis=-1, *args):

    x_r = torch.concat([j.r for j in args], axis=axis)
    x_i = torch.concat([j.i for j in args], axis=axis)

    return CPLX(x_r, x_i)

def loss(labels, predictions, loss_function):
    if isinstance(labels, CPLX):
        return  loss_function(labels.r, predictions.r) + loss_function(labels.i, predictions.i)
    else : 
        raise 'Labels should be CPLX type'
