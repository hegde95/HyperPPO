import numpy as np
import torch

def default_device():
    return 'cuda' if torch.cuda.is_available() else 'cpu'

PRIMITIVES_DEEPNETS1M = [
    'max_pool',
    'avg_pool',
    'sep_conv',
    'dil_conv',
    'conv',
    'msa',
    'cse',
    'sum',
    'concat',
    'input',
    'bias',
    'bn',
    'ln',
    'pos_enc',
    'glob_avg',
]


def capacity(model):
    c, n = 0, 0
    for name, p in model.named_parameters():
        if p.requires_grad:
            c += 1
            n += np.prod(p.shape)
    return c, int(n)