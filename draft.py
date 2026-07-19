import torch


def gen_pos_enc(ndim, pos):
    a = torch.empty([ndim, ])
