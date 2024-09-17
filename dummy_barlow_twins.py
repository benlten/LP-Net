import torch
from torch import nn
from torch.nn import functional as F
import tqdm

def off_diagonal(x): 
    #################### from https://github.com/facebookresearch/barlowtwins/blob/main/main.py ###########################
    ####################################################### begin reference ###############################################
    # return a flattened view of the off-diagonal elements of a square matrix
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()
    ####################################################### end reference #################################################

def shared_step(z1, z2, lambda_coeff = 5e-4):
    # empirical cross-correlation matrix
    z1_norm = (z1 - torch.mean(z1, dim=0)) / torch.std(z1, dim=0)
    z2_norm = (z2 - torch.mean(z2, dim=0)) / torch.std(z2, dim=0)
    cross_corr = torch.matmul(z1_norm.T, z2_norm) / z1.shape[0]

    on_diag = torch.diagonal(cross_corr).add_(-1).pow_(2).sum()
    off_diag = off_diagonal(cross_corr).pow_(2).sum()
    loss = on_diag + lambda_coeff * off_diag
    ####################################################### end reference #################################################
    return loss


z1 = torch.randn(128, 16).cuda().requires_grad_()
z2 = torch.randn(128, 16).cuda().requires_grad_()
optimizer = torch.optim.Adam([z1, z2], lr=1e-1)

for _ in tqdm.trange(1000):
    loss = shared_step(z1, z2)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    # print(loss.item())

breakpoint()
