from typing import Callable

import torch


def generate_data(likelihood: Callable, n: int=50, n_dims: int=1, manual_seed: int=42):
    torch.manual_seed(manual_seed)
    x = torch.rand(size=(n,n_dims)) * 4 - 2
    y = likelihood(torch.sum(torch.sin(2 * x),1)/n_dims).sample()  
    return x, y


def generate_n_dimensional_data(likelihood: Callable, n: int=50, in_dims: int=1, out_dims: int=3, manual_seed: int=42):
    torch.manual_seed(manual_seed)
    W = torch.randn(in_dims, out_dims) / in_dims
    x = torch.rand(size=(n,in_dims)) * 4 - 2
    y = likelihood(torch.sin(2 * x)@W).sample()  
    return x, y
