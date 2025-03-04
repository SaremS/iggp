from typing import Callable

import torch
from scipy.special import roots_hermite
import numpy as np

TORCH_PI = torch.FloatTensor([np.pi])

def torch_gauss_hermite(f: Callable[[torch.Tensor], torch.Tensor],
                        mu: torch.Tensor, sigma_sq: torch.Tensor,
                        n_points: int=10):
    """
    Approximates the expectation of a function f(x) under a Gaussian distribution
    with mean `mu` and variance `sigma_sq` using Gauss-Hermite quadrature with PyTorch tensors.

    Parameters:
        f (callable): The function to integrate, accepting PyTorch tensors.
        mu (torch.Tensor): The mean of the Gaussian distribution.
        sigma (torch.Tensor): The standard deviation of the Gaussian distribution.
        n_points (int): The number of quadrature points to use (default: 10).

    Returns:
        torch.Tensor: The approximated expectation value.
    """
    nodes, weights = roots_hermite(n_points)
    
    nodes = torch.tensor(nodes, dtype=torch.float32)
    weights = torch.tensor(weights, dtype=torch.float32)

    sigma = torch.sqrt(sigma_sq)
    
    transformed_nodes = torch.sqrt(torch.tensor(2.0)) * sigma * nodes.unsqueeze(-1) + mu.unsqueeze(-1)
    
    f_values = f(transformed_nodes)
    
    expectation = torch.sum(weights.unsqueeze(-1) * f_values, dim=0) / torch.sqrt(TORCH_PI)
    
    return expectation


def torch_gauss_hermite_batch(f: Callable[[torch.Tensor], torch.Tensor],
                        mu: torch.Tensor, sigma_sq: torch.Tensor,
                        *args, 
                        n_points: int=10):
    """
    Approximates the expectation of a function f(x) under a Gaussian distribution
    with mean `mu` and variance `sigma_sq` using Gauss-Hermite quadrature with PyTorch tensors.

    Batched Version for faster inference 

    Parameters:
        f (callable): The function to integrate, accepting PyTorch tensors.
        mu (torch.Tensor): The mean of the Gaussian distribution.
        sigma (torch.Tensor): The standard deviation of the Gaussian distribution.
        *args: Torch Tensor args to define the args of the target function f - all tensors
                must have the same shape as mu and sigma_sq
        n_points (int): The number of quadrature points to use (default: 10).

    Returns:
        torch.Tensor: The approximated expectation value.

    Important:
        - Does not validate correct shape of the inputs
    """
    nodes, weights = roots_hermite(n_points)
    
    nodes = torch.tensor(nodes, dtype=torch.float32)
    weights = torch.tensor(weights, dtype=torch.float32)

    mu_shape = list(mu.shape)
    target_shape_nodes = [n_points] + [1] * len(mu.shape)

    nodes_usq = nodes.view(*target_shape_nodes)
    weights_usq = weights.view(*target_shape_nodes)

    mu_usq = mu.unsqueeze(0)
    sigma_sq_usq = sigma_sq.unsqueeze(0)

    args_usq = []
    for arg in args:
        args_usq.append(arg.unsqueeze(0))

    sigma_usq = torch.sqrt(sigma_sq_usq)
    
    transformed_nodes = torch.sqrt(torch.tensor(2.0)) * sigma_usq * nodes_usq + mu_usq   

    f_values = f(transformed_nodes, *args_usq)
    
    expectation = torch.sum(weights_usq * f_values, dim=0) / torch.sqrt(TORCH_PI)
    
    return expectation
