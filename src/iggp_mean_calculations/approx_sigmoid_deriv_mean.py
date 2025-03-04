import torch
import numpy as np
from scipy.special import stirling2, factorial

from src.iggp_mean_calculations.torch_gauss_hermite import *


### Sigmoid E[sigmoid'(X)Y] - Logistic binary classification

def sigmoid_deriv_product_mean_approx(mu: torch.Tensor, cov: torch.Tensor) -> float:
    """
    Approximate E[sigmoid'(X)*Y] via second-order Taylor-Approximation where X and Y are univariate marginals of a bi-variate Gaussian random distribution and

    Notice that sigmoid'() denotes the first derivative of the sigmoid function
    as required to calculate integrated gradients

    Args:
        mu (torch.Tensor): Mean vector of X and Y
        cov (torch.Tensor): Covariance matrix of X and Y


    Returns:
        float: Second order Taylor Approximation of E[sigmoid'(X)*Y]

    IMPORTANT:
        - The last element of the respective bi-variate vector is presumed to be Y in this implementation, so 
            arguments must be passed accordingly
        - Does not check for valid covariance matrix

    """
    m_X = mu[0]
    m_Y = mu[1]

    K_XX = cov[0,0]
    K_YY = cov[1,1]
    K_XY = cov[0,1]

    result_torch = sigmoid_deriv_product_mean_approx_batch(m_X, m_Y,
                                                           K_XX, K_YY, K_XY)

    return result_torch.item()


def sigmoid_deriv_product_mean_approx_batch(m_X: torch.Tensor, m_Y: torch.Tensor,
                                      K_XX: torch.Tensor, K_YY: torch.Tensor, K_XY: torch.Tensor) -> torch.Tensor:
    """
    Approximate E[sigma'(X)*Y] via second order Taylor approximation where X and Y are univariate marginals of a bi-variate Gaussian random distribution and
    sigma'() is the density of a standard normal; 

    Batched Version for faster inference 

    Args:
        m_X (torch.Tensor): Mean of X marginal
        m_Y (torch.Tensor): Mean of Y marginal
        K_XX (torch.Tensor): Variance of X marginal
        K_YY (torch.Tensor): Variance of Y marginal
        K_XY (torch.Tensor): Covariance of X and Y 

    Returns:
        torch.Tensor: Exact values of E[sigma'(X)*Y], for each batch

    IMPORTANT:
        - Does not check for valid covariance matrices
        - Supports arbitrary dimensions for the parameters, but all shapes must match exactly
    """
    return (sigmoid_deriv_n(m_X, 1)*m_Y \
            + 0.5*K_XX*sigmoid_deriv_n(m_X, 3)*m_Y \
            + K_XY*sigmoid_deriv_n(m_X, 2))




def sigmoid_deriv_product_mean_quadrature(mu: torch.Tensor, cov: torch.Tensor,
                                          n_quadrature_points: int=10) -> float:
    """
    Approximate E[sigmoid'(X)*Y] via Gauss-Hermite quadrature where X and Y are univariate marginals of a bi-variate Gaussian random distribution and

    Notice that sigmoid'() denotes the first derivative of the sigmoid function
    as required to calculate integrated gradients

    Args:
        mu (torch.Tensor): Mean vector of X and Y
        cov (torch.Tensor): Covariance matrix of X and Y
        n_quadrature_points (int): The number of quadrature points to use (default: 10).

    Returns:
        float: Gauss-Hermite Approximation of E[sigmoid'(X)*Y]

    IMPORTANT:
        - The last element of the respective bi-variate vector is presumed to be Y in this implementation, so 
            arguments must be passed accordingly
        - Does not check for valid covariance matrix

    """
    m_X = mu[0]
    m_Y = mu[1]

    K_XX = cov[0,0]
    K_YY = cov[1,1]
    K_XY = cov[0,1]

    target_fun = lambda x: sigmoid_deriv_n(x, 1) * (m_Y + K_XY/K_XX * (x - m_X))

    result = torch_gauss_hermite(target_fun, m_X, K_XX, n_quadrature_points)

    return result.item()


def sigmoid_deriv_product_mean_quadrature_batch(m_X: torch.Tensor, m_Y: torch.Tensor,
                                                K_XX: torch.Tensor, K_YY: torch.Tensor, K_XY: torch.Tensor,
                                                n_quadrature_points: int=10) -> torch.tensor:
    """
    Approximate E[sigmoid'(X)*Y] via Gauss-Hermite quadrature where X and Y are univariate marginals of a bi-variate Gaussian random distribution and

    Notice that sigmoid'() denotes the first derivative of the sigmoid function
    as required to calculate integrated gradients

    Batch-wise for faster inference

    Args:
        m_X (torch.Tensor): Mean of X marginal
        m_Y (torch.Tensor): Mean of Y marginal
        K_XX (torch.Tensor): Variance of X marginal
        K_YY (torch.Tensor): Variance of Y marginal
        K_XY (torch.Tensor): Covariance of X and Y 


    Returns:
        float: Gauss-Hermite Approximation of E[sigmoid'(X)*Y]

    IMPORTANT:
        - Does not check for valid covariance matrix
    """
    #target_fun = lambda x, m_X, m_Y, K_XX, K_XY: sigmoid_deriv_n(x, 1) * (m_Y + K_XY/K_XX * (x - m_X))
    target_fun = lambda x, m_X, m_Y, K_XX, K_XY: torch.sigmoid(x) *(1-torch.sigmoid(x)) * (m_Y + K_XY/K_XX * (x - m_X))


    result = torch_gauss_hermite_batch(target_fun, m_X, K_XX, m_X, m_Y, K_XX, K_XY, n_points = n_quadrature_points)

    return result


def sigmoid_deriv_n(x: torch.Tensor, n: int) -> torch.Tensor:
    """
    Evaluate the n-th derivative of the sigmoid function at x

    Args:
        x (torch.tensor): Target tensor to evaluate the sigmoid derivative at
        n (int): n-th derivative to evaluate

    Returns:
        torch.Tensor: N-th sigmoid derivative of each element of x
    """
    result = torch.zeros_like(x, dtype=torch.float)

    for k in range(1, n+2):
        result += (-1)**(k+1) \
                    * factorial(k-1)*stirling2(n+1, k) \
                    * torch.sigmoid(x)**k

    return result
    

def sigmoid_deriv_product_mean_mc(mu: torch.Tensor, cov: torch.Tensor, n_samples: int=100000) -> float:
    """
    Estimate E[sigmoid'(X)*Y] via MONTE CARLO INTEGRATION form where X and Y are univariate marginals of a bi-variate Gaussian random distribution and
    Phi'() is the density of a standard normal; i.e. the first derivative with respect to X of a standard normal CDF as in Probit Regression 

    Args:
        mu (torch.tensor): Mean vector of X and Y
        cov (torch.tensor): Covariance matrix of X and Y
        n_samples (int): Number of MC samples

    Returns:
        float: MC estimate of E[sigmoid'(X)*Y]

    IMPORTANT:
        - The last element of the respective bi-variate vector is presumed to be Y in this implementation, so 
            arguments must be passed accordingly
        - Does not check for valid covariance matrix
    """
    mvn = torch.distributions.MultivariateNormal(mu, covariance_matrix=cov)
    
    samp = mvn.sample((n_samples,))

    sigmoid_vals = torch.sigmoid(samp[:, 0])
    sigmoid_deriv_vals = sigmoid_vals * (1-sigmoid_vals)
    result = torch.mean(sigmoid_deriv_vals * samp[:, 1])
    
    return result.item()
