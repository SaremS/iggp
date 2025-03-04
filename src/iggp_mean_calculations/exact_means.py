import torch
import numpy as np

TORCH_PI = torch.FloatTensor([np.pi])


### Gaussian PDF E[Phi'(X)Y] - Probit binary classification

def gaussian_pdf_product_mean(mu: torch.tensor, cov: torch.tensor) -> float:
    """
    Calculate E[Phi'(X)*Y] in closed form where X and Y are univariate marginals of a bi-variate Gaussian random distribution and
    Phi'() is the density of a standard normal; i.e. the first derivative with respect to X of a standard normal CDF as in Probit Regression 

    Args:
        mu (torch.tensor): Mean vector of X and Y
        cov (torch.tensor): Covariance matrix of X and Y

    Returns:
        torch.tensor: Exact value of E[Phi'(X)*Y]

    IMPORTANT:
        - The last element of the respective bi-variate vector is presumed to be Y in this implementation, so 
            arguments must be passed accordingly
        - Does not check for valid covariance matrix

    """
    #Step 1: Extract relevant elements from mu and cov
    m_X = mu[0]
    m_Y = mu[1]

    K_XX = cov[0,0]
    K_YY = cov[1,1]
    K_XY = cov[0,1]

    result_torch = gaussian_pdf_product_mean_batch(m_X, m_Y, K_XX, K_YY, K_XY)

    return result_torch.item()


def gaussian_pdf_product_mean_batch(m_X: torch.Tensor, m_Y: torch.Tensor,
                                K_XX: torch.Tensor, K_YY: torch.Tensor, K_XY: torch.Tensor) -> torch.Tensor:
    """
    Calculate E[Phi'(X)*Y] in closed form where X and Y are univariate marginals of a bi-variate Gaussian random distribution and
    Phi'() is the density of a standard normal; i.e. the first derivative with respect to X of a standard normal CDF as in Probit Regression 

    Batched Version for faster inference 

    Args:
        m_X (torch.Tensor): Mean of X marginal
        m_Y (torch.Tensor): Mean of Y marginal
        K_XX (torch.Tensor): Variance of X marginal
        K_YY (torch.Tensor): Variance of Y marginal
        K_XY (torch.Tensor): Covariance of X and Y 

    Returns:
        torch.Tensor: Exact values of E[Phi'(X)*Y], for each batch

    IMPORTANT:
        - Does not check for valid covariance matrices
        - Supports arbitrary dimensions for the parameters, but all shapes must match exactly
    """
    gauss_factor = 1 / torch.sqrt(2.*TORCH_PI*(1.+K_XX))
    gauss_exp_factor = torch.exp(-0.5 * m_X**2. / (1+K_XX))
    rem_factor = m_Y - K_XY/K_XX*m_X + K_XY/K_XX*m_X/(1+K_XX)

    return gauss_factor * gauss_exp_factor * rem_factor


    
def gaussian_pdf_product_mean_mc(mu: torch.tensor, cov: torch.tensor, n_samples: int=100000) -> torch.tensor:
    """
    Estimate E[Phi'(X)*Y] via MONTE CARLO INTEGRATION form where X and Y are univariate marginals of a bi-variate Gaussian random distribution and
    Phi'() is the density of a standard normal; i.e. the first derivative with respect to X of a standard normal CDF as in Probit Regression 

    Args:
        mu (torch.tensor): Mean vector of X and Y
        cov (torch.tensor): Covariance matrix of X and Y
        n_samples (int): Number of MC samples

    Returns:
        torch.tensor: MC estimate of E[Phi'(X)*Y]

    IMPORTANT:
        - The last element of the respective bi-variate vector is presumed to be Y in this implementation, so 
            arguments must be passed accordingly
        - Does not check for valid covariance matrix
    """
    mvn = torch.distributions.MultivariateNormal(mu, covariance_matrix=cov)
    
    samp = mvn.sample((n_samples,))

    pdf_vals = torch.exp(-0.5 * samp[:, 0]**2) / torch.sqrt(2 * TORCH_PI)
    result = torch.mean(pdf_vals * samp[:, 1])
    
    return result.item()


### Exponential function E[exp(X)Y] - Any likelihood with positive parameter 

def exp_product_mean(mu: torch.tensor, cov: torch.tensor) -> torch.tensor:
    """
    Calculate E[exp(X)*Y] in closed form where X and Y are univariate marginals of a bi-variate Gaussian random distribution and

    Args:
        mu (torch.tensor): Mean vector of X and Y
        cov (torch.tensor): Covariance matrix of X and Y

    Returns:
        torch.tensor: Exact value of E[exp(X)*Y]

    IMPORTANT:
        - The last element of the respective bi-variate vector is presumed to be Y in this implementation, so 
            arguments must be passed accordingly
        - Does not check for valid covariance matrix

    """
    #Step 1: Extract relevant elements from mu and cov
    m_X = mu[0]
    m_Y = mu[1]

    K_XX = cov[0,0]
    K_YY = cov[1,1]
    K_XY = cov[0,1]

    result = exp_product_mean_batch(m_X, m_Y, K_XX, K_YY, K_XY)

    return result.item()


def exp_product_mean_batch(m_X: torch.Tensor, m_Y: torch.Tensor,
                                K_XX: torch.Tensor, K_YY: torch.Tensor, K_XY: torch.Tensor) -> torch.Tensor:
    """
    Calculate E[exp(X)*Y] in closed form where X and Y are univariate marginals of a bi-variate Gaussian random distribution
    
    Batched Version for faster inference 

    Args:
        m_X (torch.Tensor): Mean of X marginal
        m_Y (torch.Tensor): Mean of Y marginal
        K_XX (torch.Tensor): Variance of X marginal
        K_YY (torch.Tensor): Variance of Y marginal
        K_XY (torch.Tensor): Covariance of X and Y 

    Returns:
        torch.Tensor: Exact values of E[exp(X)*Y], for each batch

    IMPORTANT:
        - Does not check for valid covariance matrices
        - Supports arbitrary dimensions for the parameters, but all shapes must match exactly
    """
    exp_factor = torch.exp(m_X+0.5*K_XX)
    rem_factor = (m_Y - K_XY/K_XX*m_X + K_XY/K_XX * (m_X+K_XX))

    return exp_factor*rem_factor


def exp_product_mean_mc(mu: torch.tensor, cov: torch.tensor, n_samples: int=100000) -> float:
    """
    Estimate E[exp(X)*Y] via MONTE CARLO INTEGRATION form where X and Y are univariate marginals of a bi-variate Gaussian random distribution and

    Args:
        mu (torch.tensor): Mean vector of X and Y
        cov (torch.tensor): Covariance matrix of X and Y
        n_samples (int): Number of MC samples

    Returns:
        torch.tensor: MC estimate of E[exp(X)*Y]

    IMPORTANT:
        - The last element of the respective bi-variate vector is presumed to be Y in this implementation, so 
            arguments must be passed accordingly
        - Does not check for valid covariance matrix
    """
    mvn = torch.distributions.MultivariateNormal(mu, covariance_matrix=cov)
    
    samp = mvn.sample((n_samples,))

    exp_vals = torch.exp(samp[:, 0])
    result = torch.mean(exp_vals * samp[:, 1])
    
    return result.item()



### Derivative of Square inverse link-function 2E[X*Y] - Probit binary classification

def square_deriv_product_mean(mu: torch.tensor, cov: torch.tensor) -> float:
    """
    Calculate 2E[X*Y] in closed form where X and Y are univariate marginals of a bi-variate Gaussian random distribution and

    Args:
        mu (torch.tensor): Mean vector of X and Y
        cov (torch.tensor): Covariance matrix of X and Y

    Returns:
        torch.tensor: Exact value of 2E[X*Y]

    IMPORTANT:
        - The last element of the respective bi-variate vector is presumed to be Y in this implementation, so 
            arguments must be passed accordingly
        - Does not check for valid covariance matrix

    """
    #Step 1: Extract relevant elements from mu and cov
    m_X = mu[0]
    m_Y = mu[1]

    K_XX = cov[0,0]
    K_YY = cov[1,1]
    K_XY = cov[0,1]

    result_torch = square_deriv_product_mean_batch(m_X, m_Y, K_XX, K_YY, K_XY)

    return result_torch.item()


def square_deriv_product_mean_batch(m_X: torch.Tensor, m_Y: torch.Tensor,
                                K_XX: torch.Tensor, K_YY: torch.Tensor, K_XY: torch.Tensor) -> torch.Tensor:
    """
    Calculate 2E[X*Y] in closed form where X and Y are univariate marginals of a bi-variate Gaussian random distribution and

    Batched Version for faster inference 

    Args:
        m_X (torch.Tensor): Mean of X marginal
        m_Y (torch.Tensor): Mean of Y marginal
        K_XX (torch.Tensor): Variance of X marginal
        K_YY (torch.Tensor): Variance of Y marginal
        K_XY (torch.Tensor): Covariance of X and Y 

    Returns:
        torch.Tensor: Exact values of 2E[X*Y], for each batch

    IMPORTANT:
        - Does not check for valid covariance matrices
        - Supports arbitrary dimensions for the parameters, but all shapes must match exactly
    """
    left_summand = 2*m_X*m_Y
    right_summand = 2*K_XY

    return left_summand + right_summand 


def square_deriv_product_mean_mc(mu: torch.tensor, cov: torch.tensor, n_samples: int=100000) -> torch.tensor:
    """
    Estimate 2E[X*Y] via MONTE CARLO INTEGRATION form where X and Y are univariate marginals of a bi-variate Gaussian random distribution and

    Args:
        mu (torch.tensor): Mean vector of X and Y
        cov (torch.tensor): Covariance matrix of X and Y
        n_samples (int): Number of MC samples

    Returns:
        torch.tensor: MC estimate of 2E[X*Y]

    IMPORTANT:
        - The last element of the respective bi-variate vector is presumed to be Y in this implementation, so 
            arguments must be passed accordingly
        - Does not check for valid covariance matrix
    """
    mvn = torch.distributions.MultivariateNormal(mu, covariance_matrix=cov)
    
    samp = mvn.sample((n_samples,))

    result = 2*torch.mean(samp[:, 0] * samp[:, 1])
    
    return result.item()
