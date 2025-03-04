import torch
import numpy as np

def softmax_deriv_c_product_mean_approx(mu: torch.Tensor, cov: torch.Tensor, c: int) -> float:
    """
    Approximates the E[g_c(X)*Y] target mean through a multivariate, second order Taylor expansion;
    here g_c() denotes the derivative of the softmax function with respect to the c-th output

    Args:
        mu (torch.Tensor): Mean vector of X and Y
        cov (torch.Tensor): Covariance matrix of X and Y
        c: Target output element with respect to which the softmax derivative should be calculated

    Returns:
        float: The approximated mean 

    IMPORTANT:
        - The last element of the respective bi-variate vector is presumed to be Y in this implementation, so 
            arguments must be passed accordingly
        - Does not check for valid covariance matrix
    """
    m_X = mu[:-1]
    m_Y = mu[-1]

    K_XX = cov[:-1,:-1]

    scx_d = softmax_c_derivative(m_X, c)

    zero_order_summand = scx_d * m_Y #zero-order taylor summand 

    hessian = softmax_target_fun_hessian(mu, cov, m_X, c)
    hess_diag = hessian.diagonal()
    K_XX_diag = K_XX.diagonal()

    second_order_summand = 0.5 * torch.sum(hess_diag * K_XX_diag) #second-order taylor summand

    return (zero_order_summand + second_order_summand).item()


def softmax_deriv_cj_product_mean_approx(mu: torch.Tensor, cov: torch.Tensor, c: int, j: int) -> float:
    """
    Approximates the E[g_cj(X)*Y] target mean through a multivariate, second order Taylor expansion;
    here g_cj() denotes the derivative of the softmax function with respect to the c-th output
    and the j-th input

    Args:
        mu (torch.Tensor): Mean vector of X and Y
        cov (torch.Tensor): Covariance matrix of X and Y
        c: Target output element with respect to which the softmax derivative should be calculated
        j: Target input element with respect to which the softmax derivative should be calculated


    Returns:
        float: The approximated mean 

    IMPORTANT:
        - The last element of the respective bi-variate vector is presumed to be Y in this implementation, so 
            arguments must be passed accordingly
        - Does not check for valid covariance matrix
    """
    m_X = mu[:-1]
    m_Y = mu[-1]

    K_XX = cov[:-1,:-1]

    scjx_d = softmax_cj_derivative(m_X, c, j)

    zero_order_summand = scjx_d * m_Y #zero-order taylor summand 

    hessian = softmax_target_fun_cj_hessian(mu, cov, m_X, c, j)
    hess_diag = hessian.diagonal()
    K_XX_diag = K_XX.diagonal()

    second_order_summand = 0.5 * torch.sum(hess_diag * K_XX_diag) #second-order taylor summand

    return (zero_order_summand + second_order_summand).item()


def softmax_deriv_c_product_mean_mc(mu: torch.Tensor, cov: torch.Tensor, c: int, n_samples: int=1000000) -> torch.Tensor:
    """
    Estimates the E[g_c(X)*Y] target mean through Monte Carlo Integration;
    here g_c() denotes the derivative of the softmax function with respect to the c-th output

    Args:
        mu (torch.Tensor): Mean vector of X and Y
        cov (torch.Tensor): Covariance matrix of X and Y
        c (int): Target output element with respect to which the softmax derivative should be calculated

    Returns:
        float: The estimated mean 

    IMPORTANT:
        - The last element of the respective bi-variate vector is presumed to be Y in this implementation, so 
            arguments must be passed accordingly
        - Does not check for valid covariance matrix
    """
    mvn = torch.distributions.MultivariateNormal(mu, covariance_matrix=cov)
    samp = mvn.sample((n_samples,))

    scx = torch.softmax(samp[:,:-1], 1)[:,c]
    scx_derivative = scx * (1. - scx)

    return torch.mean(scx_derivative * samp[:,-1]).item()


def softmax_deriv_cj_product_mean_mc(mu: torch.Tensor, cov: torch.Tensor, c: int, j: int, n_samples: int=1000000) -> torch.Tensor:
    """
    Estimates the E[g_cj(X)*Y] target mean through Monte Carlo Integration;
    here g_cj() denotes the derivative of the softmax function with respect to the c-th output and the j-th input

    Args:
        mu (torch.Tensor): Mean vector of X and Y
        cov (torch.Tensor): Covariance matrix of X and Y
        c (int): Target output element with respect to which the softmax derivative should be calculated
        j (int): Target input element with respect to which the softmax derivative should be calculated


    Returns:
        float: The estimated mean 

    IMPORTANT:
        - The last element of the respective bi-variate vector is presumed to be Y in this implementation, so 
            arguments must be passed accordingly
        - Does not check for valid covariance matrix
    """
    mvn = torch.distributions.MultivariateNormal(mu, covariance_matrix=cov)
    samp = mvn.sample((n_samples,))

    scx = torch.softmax(samp[:,:-1], 1)[:,c]
    sjx = torch.softmax(samp[:,:-1], 1)[:,j]

    return torch.mean(-scx * sjx * samp[:,-1]).item()


def softmax_c_derivative(x: torch.Tensor, c: int) -> torch.Tensor:
    """
    Evaluate the derivative of the softmax function with respect to c at the c-th element

    Args:
        x (torch.Tensor): Target input at which to evaluate the derivative
        c (int): Output with respect to which to take the derivative

    Returns:
        torch.tensor: c-th element of the gradient of the softmax with respect to the c-th output
    """
    scx = torch.softmax(x, 0)[c]
    return scx * (1-scx)


def softmax_cj_derivative(x: torch.Tensor, c: int, j: int) -> torch.Tensor:
    """
    Evaluate the derivative of the softmax function with respect to the c-th output
    and the j-th input at the j-th element

    Args:
        x (torch.Tensor): Target input at which to evaluate the derivative
        c (int): Output with respect to which to take the derivative
        j (int): Input with respect to which to take the derivative

    Returns:
        torch.tensor: j-th element of the gradient of the softmax with respect to the c-th output
                        and the j-th input
    """
    scx = torch.softmax(x, 0)[c]
    sjx = torch.softmax(x, 0)[j]

    return -scx*sjx 

def softmax_target_fun(mu: torch.Tensor, cov: torch.Tensor, x: torch.Tensor, c: int) -> torch.Tensor:
    """
    Evaluates the g_c(x)E[Y|X=x] target function at x; here g_c() denotes the derivative of the softmax function
    with respect to the c-th output

    Args:
        mu (torch.tensor): Mean vector of X and Y
        cov (torch.tensor): Covariance matrix of X and Y
        x: Vector to evaluate the target function at
        c: Target output element with respect to which the softmax derivative should be calculated

    Returns:
        torch.Tensor: The evaluation of g_c(x)E[Y|X=x]

    IMPORTANT:
        - The last element of the respective bi-variate vector is presumed to be Y in this implementation, so 
            arguments must be passed accordingly
        - Does not check for valid covariance matrix
    """
    mu_X = mu[:-1]
    mu_Y = mu[-1]

    K_XY = cov[-1,:-1]
    K_XX = cov[:-1,:-1]

    K = K_XY@torch.linalg.inv(K_XX)

    scx = torch.softmax(x,0)[c]

    return scx*mu_Y - scx**2 * mu_Y + K@x*scx - K@x*scx**2 - K@mu_X*scx + K@mu_X*scx**2 


def softmax_target_fun_cj(mu: torch.Tensor, cov: torch.Tensor, x: torch.Tensor, c: int, j: int) -> torch.Tensor:
    """
    Evaluates the g_cj(x)E[Y|X=x] target function at x; here g_cj() denotes the derivative of the softmax function
    with respect to the c-th output and the j-th input

    Args:
        mu (torch.tensor): Mean vector of X and Y
        cov (torch.tensor): Covariance matrix of X and Y
        x: Vector to evaluate the target function at
        c: Target output element with respect to which the softmax derivative should be calculated
        j: Target input element with respect to which the softmax derivative should be calculated

    Returns:
        torch.Tensor: The evaluation of g_cj(x)E[Y|X=x]

    IMPORTANT:
        - The last element of the respective bi-variate vector is presumed to be Y in this implementation, so 
            arguments must be passed accordingly
        - Does not check for valid covariance matrix
    """
    mu_X = mu[:-1]
    mu_Y = mu[-1]

    K_XY = cov[-1,:-1]
    K_XX = cov[:-1,:-1]

    K = K_XY@torch.linalg.inv(K_XX)

    scx = torch.softmax(x,0)[c]
    sjx = torch.softmax(x,0)[j]

    return - scx*sjx*mu_Y - K@x*scx*sjx + K@mu_X*scx*sjx


def softmax_target_fun_gradient(mu: torch.Tensor, cov: torch.Tensor, x: torch.Tensor, c: int) -> torch.Tensor:
    """
    Evaluates the gradient of the g_c(x)E[Y|X=x] target function at x;
    here g_c() denotes the derivative of the softmax function with respect to the c-th output

    Args:
        mu (torch.Tensor): Mean vector of X and Y
        cov (torch.Tensor): Covariance matrix of X and Y
        x (torch.Tensor): Vector to evaluate the target function at
        c (int): Target output element with respect to which the softmax derivative should be calculated

    Returns:
        torch.Tensor: The evaluation of the gradient of g_c(x)E[Y|X=x]

    IMPORTANT:
        - The last element of the respective bi-variate vector is presumed to be Y in this implementation, so 
            arguments must be passed accordingly
        - Does not check for valid covariance matrix
    """
    mu_X = mu[:-1]
    mu_Y = mu[-1]

    K_XY = cov[-1,:-1]
    K_XX = cov[:-1,:-1]

    K = K_XY@torch.linalg.inv(K_XX)
    
    sx = torch.softmax(x,0)
    scx = sx[c]

    scx_grad = softmax_c_gradient(x, c)

    return scx_grad*mu_Y - 2*scx*scx_grad*mu_Y \
            + K*scx + K@x*scx_grad - K*scx**2 - K@x*2*scx*scx_grad \
            - K@mu_X*scx_grad + K@mu_X*2*scx*scx_grad


def softmax_target_fun_cj_gradient(mu: torch.Tensor, cov: torch.Tensor, x: torch.Tensor, c: int, j: int) -> torch.Tensor:
    """
    Evaluates the gradient of the g_cj(x)E[Y|X=x] target function at x;
    here g_c() denotes the derivative of the softmax function with respect to the c-th output and j-th input

    Args:
        mu (torch.Tensor): Mean vector of X and Y
        cov (torch.Tensor): Covariance matrix of X and Y
        x (torch.Tensor): Vector to evaluate the target function at
        c (int): Target output element with respect to which the softmax derivative should be calculated
        j (int): Target input element with respect to which the softmax derivative should be calculated

    Returns:
        torch.Tensor: The evaluation of the gradient of g_c(x)E[Y|X=x]

    IMPORTANT:
        - The last element of the respective bi-variate vector is presumed to be Y in this implementation, so 
            arguments must be passed accordingly
        - Does not check for valid covariance matrix
    """
    mu_X = mu[:-1]
    mu_Y = mu[-1]

    K_XY = cov[-1,:-1]
    K_XX = cov[:-1,:-1]

    K = K_XY@torch.linalg.inv(K_XX)
    
    sx = torch.softmax(x,0)
    scx = sx[c]
    sjx = sx[j]

    scx_grad = softmax_c_gradient(x, c)
    sjx_grad = softmax_c_gradient(x, j)
    
    return  - scx_grad*sjx*mu_Y - scx*sjx_grad*mu_Y \
            - scx_grad*sjx*(K@x) - scx*sjx_grad*(K@x) - scx*sjx*K \
            + scx_grad*sjx*(K@mu_X) + scx*sjx_grad*(K@mu_X)


def softmax_target_fun_hessian(mu: torch.Tensor, cov: torch.Tensor, x: torch.Tensor, c: int) -> torch.Tensor:
    """
    Evaluates the hessian of the g_c(x)E[Y|X=x] target function at x;
    here g_c() denotes the derivative of the softmax function with respect to the c-th output

    Args:
        mu (torch.Tensor): Mean vector of X and Y
        cov (torch.Tensor): Covariance matrix of X and Y
        x (torch.Tensor): Vector to evaluate the target function at
        c (torch.Tensor): Target output element with respect to which the softmax derivative should be calculated

    Returns:
        torch.Tensor: The evaluation of the hessian of g_c(x)E[Y|X=x]

    IMPORTANT:
        - The last element of the respective bi-variate vector is presumed to be Y in this implementation, so 
            arguments must be passed accordingly
        - Does not check for valid covariance matrix
    """
    N = len(x)
    mu_X = mu[:-1]
    mu_Y = mu[-1]

    K_XY = cov[-1,:-1]
    K_XX = cov[:-1,:-1]

    K = K_XY@torch.linalg.inv(K_XX)
    Km = K.reshape(1,-1)

    sx = torch.softmax(x,0)
    scx = sx[c]

    scx_grad = softmax_c_gradient(x, c).reshape(-1,1)
    scx_hess = softmax_c_hessian(x, c)


    return scx_hess*mu_Y - 2*scx_grad@scx_grad.T*mu_Y - 2*scx*scx_hess*mu_Y \
            + Km.T@scx_grad.T + scx_hess*(Km@x) + scx_grad@Km - 2*scx*Km.T@scx_grad.T \
            - 2*scx*scx_grad@Km - 2*(Km@x)*scx_grad@scx_grad.T - 2*(Km@x)*scx*scx_hess \
            - (Km@mu_X)*scx_hess + 2*(Km@mu_X)*scx_grad@scx_grad.T + 2*(Km@mu_X)*scx*scx_hess


def softmax_target_fun_cj_hessian(mu: torch.Tensor, cov: torch.Tensor, x: torch.Tensor, c: int, j: int) -> torch.Tensor:
    """
    Evaluates the hessian of the g_cj(x)E[Y|X=x] target function at x;
    here g_cj() denotes the derivative of the softmax function with respect to the c-th output and j-th input

    Args:
        mu (torch.Tensor): Mean vector of X and Y
        cov (torch.Tensor): Covariance matrix of X and Y
        x (torch.Tensor): Vector to evaluate the target function at
        c (int): Target output element with respect to which the softmax derivative should be calculated
        j (int): Target input element with respect to which the softmax derivative should be calculated

    Returns:
        torch.Tensor: The evaluation of the hessian of g_cj(x)E[Y|X=x]

    IMPORTANT:
        - The last element of the respective bi-variate vector is presumed to be Y in this implementation, so 
            arguments must be passed accordingly
        - Does not check for valid covariance matrix
    """
    N = len(x)
    mu_X = mu[:-1]
    mu_Y = mu[-1]

    K_XY = cov[-1,:-1]
    K_XX = cov[:-1,:-1]

    K = K_XY@torch.linalg.inv(K_XX)
    Km = K.reshape(1,-1)

    sx = torch.softmax(x,0)

    scx = sx[c]
    sjx = sx[j]

    scx_grad = softmax_c_gradient(x, c).reshape(-1,1)
    scx_hess = softmax_c_hessian(x, c)
    sjx_grad = softmax_c_gradient(x, j).reshape(-1,1)
    sjx_hess = softmax_c_hessian(x, j)


    return - (scx_grad@sjx_grad.T)*mu_Y - scx_hess*sjx*mu_Y - (sjx_grad@scx_grad.T)*mu_Y - scx*sjx_hess*mu_Y \
            - scx_hess*sjx*(K@x) - (scx_grad@sjx_grad.T)*(K@x) - sjx*(scx_grad@Km) \
            - scx*sjx_hess*(K@x) - (sjx_grad@scx_grad.T)*(K@x) - scx*(sjx_grad@Km) \
            - sjx*(Km.T@scx_grad.T) - scx*(Km.T@sjx_grad.T) \
            + scx_hess*sjx*(K@mu_X) + scx_grad@sjx_grad.T*(K@mu_X) \
            + sjx_grad@scx_grad.T*(K@mu_X) + scx*sjx_hess*(K@mu_X)


def softmax_c_gradient(x: torch.Tensor, c: int) -> torch.Tensor:
    """
    Evaluates the gradient of the softmax with respect to the c-th element at x.

    Args:
        x (torch.Tensor): Vector to evaluate the softmax gradient at
        c (torch.Tensor): Target output element with respect to which the gradient should be calculated

    Returns:
        torch.Tensor: Gradient of softmax with respect to c-th output, evaluated at x
    """

    sx = torch.softmax(x,0)
    scx = sx[c]
    
    scx_grad = -scx*sx
    scx_grad[c] += scx

    return scx_grad


def softmax_c_hessian(x: torch.Tensor, c: int) -> torch.Tensor:
    """
    Evaluates the hessian of the softmax with respect to the c-th element at x.

    Args:
        x (torch.Tensor): Vector to evaluate the softmax hessian at
        c (int): Target output element with respect to which the hessian should be calculated

    Returns:
        torch.Tensor: Hessian of softmax with respect to c-th output, evaluated at x   
    """
    sx = torch.softmax(x,0)
    scx = sx[c]
    scx_grad = -sx[c]*sx
    scx_grad[c] += sx[c]

    sx_col = sx.reshape(-1,1)
    sx_row = sx.reshape(1,-1)

    hess_base = 2*sx_col * sx_row * scx
    hess_base[c,c] += scx 

    hess_base[c,:] -= scx * sx
    hess_base[:,c] -= scx * sx

    hess_base.diagonal().copy_(hess_base.diagonal() - scx*sx)

    return hess_base
