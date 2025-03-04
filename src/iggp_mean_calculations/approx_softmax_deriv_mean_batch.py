import torch
import numpy as np

def softmax_deriv_c_product_mean_approx_batch(mu_X: torch.Tensor, mu_Y: torch.Tensor,
                             K_XX: torch.Tensor, K_XY: torch.Tensor,
                             c: int) -> torch.Tensor:
    """
    Approximates the E[g_c(X)*Y] target mean through a multivariate, second order Taylor expansion;
    here g_c() denotes the derivative of the softmax function with respect to the c-th output

    Use on batched input for faster inference (batch dimension = dim 0)

    Args:
        mu_X (torch.Tensor): Mean vector batches of X 
        mu_Y (torch.Tensor): Mean batches of Y
        K_XX (torch.Tensor): Covariance matrix batches of X
        K_XY (torch.Tensor): Covariance vector batches of X and Y
        x (torch.Tensor): Vector batches to evaluate the target function at
        c (torch.Tensor): Target output element with respect to which the softmax derivative should be calculated

    Returns:
        torch.Tensor: Batches of approximated means

    IMPORTANT:
        - The last element of the respective bi-variate vector is presumed to be Y in this implementation, so 
            arguments must be passed accordingly
        - Does not check for valid covariance matrix
    """
    scx_d = softmax_c_derivative_batch(mu_X, c)

    zero_order_summand = scx_d[:,0] * mu_Y #zero-order taylor summand 

    hessian = softmax_target_fun_hessian_batch(mu_X, mu_Y, K_XX, K_XY, mu_X, c)
    hess_diag = hessian.diagonal(dim1=1, dim2=2)
    K_XX_diag = K_XX.diagonal(dim1=1,dim2=2)

    second_order_summand = 0.5 * torch.sum(hess_diag * K_XX_diag, dim=1) #second-order taylor summand

    return (zero_order_summand + second_order_summand)


def softmax_deriv_cj_product_mean_approx_batch(mu_X: torch.Tensor, mu_Y: torch.Tensor,
                             K_XX: torch.Tensor, K_XY: torch.Tensor,
                             c: int, j: int) -> torch.Tensor:
    """
    Approximates the E[g_cj(X)*Y] target mean through a multivariate, second order Taylor expansion;
    here g_cj() denotes the derivative of the softmax function with respect to the c-th output and the
    j-th input.

    Use on batched input for faster inference (batch dimension = dim 0)

    Args:
        mu_X (torch.Tensor): Mean vector batches of X 
        mu_Y (torch.Tensor): Mean batches of Y
        K_XX (torch.Tensor): Covariance matrix batches of X
        K_XY (torch.Tensor): Covariance vector batches of X and Y
        x (torch.Tensor): Vector batches to evaluate the target function at
        c (int): Target output element with respect to which the softmax derivative should be calculated
        j (int): Target input element with respect to which the softmax derivative should be calculated


    Returns:
        torch.Tensor: Batches of approximated means

    IMPORTANT:
        - Does not check for valid covariance matrix
    """
    scjx_d = softmax_cj_derivative_batch(mu_X, c, j)

    zero_order_summand = scjx_d[:,0] * mu_Y #zero-order taylor summand 

    hessian = softmax_target_fun_cj_hessian_batch(mu_X, mu_Y, K_XX, K_XY, mu_X, c, j)
    hess_diag = hessian.diagonal(dim1=1, dim2=2)
    K_XX_diag = K_XX.diagonal(dim1=1,dim2=2)

    second_order_summand = 0.5 * torch.sum(hess_diag * K_XX_diag, dim=1) #second-order taylor summand

    return (zero_order_summand + second_order_summand)


def softmax_c_derivative_batch(x: torch.Tensor, c: int) -> torch.Tensor:
    """
    Evaluate the derivative of the softmax function with respect to c at the c-th element

    Use on batched input for faster inference (batch dimension = dim 0)

    Args:
        x (torch.Tensor): Target input at which to evaluate the derivative
        c (int): Output with respect to which to take the derivative

    Returns:
        torch.tensor: Batch of c-th elements of the gradient of the softmax with respect to the c-th output
    """
    scx = torch.softmax(x, 1)[:,c].unsqueeze(-1)
    return scx * (1-scx)


def softmax_cj_derivative_batch(x: torch.Tensor, c: int, j: int) -> torch.Tensor:
    """
    Evaluate the derivative of the softmax function with respect to c at the c-th output element and the j-th input element

    Use on batched input for faster inference (batch dimension = dim 0)

    Args:
        x (torch.Tensor): Target input at which to evaluate the derivative
        c (int): Output with respect to which to take the derivative
        j (int): Input with respect to which to take the derivative

    Returns:
        torch.tensor: Batch of c-th elements of the gradient of the softmax with respect to the j-th input 
    """
    scx = torch.softmax(x, 1)[:,c].unsqueeze(-1)
    sjx = torch.softmax(x, 1)[:,j].unsqueeze(-1)

    return -scx*sjx 


def softmax_target_fun_batch(mu_X: torch.Tensor, mu_Y: torch.Tensor,
                             K_XX: torch.Tensor, K_XY: torch.Tensor,
                             x: torch.Tensor, c: int) -> torch.Tensor:
    """
    Evaluates the g_c(x)E[Y|X=x] target function at x; here g_c() denotes the derivative of the softmax function
    with respect to the c-th output

    Use on batched input for faster inference (batch dimension = dim 0)

    Args:
        mu_X (torch.Tensor): Mean vector batches of X 
        mu_Y (torch.Tensor): Mean batches of Y
        K_XX (torch.Tensor): Covariance matrix batches of X
        K_XY (torch.Tensor): Covariance vector batches of X and Y
        x (torch.Tensor): Vector batches to evaluate the target function at
        c (torch.Tensor): Target output element with respect to which the softmax derivative should be calculated

    Returns:
        torch.Tensor: The evaluation batches of g_c(x)E[Y|X=x]

    IMPORTANT:
        - Does not check for valid covariance matrix
    """
    K_XXi = torch.inverse(K_XX) #torch inverse computes batch-wise on BxNxN matrices
    K = torch.einsum("bm,bnm->bn", K_XY, K_XXi)

    scx = torch.softmax(x,1)[:,c]

    return scx*mu_Y - scx**2*mu_Y + (K*x).sum(1)*scx - (K*x).sum(1)*scx**2 - (K*mu_X).sum(1)*scx + (K*mu_X).sum(1)*scx**2 


def softmax_target_fun_cj_batch(mu_X: torch.Tensor, mu_Y: torch.Tensor,
                             K_XX: torch.Tensor, K_XY: torch.Tensor,
                                x: torch.Tensor, c: int, j: int) -> torch.Tensor:
    """
    Evaluates the g_cj(x)E[Y|X=x] target function at x; here g_cj() denotes the derivative of the softmax function
    with respect to the c-th output and the j-th input

    Use on batched input for faster inference (batch dimension = dim 0)

    Args:
        mu_X (torch.Tensor): Mean vector batches of X 
        mu_Y (torch.Tensor): Mean batches of Y
        K_XX (torch.Tensor): Covariance matrix batches of X
        K_XY (torch.Tensor): Covariance vector batches of X and Y
        x (torch.Tensor): Vector batches to evaluate the target function at
        c (int): Target output element with respect to which the softmax derivative should be calculated
        j (int): Target input element with respect to which the softmax derivative should be calculated

    Returns:
        torch.Tensor: The evaluation batches of g_cj(x)E[Y|X=x]

    IMPORTANT:
        - Does not check for valid covariance matrix
    """
    K_XXi = torch.inverse(K_XX) #torch inverse computes batch-wise on BxNxN matrices
    K = torch.einsum("bm,bnm->bn", K_XY, K_XXi)

    scx = torch.softmax(x,1)[:,c]
    sjx = torch.softmax(x,1)[:,j]

    return - scx*sjx*mu_Y - (K*x).sum(1)*scx*sjx + (K*mu_X).sum(1)*scx*sjx




def softmax_target_fun_gradient_batch(mu_X: torch.Tensor, mu_Y: torch.Tensor,
                             K_XX: torch.Tensor, K_XY: torch.Tensor,
                             x: torch.Tensor, c: int) -> torch.Tensor:
    """
    Evaluates the gradient of the g_c(x)E[Y|X=x] target function at x;
    here g_c() denotes the derivative of the softmax function with respect to the c-th output

    Use on batched input for faster inference (batch dimension = dim 0)

    Args:
        mu_X (torch.Tensor): Mean vector batches of X 
        mu_Y (torch.Tensor): Mean batches of Y
        K_XX (torch.Tensor): Covariance matrix batches of X
        K_XY (torch.Tensor): Covariance vector batches of X and Y
        x (torch.Tensor): Vector batches to evaluate the target function at
        c (torch.Tensor): Target output element with respect to which the softmax derivative should be calculated

    Returns:
        torch.Tensor: The evaluation batches of the gradient of g_c(x)E[Y|X=x]

    IMPORTANT:
        - Does not check for valid covariance matrix
    """
    K_XXi = torch.inverse(K_XX) #torch inverse computes batch-wise on BxNxN matrices
    K = torch.einsum("bm,bnm->bn", K_XY, K_XXi)

    scx = torch.softmax(x,1)[:,c].unsqueeze(-1)

    scx_grad = softmax_c_gradient_batch(x, c)
    
    mu_Y_usq = mu_Y.unsqueeze(-1)

    return scx_grad*mu_Y_usq - 2*scx*scx_grad*mu_Y_usq \
            + K*scx + (K*x).sum(1).unsqueeze(-1)*scx_grad - K*scx**2 - (K*x).sum(1).unsqueeze(-1)*2*scx*scx_grad \
            - (K*mu_X).sum(1).unsqueeze(-1)*scx_grad + (K*mu_X).sum(1).unsqueeze(-1)*2*scx*scx_grad


def softmax_target_fun_cj_gradient_batch(mu_X: torch.Tensor, mu_Y: torch.Tensor,
                                         K_XX: torch.Tensor, K_XY: torch.Tensor,
                                         x: torch.Tensor, c: int, j: int) -> torch.Tensor:
    """
    Evaluates the gradient of the g_cj(x)E[Y|X=x] target function at x;
    here g_cj() denotes the derivative of the softmax function with respect to the c-th output and
    the j-th input

    Use on batched input for faster inference (batch dimension = dim 0)

    Args:
        mu_X (torch.Tensor): Mean vector batches of X 
        mu_Y (torch.Tensor): Mean batches of Y
        K_XX (torch.Tensor): Covariance matrix batches of X
        K_XY (torch.Tensor): Covariance vector batches of X and Y
        x (torch.Tensor): Vector batches to evaluate the target function at
        c (int): Target output element with respect to which the softmax derivative should be calculated
        j (int): Target input element with respect to which the softmax derivative should be calculated

    Returns:
        torch.Tensor: The evaluation batches of the gradient of g_cj(x)E[Y|X=x]

    IMPORTANT:
        - Does not check for valid covariance matrix
    """
    K_XXi = torch.inverse(K_XX) #torch inverse computes batch-wise on BxNxN matrices
    K = torch.einsum("bm,bnm->bn", K_XY, K_XXi)

    scx = torch.softmax(x,1)[:,c].unsqueeze(-1)
    sjx = torch.softmax(x,1)[:,j].unsqueeze(-1)

    scx_grad = softmax_c_gradient_batch(x, c)
    sjx_grad = softmax_c_gradient_batch(x, j)

    mu_Y_usq = mu_Y.unsqueeze(-1)

    Kx_usq = (K*x).sum(1).unsqueeze(-1)
    KmuX_usq = (K*mu_X).sum(1).unsqueeze(-1)

    return - scx_grad*sjx*mu_Y_usq - scx*sjx_grad*mu_Y_usq \
            - scx_grad*sjx*Kx_usq - scx*sjx_grad*Kx_usq - scx*sjx*K \
            + scx_grad*sjx*KmuX_usq + scx*sjx_grad*KmuX_usq


def softmax_target_fun_hessian_batch(mu_X: torch.Tensor, mu_Y: torch.Tensor,
                             K_XX: torch.Tensor, K_XY: torch.Tensor,
                             x: torch.Tensor, c: int) -> torch.Tensor:
    """
    Evaluates the hessian of the g_c(x)E[Y|X=x] target function at x;
    here g_c() denotes the derivative of the softmax function with respect to the c-th output

    Use on batched input for faster inference (batch dimension = dim 0)

    Args:
        mu_X (torch.Tensor): Mean vector batches of X 
        mu_Y (torch.Tensor): Mean batches of Y
        K_XX (torch.Tensor): Covariance matrix batches of X
        K_XY (torch.Tensor): Covariance vector batches of X and Y
        x (torch.Tensor): Vector batches to evaluate the target function at
        c (torch.Tensor): Target output element with respect to which the softmax derivative should be calculated


    Returns:
        torch.Tensor: The evaluation of the hessian of g_c(x)E[Y|X=x]

    IMPORTANT:
        - The last element of the respective bi-variate vector is presumed to be Y in this implementation, so 
            arguments must be passed accordingly
        - Does not check for valid covariance matrix
    """
    K_XXi = torch.inverse(K_XX) #torch inverse computes batch-wise on BxNxN matrices
    K = torch.einsum("bm,bnm->bn", K_XY, K_XXi)

    scx = torch.softmax(x,1)[:,c].unsqueeze(-1)
    scx_usq = scx.unsqueeze(-1)

    scx_grad = softmax_c_gradient_batch(x, c)
    scx_hess = softmax_c_hessian_batch(x, c)
    
    mu_Y_usq = mu_Y.unsqueeze(-1)
    mu_Y_usq2 = mu_Y_usq.unsqueeze(-1)

    Kx_sum_usq2 = (K*x).sum(1).unsqueeze(-1).unsqueeze(-1)
    KmuX_sum_usq2 = (K*mu_X).sum(1).unsqueeze(-1).unsqueeze(-1)

    return scx_hess*mu_Y_usq2 - 2*torch.einsum("bj,bi->bij",scx_grad,scx_grad)*mu_Y_usq2 \
            - 2*scx.unsqueeze(-1)*scx_hess*mu_Y_usq2 + torch.einsum("bj,bi->bji",K,scx_grad) \
            + scx_hess*Kx_sum_usq2 \
            + torch.einsum("bj,bi->bji",scx_grad,K) \
            - 2*scx.unsqueeze(-1)*torch.einsum("bj,bi->bji",K,scx_grad) \
            - 2*scx.unsqueeze(-1)*torch.einsum("bj,bi->bji",scx_grad,K) \
            - 2*Kx_sum_usq2*torch.einsum("bi,bj->bij",scx_grad,scx_grad) \
            - 2*Kx_sum_usq2*scx_usq*scx_hess \
            - KmuX_sum_usq2*scx_hess \
            + 2*KmuX_sum_usq2*torch.einsum("bj,bi->bji",scx_grad,scx_grad) \
            + 2*KmuX_sum_usq2*scx_usq*scx_hess


def softmax_target_fun_cj_hessian_batch(mu_X: torch.Tensor, mu_Y: torch.Tensor,
                             K_XX: torch.Tensor, K_XY: torch.Tensor,
                             x: torch.Tensor, c: int, j: int) -> torch.Tensor:
    """
    Evaluates the hessian of the g_cj(x)E[Y|X=x] target function at x;
    here g_cj() denotes the derivative of the softmax function with respect to the c-th output and the
    j-th input

    Use on batched input for faster inference (batch dimension = dim 0)

    Args:
        mu_X (torch.Tensor): Mean vector batches of X 
        mu_Y (torch.Tensor): Mean batches of Y
        K_XX (torch.Tensor): Covariance matrix batches of X
        K_XY (torch.Tensor): Covariance vector batches of X and Y
        x (torch.Tensor): Vector batches to evaluate the target function at
        c (int): Target output element with respect to which the softmax derivative should be calculated
        j (int): Target input element with respect to which the softmax derivative should be calculated


    Returns:
        torch.Tensor: The evaluation batches of the hessian of g_cj(x)E[Y|X=x]

    IMPORTANT:
        - Does not check for valid covariance matrix
    """
    K_XXi = torch.inverse(K_XX) #torch inverse computes batch-wise on BxNxN matrices
    K = torch.einsum("bm,bnm->bn", K_XY, K_XXi)

    scx = torch.softmax(x,1)[:,c].unsqueeze(-1)
    scx_usq = scx.unsqueeze(-1)

    sjx = torch.softmax(x,1)[:,j].unsqueeze(-1)
    sjx_usq = sjx.unsqueeze(-1)

    scx_grad = softmax_c_gradient_batch(x, c)
    scx_hess = softmax_c_hessian_batch(x, c)

    sjx_grad = softmax_c_gradient_batch(x, j)
    sjx_hess = softmax_c_hessian_batch(x, j)
    
    mu_Y_usq = mu_Y.unsqueeze(-1)
    mu_Y_usq2 = mu_Y_usq.unsqueeze(-1)

    Kx_sum_usq2 = (K*x).sum(1).unsqueeze(-1).unsqueeze(-1)
    KmuX_sum_usq2 = (K*mu_X).sum(1).unsqueeze(-1).unsqueeze(-1)

    scjx_gT = torch.einsum("bj,bi->bij",scx_grad,sjx_grad)
    sjcx_gT = torch.einsum("bj,bi->bij",sjx_grad,scx_grad)

    return  - scjx_gT*mu_Y_usq2 - scx_hess*sjx_usq*mu_Y_usq2 \
            - sjcx_gT*mu_Y_usq2 - scx_usq*sjx_hess*mu_Y_usq2 \
            - scx_hess*sjx_usq*Kx_sum_usq2 \
            - scjx_gT*Kx_sum_usq2 \
            - sjx_usq*torch.einsum("bj,bi->bij",K,scx_grad) \
            - scx_usq*sjx_hess*Kx_sum_usq2 \
            - sjcx_gT*Kx_sum_usq2 \
            - scx_usq*torch.einsum("bj,bi->bij",K,sjx_grad) \
            - sjx_usq*torch.einsum("bj,bi->bij",scx_grad,K) \
            - scx_usq*torch.einsum("bj,bi->bij",sjx_grad,K) \
            + scx_hess*sjx_usq*KmuX_sum_usq2 + scjx_gT*KmuX_sum_usq2 \
            + sjcx_gT*KmuX_sum_usq2 + scx_usq*sjx_hess*KmuX_sum_usq2



def softmax_c_gradient_batch(x: torch.Tensor, c: int) -> torch.Tensor:
    """
    Evaluates the gradient of the softmax with respect to the c-th element at x.

    Use on batched input for faster inference (batch dimension = dim 0)

    Args:
        x (torch.Tensor): Vector batches to evaluate the softmax gradient at
        c (torch.Tensor): Target output element with respect to which the gradient should be calculated

    Returns:
        torch.Tensor: Gradient batches of softmax with respect to c-th output, evaluated at x
    """

    sx = torch.softmax(x,1)
    scx = sx[:,c].reshape(-1,1)
    
    scx_grad = -scx*sx
    scx_grad[:,c] += scx[:,0]

    return scx_grad


def softmax_c_hessian_batch(x: torch.Tensor, c: int) -> torch.Tensor:
    """
    Evaluates the hessian of the softmax with respect to the c-th element at x.

    Use on batched input for faster inference (batch dimension = dim 0)

    Args:
        x (torch.Tensor): Vector to evaluate the softmax hessian at
        c (int): Target output element with respect to which the hessian should be calculated

    Returns:
        torch.Tensor: Hessian batches of softmax with respect to c-th output, evaluated at x   
    """
    n_batches, n_elements = x.shape 

    sx = torch.softmax(x,1)
    scx = sx[:,c].unsqueeze(-1)
    scx_grad = -scx*sx

    scx_grad[:,c] += scx[:,0]

    sx_col = sx.unsqueeze(-1)
    sx_row = sx.unsqueeze(-2)

    hess_base = 2*sx_col * sx_row * scx.unsqueeze(-1)
    hess_base[:,c,c] += scx[:,0]

    hess_base[:,c,:] -= scx * sx
    hess_base[:,:,c] -= scx * sx

    hess_base.diagonal(dim1=1, dim2=2).copy_(hess_base.diagonal(dim1=1, dim2=2) - scx*sx)

    return hess_base
