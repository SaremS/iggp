import torch
import gpytorch
from gpytorch.kernels.rbf_kernel import postprocess_rbf


class RBFKernelDerivative(gpytorch.kernels.Kernel):
    """
    Computes the first kernel derivative of an RBF Kernel; `derivative_dim` determines the input dimension
    with respect to which to differentiate the function to. `derivative_first_dim` determines whether to
    calculate the kernel derivative with respect to the first or second kernel input.
    """
    has_lengthscale = True

    def __init__(self, derivative_dim: int=0, derivative_first_input: bool=True, **kwargs):
        super().__init__(**kwargs)
        
        self.derivative_dim = derivative_dim
        self.derivative_first_input = derivative_first_input

    
    def forward(self, x1: torch.Tensor, x2: torch.Tensor, **params):
        batch_shape = x1.shape[:-2]
        n_batch_dims = len(batch_shape)
        n1, d = x1.shape[-2:]
        n2 = x2.shape[-2]
        
        
        x1_ = x1.div(self.lengthscale)
        x2_ = x2.div(self.lengthscale)
        
        diff = self.covar_dist(x1_, x2_, square_dist=True)
        K_11 = postprocess_rbf(diff)
        
        x1d_ = x1_[:,self.derivative_dim]
        x2d_ = x2_[:,self.derivative_dim]
        lscd = self.lengthscale[0,0]
        
        derivative_adj = (x1d_.unsqueeze(1) - x2d_.unsqueeze(0)) / lscd
        
        K_11_d = -derivative_adj * K_11

        if not self.derivative_first_input:
            return -K_11_d

        return K_11_d


class RBFKernelDerivative2(gpytorch.kernels.Kernel):
    """
    Computes the second kernel derivative of an RBF Kernel with respect to the first and second kernel input element 
    and with respect to the same function input dimension; `derivative_dim` determines the input dimension
    with respect to which to differentiate the function to. 
    """
    has_lengthscale = True 

    def __init__(self, derivative_dim: int=0, **kwargs):
        super().__init__(**kwargs)
        
        self.derivative_dim = derivative_dim

    
    def forward(self, x1: torch.Tensor, x2: torch.Tensor, **params):
        batch_shape = x1.shape[:-2]
        n_batch_dims = len(batch_shape)
        n1, d = x1.shape[-2:]
        n2 = x2.shape[-2]
        
        K = torch.zeros(*batch_shape, n1, n2, device=x1.device, dtype=x1.dtype)
        
        K = torch.zeros(*batch_shape, n1, n2, device=x1.device, dtype=x1.dtype)
        
        x1_ = x1.div(self.lengthscale)
        x2_ = x2.div(self.lengthscale)
        
        diff = self.covar_dist(x1_, x2_, square_dist=True)
        K_11 = postprocess_rbf(diff)
        K[..., :n1, :n2] = K_11
        
        x1d_ = x1_[:,self.derivative_dim]
        x2d_ = x2_[:,self.derivative_dim]
        lscd = self.lengthscale[0, 0]
        
        derivative_adj = 1 / lscd**2 + ((x1d_.unsqueeze(1) - x2d_.unsqueeze(0)) / lscd)**2
        
        K_11_d = derivative_adj * K_11

        return K_11_d
