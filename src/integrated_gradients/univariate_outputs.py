import torch
import gpytorch
from gpytorch.models import ApproximateGP
import linear_operator

#presume that we run from the root directory
from src.integrated_gradients.gp_posteriors import GPPosteriors 
from src.kernels.rbf_derivative_kernels import *


def get_gp_posteriors_univariate_output(model: ApproximateGP, riemann_points: torch.Tensor) -> GPPosteriors:
    if not isinstance(model, ApproximateGP):
        raise AssertionError("get_gp_posteriors_single_output only works with gpytorch.models.ApproximateGP models")
    
    if len(riemann_points.shape) != 2:
        raise AssertionError("Eval points needs to be a 2D matrix")
    
    n_obs, n_dims = riemann_points.shape

    #initialize output containers
    post_mus = torch.zeros(n_obs, n_dims)
    post_sigsqs = torch.zeros(n_obs, n_dims)
    post_dx_mus = torch.zeros(n_obs, n_dims)
    post_dx_sigsqs = torch.zeros(n_obs, n_dims)
    post_dxx_covs = torch.zeros(n_obs, n_dims)

    #extract kernel parameters to apply to derivative kernels
    kern = model.covar_module
    kern_outputscale = model.covar_module.outputscale.detach()
    kern_lengthscale = model.covar_module.base_kernel.lengthscale.detach()

    #extract variational distribution parameters
    v_mean = model.variational_strategy._variational_distribution._parameters["variational_mean"].detach()
    v_chol_cov = model.variational_strategy._variational_distribution._parameters["chol_variational_covar"].detach()
    v_id_points = model.variational_strategy.inducing_points.detach()

    #calculate posterior distributions
    Lambda = kern(v_id_points).add_jitter(0.0001).cholesky().inverse()@kern(v_id_points, riemann_points)

    id_operator = linear_operator.operators.IdentityLinearOperator(len(v_id_points))
    post_inner_cov = id_operator - v_chol_cov@v_chol_cov.T

    riemann_points_dist = model(riemann_points)
    post_mus += riemann_points_dist.loc.reshape(-1,1)
    post_sigsqs += (riemann_points_dist.stddev**2).reshape(-1,1)

    #derivative GP
    for d in range(n_dims):

        #cross-derivative kernel (=covariance between observation and derivative GP) w.r.t. second input
        dxx_kern = gpytorch.kernels.ScaleKernel(RBFKernelDerivative(derivative_dim=d, derivative_first_input=True))
        dxx_kern.outputscale = kern_outputscale
        dxx_kern.base_kernel.lengthscale = kern_lengthscale

        #cross-derivative kernel (=covariance between observation and derivative GP) w.r.t. second input
        xdx_kern = gpytorch.kernels.ScaleKernel(RBFKernelDerivative(derivative_dim=d, derivative_first_input=False))
        xdx_kern.outputscale = kern_outputscale
        xdx_kern.base_kernel.lengthscale = kern_lengthscale
    
        #derivative kernel (=variance of derivative GP (=variance of derivative GP) 
        dxdx_kern = gpytorch.kernels.ScaleKernel(RBFKernelDerivative2(derivative_dim=d))
        dxdx_kern.outputscale = kern_outputscale
        dxdx_kern.base_kernel.lengthscale = kern_lengthscale
        
        Lambda_dxx = dxx_kern(riemann_points,v_id_points)@kern(v_id_points).add_jitter(0.0001).cholesky().inverse().T
        Lambda_xdx = kern(v_id_points).add_jitter(0.0001).cholesky().inverse()@xdx_kern(v_id_points, riemann_points)
        Lambda_dxdx = kern(v_id_points).add_jitter(0.0001).cholesky().inverse()@dxdx_kern(riemann_points, v_id_points).T
    
        K_dxx = dxx_kern(riemann_points)
        K_xdx = xdx_kern(riemann_points)
        K_dxdx = dxdx_kern(riemann_points)
    
        post_dx_mus[:,d] = Lambda_dxx@v_mean
        post_dx_sigsqs[:,d] = torch.linalg.diagonal((K_dxdx - Lambda_dxx@(post_inner_cov)@Lambda_xdx).to_dense())
        post_dxx_covs[:,d] = torch.linalg.diagonal((K_dxx - Lambda_dxx@(post_inner_cov)@Lambda).to_dense())

    return GPPosteriors(post_mus, post_dx_mus, post_sigsqs, post_dx_sigsqs, post_dxx_covs)

