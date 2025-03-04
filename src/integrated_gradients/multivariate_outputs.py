import torch
import gpytorch
from gpytorch.models import ApproximateGP
import linear_operator

#presume that we run from the root directory
from src.integrated_gradients.gp_posteriors import GPPosteriorsMultiOutput
from src.kernels.rbf_derivative_kernels import *


def get_gp_posteriors_multivariate_output(model: ApproximateGP, riemann_points: torch.Tensor, target_output_dim: int) -> GPPosteriorsMultiOutput:
    if not isinstance(model, ApproximateGP):
        raise AssertionError("get_gp_posteriors_single_output only works with gpytorch.models.ApproximateGP models")
    
    if len(riemann_points.shape) != 2:
        raise AssertionError("Eval points needs to be a 2D matrix")

    n_riemann_appr_points, in_dims = riemann_points.shape
    out_dims = len(model.covar_module.outputscale)

    mu_X = torch.zeros(n_riemann_appr_points, out_dims)
    mu_Y = torch.zeros(n_riemann_appr_points, in_dims, out_dims)

    K_XX = torch.zeros(n_riemann_appr_points, out_dims, out_dims)
    K_XY = torch.zeros(n_riemann_appr_points, in_dims, out_dims, out_dims)


    for od in range(out_dims):

        fullkern = model.covar_module
        fullkern_outputscale = model.covar_module.outputscale.detach()
        fullkern_lengthscale = model.covar_module.base_kernel.lengthscale.detach()

        kern_outputscale = fullkern_outputscale[od]
        kern_lengthscale = fullkern_lengthscale[od,:,:]
        
        kern = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())
        kern.outputscale = kern_outputscale
        kern.base_kernel.lengthscale = kern_lengthscale

        v_mean= model.variational_strategy.base_variational_strategy._variational_distribution._parameters["variational_mean"].detach()[od,:]
        v_chol_cov = model.variational_strategy.base_variational_strategy._variational_distribution._parameters["chol_variational_covar"].detach()[od,:,:]
        v_id_points = model.variational_strategy.base_variational_strategy.inducing_points.detach()[od,:,:]

        Lambda = kern(v_id_points).add_jitter(0.0001).cholesky().inverse()@kern(v_id_points, riemann_points)

        id_operator = linear_operator.operators.IdentityLinearOperator(len(v_id_points))
        post_inner_cov = id_operator - v_chol_cov@v_chol_cov.T

        riemann_points_dist = model(riemann_points)
        mu_X[:,od] = riemann_points_dist.mean[:,od]
        K_XX[:,od,od] = (riemann_points_dist.stddev[:,od]**2)

        for d in range(in_dims):

            dxx_kern = gpytorch.kernels.ScaleKernel(RBFKernelDerivative(derivative_dim=d, derivative_first_input=True))
            dxx_kern.outputscale = kern_outputscale
            dxx_kern.base_kernel.lengthscale = kern_lengthscale
        
            xdx_kern = gpytorch.kernels.ScaleKernel(RBFKernelDerivative(derivative_dim=d, derivative_first_input=False))
            xdx_kern.outputscale = kern_outputscale
            xdx_kern.base_kernel.lengthscale = kern_lengthscale
        
            dxdx_kern = gpytorch.kernels.ScaleKernel(RBFKernelDerivative2(derivative_dim=d))
            dxdx_kern.outputscale = kern_outputscale
            dxdx_kern.base_kernel.lengthscale = kern_lengthscale

            Lambda_dxx = dxx_kern(riemann_points, v_id_points)@kern(v_id_points).add_jitter(0.0001).cholesky().inverse().T
            Lambda_xdx = kern(v_id_points).add_jitter(0.0001).cholesky().inverse()@xdx_kern(v_id_points, riemann_points)
            Lambda_dxdx = kern(v_id_points).add_jitter(0.0001).cholesky().inverse()@dxdx_kern(riemann_points, v_id_points).T

            K_dxx = dxx_kern(riemann_points)
            K_xdx = xdx_kern(riemann_points)
            K_dxdx = dxdx_kern(riemann_points)

            mu_Y[:,d,od] = Lambda_dxx@v_mean
            K_XY[:,d,od,od] = torch.linalg.diagonal((K_dxx - Lambda_dxx@(post_inner_cov)@Lambda).to_dense())

    return GPPosteriorsMultiOutput(mu_X, mu_Y, K_XX, K_XY)
