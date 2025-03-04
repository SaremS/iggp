import torch
import gc

from gpytorch.models import ApproximateGP

#presume that we run from the root directory
from src.integrated_gradients.gp_posteriors import GPPosteriorsMultiOutput
from src.integrated_gradients.multivariate_outputs import get_gp_posteriors_multivariate_output 

from src.iggp_mean_calculations.approx_softmax_deriv_mean_batch import *
from src.svgp_models.multioutput_svgp import MultiOutputSVGP


def get_iggp_softmax_riemann_batched(model: MultiOutputSVGP, 
                                     eval_point: torch.Tensor,
                                     base_point: torch.Tensor,
                                     target_output_dim: int,
                                     n_riemann_approx_points: int=250,
                                     points_per_batch: int=5) -> torch.Tensor:
    #for larger input dimensionality (images), to avoid OOM errors

    out_dims = len(model.covar_module.outputscale)
    in_dims = len(eval_point)

    eval_point_usq = eval_point.unsqueeze(0)
    baseline_point_usq = base_point.unsqueeze(0)

    riemann_coeffs = torch.arange(1,n_riemann_approx_points+1).reshape(-1,1) / n_riemann_approx_points

    riemann_points = baseline_point_usq + \
                        riemann_coeffs * (eval_point_usq - baseline_point_usq)

    totals = torch.zeros(in_dims)
    
    gc.freeze()
    for i in range(0, n_riemann_approx_points, points_per_batch):
        riemann_batch = riemann_points[i:min(i + points_per_batch, n_riemann_approx_points)]

        gp_posteriors = get_gp_posteriors_multivariate_output(model, riemann_batch, target_output_dim)

        totals += apply_approximation_riemann_batched(gp_posteriors, in_dims, out_dims, target_output_dim)

        gc.collect()

    approximations = totals / n_riemann_approx_points

    return approximations * (eval_point - base_point)



def apply_approximation_riemann_batched(gp_posteriors: GPPosteriorsMultiOutput, in_dims: int, out_dims: int, target_output_dim: int) -> torch.Tensor:
    totals = torch.zeros(in_dims)

    mu_X = gp_posteriors.mu_X
    mu_Y = gp_posteriors.mu_Y

    K_XX = gp_posteriors.K_XX
    K_XY = gp_posteriors.K_XY

    for id in range(in_dims):
        h_c_k = softmax_deriv_c_product_mean_approx_batch(mu_X, mu_Y[:,id,target_output_dim], K_XX, K_XY[:,id,:,target_output_dim], target_output_dim).unsqueeze(0)

        js = [d for d in range(out_dims) if d!=target_output_dim]

        h_cj_ks = [softmax_deriv_cj_product_mean_approx_batch(mu_X, mu_Y[:,id,j], K_XX, K_XY[:,id,:,j], target_output_dim, j).unsqueeze(0) for j in js]

        totals[id] = torch.cat(h_cj_ks + [h_c_k],).sum(0).sum() #sum instead of mean - we calculate the average in the outer function

    return totals 




def get_iggp_softmax(model: MultiOutputSVGP, 
                          eval_point: torch.Tensor,
                          base_point: torch.Tensor,
                          target_output_dim: int,
                          n_riemann_approx_points: int=250) -> torch.Tensor:

    out_dims = len(model.covar_module.outputscale)
    in_dims = len(eval_point)

    eval_point_usq = eval_point.unsqueeze(0)
    baseline_point_usq = base_point.unsqueeze(0)

    riemann_coeffs = torch.arange(1,n_riemann_approx_points+1).reshape(-1,1) / n_riemann_approx_points

    riemann_points = baseline_point_usq + \
                        riemann_coeffs * (eval_point_usq - baseline_point_usq)

    gp_posteriors = get_gp_posteriors_multivariate_output(model, riemann_points, target_output_dim)

    approximations = apply_approximation(gp_posteriors, in_dims, out_dims, target_output_dim)

    return approximations * (eval_point - base_point)


def apply_approximation(gp_posteriors: GPPosteriorsMultiOutput, in_dims: int, out_dims: int, target_output_dim: int) -> torch.Tensor:
    totals = torch.zeros(in_dims)

    mu_X = gp_posteriors.mu_X
    mu_Y = gp_posteriors.mu_Y

    K_XX = gp_posteriors.K_XX
    K_XY = gp_posteriors.K_XY

    for id in range(in_dims):
        h_c_k = softmax_deriv_c_product_mean_approx_batch(mu_X, mu_Y[:,id,target_output_dim], K_XX, K_XY[:,id,:,target_output_dim], target_output_dim).unsqueeze(0)

        js = [d for d in range(out_dims) if d!=target_output_dim]

        h_cj_ks = [softmax_deriv_cj_product_mean_approx_batch(mu_X, mu_Y[:,id,j], K_XX, K_XY[:,id,:,j], target_output_dim, j).unsqueeze(0) for j in js]

        totals[id] = torch.cat(h_cj_ks + [h_c_k],).sum(0).mean()

    return totals 

