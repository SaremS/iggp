import torch
from gpytorch.models import ApproximateGP

#presume that we run from the root directory
from src.integrated_gradients.gp_posteriors import GPPosteriors
from src.integrated_gradients.univariate_outputs import get_gp_posteriors_univariate_output 

from src.iggp_mean_calculations.exact_means import \
        gaussian_pdf_product_mean_batch, \
        exp_product_mean_batch, \
        square_deriv_product_mean_batch

from src.iggp_mean_calculations.approx_sigmoid_deriv_mean import \
        sigmoid_deriv_product_mean_approx_batch, \
        sigmoid_deriv_product_mean_quadrature_batch


def get_iggp_univariate(model: ApproximateGP,
                        eval_point: torch.Tensor,
                        baseline_point: torch.Tensor,
                        link_approximation: str,
                        n_riemann_approx_points: int=250) -> torch.Tensor:
    
    valid_links = ["square_deriv", "exp", "gaussian_pdf",
                   "sigmoid_deriv_taylor", "sigmoid_deriv_quadrature"]
    
    if not link_approximation in valid_links:
        raise AssertionError('''link_approximation must be one of "square_eriv", 
                                "exp", "gaussian_pdf", "sigmoid_deriv_taylor",
                                "sigmoid_deriv_quadrature"''')

    eval_point_usq = eval_point.unsqueeze(0)
    baseline_point_usq = baseline_point.unsqueeze(0)

    riemann_coeffs = torch.arange(1,n_riemann_approx_points+1).reshape(-1,1) / n_riemann_approx_points

    riemann_points = baseline_point_usq + \
                        riemann_coeffs * (eval_point_usq - baseline_point_usq)

    gp_posteriors = get_gp_posteriors_univariate_output(model, riemann_points)

    approximations = apply_approximation(gp_posteriors, link_approximation)

    appr_mean = torch.mean(approximations, 0)

    return (eval_point-baseline_point) * appr_mean


def apply_approximation(gp_posteriors: GPPosteriors, link_approximation: str) -> torch.Tensor:

    gp_mus = gp_posteriors.gp_mus
    gp_dx_mus = gp_posteriors.gp_dx_mus

    gp_sigma_sqs = gp_posteriors.gp_sigma_sqs
    gp_dx_sigma_sqs = gp_posteriors.gp_dx_sigma_sqs

    gp_xdx_cross_covs = gp_posteriors.gp_xdx_cross_covs


    if link_approximation == "square_deriv":
        return square_deriv_product_mean_batch(gp_mus, gp_dx_mus,
                                               gp_sigma_sqs, gp_dx_sigma_sqs,
                                               gp_xdx_cross_covs)

    elif link_approximation == "exp":
        return exp_product_mean_batch(gp_mus, gp_dx_mus,
                                      gp_sigma_sqs, gp_dx_sigma_sqs,
                                      gp_xdx_cross_covs)

    elif link_approximation == "gaussian_pdf":
        return gaussian_pdf_product_mean_batch(gp_mus, gp_dx_mus,
                                               gp_sigma_sqs, gp_dx_sigma_sqs,
                                               gp_xdx_cross_covs)
 
    elif link_approximation == "sigmoid_deriv_taylor":
        return sigmoid_deriv_product_mean_approx_batch(gp_mus, gp_dx_mus,
                                                       gp_sigma_sqs, gp_dx_sigma_sqs,
                                                       gp_xdx_cross_covs)

    elif link_approximation == "sigmoid_deriv_quadrature":
        return sigmoid_deriv_product_mean_quadrature_batch(gp_mus, gp_dx_mus,
                                                           gp_sigma_sqs, gp_dx_sigma_sqs,
                                                           gp_xdx_cross_covs)


