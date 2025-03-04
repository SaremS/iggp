import torch
from gpytorch.utils.quadrature import GaussHermiteQuadrature1D
from gpytorch.likelihoods import BernoulliLikelihood

from src.svgp_models.single_output_svgp import SingleOutputSVGP
from src.svgp_models.multioutput_svgp import MultiOutputSVGP



def square_mtb_difference(model: SingleOutputSVGP, eval_point: torch.Tensor, base_point: torch.Tensor) -> torch.Tensor:
    with torch.no_grad():
        dist_target = model(eval_point)
        mu_target = dist_target.loc
        sigsq_target = dist_target.stddev**2
        dist_square_mean = mu_target**2 + sigsq_target 

        dist_base = model(base_point)
        mu_base = dist_base.loc
        sigsq_base = dist_base.stddev**2
        base_square_mean = mu_base**2 + sigsq_base 

    return dist_square_mean - base_square_mean


def exp_mtb_difference(model: SingleOutputSVGP, eval_point: torch.Tensor, base_point: torch.Tensor) -> torch.Tensor:
    with torch.no_grad():
        dist_target = model(eval_point)
        mu_target = dist_target.loc
        sigsq_target = dist_target.stddev**2
        dist_exp_mean = torch.exp(mu_target + sigsq_target/2)

        dist_base = model(base_point)
        mu_base = dist_base.loc
        sigsq_base = dist_base.stddev**2
        base_exp_mean = torch.exp(mu_base + sigsq_base/2)

    return dist_exp_mean - base_exp_mean


def probit_mtb_difference(model: SingleOutputSVGP, eval_point: torch.Tensor, base_point: torch.Tensor) -> torch.Tensor:
    likelihood = BernoulliLikelihood()
    with torch.no_grad():
        result = torch.mean(likelihood(model(eval_point)).mean) - torch.mean(likelihood(model(base_point)).mean)

    return result


def sigmoid_mtb_difference(model: SingleOutputSVGP, eval_point: torch.Tensor, base_point: torch.Tensor) -> torch.Tensor:
    with torch.no_grad():
        dist_target = model(eval_point)
        dist_base = model(base_point)

        ghq = GaussHermiteQuadrature1D(50)

        result = ghq.forward(lambda x: torch.sigmoid(x), dist_target) - ghq.forward(lambda x: torch.sigmoid(x), dist_base)
    
    return result


def softmax_mtb_difference(model: MultiOutputSVGP, 
                           eval_point: torch.Tensor,
                           base_point: torch.Tensor,
                           target_class: int,
                           size_mc_sample: int = int(1e7),
                           manual_seed: int = 42) -> torch.Tensor:

    torch.manual_seed(manual_seed)

    eval_output = torch.mean(torch.softmax(model(eval_point).sample(torch.Size([size_mc_sample])),2)[:,0,:][:,target_class])
    base_output = torch.mean(torch.softmax(model(base_point).sample(torch.Size([size_mc_sample])),2)[:,0,:][:,target_class])

    return eval_output - base_output





