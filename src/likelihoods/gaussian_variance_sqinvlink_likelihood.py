from typing import Any

import torch
from gpytorch.likelihoods import _OneDimensionalLikelihood


class GaussianVarianceSqInvLinkLikelihood(_OneDimensionalLikelihood):
    has_analytic_marginal: bool = False 

    def __init__(self) -> None:
        super().__init__()

    def forward(self, function_samples: torch.Tensor, *args: Any, **kwargs: Any) -> torch.distributions.Normal:
        variance = function_samples**2
        return torch.distributions.Normal(loc=0, scale=torch.sqrt(variance))
