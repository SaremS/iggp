from typing import Any

import torch
from gpytorch.likelihoods import _OneDimensionalLikelihood


class BernoulliSigmoidInvLinkLikelihood(_OneDimensionalLikelihood):
    has_analytic_marginal: bool = False 

    def __init__(self) -> None:
        super().__init__()

    def forward(self, function_samples: torch.Tensor, *args: Any, **kwargs: Any) -> torch.distributions.Bernoulli:
        output_probs = torch.sigmoid(function_samples)
        return torch.distributions.Bernoulli(probs=output_probs)
