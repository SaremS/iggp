from typing import Any

import torch
from gpytorch.likelihoods import _OneDimensionalLikelihood


class PoissonLikelihood(_OneDimensionalLikelihood):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, function_samples: torch.Tensor, *args: Any, **kwargs: Any) -> torch.distributions.Poisson:
        output_rates = torch.exp(function_samples)
        return torch.distributions.Poisson(rate=output_rates)
