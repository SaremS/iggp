import torch
import gpytorch

from gpytorch.likelihoods import _OneDimensionalLikelihood


class SingleOutputSVGP(gpytorch.models.ApproximateGP):
    def __init__(self, train_x: torch.Tensor, train_y: torch.Tensor,
                 likelihood: _OneDimensionalLikelihood, n_inducing = 20):
        variational_distribution = gpytorch.variational.CholeskyVariationalDistribution(n_inducing)
        variational_strategy = gpytorch.variational.VariationalStrategy(
            self, train_x[:n_inducing], variational_distribution, learn_inducing_locations=True
        )
        super(SingleOutputSVGP, self).__init__(variational_strategy)
        self.mean_module = gpytorch.means.ZeroMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
