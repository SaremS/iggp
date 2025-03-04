import gpytorch
import torch

class MultiOutputSVGP(gpytorch.models.ApproximateGP):
    def __init__(self, train_x: torch.Tensor, out_dims: int, n_inducing: int = 20):
        i_shape = train_x[:n_inducing].shape
        inducing_points = torch.ones(out_dims, 1, 1) * train_x[:n_inducing].view(1,n_inducing,i_shape[-1])
        
        variational_distribution = gpytorch.variational.CholeskyVariationalDistribution(
            inducing_points.size(-2), batch_shape=torch.Size([out_dims])
        )
        
        variational_strategy = gpytorch.variational.IndependentMultitaskVariationalStrategy(
            gpytorch.variational.VariationalStrategy(
                 self, inducing_points, variational_distribution, learn_inducing_locations=True,
            ),
            num_tasks=out_dims
         )
        
        # Each latent function has its own mean/kernel function
        super().__init__(variational_strategy)
        self.mean_module = gpytorch.means.ZeroMean(batch_shape=torch.Size([out_dims]))
        self.covar_module = gpytorch.kernels.ScaleKernel(
             gpytorch.kernels.RBFKernel(batch_shape=torch.Size([out_dims])),
             batch_shape=torch.Size([out_dims]),
         )
     
    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
