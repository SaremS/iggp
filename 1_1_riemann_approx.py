# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.16.7
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# +
import torch
import gpytorch
import numpy as np
import pandas as pd

from src.likelihoods.gaussian_variance_sqinvlink_likelihood import GaussianVarianceSqInvLinkLikelihood
from src.likelihoods.poisson_likelihood import PoissonLikelihood
from src.likelihoods.bernoulli_sigmoid_invlink_likelihood import BernoulliSigmoidInvLinkLikelihood

from src.evaluate.generate_data import *
from src.evaluate.mean_target_base_differences import *
from src.kernels.rbf_derivative_kernels import *

from src.svgp_models.single_output_svgp import SingleOutputSVGP
from src.svgp_models.multioutput_svgp import MultiOutputSVGP

from src.kernels.rbf_derivative_kernels import *
from src.iggp_mean_calculations.exact_means import *

from src.integrated_gradients.iggp_univariate import get_iggp_univariate
from src.integrated_gradients.iggp_softmax import get_iggp_softmax
# -

experiments_index = ["Square", "Exponential", "Probit", "Sigmoid (Gauss-Hermite)", "Sigmoid (Taylor)", "Softmax (Taylor)"]
riemann_points = [50, 500, 1000, 5000]
n_dims = 5

result = pd.DataFrame(index=experiments_index, columns = riemann_points)

# # Gaussian Variance - Square inverse link

# +
ll = lambda x: torch.distributions.Normal(0,x**2)

x,y = generate_data(ll, 500, n_dims, manual_seed=42)

likelihood = GaussianVarianceSqInvLinkLikelihood()
model = SingleOutputSVGP(x, y, likelihood, 50)

optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

mll = gpytorch.mlls.VariationalELBO(likelihood, model, y.numel())

for i in range(100):
    optimizer.zero_grad()
    output = model(x)
    loss = -mll(output, y)
    loss.backward()

    optimizer.step()

draws = {k: [] for k in riemann_points}

base_point = torch.zeros(x.shape[1])

n_samples = 500

np.random.seed(42)
for r in riemann_points:
    idx = np.random.choice(np.arange(len(x)),50,replace=False)
    for i in idx:
        eval_point = x[i,:]
    
        with torch.no_grad():
            iggp = get_iggp_univariate(model, eval_point, base_point, "square_deriv", r).sum()
            mean_target_base_diff = square_mtb_difference(model, eval_point.unsqueeze(0), base_point.unsqueeze(0))
    
        abs_diff = torch.abs(iggp-mean_target_base_diff) 
        
        draws[r].append(abs_diff)


for k, v in draws.items():
    result.loc["Square",k] = np.mean(v)
# -

# # Poisson - Exponential inverse link

# +
ll = lambda x: torch.distributions.Poisson(torch.exp(x))

x,y = generate_data(ll, 500, n_dims, manual_seed=42)

likelihood = PoissonLikelihood()
model = SingleOutputSVGP(x, y, likelihood, 50)

optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

mll = gpytorch.mlls.VariationalELBO(likelihood, model, y.numel())

for i in range(100):
    optimizer.zero_grad()
    output = model(x)
    loss = -mll(output, y)
    loss.backward()

    optimizer.step()

draws = {k: [] for k in riemann_points}

base_point = torch.zeros(x.shape[1])

n_samples = 500

np.random.seed(42)
for r in riemann_points:
    idx = np.random.choice(np.arange(len(x)),50,replace=False)
    for i in idx:
        eval_point = x[i,:]
    
        with torch.no_grad():
            iggp = get_iggp_univariate(model, eval_point, base_point, "exp", r).sum()
            mean_target_base_diff = exp_mtb_difference(model, eval_point.unsqueeze(0), base_point.unsqueeze(0))
    
        abs_diff = torch.abs(iggp-mean_target_base_diff) 
        
        draws[r].append(abs_diff)

for k, v in draws.items():
    result.loc["Exponential",k] = np.mean(v)
# -

# # Bernoulli - Probit inverse link

# +
ll = lambda x: torch.distributions.Bernoulli(torch.sigmoid(x))

x,y = generate_data(ll, 500, n_dims, manual_seed=42)

likelihood = BernoulliLikelihood()
model = SingleOutputSVGP(x, y, likelihood, 50)

optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

mll = gpytorch.mlls.VariationalELBO(likelihood, model, y.numel())

for i in range(100):
    optimizer.zero_grad()
    output = model(x)
    loss = -mll(output, y)
    loss.backward()

    optimizer.step()

draws = {k: [] for k in riemann_points}

base_point = torch.zeros(x.shape[1])

n_samples = 500

np.random.seed(42)
for r in riemann_points:
    idx = np.random.choice(np.arange(len(x)),50,replace=False)
    for i in idx:
        eval_point = x[i,:]
    
        with torch.no_grad():
            iggp = get_iggp_univariate(model, eval_point, base_point, "gaussian_pdf", r).sum()
            mean_target_base_diff = probit_mtb_difference(model, eval_point.unsqueeze(0), base_point.unsqueeze(0))
    
        abs_diff = torch.abs(iggp-mean_target_base_diff) 
        
        draws[r].append(abs_diff)


for k, v in draws.items():
    result.loc["Probit",k] = np.mean(v)
# -

# # Bernoulli - Sigmoid inverse link (Gauss-Hermite approx.)

# +
ll = lambda x: torch.distributions.Bernoulli(torch.sigmoid(x))

x,y = generate_data(ll, 500, n_dims, manual_seed=42)

likelihood = BernoulliSigmoidInvLinkLikelihood()
model = SingleOutputSVGP(x, y, likelihood, 50)

optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

mll = gpytorch.mlls.VariationalELBO(likelihood, model, y.numel())

for i in range(100):
    optimizer.zero_grad()
    output = model(x)
    loss = -mll(output, y)
    loss.backward()

    optimizer.step()

draws = {k: [] for k in riemann_points}

base_point = torch.zeros(x.shape[1])

n_samples = 500

np.random.seed(42)
for r in riemann_points:
    idx = np.random.choice(np.arange(len(x)),50,replace=False)
    for i in idx:
        eval_point = x[i,:]
    
        with torch.no_grad():
            iggp = get_iggp_univariate(model, eval_point, base_point, "sigmoid_deriv_quadrature", r).sum()
            mean_target_base_diff = sigmoid_mtb_difference(model, eval_point.unsqueeze(0), base_point.unsqueeze(0))
    
        abs_diff = torch.abs(iggp-mean_target_base_diff) 
        
        draws[r].append(abs_diff)

for k, v in draws.items():
    result.loc["Sigmoid (Gauss-Hermite)",k] = np.mean(v)
# -

# # Bernoulli - Sigmoid inverse link (Taylor approx.)

# +
n_samples = 500

np.random.seed(42)
for r in riemann_points:
    idx = np.random.choice(np.arange(len(x)),50,replace=False)
    for i in idx:
        eval_point = x[i,:]
    
        with torch.no_grad():
            iggp = get_iggp_univariate(model, eval_point, base_point, "sigmoid_deriv_taylor", r).sum()
            mean_target_base_diff = sigmoid_mtb_difference(model, eval_point.unsqueeze(0), base_point.unsqueeze(0))
    
        abs_diff = torch.abs(iggp-mean_target_base_diff) 
        
        draws[r].append(abs_diff)


for k, v in draws.items():
    result.loc["Sigmoid (Taylor)",k] = np.mean(v)
# -

# # Multinomial / Categorical - Softmax inverse link (Taylor approx.)

# +
ll = lambda x: torch.distributions.Categorical(torch.softmax(x,0))
out_dims = 5

x,y = generate_n_dimensional_data(ll, 500, n_dims, out_dims, manual_seed=42)

likelihood = gpytorch.likelihoods.SoftmaxLikelihood(num_features=out_dims, num_classes=out_dims, mixing_weights=None)
model = MultiOutputSVGP(x, out_dims, 50)

optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

mll = gpytorch.mlls.VariationalELBO(likelihood, model, y.numel())

for i in range(100):
    optimizer.zero_grad()
    output = model(x)
    loss = -mll(output, y)
    loss.backward()

    optimizer.step()

draws = {k: [] for k in riemann_points}

base_point = torch.zeros(x.shape[1])

n_samples = 100

np.random.seed(42)
for r in riemann_points:
    for od in range(out_dims): 
        idx = np.random.choice(np.arange(len(x)),n_samples,replace=False)
        for i in idx:
            eval_point = x[i,:]
        
            with torch.no_grad():
                iggp = get_iggp_softmax(model, eval_point, base_point, od, r).sum()
                mean_target_base_diff = softmax_mtb_difference(model, eval_point.unsqueeze(0), base_point.unsqueeze(0), od)
        
            abs_diff = torch.abs(iggp-mean_target_base_diff) 
            
            draws[r].append(abs_diff)


for k, v in draws.items():
    result.loc["Softmax (Taylor)",k] = np.mean(v)
# -

with open("../paper/other_exports/riemann_approx_table.tex", "w") as f:
    result.map(lambda x: f"{x:.4f}").to_latex(f)
