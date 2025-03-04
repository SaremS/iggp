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
from matplotlib import pyplot as plt
import seaborn as sns

sns.set_style("white")

from src.likelihoods.poisson_likelihood import PoissonLikelihood
from src.evaluate.generate_data import *
from src.evaluate.mean_target_base_differences import *
from src.kernels.rbf_derivative_kernels import *
from src.svgp_models.single_output_svgp import SingleOutputSVGP
from src.kernels.rbf_derivative_kernels import *
from src.iggp_mean_calculations.exact_means import *
from src.integrated_gradients.iggp_univariate import get_iggp_univariate
# -

df_train = pd.read_csv("./data/bike_sharing_demand_train.csv")


def preprocess(df):    
    df_hour_of_day_sin = np.sin(2*np.pi*pd.to_datetime(df["datetime"]).apply(lambda x: x.hour) / 24)
    df_hour_of_day_cos = np.cos(2*np.pi*pd.to_datetime(df["datetime"]).apply(lambda x: x.hour) / 24)
    df_month_sin = np.sin(2*np.pi*pd.to_datetime(df["datetime"]).apply(lambda x: x.month) / 12)
    df_month_cos = np.sin(2*np.pi*pd.to_datetime(df["datetime"]).apply(lambda x: x.month) / 12)

    
    df_dropped = df.drop(["atemp", "season", "weather", "datetime", "casual", "registered", "count"], axis=1)
    X = pd.concat([df_dropped, df_hour_of_day_sin, df_hour_of_day_cos, df_month_sin, df_month_cos], axis=1)
    X.columns = df_dropped.columns.tolist() + ["hour_of_day_sin", "hour_of_day_cos", "month_sin", "month_cos"]

    X_max = X.max(0)
    X = X/X_max

    y = df["count"]

    X_torch = torch.tensor(X.values, dtype=torch.float32)
    y_torch = torch.tensor(y.values, dtype=torch.float32)

    return X_torch, torch.tensor(X_max.values, dtype=torch.float32), y_torch, X.columns.tolist()

x, x_max, y, colnames = preprocess(df_train)

# +
ll = lambda x: torch.distributions.Poisson(torch.exp(x))
n_dims = 1

likelihood = PoissonLikelihood()
model = SingleOutputSVGP(x, y, likelihood, 50)

optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

mll = gpytorch.mlls.VariationalELBO(likelihood, model, y.numel())

np.random.seed(123)
torch.manual_seed(123)
losses = []
for i in range(500):
    samp_idx = np.random.choice(np.arange(len(x)),250)
    optimizer.zero_grad()
    output = model(x[samp_idx,:])
    loss = -mll(output, y[samp_idx])
    loss.backward()
    losses.append(loss.detach().numpy())

    optimizer.step()

base_point = torch.zeros(x.shape[1])
plt.plot(np.array(losses))

# +
i = 100

base_point = x.mean(0)
base_point[0]  = 0 #no holiday
base_point[1]  = 1 #workingday
base_point[-4] = 12 #noon
base_point[-3] = 12 #noon
base_point[-2] = 6 #june
base_point[-1] = 6 #june

base_point_pd = pd.Series((base_point*x_max)[:-4]).astype(object) #retransform for interpretability
base_point_pd.index = colnames[:-4]
base_point_pd["holiday"] = np.int64(base_point_pd["holiday"])
base_point_pd["workingday"] = np.int64(base_point_pd["workingday"])
base_point_pd["temp"] = np.round(np.float64(base_point_pd["temp"]),2)
base_point_pd["humidity"] = np.round(np.float64(base_point_pd["humidity"]),2)
base_point_pd["windspeed"] = np.round(np.float64(base_point_pd["windspeed"]),2)
base_point_pd["hour_of_day"] = np.int64(12)
base_point_pd["month"] = np.int64(6)

base_point[-4] = np.sin(2* np.pi * 12 / 24) #noon
base_point[-3] = np.cos(2* np.pi * 12 / 24) #noon
base_point[-2] = np.sin(2* np.pi * 6 / 12) #june
base_point[-1] = np.sin(2* np.pi * 6 / 12) #june

eval_point = x[i,:]

iggp_full = get_iggp_univariate(model, eval_point, base_point, "exp", 1000).detach().numpy()
iggp = np.concatenate([iggp_full[:-4],np.array(iggp_full[-4:-2].sum()).reshape(-1), np.array(iggp_full[-2:].sum()).reshape(-1)])

xlim = np.max(np.abs(iggp))

target_pd = df_train.iloc[i,:][["holiday", "workingday", "temp", "humidity", "windspeed"]]
target_pd["hour_of_day"] = pd.to_datetime(df_train.iloc[i,:]["datetime"]).hour
target_pd["month"] = pd.to_datetime(df_train.iloc[i,:]["datetime"]).month

df_plot = pd.concat([target_pd.apply(lambda x: f"Target: {x:.2f}"), 
                     base_point_pd.apply(lambda x: f"Baseline: {x:.2f}")], axis=1).reset_index()
df_plot["iggp"] = iggp
df_plot.columns = ["colname", "target_point", "base_point", "iggp"]

plt.close()
fig, ax = plt.subplots(figsize=(20, 10))

bars = ax.barh(df_plot["colname"], df_plot["iggp"], zorder=50, color="blue", height=1.0)
ax.grid(linestyle="dotted")
ax.axvline(0.0, c="black", linestyle="dashed", alpha=0.5)
ax.tick_params(axis="both", which="major", labelsize=15)
ax.set_xlim((-xlim*1.1,xlim*1.1))
ax.set_xlabel("Expected Integrated Gradient", fontsize=25, fontweight="bold")
ax.set_ylabel("Input feature", fontsize=25, fontweight="bold")

plt.yticks(fontsize=25)
plt.xticks(fontsize=25)

plt.savefig(f"../paper/img/experiments/bikeshare_exp1.png", transparent=False, bbox_inches="tight")
plt.close()


# +
i = 3300

base_point = x.mean(0)
base_point[0]  = 0 #no holiday
base_point[1]  = 1 #workingday
base_point[-4] = 12 #noon
base_point[-3] = 12 #noon
base_point[-2] = 6 #june
base_point[-1] = 6 #june

base_point_pd = pd.Series((base_point*x_max)[:-4]).astype(object) #retransform for interpretability
base_point_pd.index = colnames[:-4]
base_point_pd["holiday"] = np.int64(base_point_pd["holiday"])
base_point_pd["workingday"] = np.int64(base_point_pd["workingday"])
base_point_pd["temp"] = np.round(np.float64(base_point_pd["temp"]),2)
base_point_pd["humidity"] = np.round(np.float64(base_point_pd["humidity"]),2)
base_point_pd["windspeed"] = np.round(np.float64(base_point_pd["windspeed"]),2)
base_point_pd["hour_of_day"] = np.int64(12)
base_point_pd["month"] = np.int64(6)

base_point[-4] = np.sin(2* np.pi * 12 / 24) #noon
base_point[-3] = np.cos(2* np.pi * 12 / 24) #noon
base_point[-2] = np.sin(2* np.pi * 6 / 12) #june
base_point[-1] = np.sin(2* np.pi * 6 / 12) #june

eval_point = x[i,:]

iggp_full = get_iggp_univariate(model, eval_point, base_point, "exp", 1000).detach().numpy()
iggp = np.concatenate([iggp_full[:-4],np.array(iggp_full[-4:-2].sum()).reshape(-1), np.array(iggp_full[-2:].sum()).reshape(-1)])


target_pd = df_train.iloc[i,:][["holiday", "workingday", "temp", "humidity", "windspeed"]]
target_pd["hour_of_day"] = pd.to_datetime(df_train.iloc[i,:]["datetime"]).hour
target_pd["month"] = pd.to_datetime(df_train.iloc[i,:]["datetime"]).month

df_plot = pd.concat([target_pd.apply(lambda x: f"Target: {x:.2f}"), 
                     base_point_pd.apply(lambda x: f"Baseline: {x:.2f}")], axis=1).reset_index()
df_plot["iggp"] = iggp
df_plot.columns = ["colname", "target_point", "base_point", "iggp"]


xlim = np.max(np.abs(iggp))

target_pd = df_train.iloc[i,:][["holiday", "workingday", "temp", "humidity", "windspeed"]]
target_pd["hour_of_day"] = pd.to_datetime(df_train.iloc[i,:]["datetime"]).hour
target_pd["month"] = pd.to_datetime(df_train.iloc[i,:]["datetime"]).month

df_plot = pd.concat([target_pd.apply(lambda x: f"Target: {x:.2f}"), 
                     base_point_pd.apply(lambda x: f"Baseline: {x:.2f}")], axis=1).reset_index()
df_plot["iggp"] = iggp
df_plot.columns = ["colname", "target_point", "base_point", "iggp"]


plt.close()
fig, ax = plt.subplots(figsize=(20, 10))

bars = ax.barh(df_plot["colname"], df_plot["iggp"], zorder=50, color="blue", height=1.0)
ax.grid(linestyle="dotted")
ax.axvline(0.0, c="black", linestyle="dashed", alpha=0.5)
ax.tick_params(axis="both", which="major", labelsize=15)
ax.set_xlim((-xlim*1.1,xlim*1.1))
ax.set_xlabel("Expected Integrated Gradient", fontsize=25, fontweight="bold")
ax.set_ylabel("Input feature", fontsize=25, fontweight="bold")

plt.yticks(fontsize=25)
plt.xticks(fontsize=25)

plt.savefig(f"../paper/img/experiments/bikeshare_exp2.png", transparent=False, bbox_inches="tight")
plt.close()

# -




