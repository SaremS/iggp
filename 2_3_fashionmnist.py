#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
import gpytorch
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns

sns.set_style("white")

import torchvision
import torchvision.transforms as transforms

from src.evaluate.generate_data import *
from src.evaluate.mean_target_base_differences import *
from src.svgp_models.multioutput_svgp import MultiOutputSVGP
from src.integrated_gradients.iggp_softmax import get_iggp_softmax, get_iggp_softmax_riemann_batched


# In[2]:


transform = transforms.Compose([transforms.ToTensor()])
train_dataset = torchvision.datasets.FashionMNIST(root="./data", train=True, transform=transform, download=True)


# In[3]:


x = (train_dataset.data / 255).view(len(train_dataset), -1)
y = train_dataset.targets


# In[ ]:


print("Training model")


# In[4]:


out_dims = 10

likelihood = gpytorch.likelihoods.SoftmaxLikelihood(num_features=out_dims, num_classes=out_dims, mixing_weights=None)
model = MultiOutputSVGP(x, out_dims, 100)

optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

mll = gpytorch.mlls.VariationalELBO(likelihood, model, y.numel())


np.random.seed(123)
torch.manual_seed(123)
losses = []
for i in range(1000):
    samp_idx = np.random.choice(np.arange(len(x)),250)
    optimizer.zero_grad()
    output = model(x[samp_idx,:])
    loss = -mll(output, y[samp_idx])
    loss.backward()
    losses.append(loss.detach().numpy())

    optimizer.step()

plt.plot(np.array(losses))


# In[5]:


N_RIEMANN_POINTS = 100
RIEMANN_BATCH_SIZE = 5

N_ABLATION_SAMPLES = 1000
N_MC_LIKELIHOOD_SAMPLES = 2500


# # Target = 0

# In[6]:


target_class = 0
print(f"Target: {target_class}")


eval_point = x[y==target_class,:][0]
base_point = torch.zeros(x.shape[1])

with torch.no_grad():
    iggp = get_iggp_softmax_riemann_batched(model, eval_point, base_point, target_class, 
                                            N_RIEMANN_POINTS, RIEMANN_BATCH_SIZE).detach().numpy()

vm = np.max(np.abs(iggp))
plt.close()
cmap = plt.cm.seismic 
plt.imshow(iggp.reshape(28,28), cmap=cmap, vmin=-vm, vmax=vm)
plt.xticks([])
plt.yticks([])
plt.savefig(f"../paper/img/experiments/fashionmnist_{target_class}_image.png", transparent=False)
plt.close()

# In[7]:


theo_diff = softmax_mtb_difference(model, eval_point.unsqueeze(0), torch.zeros(len(eval_point)).unsqueeze(0), target_class)

argsort_iggp = np.argsort(iggp).tolist()[::-1]
sort_iggp = (np.array(argsort_iggp)[eval_point[argsort_iggp] > 0]).tolist()

neq_zero_pixels = np.where(eval_point > 0)[0]

maxk = len(neq_zero_pixels)

torch.manual_seed(123)
np.random.seed(123)
with gpytorch.settings.num_likelihood_samples(N_MC_LIKELIHOOD_SAMPLES):
    eval_point_prob = likelihood(model(eval_point.reshape(1,-1))).probs.mean(0)[:,target_class]

prob_diffs = torch.empty(maxk+1, 3)
prob_diffs[:] = float('nan')

with torch.no_grad():
    for k in range(maxk+1):
        #remove pixels with positive attribution
        eval_points_abl = eval_point.reshape(1,-1) * 1
        eval_points_abl[0,sort_iggp[:k]] = 0
    
        with gpytorch.settings.num_likelihood_samples(N_MC_LIKELIHOOD_SAMPLES):
            abl_points_prob = (likelihood(model(eval_points_abl)).probs.mean(0)[:,target_class])

        prob_diffs[k,0] = (eval_point_prob - abl_points_prob)


        #random points
        eval_points_abl = eval_point.reshape(1,-1) * torch.ones(N_ABLATION_SAMPLES,1)

        rand_idx = np.concatenate(
            [np.random.choice(neq_zero_pixels, (k,), replace = False).reshape(1,-1) for _ in range(N_ABLATION_SAMPLES)],
            axis=0
        )

        eval_points_abl[torch.arange(N_ABLATION_SAMPLES).unsqueeze(1), rand_idx] = 0

        with gpytorch.settings.num_likelihood_samples(N_MC_LIKELIHOOD_SAMPLES):
            abl_points_prob = (likelihood(model(eval_points_abl)).probs.mean(0)[:,target_class])
    
        prob_diffs[k,2] = (eval_point_prob - abl_points_prob).mean()


# In[8]:
n_pos_iggps = (iggp > 0).sum()

plt.close()
plt.plot(prob_diffs.numpy()[:,0], c="blue",lw=2.5)
plt.axhline(theo_diff.numpy(), linestyle="dashed", c="red",lw=2.5)
plt.axvline(n_pos_iggps, linestyle="dashdot", c="black",lw=2.5)
plt.plot(prob_diffs.numpy()[:,2], c="green",lw=2.5)
plt.axhline(0, linestyle="dotted", c="black")
plt.xlim((0,maxk))
plt.yticks(fontsize=20)
plt.xticks(fontsize=20)
plt.gca().xaxis.set_major_locator(ticker.MaxNLocator(nbins=5))
plt.xlabel("# Pixels replaced by baseline", fontsize=20, fontweight="bold")
plt.ylabel("Class Prob. difference", fontsize=20, fontweight="bold")
plt.savefig(f"../paper/img/experiments/fashionmnist_{target_class}_chart.png", transparent=False, bbox_inches="tight")
plt.close()
# # Target = 1

# In[9]:


target_class = 1
print(f"Target: {target_class}")


eval_point = x[y==target_class,:][0]
base_point = torch.zeros(x.shape[1])

with torch.no_grad():
    iggp = get_iggp_softmax_riemann_batched(model, eval_point, base_point, target_class, 
                                            N_RIEMANN_POINTS, RIEMANN_BATCH_SIZE).detach().numpy()

vm = np.max(np.abs(iggp))
plt.close()
cmap = plt.cm.seismic 
plt.imshow(iggp.reshape(28,28), cmap=cmap, vmin=-vm, vmax=vm)
plt.xticks([])
plt.yticks([])
plt.savefig(f"../paper/img/experiments/fashionmnist_{target_class}_image.png", transparent=False)
plt.close()

# In[10]:


theo_diff = softmax_mtb_difference(model, eval_point.unsqueeze(0), torch.zeros(len(eval_point)).unsqueeze(0), target_class)

argsort_iggp = np.argsort(iggp).tolist()[::-1]
sort_iggp = (np.array(argsort_iggp)[eval_point[argsort_iggp] > 0]).tolist()

neq_zero_pixels = np.where(eval_point > 0)[0]

maxk = len(neq_zero_pixels)

torch.manual_seed(123)
np.random.seed(123)
with gpytorch.settings.num_likelihood_samples(N_MC_LIKELIHOOD_SAMPLES):
    eval_point_prob = likelihood(model(eval_point.reshape(1,-1))).probs.mean(0)[:,target_class]

prob_diffs = torch.empty(maxk+1, 3)
prob_diffs[:] = float('nan')

with torch.no_grad():
    for k in range(maxk+1):
        #remove pixels with positive attribution
        eval_points_abl = eval_point.reshape(1,-1) * 1
        eval_points_abl[0,sort_iggp[:k]] = 0
    
        with gpytorch.settings.num_likelihood_samples(N_MC_LIKELIHOOD_SAMPLES):
            abl_points_prob = (likelihood(model(eval_points_abl)).probs.mean(0)[:,target_class])

        prob_diffs[k,0] = (eval_point_prob - abl_points_prob)


        #random points
        eval_points_abl = eval_point.reshape(1,-1) * torch.ones(N_ABLATION_SAMPLES,1)

        rand_idx = np.concatenate(
            [np.random.choice(neq_zero_pixels, (k,), replace = False).reshape(1,-1) for _ in range(N_ABLATION_SAMPLES)],
            axis=0
        )

        eval_points_abl[torch.arange(N_ABLATION_SAMPLES).unsqueeze(1), rand_idx] = 0

        with gpytorch.settings.num_likelihood_samples(N_MC_LIKELIHOOD_SAMPLES):
            abl_points_prob = (likelihood(model(eval_points_abl)).probs.mean(0)[:,target_class])
    
        prob_diffs[k,2] = (eval_point_prob - abl_points_prob).mean()


# In[11]:
n_pos_iggps = (iggp > 0).sum()

plt.close()
plt.plot(prob_diffs.numpy()[:,0], c="blue",lw=2.5)
plt.axhline(theo_diff.numpy(), linestyle="dashed", c="red",lw=2.5)
plt.axvline(n_pos_iggps, linestyle="dashdot", c="black",lw=2.5)
plt.plot(prob_diffs.numpy()[:,2], c="green",lw=2.5)
plt.axhline(0, linestyle="dotted", c="black")
plt.xlim((0,maxk))
plt.yticks(fontsize=20)
plt.xticks(fontsize=20)
plt.gca().xaxis.set_major_locator(ticker.MaxNLocator(nbins=5))
plt.xlabel("# Pixels replaced by baseline", fontsize=20, fontweight="bold")
plt.ylabel("Class Prob. difference", fontsize=20, fontweight="bold")
plt.savefig(f"../paper/img/experiments/fashionmnist_{target_class}_chart.png", transparent=False, bbox_inches="tight")
plt.close()

# # Target = 2

# In[12]:


target_class = 2
print(f"Target: {target_class}")


eval_point = x[y==target_class,:][0]
base_point = torch.zeros(x.shape[1])

with torch.no_grad():
    iggp = get_iggp_softmax_riemann_batched(model, eval_point, base_point, target_class, 
                                            N_RIEMANN_POINTS, RIEMANN_BATCH_SIZE).detach().numpy()

vm = np.max(np.abs(iggp))
plt.close()
cmap = plt.cm.seismic 
plt.imshow(iggp.reshape(28,28), cmap=cmap, vmin=-vm, vmax=vm)
plt.xticks([])
plt.yticks([])
plt.savefig(f"../paper/img/experiments/fashionmnist_{target_class}_image.png", transparent=False)
plt.close()

# In[13]:


theo_diff = softmax_mtb_difference(model, eval_point.unsqueeze(0), torch.zeros(len(eval_point)).unsqueeze(0), target_class)

argsort_iggp = np.argsort(iggp).tolist()[::-1]
sort_iggp = (np.array(argsort_iggp)[eval_point[argsort_iggp] > 0]).tolist()

neq_zero_pixels = np.where(eval_point > 0)[0]

maxk = len(neq_zero_pixels)

torch.manual_seed(123)
np.random.seed(123)
with gpytorch.settings.num_likelihood_samples(N_MC_LIKELIHOOD_SAMPLES):
    eval_point_prob = likelihood(model(eval_point.reshape(1,-1))).probs.mean(0)[:,target_class]

prob_diffs = torch.empty(maxk+1, 3)
prob_diffs[:] = float('nan')

with torch.no_grad():
    for k in range(maxk+1):
        #remove pixels with positive attribution
        eval_points_abl = eval_point.reshape(1,-1) * 1
        eval_points_abl[0,sort_iggp[:k]] = 0
    
        with gpytorch.settings.num_likelihood_samples(N_MC_LIKELIHOOD_SAMPLES):
            abl_points_prob = (likelihood(model(eval_points_abl)).probs.mean(0)[:,target_class])

        prob_diffs[k,0] = (eval_point_prob - abl_points_prob)


        #random points
        eval_points_abl = eval_point.reshape(1,-1) * torch.ones(N_ABLATION_SAMPLES,1)

        rand_idx = np.concatenate(
            [np.random.choice(neq_zero_pixels, (k,), replace = False).reshape(1,-1) for _ in range(N_ABLATION_SAMPLES)],
            axis=0
        )

        eval_points_abl[torch.arange(N_ABLATION_SAMPLES).unsqueeze(1), rand_idx] = 0

        with gpytorch.settings.num_likelihood_samples(N_MC_LIKELIHOOD_SAMPLES):
            abl_points_prob = (likelihood(model(eval_points_abl)).probs.mean(0)[:,target_class])
    
        prob_diffs[k,2] = (eval_point_prob - abl_points_prob).mean()


# In[14]:
n_pos_iggps = (iggp > 0).sum()

plt.close()
plt.plot(prob_diffs.numpy()[:,0], c="blue",lw=2.5)
plt.axhline(theo_diff.numpy(), linestyle="dashed", c="red",lw=2.5)
plt.axvline(n_pos_iggps, linestyle="dashdot", c="black",lw=2.5)
plt.plot(prob_diffs.numpy()[:,2], c="green",lw=2.5)
plt.axhline(0, linestyle="dotted", c="black")
plt.xlim((0,maxk))
plt.yticks(fontsize=20)
plt.xticks(fontsize=20)
plt.gca().xaxis.set_major_locator(ticker.MaxNLocator(nbins=5))
plt.xlabel("# Pixels replaced by baseline", fontsize=20, fontweight="bold")
plt.ylabel("Class Prob. difference", fontsize=20, fontweight="bold")
plt.savefig(f"../paper/img/experiments/fashionmnist_{target_class}_chart.png", transparent=False, bbox_inches="tight")
plt.close()

# # Target = 3

# In[15]:


target_class = 3
print(f"Target: {target_class}")


eval_point = x[y==target_class,:][0]
base_point = torch.zeros(x.shape[1])

with torch.no_grad():
    iggp = get_iggp_softmax_riemann_batched(model, eval_point, base_point, target_class, 
                                            N_RIEMANN_POINTS, RIEMANN_BATCH_SIZE).detach().numpy()

vm = np.max(np.abs(iggp))
plt.close()
cmap = plt.cm.seismic 
plt.imshow(iggp.reshape(28,28), cmap=cmap, vmin=-vm, vmax=vm)
plt.xticks([])
plt.yticks([])
plt.savefig(f"../paper/img/experiments/fashionmnist_{target_class}_image.png", transparent=False)
plt.close()

# In[16]:


theo_diff = softmax_mtb_difference(model, eval_point.unsqueeze(0), torch.zeros(len(eval_point)).unsqueeze(0), target_class)

argsort_iggp = np.argsort(iggp).tolist()[::-1]
sort_iggp = (np.array(argsort_iggp)[eval_point[argsort_iggp] > 0]).tolist()

neq_zero_pixels = np.where(eval_point > 0)[0]

maxk = len(neq_zero_pixels)

torch.manual_seed(123)
np.random.seed(123)
with gpytorch.settings.num_likelihood_samples(N_MC_LIKELIHOOD_SAMPLES):
    eval_point_prob = likelihood(model(eval_point.reshape(1,-1))).probs.mean(0)[:,target_class]

prob_diffs = torch.empty(maxk+1, 3)
prob_diffs[:] = float('nan')

with torch.no_grad():
    for k in range(maxk+1):
        #remove pixels with positive attribution
        eval_points_abl = eval_point.reshape(1,-1) * 1
        eval_points_abl[0,sort_iggp[:k]] = 0
    
        with gpytorch.settings.num_likelihood_samples(N_MC_LIKELIHOOD_SAMPLES):
            abl_points_prob = (likelihood(model(eval_points_abl)).probs.mean(0)[:,target_class])

        prob_diffs[k,0] = (eval_point_prob - abl_points_prob)


        #random points
        eval_points_abl = eval_point.reshape(1,-1) * torch.ones(N_ABLATION_SAMPLES,1)

        rand_idx = np.concatenate(
            [np.random.choice(neq_zero_pixels, (k,), replace = False).reshape(1,-1) for _ in range(N_ABLATION_SAMPLES)],
            axis=0
        )

        eval_points_abl[torch.arange(N_ABLATION_SAMPLES).unsqueeze(1), rand_idx] = 0

        with gpytorch.settings.num_likelihood_samples(N_MC_LIKELIHOOD_SAMPLES):
            abl_points_prob = (likelihood(model(eval_points_abl)).probs.mean(0)[:,target_class])
    
        prob_diffs[k,2] = (eval_point_prob - abl_points_prob).mean()


# In[17]:
n_pos_iggps = (iggp > 0).sum()

plt.close()
plt.plot(prob_diffs.numpy()[:,0], c="blue",lw=2.5)
plt.axhline(theo_diff.numpy(), linestyle="dashed", c="red",lw=2.5)
plt.axvline(n_pos_iggps, linestyle="dashdot", c="black",lw=2.5)
plt.plot(prob_diffs.numpy()[:,2], c="green",lw=2.5)
plt.axhline(0, linestyle="dotted", c="black")
plt.xlim((0,maxk))
plt.yticks(fontsize=20)
plt.xticks(fontsize=20)
plt.gca().xaxis.set_major_locator(ticker.MaxNLocator(nbins=5))
plt.xlabel("# Pixels replaced by baseline", fontsize=20, fontweight="bold")
plt.ylabel("Class Prob. difference", fontsize=20, fontweight="bold")
plt.savefig(f"../paper/img/experiments/fashionmnist_{target_class}_chart.png", transparent=False, bbox_inches="tight")
plt.close()

# # Target = 4

# In[18]:


target_class = 4
print(f"Target: {target_class}")


eval_point = x[y==target_class,:][0]
base_point = torch.zeros(x.shape[1])

with torch.no_grad():
    iggp = get_iggp_softmax_riemann_batched(model, eval_point, base_point, target_class, 
                                            N_RIEMANN_POINTS, RIEMANN_BATCH_SIZE).detach().numpy()

vm = np.max(np.abs(iggp))
plt.close()
cmap = plt.cm.seismic 
plt.imshow(iggp.reshape(28,28), cmap=cmap, vmin=-vm, vmax=vm)
plt.xticks([])
plt.yticks([])
plt.savefig(f"../paper/img/experiments/fashionmnist_{target_class}_image.png", transparent=False)
plt.close()

# In[ ]:


theo_diff = softmax_mtb_difference(model, eval_point.unsqueeze(0), torch.zeros(len(eval_point)).unsqueeze(0), target_class)

argsort_iggp = np.argsort(iggp).tolist()[::-1]
sort_iggp = (np.array(argsort_iggp)[eval_point[argsort_iggp] > 0]).tolist()

neq_zero_pixels = np.where(eval_point > 0)[0]

maxk = len(neq_zero_pixels)

torch.manual_seed(123)
np.random.seed(123)
with gpytorch.settings.num_likelihood_samples(N_MC_LIKELIHOOD_SAMPLES):
    eval_point_prob = likelihood(model(eval_point.reshape(1,-1))).probs.mean(0)[:,target_class]

prob_diffs = torch.empty(maxk+1, 3)
prob_diffs[:] = float('nan')

with torch.no_grad():
    for k in range(maxk+1):
        #remove pixels with positive attribution
        eval_points_abl = eval_point.reshape(1,-1) * 1
        eval_points_abl[0,sort_iggp[:k]] = 0
    
        with gpytorch.settings.num_likelihood_samples(N_MC_LIKELIHOOD_SAMPLES):
            abl_points_prob = (likelihood(model(eval_points_abl)).probs.mean(0)[:,target_class])

        prob_diffs[k,0] = (eval_point_prob - abl_points_prob)


        #random points
        eval_points_abl = eval_point.reshape(1,-1) * torch.ones(N_ABLATION_SAMPLES,1)

        rand_idx = np.concatenate(
            [np.random.choice(neq_zero_pixels, (k,), replace = False).reshape(1,-1) for _ in range(N_ABLATION_SAMPLES)],
            axis=0
        )

        eval_points_abl[torch.arange(N_ABLATION_SAMPLES).unsqueeze(1), rand_idx] = 0

        with gpytorch.settings.num_likelihood_samples(N_MC_LIKELIHOOD_SAMPLES):
            abl_points_prob = (likelihood(model(eval_points_abl)).probs.mean(0)[:,target_class])
    
        prob_diffs[k,2] = (eval_point_prob - abl_points_prob).mean()


# In[ ]:
n_pos_iggps = (iggp > 0).sum()

plt.close()
plt.plot(prob_diffs.numpy()[:,0], c="blue",lw=2.5)
plt.axhline(theo_diff.numpy(), linestyle="dashed", c="red",lw=2.5)
plt.axvline(n_pos_iggps, linestyle="dashdot", c="black",lw=2.5)
plt.plot(prob_diffs.numpy()[:,2], c="green",lw=2.5)
plt.axhline(0, linestyle="dotted", c="black")
plt.xlim((0,maxk))
plt.yticks(fontsize=20)
plt.xticks(fontsize=20)
plt.gca().xaxis.set_major_locator(ticker.MaxNLocator(nbins=5))
plt.xlabel("# Pixels replaced by baseline", fontsize=20, fontweight="bold")
plt.ylabel("Class Prob. difference", fontsize=20, fontweight="bold")
plt.savefig(f"../paper/img/experiments/fashionmnist_{target_class}_chart.png", transparent=False, bbox_inches="tight")
plt.close()

# # Target = 5

# In[ ]:


target_class = 5
print(f"Target: {target_class}")


eval_point = x[y==target_class,:][0]
base_point = torch.zeros(x.shape[1])

with torch.no_grad():
    iggp = get_iggp_softmax_riemann_batched(model, eval_point, base_point, target_class, 
                                            N_RIEMANN_POINTS, RIEMANN_BATCH_SIZE).detach().numpy()

vm = np.max(np.abs(iggp))
plt.close()
cmap = plt.cm.seismic 
plt.imshow(iggp.reshape(28,28), cmap=cmap, vmin=-vm, vmax=vm)
plt.xticks([])
plt.yticks([])
plt.savefig(f"../paper/img/experiments/fashionmnist_{target_class}_image.png", transparent=False)
plt.close()

# In[ ]:


theo_diff = softmax_mtb_difference(model, eval_point.unsqueeze(0), torch.zeros(len(eval_point)).unsqueeze(0), target_class)

argsort_iggp = np.argsort(iggp).tolist()[::-1]
sort_iggp = (np.array(argsort_iggp)[eval_point[argsort_iggp] > 0]).tolist()

neq_zero_pixels = np.where(eval_point > 0)[0]

maxk = len(neq_zero_pixels)

torch.manual_seed(123)
np.random.seed(123)
with gpytorch.settings.num_likelihood_samples(N_MC_LIKELIHOOD_SAMPLES):
    eval_point_prob = likelihood(model(eval_point.reshape(1,-1))).probs.mean(0)[:,target_class]

prob_diffs = torch.empty(maxk+1, 3)
prob_diffs[:] = float('nan')

with torch.no_grad():
    for k in range(maxk+1):
        #remove pixels with positive attribution
        eval_points_abl = eval_point.reshape(1,-1) * 1
        eval_points_abl[0,sort_iggp[:k]] = 0
    
        with gpytorch.settings.num_likelihood_samples(N_MC_LIKELIHOOD_SAMPLES):
            abl_points_prob = (likelihood(model(eval_points_abl)).probs.mean(0)[:,target_class])

        prob_diffs[k,0] = (eval_point_prob - abl_points_prob)


        #random points
        eval_points_abl = eval_point.reshape(1,-1) * torch.ones(N_ABLATION_SAMPLES,1)

        rand_idx = np.concatenate(
            [np.random.choice(neq_zero_pixels, (k,), replace = False).reshape(1,-1) for _ in range(N_ABLATION_SAMPLES)],
            axis=0
        )

        eval_points_abl[torch.arange(N_ABLATION_SAMPLES).unsqueeze(1), rand_idx] = 0

        with gpytorch.settings.num_likelihood_samples(N_MC_LIKELIHOOD_SAMPLES):
            abl_points_prob = (likelihood(model(eval_points_abl)).probs.mean(0)[:,target_class])
    
        prob_diffs[k,2] = (eval_point_prob - abl_points_prob).mean()


# In[ ]:
n_pos_iggps = (iggp > 0).sum()

plt.close()
plt.plot(prob_diffs.numpy()[:,0], c="blue",lw=2.5)
plt.axhline(theo_diff.numpy(), linestyle="dashed", c="red",lw=2.5)
plt.axvline(n_pos_iggps, linestyle="dashdot", c="black",lw=2.5)
plt.plot(prob_diffs.numpy()[:,2], c="green",lw=2.5)
plt.axhline(0, linestyle="dotted", c="black")
plt.xlim((0,maxk))
plt.yticks(fontsize=20)
plt.xticks(fontsize=20)
plt.gca().xaxis.set_major_locator(ticker.MaxNLocator(nbins=5))
plt.xlabel("# Pixels replaced by baseline", fontsize=20, fontweight="bold")
plt.ylabel("Class Prob. difference", fontsize=20, fontweight="bold")
plt.savefig(f"../paper/img/experiments/fashionmnist_{target_class}_chart.png", transparent=False, bbox_inches="tight")
plt.close()

# # Target = 6

# In[ ]:


target_class = 6
print(f"Target: {target_class}")


eval_point = x[y==target_class,:][0]
base_point = torch.zeros(x.shape[1])

with torch.no_grad():
    iggp = get_iggp_softmax_riemann_batched(model, eval_point, base_point, target_class, 
                                            N_RIEMANN_POINTS, RIEMANN_BATCH_SIZE).detach().numpy()

vm = np.max(np.abs(iggp))
plt.close()
cmap = plt.cm.seismic 
plt.imshow(iggp.reshape(28,28), cmap=cmap, vmin=-vm, vmax=vm)
plt.xticks([])
plt.yticks([])
plt.savefig(f"../paper/img/experiments/fashionmnist_{target_class}_image.png", transparent=False)
plt.close()

# In[ ]:


theo_diff = softmax_mtb_difference(model, eval_point.unsqueeze(0), torch.zeros(len(eval_point)).unsqueeze(0), target_class)

argsort_iggp = np.argsort(iggp).tolist()[::-1]
sort_iggp = (np.array(argsort_iggp)[eval_point[argsort_iggp] > 0]).tolist()

neq_zero_pixels = np.where(eval_point > 0)[0]

maxk = len(neq_zero_pixels)

torch.manual_seed(123)
np.random.seed(123)
with gpytorch.settings.num_likelihood_samples(N_MC_LIKELIHOOD_SAMPLES):
    eval_point_prob = likelihood(model(eval_point.reshape(1,-1))).probs.mean(0)[:,target_class]

prob_diffs = torch.empty(maxk+1, 3)
prob_diffs[:] = float('nan')

with torch.no_grad():
    for k in range(maxk+1):
        #remove pixels with positive attribution
        eval_points_abl = eval_point.reshape(1,-1) * 1
        eval_points_abl[0,sort_iggp[:k]] = 0
    
        with gpytorch.settings.num_likelihood_samples(N_MC_LIKELIHOOD_SAMPLES):
            abl_points_prob = (likelihood(model(eval_points_abl)).probs.mean(0)[:,target_class])

        prob_diffs[k,0] = (eval_point_prob - abl_points_prob)


        #random points
        eval_points_abl = eval_point.reshape(1,-1) * torch.ones(N_ABLATION_SAMPLES,1)

        rand_idx = np.concatenate(
            [np.random.choice(neq_zero_pixels, (k,), replace = False).reshape(1,-1) for _ in range(N_ABLATION_SAMPLES)],
            axis=0
        )

        eval_points_abl[torch.arange(N_ABLATION_SAMPLES).unsqueeze(1), rand_idx] = 0

        with gpytorch.settings.num_likelihood_samples(N_MC_LIKELIHOOD_SAMPLES):
            abl_points_prob = (likelihood(model(eval_points_abl)).probs.mean(0)[:,target_class])
    
        prob_diffs[k,2] = (eval_point_prob - abl_points_prob).mean()


# In[ ]:
n_pos_iggps = (iggp > 0).sum()

plt.close()
plt.plot(prob_diffs.numpy()[:,0], c="blue",lw=2.5)
plt.axhline(theo_diff.numpy(), linestyle="dashed", c="red",lw=2.5)
plt.axvline(n_pos_iggps, linestyle="dashdot", c="black",lw=2.5)
plt.plot(prob_diffs.numpy()[:,2], c="green",lw=2.5)
plt.axhline(0, linestyle="dotted", c="black")
plt.xlim((0,maxk))
plt.yticks(fontsize=20)
plt.xticks(fontsize=20)
plt.gca().xaxis.set_major_locator(ticker.MaxNLocator(nbins=5))
plt.xlabel("# Pixels replaced by baseline", fontsize=20, fontweight="bold")
plt.ylabel("Class Prob. difference", fontsize=20, fontweight="bold")
plt.savefig(f"../paper/img/experiments/fashionmnist_{target_class}_chart.png", transparent=False, bbox_inches="tight")
plt.close()

# # Target = 7

# In[ ]:


target_class = 7
print(f"Target: {target_class}")


eval_point = x[y==target_class,:][0]
base_point = torch.zeros(x.shape[1])

with torch.no_grad():
    iggp = get_iggp_softmax_riemann_batched(model, eval_point, base_point, target_class, 
                                            N_RIEMANN_POINTS, RIEMANN_BATCH_SIZE).detach().numpy()

vm = np.max(np.abs(iggp))
plt.close()
cmap = plt.cm.seismic 
plt.imshow(iggp.reshape(28,28), cmap=cmap, vmin=-vm, vmax=vm)
plt.xticks([])
plt.yticks([])
plt.savefig(f"../paper/img/experiments/fashionmnist_{target_class}_image.png", transparent=False)
plt.close()

# In[ ]:


theo_diff = softmax_mtb_difference(model, eval_point.unsqueeze(0), torch.zeros(len(eval_point)).unsqueeze(0), target_class)

argsort_iggp = np.argsort(iggp).tolist()[::-1]
sort_iggp = (np.array(argsort_iggp)[eval_point[argsort_iggp] > 0]).tolist()

neq_zero_pixels = np.where(eval_point > 0)[0]

maxk = len(neq_zero_pixels)

torch.manual_seed(123)
np.random.seed(123)
with gpytorch.settings.num_likelihood_samples(N_MC_LIKELIHOOD_SAMPLES):
    eval_point_prob = likelihood(model(eval_point.reshape(1,-1))).probs.mean(0)[:,target_class]

prob_diffs = torch.empty(maxk+1, 3)
prob_diffs[:] = float('nan')

with torch.no_grad():
    for k in range(maxk+1):
        #remove pixels with positive attribution
        eval_points_abl = eval_point.reshape(1,-1) * 1
        eval_points_abl[0,sort_iggp[:k]] = 0
    
        with gpytorch.settings.num_likelihood_samples(N_MC_LIKELIHOOD_SAMPLES):
            abl_points_prob = (likelihood(model(eval_points_abl)).probs.mean(0)[:,target_class])

        prob_diffs[k,0] = (eval_point_prob - abl_points_prob)


        #random points
        eval_points_abl = eval_point.reshape(1,-1) * torch.ones(N_ABLATION_SAMPLES,1)

        rand_idx = np.concatenate(
            [np.random.choice(neq_zero_pixels, (k,), replace = False).reshape(1,-1) for _ in range(N_ABLATION_SAMPLES)],
            axis=0
        )

        eval_points_abl[torch.arange(N_ABLATION_SAMPLES).unsqueeze(1), rand_idx] = 0

        with gpytorch.settings.num_likelihood_samples(N_MC_LIKELIHOOD_SAMPLES):
            abl_points_prob = (likelihood(model(eval_points_abl)).probs.mean(0)[:,target_class])
    
        prob_diffs[k,2] = (eval_point_prob - abl_points_prob).mean()


# In[ ]:
n_pos_iggps = (iggp > 0).sum()

plt.close()
plt.plot(prob_diffs.numpy()[:,0], c="blue",lw=2.5)
plt.axhline(theo_diff.numpy(), linestyle="dashed", c="red",lw=2.5)
plt.axvline(n_pos_iggps, linestyle="dashdot", c="black",lw=2.5)
plt.plot(prob_diffs.numpy()[:,2], c="green",lw=2.5)
plt.axhline(0, linestyle="dotted", c="black")
plt.xlim((0,maxk))
plt.yticks(fontsize=20)
plt.xticks(fontsize=20)
plt.gca().xaxis.set_major_locator(ticker.MaxNLocator(nbins=5))
plt.xlabel("# Pixels replaced by baseline", fontsize=20, fontweight="bold")
plt.ylabel("Class Prob. difference", fontsize=20, fontweight="bold")
plt.savefig(f"../paper/img/experiments/fashionmnist_{target_class}_chart.png", transparent=False, bbox_inches="tight")
plt.close()

# # Target = 8

# In[ ]:


target_class = 8
print(f"Target: {target_class}")


eval_point = x[y==target_class,:][0]
base_point = torch.zeros(x.shape[1])

with torch.no_grad():
    iggp = get_iggp_softmax_riemann_batched(model, eval_point, base_point, target_class, 
                                            N_RIEMANN_POINTS, RIEMANN_BATCH_SIZE).detach().numpy()

vm = np.max(np.abs(iggp))
plt.close()
cmap = plt.cm.seismic 
plt.imshow(iggp.reshape(28,28), cmap=cmap, vmin=-vm, vmax=vm)
plt.xticks([])
plt.yticks([])
plt.savefig(f"../paper/img/experiments/fashionmnist_{target_class}_image.png", transparent=False)
plt.close()

# In[ ]:


theo_diff = softmax_mtb_difference(model, eval_point.unsqueeze(0), torch.zeros(len(eval_point)).unsqueeze(0), target_class)

argsort_iggp = np.argsort(iggp).tolist()[::-1]
sort_iggp = (np.array(argsort_iggp)[eval_point[argsort_iggp] > 0]).tolist()

neq_zero_pixels = np.where(eval_point > 0)[0]

maxk = len(neq_zero_pixels)

torch.manual_seed(123)
np.random.seed(123)
with gpytorch.settings.num_likelihood_samples(N_MC_LIKELIHOOD_SAMPLES):
    eval_point_prob = likelihood(model(eval_point.reshape(1,-1))).probs.mean(0)[:,target_class]

prob_diffs = torch.empty(maxk+1, 3)
prob_diffs[:] = float('nan')

with torch.no_grad():
    for k in range(maxk+1):
        #remove pixels with positive attribution
        eval_points_abl = eval_point.reshape(1,-1) * 1
        eval_points_abl[0,sort_iggp[:k]] = 0
    
        with gpytorch.settings.num_likelihood_samples(N_MC_LIKELIHOOD_SAMPLES):
            abl_points_prob = (likelihood(model(eval_points_abl)).probs.mean(0)[:,target_class])

        prob_diffs[k,0] = (eval_point_prob - abl_points_prob)


        #random points
        eval_points_abl = eval_point.reshape(1,-1) * torch.ones(N_ABLATION_SAMPLES,1)

        rand_idx = np.concatenate(
            [np.random.choice(neq_zero_pixels, (k,), replace = False).reshape(1,-1) for _ in range(N_ABLATION_SAMPLES)],
            axis=0
        )

        eval_points_abl[torch.arange(N_ABLATION_SAMPLES).unsqueeze(1), rand_idx] = 0

        with gpytorch.settings.num_likelihood_samples(N_MC_LIKELIHOOD_SAMPLES):
            abl_points_prob = (likelihood(model(eval_points_abl)).probs.mean(0)[:,target_class])
    
        prob_diffs[k,2] = (eval_point_prob - abl_points_prob).mean()


# In[ ]:
n_pos_iggps = (iggp > 0).sum()

plt.close()
plt.plot(prob_diffs.numpy()[:,0], c="blue",lw=2.5)
plt.axhline(theo_diff.numpy(), linestyle="dashed", c="red",lw=2.5)
plt.axvline(n_pos_iggps, linestyle="dashdot", c="black",lw=2.5)
plt.plot(prob_diffs.numpy()[:,2], c="green",lw=2.5)
plt.axhline(0, linestyle="dotted", c="black")
plt.xlim((0,maxk))
plt.yticks(fontsize=20)
plt.xticks(fontsize=20)
plt.gca().xaxis.set_major_locator(ticker.MaxNLocator(nbins=5))
plt.xlabel("# Pixels replaced by baseline", fontsize=20, fontweight="bold")
plt.ylabel("Class Prob. difference", fontsize=20, fontweight="bold")
plt.savefig(f"../paper/img/experiments/fashionmnist_{target_class}_chart.png", transparent=False, bbox_inches="tight")
plt.close()

# # Target = 9

# In[ ]:


target_class = 9
print(f"Target: {target_class}")


eval_point = x[y==target_class,:][0]
base_point = torch.zeros(x.shape[1])

with torch.no_grad():
    iggp = get_iggp_softmax_riemann_batched(model, eval_point, base_point, target_class, 
                                            N_RIEMANN_POINTS, RIEMANN_BATCH_SIZE).detach().numpy()

vm = np.max(np.abs(iggp))
plt.close()
cmap = plt.cm.seismic 
plt.imshow(iggp.reshape(28,28), cmap=cmap, vmin=-vm, vmax=vm)
plt.xticks([])
plt.yticks([])
plt.savefig(f"../paper/img/experiments/fashionmnist_{target_class}_image.png", transparent=False)
plt.close()

# In[ ]:


theo_diff = softmax_mtb_difference(model, eval_point.unsqueeze(0), torch.zeros(len(eval_point)).unsqueeze(0), target_class)

argsort_iggp = np.argsort(iggp).tolist()[::-1]
sort_iggp = (np.array(argsort_iggp)[eval_point[argsort_iggp] > 0]).tolist()

neq_zero_pixels = np.where(eval_point > 0)[0]

maxk = len(neq_zero_pixels)

torch.manual_seed(123)
np.random.seed(123)
with gpytorch.settings.num_likelihood_samples(N_MC_LIKELIHOOD_SAMPLES):
    eval_point_prob = likelihood(model(eval_point.reshape(1,-1))).probs.mean(0)[:,target_class]

prob_diffs = torch.empty(maxk+1, 3)
prob_diffs[:] = float('nan')

with torch.no_grad():
    for k in range(maxk+1):
        #remove pixels with positive attribution
        eval_points_abl = eval_point.reshape(1,-1) * 1
        eval_points_abl[0,sort_iggp[:k]] = 0
    
        with gpytorch.settings.num_likelihood_samples(N_MC_LIKELIHOOD_SAMPLES):
            abl_points_prob = (likelihood(model(eval_points_abl)).probs.mean(0)[:,target_class])

        prob_diffs[k,0] = (eval_point_prob - abl_points_prob)


        #random points
        eval_points_abl = eval_point.reshape(1,-1) * torch.ones(N_ABLATION_SAMPLES,1)

        rand_idx = np.concatenate(
            [np.random.choice(neq_zero_pixels, (k,), replace = False).reshape(1,-1) for _ in range(N_ABLATION_SAMPLES)],
            axis=0
        )

        eval_points_abl[torch.arange(N_ABLATION_SAMPLES).unsqueeze(1), rand_idx] = 0

        with gpytorch.settings.num_likelihood_samples(N_MC_LIKELIHOOD_SAMPLES):
            abl_points_prob = (likelihood(model(eval_points_abl)).probs.mean(0)[:,target_class])
    
        prob_diffs[k,2] = (eval_point_prob - abl_points_prob).mean()


# In[ ]:
n_pos_iggps = (iggp > 0).sum()

plt.close()
plt.plot(prob_diffs.numpy()[:,0], c="blue",lw=2.5)
plt.axhline(theo_diff.numpy(), linestyle="dashed", c="red",lw=2.5)
plt.axvline(n_pos_iggps, linestyle="dashdot", c="black",lw=2.5)
plt.plot(prob_diffs.numpy()[:,2], c="green",lw=2.5)
plt.axhline(0, linestyle="dotted", c="black")
plt.xlim((0,maxk))
plt.yticks(fontsize=20)
plt.xticks(fontsize=20)
plt.gca().xaxis.set_major_locator(ticker.MaxNLocator(nbins=5))
plt.xlabel("# Pixels replaced by baseline", fontsize=20, fontweight="bold")
plt.ylabel("Class Prob. difference", fontsize=20, fontweight="bold")
plt.savefig(f"../paper/img/experiments/fashionmnist_{target_class}_chart.png", transparent=False, bbox_inches="tight")
plt.close()

# In[ ]:




