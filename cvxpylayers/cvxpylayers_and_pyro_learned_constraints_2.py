#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
The goal of this script is to showcase how cvxpylayers can be used to learn
constraints that are intrinsically hidden in data. This includes generating
data that adhere to upper and lower bounds and then fitting a probability
distribution in pyro that is subseded by a projection operator implemented in 
cvxpylayers. The samples are timeseries and the upper and lower bounds depend
on time.
For this, do the following:
    1. Imports and definitions
    2. Simulate data
    3. Model in cvxpylayers
    4. Model in pyro
    5. Training
    6. Plots and illustrations
"""



"""
    1. Imports and definitions
"""


# i) Imports

import cvxpy as cp
from cvxpylayers.torch import CvxpyLayer
import torch
import pyro
import pprint
import copy
import matplotlib.pyplot as plt


# ii) Definitions

n_samples = 50
n_times = 10
time = torch.linspace(0,1,n_times)
index_data = torch.linspace(0,n_samples,n_samples)



"""
    2. Simulate data
"""


# i) Set up simulation

mu = 0
sigma = 1

ub = 2*(0.25+(time-0.5)**2)
lb = -2*(0.25+(time-0.5)**2)


# ii) Simulate data

data_unbounded = torch.normal(mu, sigma, [n_samples, n_times])
data_bounded = torch.maximum(data_unbounded, lb)
data_bounded = torch.minimum(data_bounded, ub)



"""
    3. Model in cvxpylayers
"""


# i) Define forward projection

bounds = cp.Parameter(shape = ([2,n_times]))
x_in = cp.Parameter(shape = (n_times,))
x_out = cp.Variable(shape = (n_times,))

# x_out = argmin ||x_in - x||  s.t. x in bounds
# goal: project x_in into bounds
objective = cp.Minimize(cp.norm(x_in - x_out, p =1) + cp.norm(bounds,p = 1))
cons = [x_out >= bounds[0,:], x_out <= bounds[1,:]]

problem = cp.Problem(objective, cons)


# ii) Convert to differentiable layer

layer = CvxpyLayer(problem = problem, parameters = [bounds, x_in], variables = [x_out])

# Initialize bounds_torch as the parameter w.r.t which grads are computed
# Parameter bounds is placeholder, later on process bounds_torch into bounds
# and pass to layer to compute loss.
bounds_torch_ub = 0.5*torch.ones([1,n_times])
bounds_torch_lb = -0.5*torch.ones([1,n_times])

bounds_torch = torch.vstack((bounds_torch_lb, bounds_torch_ub)).requires_grad_(True)



"""
    4. Model in pyro
"""


# i) Forward model

def model(data = None):
    # Set up parameters
    mu = pyro.param("mu", init_tensor = torch.tensor([1.0]))
    sigma = pyro.param("sigma", init_tensor = torch.tensor([3.0]))
    bounds_torch_ub = 0.5*torch.ones([1,n_times])
    bounds_torch_lb = -0.5*torch.ones([1,n_times])
    bounds = pyro.param("bounds", init_tensor = torch.vstack((bounds_torch_lb, bounds_torch_ub)))
    
    # Define intermediate observation distribution (pre-projection)
    extension_tensor = torch.ones([n_samples,n_times])
    subobs_dist = pyro.distributions.Normal(extension_tensor*mu, sigma).to_event(1)
    with pyro.plate("batch_plate", size = n_samples, dim = -1):
        subobs = pyro.sample("subobs", subobs_dist)
        
        # Define observation distribution (post-projection)
        subobs_proj = layer(bounds, subobs)[0]
        reg_noise = torch.tensor([0.001])
        obs_dist = pyro.distributions.Normal(subobs_proj, reg_noise).to_event(1)
        obs = pyro.sample("obs", obs_dist, obs = data)

    return obs

# model_trace = pyro.poutine.trace(model).get_trace()
# print(model_trace.format_shapes())
# pprint.pprint(model_trace.nodes)


# ii) Guide

guide = pyro.infer.autoguide.AutoNormal(model)





"""
    5. Training
"""


# i) Set up iteration

pyro.clear_param_store()

# specifying scalar options
learning_rate = 3e-2
num_epochs = 200
adam_args = {"lr" : learning_rate}

# Setting up svi
optimizer = pyro.optim.Adam(adam_args)
elbo_loss = pyro.infer.Trace_ELBO()
svi = pyro.infer.SVI(model = model, guide = guide, optim = optimizer, loss= elbo_loss)


# ii) Iterate

loss_history = []
bounds_history = []
x_out_history = []

for k in range(num_epochs):
        
    # record data
    loss = svi.step(data_bounded)
    loss_history.append(loss)
    x_out_history.append(copy.copy(model().detach()))
    bounds_history.append(copy.copy(pyro.get_param_store()['bounds'].detach().numpy()))
        
    if k % 10 == 0:
        print('Epoch = {}, Loss = {}'.format(k, loss))

for name, value in pyro.get_param_store().items():
    print(name, pyro.param(name).data.cpu().numpy())



"""
    6. Plots and illustrations
"""


# i) Plot overall loss

plt.figure(num = 1, figsize = (10,5), dpi = 300)
plt.plot(loss_history)
plt.title('Loss')
plt.xlabel('epoch nr')


# ii) Plot overall bounds

plt.figure(num = 2, figsize = (10,5), dpi = 300)
plt.plot([bd[0] for bd in bounds_history])
plt.plot([bd[1] for bd in bounds_history])
plt.title('bounds')
plt.xlabel('epoch nr')



# iii) Plot several projection realizations

k_1 = 0
k_2 = round(num_epochs/3)
k_3 = num_epochs-1

fig, axs = plt.subplots(4, 1, figsize=(12, 12))
axs[0].plot(time, data_bounded.detach().T)
axs[0].set_title('Original data')

axs[1].plot(time, x_out_history[k_1].detach().T)
axs[1].set_title('Projected data; epoch = {}'.format(k_1))

axs[2].plot(time, x_out_history[k_2].detach().T)
axs[2].set_title('Projected data; epoch = {}'.format(k_2))

axs[3].plot(time, x_out_history[k_3].detach().T)
axs[3].set_title('Projected data; epoch = {}'.format(k_3))

plt.tight_layout()
plt.show()




















