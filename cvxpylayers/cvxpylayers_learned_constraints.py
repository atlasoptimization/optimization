#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
The goal of this script is to showcase how cvxpylayers can be used to learn
constraints that are intrinsically hidden in data. This includes generating
data that adhere to upper and lower bounds and then fitting appropriate bounds
implemented by projection that is cast as an optimization problem in cvxpylayers.
For this, do the following:
    1. Imports and definitions
    2. Simulate data
    3. Model in cvxpylayers
    4. Training
    5. Plots and illustrations
"""

"""
    1. Imports and definitions
"""


# i) Imports

import cvxpy as cp
from cvxpylayers.torch import CvxpyLayer
import torch
import matplotlib.pyplot as plt
import copy


# ii) Definitions

n_data = 100
reg_coeff = 0.1
index_data = torch.linspace(0,n_data,n_data)



"""
    2. Simulate data
"""


# i) Set up simulation

mu = 0
sigma = 1

ub = 1
lb = -1


# ii) Simulate data

data_unbounded = torch.normal(mu, sigma, [n_data,1])
data_bounded = torch.maximum(data_unbounded, torch.tensor(-1))
data_bounded = torch.minimum(data_bounded, torch.tensor(1))



"""
    3. Model in cvxpylayers
"""


# i) Define forward projection

bounds = cp.Parameter(shape = (2,))
x_in = cp.Parameter(shape = (1,))
x_out = cp.Variable(shape = (1,))

# x_out = argmin ||x_in - x||  s.t. x in bounds
# goal: project x_in into bounds
objective = cp.Minimize(cp.norm(x_in - x_out, p =1) + cp.norm(bounds,p = 1))
cons = [x_out >= bounds[0], x_out <= bounds[1]]

problem = cp.Problem(objective, cons)


# ii) Convert to differentiable layer

layer = CvxpyLayer(problem = problem, parameters = [bounds, x_in], variables = [x_out])

# Initialize bounds_torch as the parameter w.r.t which grads are computed
# Parameter bounds is placeholder, later on process bounds_torch into bounds
# and pass to layer to compute loss.
bounds_torch = torch.tensor([-0.5, +0.5], requires_grad = True)



"""
    4. Training
"""


# i) Set up iteration

learning_rate = 5*1e-2
num_epochs = 50
adam_args = {"lr" : learning_rate}

# Setting up optimizer
optimizer = torch.optim.Adam([bounds_torch], lr = learning_rate)


# ii) Loss function

def loss_fun(bounds_torch, x_in):
    # loss = ||x_in - x_out|| + reg_coeff * ||bounds|| 
    # where x_out is the result of the projection layers
    # goal: penalize projection residuals and absolute value of bounds to find
    # the tightest bounds explaining the data.
    x_out = layer(bounds_torch, x_in)[0]
    loss = (1/n_data)*torch.norm(x_out - x_in, p = 1) + reg_coeff * torch.norm(bounds_torch, p = 1)
    return loss, x_out

print('bounds_init = {}'.format(bounds_torch))


# iii) Iterate

loss_history = []
bounds_history = []
x_out_history = []

for k in range(num_epochs):
        
    # record data
    loss, x_out = loss_fun(bounds_torch, data_bounded)
    loss_history.append(loss.item())
    x_out_history.append(copy.copy(x_out.detach()))
    bounds_history.append(copy.copy(bounds_torch.detach().numpy()))
    
    # optimize
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    if k % 1 == 0:
        print('Epoch = {}, Loss = {}'.format(k, loss.item()))



"""
    5. Plots and illustrations
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
axs[0].plot(index_data, data_bounded.flatten().detach())
axs[0].set_title('Original data')

axs[1].plot(index_data, x_out_history[k_1].flatten().detach())
axs[1].set_title('Projected data; epoch = {}'.format(k_1))

axs[2].plot(index_data, x_out_history[k_2].flatten().detach())
axs[2].set_title('Projected data; epoch = {}'.format(k_2))

axs[3].plot(index_data, x_out_history[k_3].flatten().detach())
axs[3].set_title('Projected data; epoch = {}'.format(k_3))

plt.tight_layout()
plt.show()

















