#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
The goal of this script is to test cvpylayer basic functionality on a simple 
example. The example consists in a linear programming problem modelling some
production planning problem. The costs in tha LP are nonlinearly dependent on
some parameter theta that is to be found by gradient descent to enable lower
cost minima.
For this, do the following:
    1. Imports and definitions
    2. Formulate the problem
    3. Solution process
    
The script is meant solely for educational and illustrative purposes. Written by
Jemil Avers Butt, Atlas optimization GmbH, www.atlasoptimization.com.
"""

"""
    1. Imports and definitions
"""


# i) Imports

import torch
import numpy as np
import cvxpy as cp
from cvxpylayers.torch import CvxpyLayer


# ii) Definitions





"""
    2. Formulate the problem
"""


# i) Define forward problem

c = cp.Parameter(shape = (2,))
x = cp.Variable(shape = (2,))

objective = cp.Minimize(c.T@x)
cons = [x >=0, cp.sum(x) ==2]

problem = cp.Problem(objective, cons)


# ii) Convert to differentiable layer

layer = CvxpyLayer(problem = problem, parameters = [c], variables = [x])

# Initialize theta as the parameter w.r.t which grads are computed
# Parameter c is placeholder, later on process theta into c and pass to layer
# to compute loss.
theta = torch.tensor([0.7], requires_grad = True)



"""
    3. Solution process
"""


# i) Set up iteration

learning_rate = 1*1e-1
num_epochs = 500
adam_args = {"lr" : learning_rate}

# Setting up optimizer
optimizer = torch.optim.Adam([theta], lr = learning_rate)


# ii) Loss function

def loss_fun(theta):
    # Relationship : cost_vector = [theta^2, (1-theta)^2]
    c_torch = torch.stack([theta**2, (1-theta)**2]).flatten()
    x_star = layer(c_torch)[0]
    loss = c_torch.T@x_star
    return loss

print('theta_init = {}, objective_init = {}'.format(theta, loss_fun(theta)))


# iii) Iterate

for k in range(num_epochs):
    optimizer.zero_grad()
    loss = loss_fun(theta)
    loss.backward()
    optimizer.step()
    
    if k % 100 == 0:
        print('Epoch = {}, Loss = {}'.format(k, loss.item()))

print('theta_end = {}, objective_end = {}'.format(theta, loss_fun(theta)))

