#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Monotonic Output Regression
"""


import torch
import numpy as np
import cvxpy as cp
from cvxpylayers.torch import CvxpyLayer
import cvxpylayers
from algorithms import fit
import matplotlib.pyplot as plt
from latexify import latexify
cp.__version__, cvxpylayers.__version__


# Define convex optimization model
n = 20
m = 10

y = cp.Variable(m)
yhat = cp.Parameter(m)

objective = cp.norm2(y-yhat)
constraints = [cp.diff(y) >= 0]
prob = cp.Problem(cp.Minimize(objective), constraints)
layer = CvxpyLayer(prob, [yhat], [y])

# Get data
def get_data(N, n, m, theta):
    X = torch.randn(N, n)
    Y = layer(X @ theta + torch.randn(N, m))[0]
    return X, Y

torch.manual_seed(0)
theta_true = torch.randn(n, m)
X, Y = get_data(100, n, m, theta_true)
Xval, Yval = get_data(50, n, m, theta_true)

mse_loss = torch.nn.MSELoss()
theta_lstsq = torch.solve(X.t() @ Y, X.t() @ X).solution
lstsq_val_loss = mse_loss(Xval @ theta_lstsq, Yval).item()
bayes_val_loss = mse_loss(layer(Xval @ theta_true)[0], Yval).item()

theta = torch.zeros_like(theta_lstsq)
theta.requires_grad_(True)
def loss(X, Y, theta):
    return mse_loss(layer(X @ theta)[0], Y)

val_losses, train_losses = fit(lambda X, Y: loss(X, Y, theta), [theta], X, Y, Xval, Yval,
                               opt=torch.optim.Adam, opt_kwargs={"lr": 1e-1},
                               batch_size=16, epochs=20, verbose=True)

latexify(5.485, 1.8)
fig, ax = plt.subplots(1, 2)

ax[0].axhline(lstsq_val_loss, linestyle='-.', c='black', label='LR')
ax[0].plot(np.arange(1, 21), val_losses, c='k', label='COM')
ax[0].axhline(bayes_val_loss, linestyle='--', c='black', label='true')
ax[0].set_xlabel("iteration")
ax[0].set_ylabel("validation loss")
ax[0].set_ylim(0)
ax[0].legend()

ax[1].plot(Xval[13] @ theta_lstsq, '-.', c='k', label='LR')
ax[1].plot(layer(Xval[13] @ theta)[0].detach().numpy(), c='k', label='COM')
ax[1].plot(Yval[13].numpy(), '--', c='k', label='true')
ax[1].legend()
ax[1].set_xlabel('$i$')
ax[1].set_ylabel("$\phi(x;\\theta)$")

plt.tight_layout()
plt.savefig("figures/mono_regression.pdf")
plt.show()



























































