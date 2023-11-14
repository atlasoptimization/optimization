#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
The goal of this script is to test cvpylayer basic functionality on a simple 
example. The example consists in a constrained least squares problem and we will
use cvxpylayers to figure out an additional datapoint that would lead to a certain
value for the adjusted line.
"""

# import cvxpy as cp

# m , n = 20 , 10
# x = cp.Variable((n , 1))
# F = cp.Parameter(( m , n ))
# g = cp.Parameter(( m , 1))
# lambd = cp.Parameter((1 , 1), nonneg = True)
# objective_fn = cp.norm(F @ x - g) + lambd * cp.norm(x)
# constraints = [ x >= 0]
# problem = cp.Problem (cp.Minimize (objective_fn), constraints)
# assert problem.is_dpp()


# import cvxpy as cp
# import torch
# from cvxpylayers.torch import CvxpyLayer

# n, m = 2, 3
# x = cp.Variable(n)
# A = cp.Parameter((m, n))
# b = cp.Parameter(m)
# constraints = [x >= 0]
# objective = cp.Minimize(0.5 * cp.pnorm(A @ x - b, p=1))
# problem = cp.Problem(objective, constraints)
# assert problem.is_dpp()

# cvxpylayer = CvxpyLayer(problem, parameters=[A, b], variables=[x])
# A_tch = torch.randn(m, n, requires_grad=True)
# b_tch = torch.randn(m, requires_grad=True)

# # solve the problem
# solution, = cvxpylayer(A_tch, b_tch)

# # compute the gradient of the sum of the solution with respect to A, b
# solution.sum().backward()

import torch
import cvxpy as cp
from cvxpylayers.torch import CvxpyLayer

# Define the forward problem: linear regression
m = cp.Variable()
c = cp.Variable()
x_data = cp.Parameter(shape=(11,))
y_data = cp.Parameter(shape=(11,))
objective = cp.Minimize(cp.sum_squares(y_data - m * x_data - c))
problem = cp.Problem(objective)

# Transform the problem into a cvxpylayer
layer = CvxpyLayer(problem, parameters=[x_data, y_data], variables=[m, c])

# Generate synthetic data
torch.manual_seed(42)  # for reproducibility
x_values = torch.linspace(0, 10, 10)
y_values = 2 * x_values + 1 + torch.randn_like(x_values)  # y = 2x + 1 + noise

# Given observed m and c, compute the x_data and y_data that would make m and c optimal
observed_m = torch.tensor([1.8], requires_grad=True)  # close to 2 (the true slope)
observed_c = torch.tensor([1.2], requires_grad=True)  # close to 1 (the true intercept)

# Placeholder for x_star and y_star, the influential data point we want to infer
x_star = torch.tensor([5.0], requires_grad=True)
y_star = torch.tensor([11.0], requires_grad=True)

# Use gradient descent to solve the inverse problem
optimizer = torch.optim.Adam([x_star, y_star], lr=0.01)
for i in range(500):
    optimizer.zero_grad()
    
    # Use observed data + x_star and y_star
    x_combined = torch.cat([x_values, x_star])
    y_combined = torch.cat([y_values, y_star])
    
    predicted_m, predicted_c = layer(x_combined, y_combined)
    
    loss = (predicted_m - observed_m)**2 + (predicted_c - observed_c)**2
    loss.backward()
    optimizer.step()

x_star_value, y_star_value = x_star.item(), y_star.item()

x_star_value, y_star_value



import numpy as np
import matplotlib.pyplot as plt

# Given the previous setup, let's use the values directly for illustration:
# (For this example, I'll assume synthetic values for x_star and y_star, and also for observed_m and observed_c)

# x_values = np.linspace(0, 10, 10)
# y_values = 2 * x_values + 1 + np.random.randn(10)  # y = 2x + 1 + noise

# observed_m = 1.8
# observed_c = 1.2

# # Synthetic values for x_star and y_star
# x_star_value = 5.0
# y_star_value = 11.0

# Plotting
plt.figure(figsize=(10, 6))
plt.scatter(x_values, y_values, label='Original Data', color='blue')
plt.scatter(x_star_value, y_star_value, label='Inferred Point $(x^*, y^*)$', color='red', marker='*')
plt.plot(x_values.detach(), observed_m.detach() * x_values.detach() + observed_c.detach(), label='Observed Line', color='green')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Inverse Optimization with CVXPYLayers')
plt.legend()
plt.grid(True)
plt.show()