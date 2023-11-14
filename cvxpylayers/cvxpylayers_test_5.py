#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 10 10:07:26 2023

@author: jemil
"""


import torch
import torch.nn as nn
import torch.optim as optim
from cvxpylayers.torch import CvxpyLayer
import numpy as np
import matplotlib.pyplot as plt
import cvxpy as cp

# 1. Define a simple neural network to predict b
class SimplePredictor(nn.Module):
    def __init__(self):
        super(SimplePredictor, self).__init__()
        self.fc = nn.Linear(1, 1)

    def forward(self, x):
        return self.fc(x)

model = SimplePredictor()

# 2. Define the cvxpy optimization layer
A = cp.Parameter(1)
x = cp.Variable(1)
b_param = cp.Parameter(1)
objective = cp.Minimize(cp.abs(x))
constraints = [A @ x <= b_param]
problem = cp.Problem(objective, constraints)
optim_layer = CvxpyLayer(problem, parameters=[A, b_param], variables=[x])

# Fixed A for simplicity
A_value = torch.Tensor([1.0])

# 3. Generate synthetic data for training
X_train = torch.linspace(-1, 1, 100).unsqueeze(1)
y_train = 2.0 - 2.0 * X_train

# 4. Train the model
optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
losses = []

for k in range(10):  # 100 epochs for demonstration
    print("Epoch {}".format(k))
    optimizer.zero_grad()
    
    # Predict b
    predicted_b = model(X_train)
    
    # Solve optimization problem to get x
    x_val, = optim_layer(A_value, predicted_b)
    
    # Loss is x (since we want to minimize x)
    loss = x_val.mean()
    loss.backward()
    
    optimizer.step()
    losses.append(loss.item())

# 5. Plot the training loss
plt.plot(losses)
plt.xlabel('Epoch')
plt.ylabel('Value of |x|')
plt.title('Training Loss')
plt.show()