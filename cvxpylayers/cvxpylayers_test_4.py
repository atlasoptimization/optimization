#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Problem: Portfolio Optimization with Neural Network Predictions
Imagine you have a portfolio of two assets, and you want to decide how much to invest in each to maximize expected returns while managing risk.

You use a neural network to predict the expected returns of the two assets based on some historical data. With this prediction, you use a convex optimization problem to determine how to allocate your investment.

Here's the setup:

Prediction: A neural network predicts the expected returns of the two assets.
Optimization: Given the predicted expected returns, determine the optimal investment allocation to maximize returns while keeping risk (variance) below a certain threshold.

"""


import torch
import torch.nn as nn
import torch.optim as optim
from cvxpylayers.torch import CvxpyLayer
import numpy as np
import matplotlib.pyplot as plt

# 1. Define the neural network to predict expected returns
class ReturnPredictor(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(ReturnPredictor, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 2)
        )

    def forward(self, x):
        return self.fc(x)

model = ReturnPredictor(input_size=5, hidden_size=10)

# 2. Define the cvxpy optimization layer for portfolio allocation
predicted_returns = cp.Parameter(2)
allocation = cp.Variable(2, nonneg=True)
portfolio_return = cp.sum(predicted_returns * allocation)
risk_threshold = 0.05
portfolio_variance = cp.quad_form(allocation, np.array([[0.1, 0.03], [0.03, 0.12]]))
objective = cp.Maximize(portfolio_return)
constraints = [cp.sum(allocation) == 1, portfolio_variance <= risk_threshold]
problem = cp.Problem(objective, constraints)
allocation_layer = CvxpyLayer(problem, parameters=[predicted_returns], variables=[allocation])

# 3. Generate synthetic data for training
def generate_data(n_samples):
    X = np.random.randn(n_samples, 5)
    y = 0.1 * X.sum(axis=1) + np.random.normal(0, 0.1, n_samples)
    return torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)

X_train, y_train = generate_data(1000)

# 4. Train the model end-to-end
optimizer = optim.Adam(model.parameters(), lr=0.01)
losses = []

for _ in range(100):  # 100 epochs for demonstration
    optimizer.zero_grad()
    
    # Predict expected returns
    predicted = model(X_train)
    
    # Get optimal allocation based on predicted returns
    allocated, = allocation_layer(predicted)
    
    # Loss is negative of expected portfolio return (we want to maximize returns)
    loss = -torch.sum(allocated * predicted)
    loss.backward()
    
    optimizer.step()
    losses.append(loss.item())

# 5. Plot the training loss
plt.plot(losses)
plt.xlabel('Epoch')
plt.ylabel('Negative Expected Return')
plt.title('Training Loss')
plt.show()