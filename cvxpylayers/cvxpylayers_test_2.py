#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
The goal of this script is to showcase how a two-stage process can be employed
to augment a feed-forward network with a bound via projection.
Then we use cvxpylayers to train the network in such a way that the final 
product (ann + cvpy projection) fits best.
"""


# import torch
# import torch.nn as nn
# import cvxpy as cp
# from cvxpylayers.torch import CvxpyLayer

# # 1. Neural Network Definition
# class WeightPredictor(nn.Module):
#     def __init__(self):
#         super(WeightPredictor, self).__init__()
#         self.fc = nn.Sequential(
#             nn.Linear(2, 10),  # Input: [height, age]
#             nn.ReLU(),
#             nn.Linear(10, 1)
#         )

#     def forward(self, x):
#         return self.fc(x)

# # Initialize the neural network
# model = WeightPredictor()

# # 2. CVXPY Projection Layer
# w_pred = cp.Parameter()  # input from the neural network
# w = cp.Variable()  # projected weight

# objective = cp.Minimize((w - w_pred)**2)
# constraints = [w >= 40, w <= 200]
# problem = cp.Problem(objective, constraints)

# # Convert the CVXPY problem into a CvxpyLayer
# projection_layer = CvxpyLayer(problem, parameters=[w_pred], variables=[w])

# model, projection_layer
# Import necessary libraries
import torch
import torch.nn as nn
import cvxpy as cp
from cvxpylayers.torch import CvxpyLayer
import numpy as np
import matplotlib.pyplot as plt

# Neural Network Definition
class WeightPredictor(nn.Module):
    def __init__(self):
        super(WeightPredictor, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(2, 2),
            nn.ReLU(),
            nn.Linear(2, 1)
        )

    def forward(self, x):
        return self.fc(x)

# Initialize the neural network
model = WeightPredictor()

# CVXPY Projection Layer
w_pred = cp.Parameter()  
w = cp.Variable()
objective = cp.Minimize((w - w_pred)**2)
constraints = [w >= 60, w <= 200]
problem = cp.Problem(objective, constraints)
projection_layer = CvxpyLayer(problem, parameters=[w_pred], variables=[w])

# Synthesize data
np.random.seed(0)
torch.manual_seed(0)
n_samples = 200
heights = np.random.uniform(150, 190, n_samples)
ages = np.random.uniform(20, 60, n_samples)
weights = 0.25 * heights + 0.5 * ages + np.random.normal(0, 00, n_samples)
heights = (heights - heights.mean()) / heights.std()
ages = (ages - ages.mean()) / ages.std()
X = torch.Tensor(np.column_stack((heights, ages)))
y_true = torch.Tensor(weights)

# Train the model
optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
loss_fn = nn.MSELoss()
num_epochs = 100
losses = []

for epoch in range(num_epochs):
    y_pred = model(X).squeeze()
    # y_projected = y_pred    # works just fine without projection layer
    y_projected, = projection_layer(y_pred) # does not work with projection layer
    loss = loss_fn(y_projected, y_true)
    losses.append(loss.item())
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    print('Epoch : {}, loss : {}'.format(epoch, loss.item()))

# Visualization
plt.figure(figsize=(14, 6))
plt.subplot(1, 2, 1)
plt.plot(losses)
plt.title('Training Loss')
plt.xlabel('Epoch')
plt.ylabel('MSE Loss')

plt.subplot(1, 2, 2)
plt.scatter(y_true, model(X).detach().numpy(), label='NN Predictions', alpha=0.7)
plt.scatter(y_true, projection_layer(model(X).flatten())[0].detach().numpy(), label='Projected Predictions', alpha=0.7)
plt.plot([min(y_true), max(y_true)], [min(y_true), max(y_true)], '--', color='black', label='y=x line')
plt.title('True Weight vs Predictions')
plt.xlabel('True Weight')
plt.ylabel('Predicted Weight')
plt.legend()
plt.tight_layout()
plt.show()




