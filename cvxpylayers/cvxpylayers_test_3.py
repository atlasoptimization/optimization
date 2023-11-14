#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Problem: Energy Consumption Minimization
Imagine you're trying to control the temperature of a building. You have a heater,
 and you want to minimize the energy it uses over a day. However, you also want
 to ensure the building's temperature remains comfortable.

You use a neural network to predict the outside temperature for the next 24 hours
 based on historical data. With this prediction, you use a convex optimization 
 problem to determine how to control the heater to minimize energy while keeping
 the indoor temperature within a comfort range.

Prediction: A neural network predicts the outside temperature for the next 24 hours.
Optimization: Given the predicted outside temperature, determine the heating schedule
 to minimize energy while keeping indoor temperature within a specified range
 (e.g., 20°C to 25°C).

However, the power of cvxpylayers comes into play when you want to train a model
end-to-end, where the neural network's predictions are directly informed by the
results of the optimization problem.

Let's dive deeper into a scenario where this differentiation matters:

Scenario: Predictive Control with Forecast Uncertainty
Suppose you're not just predicting the outside temperature but also some parameters
 of your building's thermal dynamics (maybe because they're not well-known, they
 change over time, or there's some uncertainty in your model).

For instance, you could use a neural network to predict:

The outside temperature for the next 24 hours.
A parameter α that captures how quickly your building loses heat to the outside.
The control problem would then be to determine the heating schedule that minimizes
 energy consumption while keeping the indoor temperature comfortable, given the
 predicted outside temperature and the predicted heat loss parameter α.

Now, suppose you have data on the actual indoor temperatures resulting from
 various heating schedules in the past. If your predicted α is off, then 
 following the optimal heating schedule based on that incorrect α would lead to
 indoor temperatures that deviate from the comfort range.

By using cvxpylayers, you can backpropagate through the optimization problem to
adjust the neural network's predictions of α. This is where the differentiability
of the convex optimization comes into play. You're effectively training the neural
network to make predictions that lead to more accurate control decisions.

Here's a rough outline:

The neural network predicts the outside temperature and α. These predictions are
fed into the convex optimization problem to determine the heating schedule.
This heating schedule is then "applied" to the building (in a simulated environment
or using historical data), resulting in indoor temperatures. The loss is based 
on deviations of these indoor temperatures from the desired comfort range.
This loss is backpropagated through the optimization problem and then through the
neural network to adjust its predictions. By doing this end-to-end training, the
 neural network learns to make predictions that are not just accurate in a conventional
sense, but also useful for the specific control task at hand.
"""
import torch
import torch.nn as nn
import numpy as np
import cvxpy as cp
from cvxpylayers.torch import CvxpyLayer
import matplotlib.pyplot as plt

# Neural Network to Predict Outside Temperature and Temperature Offset
class SimplifiedTempPredictor(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimplifiedTempPredictor, self).__init__()
        self.temp_fc = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size)
        )
        self.delta_fc = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size),
            nn.Tanh()  # Tanh ensures the offset is within a reasonable range (-1 to 1)
        )

    def forward(self, x):
        return self.temp_fc(x), self.delta_fc(x)

# Initialize the simplified temperature predictor
simplified_temp_model = SimplifiedTempPredictor(input_size=7, hidden_size=10, output_size=24)

# CVXPY Optimization Layer for Simplified Heating Control
predicted_temp = cp.Parameter(24)
delta = cp.Parameter(24)
heating_schedule = cp.Variable(24)
energy_cost = cp.sum_squares(heating_schedule)

# Adjusting the temperature prediction linearly using delta
temp_constraints = [heating_schedule + predicted_temp + delta >= 20,
                    heating_schedule + predicted_temp + delta <= 25]

heating_problem_simplified = cp.Problem(cp.Minimize(energy_cost), temp_constraints)
heating_layer_simplified = CvxpyLayer(heating_problem_simplified, parameters=[predicted_temp, delta], variables=[heating_schedule])

# Synthetic Data Generation
def generate_data(n_samples):
    heights = np.random.uniform(150, 190, n_samples)
    ages = np.random.uniform(20, 60, n_samples)
    outside_temps = 0.25 * heights + 0.5 * ages + np.random.normal(0, 5, (n_samples, 24))
    heights = (heights - heights.mean()) / heights.std()
    ages = (ages - ages.mean()) / ages.std()
    X = np.column_stack((heights, ages))
    return torch.Tensor(X), torch.Tensor(outside_temps)

X_data, y_temp_data = generate_data(n_samples=200)

# Train the model end-to-end
optimizer_simplified = torch.optim.Adam(simplified_temp_model.parameters(), lr=0.01)
loss_fn = nn.MSELoss()

num_epochs = 100  # Reduced for brevity
losses_simplified = []

for epoch in range(num_epochs):
    temp_preds, delta_preds = simplified_temp_model(X_data)
    heating_schedules, = heating_layer_simplified(temp_preds, delta_preds)
    
    # Simulated indoor temperatures based on heating schedules
    indoor_temps = heating_schedules + y_temp_data + delta_preds
    
    # Loss based on deviations from the comfort range
    loss = loss_fn(indoor_temps, 22.5 * torch.ones_like(indoor_temps))  # 22.5 is mid of the comfort range
    losses_simplified.append(loss.item())
    
    optimizer_simplified.zero_grad()
    loss.backward()
    optimizer_simplified.step()

# Visualization
plt.plot(losses_simplified)
plt.title('Simplified Training Loss')
plt.xlabel('Epoch')
plt.ylabel('MSE Loss')
plt.show()
