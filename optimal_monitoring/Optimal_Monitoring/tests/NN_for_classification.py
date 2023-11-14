"""
The goal is to train a very simple classifier using pytorch
"""


"""
Imports and Definitions
"""

import math
import random
import numpy as np
import matplotlib
import copy
import matplotlib.pyplot as plt
from collections import namedtuple, deque
from itertools import count


import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


n_state=3





device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


"""
Generate Data
"""

# Rule: When all three numbers are positive: 1, else -1

def true_fun(x):
    
    if x[0]>=0 and x[1]>=0 and x[2]>=0:
        funval=1
    else:
        funval=-1
    
    return funval




"""
Define NN
"""

class ANN(nn.Module):

    def __init__(self):
        super(ANN, self).__init__()
        self.fc1 = nn.Linear(n_state, n_state)
        self.fc2 = nn.Linear(n_state, n_state)
        self.fc3 = nn.Linear(n_state, n_state)
        self.fc4 = nn.Linear(n_state, n_state)
        self.fc5 = nn.Linear(n_state,1)



    # Define nn forward pass
    def forward(self, x):
        #x = x.to(device)
        x = torch.tanh(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        x = torch.tanh(self.fc3(x))
        x = torch.tanh(self.fc4(x))
        x = torch.tanh(self.fc5(x))
        return x

class_nn=ANN()
class_nn.train()
class_nn.zero_grad()


"""
 Train NN

"""

n_episodes=200
optimizer=optim.RMSprop(class_nn.parameters())

for k in range(n_episodes):
    x=np.random.multivariate_normal(np.zeros([3]),np.eye(3))
    x=torch.from_numpy(x)
        
    optimizer.zero_grad()
    
    label_true=torch.from_numpy(np.array([true_fun(x)]))
    label_pred=class_nn(x.float())
    
    criterion=nn.SmoothL1Loss() 
    loss=criterion(label_true.float(),label_pred.float())
    # print(loss)
    
    
    # print('fc1.bias.grad before backward')
    # print(class_nn.fc1.bias.grad)
    
    loss.backward()
    
    # print('fc1.bias.grad after backward')
    # print(class_nn.fc1.bias.grad)
    
    
    optimizer.step()
    
class_nn.eval()
n_test=20
for k in range(n_test):
    x=np.random.multivariate_normal(np.zeros([3]),np.eye(3))
    x=torch.from_numpy(x)
    
    
    label_true=torch.from_numpy(np.array([true_fun(x)]))
    label_pred=class_nn(x.float())
    
    print((label_true-label_pred).squeeze())












"""
Validate NN
"""


































