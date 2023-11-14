"""
The goal of this script is to explore reinforcement learning for a simple problem 
that deals with the distribution of measurements. A neural network is trained to
optimally choose an action that maximizes reconstruction quality of a scalar function.

For this, do the following:
    1. Definitions and imports
    2. Define the environment
    3. Set up general classes
    4. Specification of actor
    5. Optimization function
    6. Training the model
    7. Plots and illustrations 
    
"""



"""
    1. Definitions and imports
"""


# i) Imports of packages

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


# ii) Auxiliary parameters



n_state=3
n_action=2
dims=torch.tensor([n_state,n_action])



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



"""
    2. Define the environment
"""

# pass as input: z_max, dims

class Measurement_env():
    
    def __init__(self,z_max,dims):
    
        z=torch.randint(5, z_max, [1])
        x=torch.distributions.Normal(0,4).sample([1])
        v_x=torch.from_numpy(np.zeros([1]))
        
        self.state=torch.cat((z,x,v_x))
        self.epoch=0
        self.max_epoch=z
        self.state_sequence=self.state
        self.action_sequence=[]
        self.reward_sequence=torch.tensor([0])
        self.dim=dims
    
    
    def step(self,action):
        
        
        acceleration_value=(action.item())*(1/(self.dim[1]-1))-0.5
        
        state_change=torch.tensor([-1,self.state[2],acceleration_value])
        
        new_state=self.state+state_change 
        self.state_sequence=torch.vstack((self.state_sequence,new_state))
        self.action_sequence.append(acceleration_value)
        self.state=new_state

        
        reward=-np.abs(self.state[1])
        self.reward_sequence=torch.vstack((self.reward_sequence,reward))
        
        self.epoch=self.epoch+1
        if self.epoch==self.max_epoch:
            done=True
        else:
            done=False
        
        
        return reward, done
        
        
        
    def reset(self):
        z=torch.randint(5, z_max, [1])
        x=torch.distributions.Normal(0,4).sample([1])
        v_x=torch.from_numpy(np.zeros([1]))
        
        self.state=torch.cat((z,x,v_x))
        self.epoch=0
        self.max_epoch=z
        self.state_sequence=self.state
        self.action_sequence=[]
        self.reward_sequence=torch.tensor([0])
        self.dim=dims
        

# z max >=6
z_max=10
env=Measurement_env(z_max,dims)
env.reset()

# env.reset()


"""
    3. Set up general classes
"""


#############################################################################

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))


class ReplayMemory(object):

    def __init__(self, capacity):
        self.memory = deque([],maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)
    




"""
    4. Specification of actor
"""

##############################################################################

class DQN(nn.Module):

    def __init__(self):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(n_state, n_state)
        self.fc2 = nn.Linear(n_state, n_state)
        self.fc3 = nn.Linear(n_state,n_action)



    # Define nn forward pass
    def forward(self, x):
        #x = x.to(device)
        x = torch.tanh(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        x = torch.tanh(self.fc3(x))
        return x
    
    
# # Test
# my_nn=DQN()
# xxx=torch.rand([120])
# yyy=my_nn.forward(xxx)   
    
##############################################################################



BATCH_SIZE = 100
GAMMA = 0.999
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 200
TARGET_UPDATE = 10


policy_net = DQN()
#policy_net = policy_net.float()
target_net = DQN()
#target_net = target_net.float()


target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

optimizer = optim.RMSprop(policy_net.parameters())
memory = ReplayMemory(10000)


steps_done=0

def select_action(state):
    global steps_done
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * \
        math.exp(-1. * steps_done / EPS_DECAY)
    steps_done += 1
    if sample > eps_threshold:
        with torch.no_grad():
            # t.max(1) will return largest column value of each row.
            # second column on max result is index of where max element was
            # found, so we pick action with the larger expected reward.
            return policy_net(state.float()).max(0)[1].view(1, 1)
    else:
        return torch.tensor([[random.randrange(n_action)]], device=device, dtype=torch.long)


# one untrained run

sum_reward_untrained=0
env.reset()
for k in range(env.max_epoch):
    action=select_action(env.state)
    reward,done=env.step(action)
    sum_reward_untrained=sum_reward_untrained+reward


untrained_state_sequence=copy.deepcopy(env.state_sequence)
untrained_action_sequence=copy.deepcopy(env.action_sequence)



"""
    5. Optimization function
"""


def optimize_model():
    if len(memory) < BATCH_SIZE:
        return
    transitions = memory.sample(BATCH_SIZE)
    # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
    # detailed explanation). This converts batch-array of Transitions
    # to Transition of batch-arrays.
    batch = Transition(*zip(*transitions))

    # Compute a mask of non-final states and concatenate the batch elements
    # (a final state would've been the one after which simulation ended)
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                          batch.next_state)), device=device, dtype=torch.bool)
    non_final_next_states = torch.stack([s for s in batch.next_state
                                                if s is not None])
    state_batch = torch.stack(batch.state)
    action_batch = torch.stack(batch.action)
    action_batch=torch.squeeze(action_batch,2)
    reward_batch = torch.stack(batch.reward)

    # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
    # columns of actions taken. These are the actions which would've been taken
    # for each batch state according to policy_net
    state_action_values = policy_net(state_batch.float()).gather(1, action_batch)

    # Compute V(s_{t+1}) for all next states.
    # Expected values of actions for non_final_next_states are computed based
    # on the "older" target_net; selecting their best reward with max(1)[0].
    # This is merged based on the mask, such that we'll have either the expected
    # state value or 0 in case the state was final.
    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    next_state_values[non_final_mask] = target_net(non_final_next_states.float()).max(1)[0].detach()
    # Compute the expected Q values
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch.squeeze()

    # Compute Huber loss
    criterion = nn.SmoothL1Loss()
    loss = criterion(state_action_values.float(), expected_state_action_values.unsqueeze(1).float())

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    for param in policy_net.parameters():
        param.grad.data.clamp_(-1, 1)
    optimizer.step()
    
    print(loss)




"""
    6. Training the model
"""

num_episodes = 100
for i_episode in range(num_episodes):
    # Initialize the environment and state
    print(i_episode)
    env.reset()
    state = copy.deepcopy(env.state)
    for t in count():
        # Select and perform an action
        action = select_action(state)
        reward, done = env.step(action)
        reward = torch.tensor([reward], device=device)

        # Observe new state
        last_state = state
        if not done:
            next_state = env.state
        else:
            next_state = None

        # Store the transition in memory
        memory.push(state, action, next_state, reward)

        # Move to the next state
        state = copy.deepcopy(next_state)

        # Perform one step of the optimization (on the policy network)
        optimize_model()
        if done:
            break
    # Update the target network, copying all weights and biases in DQN
    if i_episode % TARGET_UPDATE == 0:
        target_net.load_state_dict(policy_net.state_dict())

print('Complete')






# """
#     7. Plots and illustrations 
# """

# # Let it run once random and once fully trained

# fully trained
sum_reward_trained=0
env.reset()
for k in range(env.max_epoch):
    action=select_action(env.state)
    reward,done=env.step(action)
    sum_reward_trained=sum_reward_trained+reward
    
    
print(untrained_state_sequence)
print(sum_reward_untrained)
print(untrained_action_sequence)

print(env.state_sequence)
print(sum_reward_trained)
print(env.action_sequence)




