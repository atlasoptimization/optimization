
##############################################################################
import gym
import math
import random
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple, deque
from itertools import count
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T



# set up matplotlib
is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    from IPython import display

plt.ion()

# if gpu is to be used
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Control_landing_env_continuous():
    
    def __init__(self, dims, x_target):
    
        z_max=5
        n_a=dims[0]
        x=3*(np.random.normal(0,1))
        vx=0*(np.random.normal(0,1))
        
        self.state=np.vstack((z_max,x,vx))
        
        self.epoch=0
        self.max_epoch=z_max
        self.state_sequence=self.state
        self.action_sequence=[]
        self.reward_sequence=[]
        self.n_action=n_a
        self.dim=np.array([3,n_a])
        self.z_max=z_max
        self.x_target=x_target
        
        
    
    def step(self,action):    
        
        acceleration_value=(action)*(1/(self.n_action-1))-0.5
        
        state_change=np.array([[-1],[self.state[2][0]],[acceleration_value]])       
        state=self.state       
        new_state=state+state_change                   
        
        self.state=new_state
        self.state_sequence=np.hstack((self.state_sequence,new_state))
        self.action_sequence.append(acceleration_value)
        
        reward=-np.abs(self.state[1]-self.x_target)
        self.reward_sequence=np.hstack((self.reward_sequence,reward))
        self.last_transition=Transition(state,action,new_state,reward)

        
        self.epoch=self.epoch+1
        if self.epoch==self.max_epoch:
            done=True
        else:
            done=False
        
        
        return reward, done
        
        
        
    def reset(self):

        x=3*(np.random.normal(0,1))
        vx=0*(np.random.normal(0,1))
        
        self.state=np.vstack((self.max_epoch,x,vx))
        
        self.epoch=0
        self.state_sequence=self.state
        self.action_sequence=[]
        self.reward_sequence=[]


env=Control_landing_env_continuous(np.array([2]), 0)



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
    
n_action = env.n_action
    
    
##############################################################################
class DQN(nn.Module):

    def __init__(self):
        super(DQN, self).__init__()
        
        n_hidden=5
        self.fc1=nn.Linear(3, n_hidden)
        self.fc2=nn.Linear(n_hidden, n_hidden)
        self.fc3=nn.Linear(n_hidden, n_hidden)
        self.fc4=nn.Linear(n_hidden, n_action)


    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        # x = x.to(device)
        x = torch.tanh(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        x = torch.tanh(self.fc3(x))
        x = torch.tanh(self.fc4(x))
        return x
    
##############################################################################

resize = T.Compose([T.ToPILImage(),
                    T.Resize(40, interpolation=Image.CUBIC),
                    T.ToTensor()])


##############################################################################

BATCH_SIZE = 128
GAMMA = 0.9
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 200
TARGET_UPDATE = 50


policy_net = DQN().to(device)
target_net = DQN().to(device)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

optimizer = optim.RMSprop(policy_net.parameters())
memory = ReplayMemory(10000)


steps_done = 0


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
            return torch.tensor([[policy_net(state.squeeze().float()).max(0)[1]]])
    else:
        return torch.tensor([[random.randrange(n_action)]], device=device, dtype=torch.long)


episode_durations = []

        
        
##############################################################################

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
    non_final_next_states = torch.cat([s for s in batch.next_state
                                                if s is not None],1)
    state_batch = torch.cat(batch.state,1)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
    # columns of actions taken. These are the actions which would've been taken
    # for each batch state according to policy_net
    state_action_values = policy_net(state_batch.T.float()).gather(1, action_batch)

    # Compute V(s_{t+1}) for all next states.
    # Expected values of actions for non_final_next_states are computed based
    # on the "older" target_net; selecting their best reward with max(1)[0].
    # This is merged based on the mask, such that we'll have either the expected
    # state value or 0 in case the state was final.
    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    next_state_values[non_final_mask] = target_net(non_final_next_states.T.float()).max(1)[0].detach()
    # Compute the expected Q values
    expected_state_action_values = (next_state_values.unsqueeze(1) * GAMMA) + reward_batch

    # Compute Huber loss
    criterion = nn.SmoothL1Loss()
    loss = criterion(state_action_values.float(), expected_state_action_values.float())

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    for param in policy_net.parameters():
        param.grad.data.clamp_(-1, 1)
    optimizer.step()
    
##############################################################################

num_episodes = 100
for i_episode in range(num_episodes):
    # Initialize the environment and state
    print(i_episode)
    env.reset()

    state = torch.from_numpy(env.state)
    for t in count():
        # Select and perform an action
        action = select_action(state)
        reward, done = env.step(action.item())
        reward = torch.tensor([reward], device=device)

        # Observe new state

        if not done:
            next_state = torch.from_numpy(env.state)
        else:
            next_state = None

        # Store the transition in memory
        memory.push(state, action, next_state, reward)

        # Move to the next state
        state = next_state

        # Perform one step of the optimization (on the policy network)
        optimize_model()
        if done:
            break
    # Update the target network, copying all weights and biases in DQN
    if i_episode % TARGET_UPDATE == 0:
        target_net.load_state_dict(policy_net.state_dict())

print('Complete')




print(env.state_sequence)
print(env.action_sequence)
print(env.reward_sequence)



# iii) Plot evaluation rounds

n_episodes_eval=10
z_trajectory_sample=np.zeros([env.z_max,n_episodes_eval])
x_trajectory_sample=np.zeros([env.z_max,n_episodes_eval])
vx_trajectory_sample=np.zeros([env.z_max,n_episodes_eval])

for k_episode in range(n_episodes_eval):
    
    env.reset()
    
    for t in count():
        
        z_trajectory_sample[t,k_episode]=env.state[0]
        x_trajectory_sample[t,k_episode]=env.state[1]
        vx_trajectory_sample[t,k_episode]=env.state[2]
        
        action =select_action(torch.from_numpy(env.state))
        reward,done=env.step(action)
        
        if done:
            break
plt.figure(1)      
plt.plot(x_trajectory_sample)


t_vals=np.linspace(-3,3,100)
x_vals=torch.from_numpy(np.array([np.linspace(5,5,100), np.linspace(-3,3,100),np.linspace(0,0,100)]))
q=policy_net(x_vals.reshape([100,3]).float())

plt.figure(2)
plt.plot(t_vals,q[:,0].detach())
plt.plot(t_vals,q[:,1].detach())
plt.show()





##############################################################################





































