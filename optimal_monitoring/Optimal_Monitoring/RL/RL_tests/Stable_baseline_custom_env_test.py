import math
import random
import numpy as np
import matplotlib
import copy
import matplotlib.pyplot as plt
from collections import namedtuple, deque
from itertools import count

import stable_baselines3 as sb
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import gym
from gym import spaces
from gym import utils
from gym.utils import seeding

from stable_baselines3.common.env_checker import check_env

standard_env=gym.make('CartPole-v0')

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))



# # Custom env = Border walking


# class CustomEnv(gym.Env):
#   """Custom Environment that follows gym interface"""
#   metadata = {'render.modes': ['human']}

#   def __init__(self):
#     super(CustomEnv, self).__init__()
#     # Define action and observation space
#     # They must be gym.spaces objects
#     # Example when using discrete actions:
#     self.state=np.random.randint(-5,5)-0.5
#     self.action_space = spaces.Discrete(2)
#     # Example for using image as input (channel-first; channel-last also works):
#     self.observation_space = spaces.Box(low=-5, high=5,
#                                         shape=(1,), dtype=np.float32)
#     self.max_epoch=10
#     self.epoch=0



#   def step(self, action):
#     state_change=2*action-1
#     new_state=np.clip(self.state+state_change,-4.5,4.5)

#     # Account for the boundaries of the grid
#     reward=np.abs(np.sign(new_state)-np.sign(self.state))
#     self.state = new_state
#     self.epoch=self.epoch+1

#     # Are we at the left of the grid?
#     done = bool(self.epoch == self.max_epoch)


#     # Optionally we can pass additional info, we are not using that for now
#     info = {}

#     return np.array([self.state]).astype(np.float32), reward, done, info


#   def reset(self):
#     self.state = np.random.randint(-5,5)-0.5
#     # here we convert to float32 to make it more general (in case we want to use continuous actions)
#     observation = np.array([self.state]).astype(np.float32)
#     self.epoch=0
    
#     return observation  # reward, done, info can't be included


#   def render(self, reward, mode='console'):
#     if mode != 'console':
#       raise NotImplementedError()
#     # agent is represented as a cross, rest as a dot
#     print(self.state,reward)
    
#   def close (self):
#       pass
      
      
# new_env=CustomEnv()
# check_env(new_env)




# from stable_baselines3 import PPO


# model = PPO("MlpPolicy", new_env, verbose=1)
# model.learn(total_timesteps=100)

# obs = new_env.reset()
# for k in range(100):
#     action, _states = model.predict(obs, deterministic=True)
#     obs, reward, done, info = new_env.step(action)
#     new_env.render(reward)
#     if done:
#       obs = new_env.reset()






# Custom env = Landing


class CustomEnv(gym.Env):
  """Custom Environment that follows gym interface"""
  metadata = {'render.modes': ['human']}

  def __init__(self,x_target):
    super(CustomEnv, self).__init__()
    # Define action and observation space
    # They must be gym.spaces objects
    # Example when using discrete actions:
    z_max=10
    x=2*(np.random.normal(0,1))
    vx=0*(np.random.normal(0,1))
    self.n_action=2
    self.x_target=x_target
        
    self.state=np.vstack((z_max,x,vx)).flatten()
    self.state_sequence=self.state
    self.action_space = spaces.Discrete(self.n_action)
    # Example for using image as input (channel-first; channel-last also works):
    self.observation_space = spaces.Box(low=-10, high=10,
                                        shape=(3,), dtype=np.float32)
    self.max_epoch=z_max
    self.epoch=0



  def step(self, action):
    acceleration_value=(action)*(1/(self.n_action-1))-0.5
        
    state_change=np.array([-1,self.state[2],acceleration_value])       
    state=self.state       
    new_state=state+state_change                   
    
    self.state=new_state
    self.state_sequence=np.vstack((self.state_sequence,new_state))

    # Account for the boundaries of the grid
    reward=-np.abs(self.state[1]-self.x_target)
    self.state = new_state
    self.epoch=self.epoch+1

    # Are we at the left of the grid?
    done = bool(self.epoch == self.max_epoch)


    # Optionally we can pass additional info, we are not using that for now
    info = {}

    return self.state.astype(np.float32), reward, done, info


  def reset(self):
    z_max=10
    x=2*(np.random.normal(0,1))
    vx=0*(np.random.normal(0,1))

    self.state=np.vstack((z_max,x,vx)).flatten()
    self.state_sequence=self.state
    observation=self.state
    self.max_epoch=z_max
    self.epoch=0
    
    return observation.astype(np.float32)  # reward, done, info can't be included


  def render(self, reward, mode='console'):
    if mode != 'console':
      raise NotImplementedError()
    # agent is represented as a cross, rest as a dot
    # print(reward)
    
  def close (self):
      pass
      
      
new_env=CustomEnv(0)
check_env(new_env)




from stable_baselines3 import PPO, A2C, DQN


model = PPO("MlpPolicy", new_env, verbose=1)
model.learn(total_timesteps=5000)

obs = new_env.reset()
n_episodes=10
x_sequence=np.zeros([10,n_episodes])
v_sequence=np.zeros([10,n_episodes])

for k in range(n_episodes):
    for l in range(10):
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done, info = new_env.step(action)
        x_sequence[l,k]=new_env.state[1]
        v_sequence[l,k]=new_env.state[2]
        new_env.render(reward)
        if done:
          obs = new_env.reset()

plt.plot(x_sequence)
# plt.plot(v_sequence)










