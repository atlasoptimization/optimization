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


import time




Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))





# Custom env = measurement game


class CustomEnv(gym.Env):
  """Custom Environment that follows gym interface"""
  metadata = {'render.modes': ['human']}

  def __init__(self):
    super(CustomEnv, self).__init__()
    # Define action and observation space
    
    self.n_meas=3
    self.n_state=2*self.n_meas
    t_max=50
    self.t_max=t_max
    self.action_space = spaces.Box(low=-1, high=1,
                                        shape=(1,), dtype=np.float32)
    # Example for using image as input (channel-first; channel-last also works):
    self.observation_space = spaces.Box(low=-1, high=1,
                                        shape=(6,), dtype=np.float32)
    self.max_epoch=4
    self.t=np.linspace(0,1,t_max)
    self.t_measured=[]
    self.measurements=[]
    self.epoch=0
    
    self.bump_fun=np.zeros([self.t_max])
    rand_index=np.random.randint(0,t_max-1)
    self.bump_center=rand_index
    
    for k in range(t_max):
        self.bump_fun[k]=self.bump_function(self.t[k],self.t[rand_index])
    
    self.state=np.zeros([self.n_state])
    
  def round_to_index(self,t):
      approx_index=t*self.t_max
      index=np.ceil(approx_index)-1
      return index.astype(int)
    
  def bump_function(self,t,s):
    # cov_val=np.exp(-np.abs((t-s)/1.2)**2)
    cov_val=np.exp(-np.abs((t-s)/0.1)**2)

    return cov_val


  def step(self, action):
       
      
    self.epoch=self.epoch+1    
    t_measured=0.5*(action)+0.5
    fun_measured=self.bump_function(t_measured,self.t[self.bump_center])
    
    augmented_t_measured=np.hstack((self.state[0:self.n_meas],t_measured))
    augmented_f_measured=np.hstack((self.state[self.n_meas:self.n_state],fun_measured))
    
    indices_largest_three=augmented_f_measured.argsort()[::-1][0:3]
    
    new_t_measured=augmented_t_measured[indices_largest_three]
    new_f_measured=augmented_f_measured[indices_largest_three]
    
    self.state=np.hstack((new_t_measured, new_f_measured))
    
    self.t_measured=np.hstack((self.t_measured,t_measured))
    self.measurements=np.hstack((self.measurements,fun_measured))
    
    reward=-np.abs(1-new_f_measured[0])
    
    if self.epoch==self.max_epoch:
        done=True
    else:
        done=False

    # Optionally we can pass additional info, we are not using that for now
    info = {}

    return self.state, reward, done, info


  def reset(self):

    self.t_measured=[]
    self.measurements=[]
    self.epoch=0
    
    self.bump_fun=np.zeros([self.t_max])
    rand_index=np.random.randint(0,self.t_max-1)
    self.bump_center=rand_index
    
    for k in range(self.t_max):
        self.bump_fun[k]=self.bump_function(self.t[k],self.t[rand_index])
    
    self.state=np.zeros([self.n_state])
    observation=self.state
    
    return observation  # reward, done, info can't be included


  def render(self, reward, mode='console'):
    if mode != 'console':
      raise NotImplementedError()
    
    # plt.plot(self.t,self.fun,linestyle='solid',color='0')
    plt.plot(self.t,self.bump_fun,linestyle='dashed',color='0')
    plt.scatter(self.t_measured,self.measurements)
    plt.title('Measured locations')
    plt.xlabel('Location')
    plt.ylabel('Bump function value')
    print(reward)
    
  def close (self):
      pass
      
      
new_env=CustomEnv()
# new_env.step(0.5)
check_env(new_env)



import stable_baselines3 as sb3
from stable_baselines3 import PPO, A2C, DQN, DDPG, TD3
# from stable_baselines3.common.env_util import make_vec_env

# env_vec=sb3.common.vec_env.VecEnv(4, new_env.observation_space,new_env.action_space)


# model = PPO("MlpPolicy", new_env,verbose=1)

from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise
# action_noise=OrnsteinUhlenbeckActionNoise(np.zeros(1), 0.1*np.ones(1))
# model = TD3("MlpPolicy", new_env,verbose=1,action_noise=action_noise)
model = TD3("MlpPolicy", new_env,verbose=1)
model.learn(total_timesteps=100)

obs = new_env.reset()

n_episodes=3

for k in range(n_episodes):
    done=False
    obs = new_env.reset()
    while done ==False:
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done, info = new_env.step(action)

        if done:
            new_env.render(reward)
            # time.sleep(0.5)
            break




