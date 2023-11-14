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


standard_env=gym.make('CartPole-v0')

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))





# Custom env = measurement game


class CustomEnv(gym.Env):
  """Custom Environment that follows gym interface"""
  metadata = {'render.modes': ['human']}

  def __init__(self, meas_cost):
    super(CustomEnv, self).__init__()
    # Define action and observation space
    t_max=50
    
    self.t_max=t_max
    self.n_action=2
    self.action_space = spaces.Discrete(self.n_action)
    # Example for using image as input (channel-first; channel-last also works):
    self.observation_space = spaces.Box(low=np.array([0,-10,0,-10]), high=np.array([t_max,10,t_max,10]),
                                        shape=(4,), dtype=np.float32)
    self.max_epoch=t_max-1
    self.epoch=0
    self.t=np.linspace(0,1,t_max)
    self.meas_cost=meas_cost
       
    K_mat=np.zeros([t_max,t_max])
    for k in range(t_max):
        for l in range(t_max):
            K_mat[k,l]=self.cov_fun(self.t[k],self.t[l])
    
    self.K_mat=K_mat
    self.fun=np.random.multivariate_normal(np.zeros([t_max]),K_mat)
    
    self.state=np.vstack((0,0,0,0)).flatten()
    self.state_sequence=self.state
    self.meas_sequence=np.empty([0,1])
    self.t_meas_sequence=np.empty([0,1])
    self.total_meas=0
    self.accum_reward=0
    self.fun_hat=0

  def cov_fun(self,t,s):
    # cov_val=np.exp(-np.abs((t-s)/1.2)**2)
    cov_val=np.exp(-np.abs((t-s)/0.2)**2)
    # cov_val=np.min(np.array([s,t]))
    return cov_val
    

  def step(self, action):
   
    if action==0:
        meas_done=0
    else:
        meas_done=1
    self.total_meas=self.total_meas+meas_done
            
    self.epoch=self.epoch+1
    done = bool(self.epoch == self.max_epoch)
     
    if meas_done==0:
        state_change=np.array([+1,0,+1,0])
        new_state=self.state+state_change
    if meas_done==1:
        new_state=np.array([0,self.fun[self.epoch],self.state[0]+1,self.state[1]])
        self.meas_sequence=np.vstack((self.meas_sequence,np.array([new_state[1]])))  
        self.t_meas_sequence=np.vstack((self.t_meas_sequence,np.array([self.t[self.epoch]])))          
    
    self.state=new_state
    self.state_sequence=np.vstack((self.state_sequence,new_state))
    
    rmse=0
    if done == True:
        
        K_t=np.zeros([self.t_max,self.total_meas])
        K_ij=np.zeros([self.total_meas,self.total_meas])
        
        for k in range(self.t_max):
            for l in range(self.total_meas):
                K_t[k,l]=self.cov_fun(self.t[k],self.t_meas_sequence[l])
                
        for k in range(self.total_meas):
            for l in range(self.total_meas):
                K_ij[k,l]=self.cov_fun(self.t_meas_sequence[k],self.t_meas_sequence[l])
        
        fun_hat=K_t@np.linalg.pinv(K_ij)@self.meas_sequence
        rmse=np.linalg.norm(self.fun-fun_hat.squeeze())
        self.fun_hat=fun_hat
    else:
        pass
    
    reward= -meas_done*self.meas_cost-done*rmse
    self.accum_reward=self.accum_reward+reward

    # Optionally we can pass additional info, we are not using that for now
    info = {}

    return self.state.astype(np.float32), reward, done, info


  def reset(self):

    t_max=self.t_max
    self.epoch=0
            
    self.fun=np.random.multivariate_normal(np.zeros([t_max]),self.K_mat)
    
    self.state=np.vstack((0,0,0,0)).flatten()
    self.state_sequence=self.state
    self.meas_sequence=np.empty([0,1])
    self.t_meas_sequence=np.empty([0,1])
    self.total_meas=0
    self.accum_reward=0
    self.fun_hat=0
    
    observation=self.state
    
    return observation.astype(np.float32)  # reward, done, info can't be included


  def render(self, reward, mode='console'):
    if mode != 'console':
      raise NotImplementedError()
    plt.figure(1,dpi=300)
    plt.plot(self.t,self.fun,linestyle='solid',color='0')
    plt.plot(self.t,self.fun_hat,linestyle='dashed',color='0')
    plt.scatter(self.t_meas_sequence,self.meas_sequence)
    plt.title('Measurement times for cost = 0.1')
    plt.xlabel('Time')
    plt.ylabel('Function value')
    print(reward)
    
  def close (self):
      pass
      
      
new_env=CustomEnv(0.001)
check_env(new_env)




from stable_baselines3 import PPO, A2C, DQN


model = PPO("MlpPolicy", new_env,verbose=1)
model.learn(total_timesteps=1000)

obs = new_env.reset()

n_episodes=3

for k in range(n_episodes):
    done=False
    obs = new_env.reset()
    while done ==False:
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done, info = new_env.step(action)

        if done:
            new_env.render(new_env.accum_reward)
            # time.sleep(0.5)
            break












