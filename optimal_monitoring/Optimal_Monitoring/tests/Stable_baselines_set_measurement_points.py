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

  def __init__(self, meas_cost):
    super(CustomEnv, self).__init__()
    # Define action and observation space
    t_max=50
    
    self.t_max=t_max
    self.n_action=2
    self.total_meas=5
    self.action_space = spaces.Box(low=-1, high=1,
                                        shape=(5,), dtype=np.float32)
    # Example for using image as input (channel-first; channel-last also works):
    self.observation_space = spaces.Discrete(2)
    self.max_epoch=t_max-1
    self.t=np.linspace(0,1,t_max)
    self.t_measured=[]
    self.measurements=[]
       
    K_mat=np.zeros([t_max,t_max])
    for k in range(t_max):
        for l in range(t_max):
            K_mat[k,l]=self.cov_fun(self.t[k],self.t[l])
    
    self.K_mat=K_mat
    self.fun=np.random.multivariate_normal(np.zeros([t_max]),K_mat)
    
    self.state=0
    self.fun_hat=0
    
  def round_to_index(self,t):
      approx_index=t*self.t_max
      index=np.ceil(approx_index)-1
      return index.astype(int)

  def cov_fun(self,t,s):
    # cov_val=np.exp(-np.abs((t-s)/1.2)**2)
    cov_val=np.exp(-np.abs((t-s)/0.2)**2)
    # cov_val=np.min(np.array([s,t]))-s*t
    # cov_val=np.min(np.array([s,t]))
    return cov_val
    

  def step(self, action):
   
    done=True
    action=(0.5)*action+0.5
    self.action=action
        
    K_t=np.zeros([self.t_max,self.total_meas])
    K_ij=np.zeros([self.total_meas,self.total_meas])
    
    for k in range(self.t_max):
        for l in range(self.total_meas):
            K_t[k,l]=self.cov_fun(self.t[k],action[l])
            
    for k in range(self.total_meas):
        for l in range(self.total_meas):
            K_ij[k,l]=self.cov_fun(action[k],action[l])
    
    meas_indices=self.round_to_index(action)
    measurements=self.fun[meas_indices]
    fun_hat=K_t@np.linalg.pinv(K_ij,rcond=1e-6,hermitian=True)@measurements
    rmse=np.linalg.norm(self.fun-fun_hat.squeeze())
    self.fun_hat=fun_hat
    self.measurements=measurements
    self.t_measured=self.t[meas_indices]

    
    reward= -rmse

    # Optionally we can pass additional info, we are not using that for now
    info = {}

    return self.state, reward, done, info


  def reset(self):
      

    self.t_measured=[]
    self.measurements=[]
       
    t_max=self.t_max
    
    K_mat=np.zeros([t_max,t_max])
    for k in range(t_max):
        for l in range(t_max):
            K_mat[k,l]=self.cov_fun(self.t[k],self.t[l])
    
    self.K_mat=K_mat
    self.fun=np.random.multivariate_normal(np.zeros([t_max]),K_mat)
    
    self.state=0
    self.fun_hat=0

    observation=self.state
    
    return observation  # reward, done, info can't be included


  def render(self, reward, mode='console'):
    if mode != 'console':
      raise NotImplementedError()
    
    plt.plot(self.t,self.fun,linestyle='solid',color='0')
    plt.plot(self.t,self.fun_hat,linestyle='dashed',color='0')
    plt.scatter(self.t_measured,self.measurements)
    plt.title('Batch choice of measurements')
    plt.xlabel('Time')
    plt.ylabel('Function value')
    print(reward)
    
  def close (self):
      pass
      
      
new_env=CustomEnv(0.05)
check_env(new_env)




from stable_baselines3 import PPO, A2C, DQN, DDPG, TD3


model = TD3("MlpPolicy", new_env,verbose=1)
model.learn(total_timesteps=200)

obs = new_env.reset()

n_episodes=1

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




