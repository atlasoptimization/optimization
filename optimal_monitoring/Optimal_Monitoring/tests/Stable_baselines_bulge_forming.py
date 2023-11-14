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
from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise


import time




Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))





# Custom env = beam bending measurements


class CustomEnv(gym.Env):
  """Custom Environment that follows gym interface"""
  metadata = {'render.modes': ['human']}

  def __init__(self):
    super(CustomEnv, self).__init__()
    # Define action and observation space
    
    self.n_meas=3
    self.n_past=3
    self.n_x_meas=self.n_meas*self.n_past
    self.n_state=2*self.n_x_meas
    self.state=np.zeros([self.n_state])
    
    x_max=50
    self.x_max=x_max
    t_max=5
    self.t_max=t_max
    self.def_hat=np.zeros([self.x_max])
    
    self.action_space = spaces.Box(low=-1, high=1,
                                        shape=(self.n_meas,), dtype=np.float32)
    # Example for using image as input (channel-first; channel-last also works):
    self.observation_space = spaces.Box(low=-1, high=1,
                                        shape=(self.n_state,), dtype=np.float32)
    self.max_epoch=self.t_max
    self.t=np.linspace(0,1,t_max)
    self.x=np.linspace(0,1,x_max)
    self.x_measured=np.empty([0,self.n_meas])
    self.measurements=np.empty([0,self.n_meas])
    self.epoch=0
    
    
    self.bump_fun=np.zeros([self.x_max])
    rand_index=np.random.randint(0,x_max-1)
    self.bump_center=rand_index
    
    for k in range(x_max):
        self.bump_fun[k]=self.bump_function(self.x[k],self.x[rand_index])
        
        
    self.def_fun=np.zeros([self.x_max,self.t_max])
    for k in range(self.t_max):
        self.def_fun[:,k]=(self.bump_fun*(k/self.t_max))

    
  def round_to_index(self,x):
      approx_index=x*self.x_max
      index=np.ceil(approx_index)-1
      return index.astype(int)

  def bump_function(self,t,s):
    # cov_val=np.exp(-np.abs((t-s)/1.2)**2)
    cov_val=np.exp(-np.abs((t-s)/0.1)**2)

    return cov_val
    
  def kernel_x(self,x,y):
    cov_val=np.exp(-np.abs((x-y)/0.1)**2)

    return cov_val


  def step(self, action):
       
         
    x_measured=0.5*(action)+0.5
    x_indices=self.round_to_index(x_measured)
    def_measured=self.def_fun[x_indices,self.epoch]
    
    augmented_x_measured=np.hstack((x_measured, self.state[0:self.n_x_meas]))
    augmented_def_measured=np.hstack((def_measured, self.state[self.n_x_meas:self.n_state]))
    
    
    new_x_measured_vec=augmented_x_measured[0:self.n_x_meas]
    new_def_measured_vec=augmented_def_measured[0:self.n_x_meas]
    
    self.state=np.hstack((new_x_measured_vec, new_def_measured_vec))
    
    self.x_measured=np.vstack((self.x_measured,x_measured))
    self.measurements=np.vstack((self.measurements,def_measured))
    
    K_ij=np.zeros([self.n_meas,self.n_meas])
    K_t=np.zeros([self.x_max,self.n_meas])
    
    for k in range(self.n_meas):
        for l in range(self.n_meas):
            K_ij[k,l]=self.kernel_x(x_measured[k],x_measured[l])
            
    for k in range(self.x_max):
        for l in range(self.n_meas):
            K_t[k,l]=self.kernel_x(self.x[k], x_measured[l])
                
    def_hat=K_t@np.linalg.pinv(K_ij,rcond=1e-6,hermitian=True)@def_measured
    def_true=self.def_fun[:,self.epoch]
    rmse=np.linalg.norm(def_hat-def_true)
    reward=-rmse
    
    self.def_hat=def_hat
    
    self.epoch=self.epoch+1 
    if self.epoch==self.max_epoch:
        done=True
    else:
        done=False

    # Optionally we can pass additional info, we are not using that for now
    info = {}

    return self.state, reward, done, info


  def reset(self):

    self.state=np.zeros([self.n_state])
    self.def_hat=np.zeros([self.x_max])
    self.x_measured=np.empty([0,self.n_meas])
    self.measurements=np.empty([0,self.n_meas])
    self.epoch=0
                
    
    self.bump_fun=np.zeros([self.x_max])
    rand_index=np.random.randint(0,self.x_max-1)
    self.bump_center=rand_index
    
    for k in range(self.x_max):
        self.bump_fun[k]=self.bump_function(self.x[k],self.x[rand_index])
        
        
    self.def_fun=np.zeros([self.x_max,self.t_max])
    for k in range(self.t_max):
        self.def_fun[:,k]=(self.bump_fun*(k/self.t_max))
    
    observation=self.state
    
    return observation  # reward, done, info can't be included


  def render(self, reward, mode='console'):
    if mode != 'console':
      raise NotImplementedError()
    
    # plt.plot(self.t,self.fun,linestyle='solid',color='0')
    plt.plot(self.x,self.def_fun[:,self.epoch-1],linestyle='solid',color='0')
    plt.plot(self.x,self.def_hat,linestyle='dashed',color='0')
    plt.scatter(self.x_measured[self.epoch-1,:],self.measurements[self.epoch-1,:])
    print(reward)
    
  def close (self):
      pass
      
      
new_env=CustomEnv()
# new_env.step(0.5)
check_env(new_env)




from stable_baselines3 import PPO, A2C, DQN, DDPG, TD3

# action_noise=NormalActionNoise(np.zeros(new_env.n_meas), 0.1*np.ones(new_env.n_meas))
action_noise=OrnsteinUhlenbeckActionNoise(np.zeros(new_env.n_meas), 0.1*np.ones(new_env.n_meas))

policy_kwargs = dict(net_arch=[100,3,100])

# model = TD3("MlpPolicy", new_env,verbose=1, policy_kwargs=policy_kwargs)
# model = PPO("MlpPolicy", new_env,verbose=1 ,use_sde=True)
model = TD3("MlpPolicy", new_env,verbose=1, action_noise=action_noise)

model.learn(total_timesteps=20000)

obs = new_env.reset()

n_episodes=1

for k in range(n_episodes):
    done=False
    obs = new_env.reset()
    while done ==False:
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done, info = new_env.step(action)
        new_env.render(reward)

        if done:
            # time.sleep(0.5)
            break




