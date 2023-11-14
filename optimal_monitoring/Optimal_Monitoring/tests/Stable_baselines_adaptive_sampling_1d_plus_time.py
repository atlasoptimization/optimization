import math
import sys
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

sys.path.insert(0, '../../../RKHS/Simulation/')
import Simulation_support_funs as sf


Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))





# Custom env = measurement game


class CustomEnv(gym.Env):
  """Custom Environment that follows gym interface"""
  metadata = {'render.modes': ['human']}

  def __init__(self):
    super(CustomEnv, self).__init__()
    # Define action and observation space
    x1_max=50
    x2_max=15
    
    self.x_max=np.array([[x1_max],[x2_max]])
    
    self.n_meas=10
    self.n_state=3*self.n_meas
    self.action_space = spaces.Box(low=-1, high=1,
                                        shape=(1,), dtype=np.float32)
    # Example for using image as input (channel-first; channel-last also works):
    self.observation_space = spaces.Box(low=-1, high=1,
                                        shape=(self.n_state,), dtype=np.float32)
    self.max_epoch=x2_max
    self.x1=np.linspace(0,1,x1_max)
    self.x2=np.linspace(0,1,x2_max)
    
    self.epoch=0
    self.x_measured=np.empty([0,2])
    self.f_measured=np.empty([0,1])
    self.x_meas_sequence=-1*np.ones([self.n_meas,2])
    self.f_meas_sequence=-1*np.ones(self.n_meas)
       
    
    rf, K_x1, K_x2 =sf.Simulate_random_field_fast(self.x1, self.x2, self.cov_fun_x, self.cov_fun_t)
    
    self.fun=rf
    
    self.state=-1*np.ones(self.n_state)

    
  def round_to_index(self,x):
      approx_index=x*self.x_max.squeeze()
      index=np.ceil(approx_index)-1
      return index.astype(int)

  def cov_fun_x(self,t,s):
    # cov_val=np.exp(-np.abs((t-s)/1.2)**2)
    cov_val = 0.2*np.exp(-np.abs((t-s)/0.2)**2)
    # cov_val=np.min(np.array([s,t]))-s*t
    # cov_val=np.min(np.array([s,t]))
    return cov_val

  def cov_fun_t(self,t,s):
    # cov_val=np.exp(-np.abs((t-s)/1.2)**2)
    cov_val =s*t* 0.2*np.exp(-np.abs((t-s)/0.2)**2)
    # cov_val=np.min(np.array([s,t]))-s*t
    # cov_val=np.min(np.array([s,t]))
    return cov_val
    

  def step(self, action):
   
    done=False
    meas_pos=(0.5)*action+0.5
    self.meas_pos=meas_pos
    
    meas_index=self.round_to_index(meas_pos)[0]
    meas_index=np.vstack((meas_index,np.array([self.epoch])))
    f_measured=self.fun[meas_index[0],meas_index[1]]


    self.f_measured=np.vstack((f_measured,self.f_measured))
    self.x_measured=np.vstack((np.array([self.x1[meas_index[0]], self.x2[meas_index[1]]]).squeeze(), self.x_measured))
    
    augmented_x_vec=np.vstack((self.x_measured, self.x_meas_sequence))
    augmented_f_vec=np.hstack((self.f_measured.squeeze(), self.f_meas_sequence))
        
    
    self.state=np.hstack((augmented_x_vec[0:self.n_meas,0], augmented_x_vec[0:self.n_meas,1], augmented_f_vec[0:self.n_meas]))

    
    # reward= -rmse
    # reward=-np.abs(np.max(self.fun)-np.max(self.fun_hat))
    reward=-np.abs(np.max(self.fun[:,self.epoch])-f_measured[0])

    # Optionally we can pass additional info, we are not using that for now
    info = {}
    
    self.epoch=self.epoch+1
    if self.epoch==self.max_epoch:
        done=True
    else:
        pass

    return self.state, reward, done, info


  def reset(self):
      

    self.epoch=0
    self.x_measured=np.empty([0,2])
    self.f_measured=np.empty([0,1])
    self.x_meas_sequence=-1*np.ones([self.n_meas,2])
    self.f_meas_sequence=-1*np.ones(self.n_meas)
       
    
    rf, K_x1, K_x2 =sf.Simulate_random_field_fast(self.x1, self.x2, self.cov_fun_x, self.cov_fun_t)
    
    self.fun=rf
    
    self.state=-1*np.ones(self.n_state)

    observation=self.state
    
    return observation  # reward, done, info can't be included


  def render(self, reward, mode='console'):
    if mode != 'console':
      raise NotImplementedError()
    
    plt.imshow(self.fun.T, extent=[0,1,1,0])
    plt.colorbar()
    plt.scatter(self.x_measured[:,0],self.x_measured[:,1])
    plt.title('Sequential spatial measurements')
    plt.xlabel('X axis')
    plt.ylabel('T axis')
    print(reward, self.x_measured,self.f_measured)
    
  def close (self):
      pass
      
      
new_env=CustomEnv()
action=0.5
new_env.step(action)
check_env(new_env)




from stable_baselines3 import PPO, A2C, DQN, DDPG, TD3
from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise

# policy_kwargs = dict(net_arch=[100,20,100])
# action_noise=OrnsteinUhlenbeckActionNoise(np.zeros(1), 0.5*np.ones(1))
action_noise=NormalActionNoise(np.zeros(1), 0.5*np.ones(1))
# model = TD3("MlpPolicy", new_env, verbose=1, action_noise=action_noise, policy_kwargs=policy_kwargs)
model = TD3("MlpPolicy", new_env, verbose=1, action_noise=action_noise)

# model = PPO("MlpPolicy", new_env,verbose=1, use_sde=True)
model.learn(total_timesteps=100)

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




