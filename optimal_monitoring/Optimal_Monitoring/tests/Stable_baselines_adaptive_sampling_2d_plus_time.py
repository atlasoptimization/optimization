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
    x2_max=30
    x3_max=10
    
    self.x_max=np.array([[x1_max],[x2_max],[x3_max]])
    self.x1_max=x1_max
    self.x2_max=x2_max
    self.x3_max=x3_max
    
    self.n_meas=7
    self.n_state=3*self.n_meas
    self.action_space = spaces.Box(low=-1, high=1,
                                        shape=(2,), dtype=np.float32)
    # Example for using image as input (channel-first; channel-last also works):
    self.observation_space = spaces.Box(low=-1, high=1,
                                        shape=(self.n_state,), dtype=np.float32)
    self.max_epoch=x3_max
    self.x1=np.linspace(0,1,x1_max)
    self.x2=np.linspace(0,1,x2_max)
    self.x3=np.linspace(0,1,x3_max)
    
    self.epoch=0
    # self.x_measured=np.empty([0,3])
    self.x_measured=np.empty([0,2])
    self.f_measured=np.empty([0,1])
    # self.x_meas_sequence=-1*np.ones([self.n_meas,3])
    self.x_meas_sequence=-1*np.ones([self.n_meas,2])
    self.f_meas_sequence=-1*np.ones(self.n_meas)
       
    
    rf, K_x1, K_x2, K_x3 =sf.Simulate_random_field_3D_fast(self.x1, self.x2, self.x3, self.cov_fun_x, self.cov_fun_x, self.cov_fun_t)
    
    self.K_x1=K_x1
    self.K_x2=K_x2
    self.K_x3=K_x3
    
    self.fun=rf
    
    self.state=-1*np.ones(self.n_state)

    
  def round_to_index(self,x):
      approx_index=x*(self.x_max[:2].squeeze())
      index=np.ceil(approx_index)-1
      return index.astype(int)

  def cov_fun_x(self,t,s):
    # cov_val=np.exp(-np.abs((t-s)/1.2)**2)
    cov_val = np.exp(-np.abs((t-s)/0.2)**2)
    # cov_val = 0.2*(s*t*np.exp(-np.abs((t-s)/0.2)**2)-s*t)
    # cov_val = 0.2*(np.min(np.array([[s],[t]])-s*t))
    # cov_val=0.2*(-s**2*t**2+s*t)
    # cov_val=np.min(np.array([s,t]))-s*t
    # cov_val=np.min(np.array([s,t]))
    return cov_val

  def cov_fun_t(self,t,s):
    # cov_val=np.exp(-np.abs((t-s)/1.2)**2)
    cov_val =s*t* 0.2*np.exp(-np.abs((t-s)/0.8)**2)
    # cov_val=np.min(np.array([s,t]))-s*t
    # cov_val=np.min(np.array([s,t]))
    return cov_val
    

  def step(self, action):
   
    done=False
    meas_pos=(0.5)*action+0.5
    self.meas_pos=meas_pos
    
    meas_index=self.round_to_index(meas_pos)
    # meas_index=np.hstack((meas_index,np.array([self.epoch])))
    # f_measured=self.fun[meas_index[0],meas_index[1],meas_index[2]]
    f_measured=self.fun[meas_index[0],meas_index[1],self.epoch]


    self.f_measured=np.vstack((f_measured,self.f_measured))
    # self.x_measured=np.vstack((np.array([self.x1[meas_index[0]], self.x2[meas_index[1]], self.x3[meas_index[2]]]).squeeze(), self.x_measured))
    self.x_measured=np.vstack((np.array([self.x1[meas_index[0]], self.x2[meas_index[1]]]).squeeze(), self.x_measured))
    
    augmented_x_vec=np.vstack((self.x_measured, self.x_meas_sequence))
    augmented_f_vec=np.hstack((self.f_measured.squeeze(), self.f_meas_sequence))
        
    
    # self.state=np.hstack((augmented_x_vec[0:self.n_meas,0], augmented_x_vec[0:self.n_meas,1], augmented_x_vec[0:self.n_meas,2], augmented_f_vec[0:self.n_meas]))
    self.state=np.hstack((augmented_x_vec[0:self.n_meas,0], augmented_x_vec[0:self.n_meas,1], augmented_f_vec[0:self.n_meas]))


    reward=-np.abs(np.max(self.fun[:,:,self.epoch])-self.f_measured[0])[0]
    self.epoch=self.epoch+1
    
    if self.epoch==self.max_epoch:
        done=True
    else:
        pass
    
    # if self.epoch==self.max_epoch:
    #     done=True
    #     # reward= -rmse
    #     # reward=-np.abs(np.max(self.fun)-np.max(self.fun_hat))
    #     reward=-np.abs(np.max(self.fun[:,:,-1])-self.f_measured[0])[0]

    # else:
    #     reward=0

    # Optionally we can pass additional info, we are not using that for now
    info = {}

    return self.state, reward, done, info


  def reset(self):
      

    self.epoch=0
    # self.x_measured=np.empty([0,3])
    self.x_measured=np.empty([0,2])
    self.f_measured=np.empty([0,1])
    # self.x_meas_sequence=-1*np.ones([self.n_meas,3])
    self.x_meas_sequence=-1*np.ones([self.n_meas,2])
    self.f_meas_sequence=-1*np.ones(self.n_meas)
       
    
    rf, K_x1, K_x2, K_x3 =sf.Simulate_random_field_3D_fast(self.x1, self.x2, self.x3, self.cov_fun_x, self.cov_fun_x, self.cov_fun_t)
    
    self.fun=rf
    
    self.state=-1*np.ones(self.n_state)

    observation=self.state
    
    return observation  # reward, done, info can't be included


  def render(self, reward, mode='console'):
    if mode != 'console':
      raise NotImplementedError()
    
    plt.imshow(self.fun[:,:,-1].squeeze().T, extent=[0,1,1,0])
    plt.colorbar()
    plt.scatter(self.x_measured[1:,0],self.x_measured[1:,1])
    plt.scatter(self.x_measured[0,0],self.x_measured[0,1])
    plt.title('Sequential 2D spatial measurements')
    plt.xlabel('X axis')
    plt.ylabel('Y axis')
    print(reward, self.x_measured,self.f_measured)
    
  def close (self):
      pass
      
      
new_env=CustomEnv()
action=np.array([0.9,0.1])
new_env.step(action)
check_env(new_env)


def eval_on_grid(environment):
    n_meas=environment.n_meas
    fun=environment.fun[:,:,-1]
    n_total=fun.shape[0]*fun.shape[1]
    index_vals=np.linspace(0,n_total-1,n_meas).round().astype(int)
    fun_vals=np.ravel(fun)[index_vals]
    reward=-np.abs(np.max(fun[:,:])-np.max(fun_vals))
    
    grid_indices=np.unravel_index(index_vals, fun.shape)
    coord_indices=np.zeros([2,n_meas])
    
    for l in range(n_meas):
        coord_indices[0,l]=environment.x1[grid_indices[0][l]]
        coord_indices[1,l]=environment.x2[grid_indices[1][l]]
        
    grid_coords=coord_indices

    return reward, grid_coords



from stable_baselines3 import PPO, A2C, DQN, DDPG, TD3
from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise

policy_kwargs = dict(net_arch=[100,30,100])
# action_noise=OrnsteinUhlenbeckActionNoise(np.zeros(1), 0.5*np.ones(1))
action_noise=NormalActionNoise(np.zeros(1), 0.5*np.ones(1))
model = TD3("MlpPolicy", new_env, verbose=1, action_noise=action_noise, policy_kwargs=policy_kwargs)
# model = TD3("MlpPolicy", new_env, verbose=1, action_noise=action_noise)

# model = PPO("MlpPolicy", new_env,verbose=1, use_sde=True)
model.learn(total_timesteps=1000)

obs = new_env.reset()

n_episodes=1

for k in range(n_episodes):
    done=False
    obs = new_env.reset()
    while done ==False:
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done, info = new_env.step(action)

        if done:
            grid_reward, grid_coords=eval_on_grid(new_env)
            print(grid_reward)
            new_env.render(reward)
            plt.scatter(grid_coords[0,:],grid_coords[1,:])
            # time.sleep(0.5)
            break




