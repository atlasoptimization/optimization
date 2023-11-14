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
    t_max=50
    
    self.t_max=t_max
    self.n_meas=5
    self.n_state=2*self.n_meas
    self.action_space = spaces.Box(low=-1, high=1,
                                        shape=(1,), dtype=np.float32)
    # Example for using image as input (channel-first; channel-last also works):
    self.observation_space = spaces.Box(low=-1, high=1,
                                        shape=(self.n_state,), dtype=np.float32)
    self.max_epoch=self.n_meas-1
    self.t=np.linspace(0,1,t_max)
    self.epoch=0
    self.t_measured=np.empty([0,1])
    self.measurements=np.empty([0,1])
    self.t_meas_sequence=-1*np.ones(self.n_meas)
    self.f_meas_sequence=-1*np.ones(self.n_meas)
       
    K_mat=np.zeros([t_max,t_max])
    for k in range(t_max):
        for l in range(t_max):
            K_mat[k,l]=self.cov_fun(self.t[k],self.t[l])
    
    self.K_mat=K_mat
    self.fun=np.random.multivariate_normal(np.zeros([t_max]),K_mat)
    
    self.state=-1*np.ones(self.n_state)
    self.fun_hat=0
    
  def round_to_index(self,t):
      approx_index=t*self.t_max
      index=np.ceil(approx_index)-1
      return index.astype(int)

  def cov_fun(self,t,s):
    # cov_val=np.exp(-np.abs((t-s)/1.2)**2)
    # cov_val = 0.2*np.exp(-np.abs((t-s)/0.2)**2)
    cov_val = s*t*0.2*np.exp(-np.abs((t-s)/0.2)**2)
    # cov_val=np.min(np.array([s,t]))-s*t
    # cov_val=np.min(np.array([s,t]))
    return cov_val
    

  def step(self, action):
   
    done=False
    meas_pos=(0.5)*action+0.5
    self.meas_pos=meas_pos
    temp_n=self.epoch+1
    
    meas_index=self.round_to_index(meas_pos)
    measurements=self.fun[meas_index]


    self.measurements=np.vstack((measurements,self.measurements))
    self.t_measured=np.vstack((self.t[meas_index],self.t_measured))
    
    augmented_t_vec=np.hstack((self.t_measured.squeeze(), self.t_meas_sequence))
    augmented_f_vec=np.hstack((self.measurements.squeeze(), self.f_meas_sequence))
    
    
    K_t=np.zeros([self.t_max,temp_n])
    K_ij=np.zeros([temp_n,temp_n])
    
    for k in range(self.t_max):
        for l in range(temp_n):
            K_t[k,l]=self.cov_fun(self.t[k],self.t_measured[l])
            
    for k in range(temp_n):
        for l in range(temp_n):
            K_ij[k,l]=self.cov_fun(self.t_measured[k],self.t_measured[l])   
    
    fun_hat=K_t@np.linalg.pinv(K_ij,rcond=1e-6,hermitian=True)@self.measurements
    rmse=np.linalg.norm(self.fun-fun_hat.squeeze())
    self.fun_hat=fun_hat
    
    self.state=np.hstack((augmented_t_vec[0:self.n_meas],augmented_f_vec[0:self.n_meas]))

    
    reward= -rmse
    # reward=-np.abs(np.max(self.fun)-np.max(self.fun_hat))
    # reward=-np.abs(np.max(self.fun)-np.max(self.measurements))

    # Optionally we can pass additional info, we are not using that for now
    info = {}
    
    self.epoch=self.epoch+1
    if self.epoch==self.max_epoch+1:
        done=True
    else:
        pass

    return self.state, reward, done, info


  def reset(self):
      

    self.t_measured=np.empty([0,1])
    self.measurements=np.empty([0,1])
    self.epoch=0
       
    t_max=self.t_max
    
    K_mat=np.zeros([t_max,t_max])
    for k in range(t_max):
        for l in range(t_max):
            K_mat[k,l]=self.cov_fun(self.t[k],self.t[l])
    
    self.K_mat=K_mat
    self.fun=np.random.multivariate_normal(np.zeros([t_max]),K_mat)
    
    self.state=-1*np.ones(self.n_state)
    self.fun_hat=0
    self.t_meas_sequence=-1*np.ones(self.n_meas)
    self.f_meas_sequence=-1*np.ones(self.n_meas)

    observation=self.state
    
    return observation  # reward, done, info can't be included


  def render(self, reward, mode='console'):
    if mode != 'console':
      raise NotImplementedError()
    
    plt.plot(self.t,self.fun,linestyle='solid',color='0')
    plt.plot(self.t,self.fun_hat,linestyle='dashed',color='0')
    plt.scatter(self.t_measured,self.measurements)
    print(reward, self.t_measured,self.measurements)
    
  def close (self):
      pass
      
      
new_env=CustomEnv()
action=0.5
new_env.step(action)
check_env(new_env)




from stable_baselines3 import PPO, A2C, DQN, DDPG, TD3
from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise


action_noise=OrnsteinUhlenbeckActionNoise(np.zeros(1), 0.1*np.ones(1))
model = TD3("MlpPolicy", new_env, verbose=1, action_noise=action_noise)

# model = PPO("MlpPolicy", new_env,verbose=1, use_sde=True)
model.learn(total_timesteps=10)

obs = new_env.reset()


# Evaluate and compare to regular grid

def reward_regular_grid(environment):
    n_meas=environment.n_meas
    fun_vals=environment.fun
    t_max=environment.t_max
    
    t=environment.t
    t_obs=np.linspace(0,1,n_meas)
    
    meas_index=np.linspace(0,t_max-1,n_meas).round().astype(int)
    measurements=environment.fun[meas_index]
    
    
    K_t=np.zeros([t_max,n_meas])
    K_ij=np.zeros([n_meas,n_meas])
    
    for k in range(t_max):
        for l in range(n_meas):
            K_t[k,l]=environment.cov_fun(t[k],t_obs[l])
            
    for k in range(n_meas):
        for l in range(n_meas):
            K_ij[k,l]=environment.cov_fun(t_obs[k],t_obs[l])   
    
    fun_hat=K_t@np.linalg.pinv(K_ij,rcond=1e-6,hermitian=True)@measurements
    reward=-np.linalg.norm(environment.fun-fun_hat.squeeze())
    # reward=-np.abs(np.max(environment.fun)-np.max(measurements))
    
    return reward, fun_hat
    
    
    
    

n_episodes=1

for k in range(n_episodes):
    done=False
    obs = new_env.reset()
    reward_grid, fun_hat_grid=reward_regular_grid(new_env)
    
    while done ==False:
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done, info = new_env.step(action)

        if done:
            print(reward_grid)
            new_env.render(reward)
            plt.plot(new_env.t,fun_hat_grid)
            
            # time.sleep(0.5)
            break




