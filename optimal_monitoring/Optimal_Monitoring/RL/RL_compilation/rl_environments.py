"""
This file provides a set of standard environments on which rl agents can be 
tested after import. These environments all support at least the following 
functionalities: 
    a) env.state: The state of the environment
    b) env.dim: The dimensions of environment and action space
    c) env.epoch: Counts the number of steps the environment was subjected to
    d) env.step() A functiont ransforming the state of the environment.
                It passes out new states and rewards
    e) env.reset() A function resetting the environment to its ground state
    
The following functionalities are optional:
    a) env.action_sequence: Sequence of actions for documentation purpose
    b) env. state_sequence: Sequence of state vectors for documentation purpose
    c) env.reward_sequence: Sequence of reward signals for documentation purpose
    
This file is to be run once upon which it provides the following classes:
    1) Control_landing_env : Provide a falling aircraft with acceleration signals
        to make it land around a certain spot
    2) Border_walking_env: Provide an agent with walking directions that
        get rewarded for an x coordinate sign change
    
"""



"""
1. Definitions and imports ---------------------------------------------------
"""

# i) Imports

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


# ii) Definitions

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))




"""
2. Control landing environment -----------------------------------------------

This environment consists of a model of a falling object that can act by 
accelerating to the left or the right. Each step it falls closer to the ground 
and is punished by its deviation on the x axis from a target value. The parameters
passed to it upon its creation are the maximum falling height, the discretization
density for x and v_x values, the action dimension and the target x value.

INPUTS
    Name                 Interpretation                             Type
z_max           The maximum falling height                      Integer > 5
dims            Array of dimensions [n_x, n_v_x, n_action]       Array
x_target        Target coordinate                               Number
action          The action to be performed during step          Index
"""

class Control_landing_env_discrete():
    
    def __init__(self, dims, x_target):
    
        z_max=5
        n_x=dims[0]
        n_vx=dims[1]
        n_a=dims[2]
        x=10*(np.random.randint(0,n_x)/n_x - 1/2)
        vx=0*(np.random.randint(0,n_vx)/n_vx -1/2)
        
        self.state=np.vstack((z_max,x,vx))
        
        self.epoch=0
        self.max_epoch=z_max
        self.state_sequence=self.state
        self.action_sequence=[]
        self.reward_sequence=[]
        self.dim=np.hstack((z_max+1,dims))
        self.n_state=self.dim[0]*self.dim[1]*self.dim[2]
        self.n_action=self.dim[3]
        self.z_max=z_max
        self.x_target=x_target
        self.index_bounds_low=np.array([[0],[0],[0]])
        self.index_bounds_high=np.array([[z_max],[n_x-1],[n_vx-1]])
        
        
        index_array=self.state_to_index_array(self.state)
        self.state_index=self.array_to_linind(index_array)
    
    def array_to_linind(self,index_array):
        linind=np.ravel_multi_index((index_array[0][0],index_array[1][0],index_array[2][0]), (self.dim[0],self.dim[1],self.dim[2]))
        return linind
    
        
    def linind_to_array(self,linind):
        index_array=np.unravel_index(linind, (self.dim[0],self.dim[1],self.dim[2]))
        index_array=np.asarray(index_array)
        return index_array
    
    def state_to_index_array(self, state):
        index_array=np.round(np.array([state[0],self.dim[1]*(state[1]/10+0.5),self.dim[2]*(state[2]/2 +0.5)]))
        index_array=np.clip(index_array, self.index_bounds_low, self.index_bounds_high).astype(int)
        return index_array
        
    def index_array_to_state(self, index_array):
        state=np.array([index_array[0], 10*(index_array[1]/self.dim[1] -1/2), 2*(index_array[2]/self.dim[2] -1/2)])
        return state
    
    
    def step(self,action):    
        
        acceleration_value=(action)*(1/(self.dim[3]-1))-0.5
        
        state_change=np.array([[-1],self.state[2],[acceleration_value]])
        
        state=self.state
        state_index_array=self.state_to_index_array(state)
        state_index=self.array_to_linind(state_index_array)
        
        new_state=state+state_change
        index_array=self.state_to_index_array(new_state)  # perform clipping in index domain
        new_state_index=self.array_to_linind(index_array)
         
        new_state=self.index_array_to_state(index_array)              
        
        self.state=new_state
        self.state_sequence=np.hstack((self.state_sequence,new_state))
        self.action_sequence.append(acceleration_value)
        self.state=new_state


        
        reward=-np.abs(self.state[1]-self.x_target)
        self.reward_sequence=np.hstack((self.reward_sequence,reward))
        self.last_transition=Transition(state_index,action,new_state_index,reward)
        
        self.state_index=new_state_index
        
        self.epoch=self.epoch+1
        if self.epoch==self.max_epoch:
            done=True
        else:
            done=False
        
        
        return reward, done
        
        
        
    def reset(self):
        z_max=self.dim[0]-1
        n_x=self.dim[1]
        n_vx=self.dim[2]
        n_a=self.dim[3]
        x=10*(np.random.randint(0,n_x)/n_x - 1/2)
        vx=0*(np.random.randint(0,n_vx)/n_vx -1/2)
        
        self.state=np.vstack((z_max,x,vx))
        index_array=self.state_to_index_array(self.state)
        self.state_index=self.array_to_linind(index_array)
        
        self.epoch=0
        self.max_epoch=z_max
        self.state_sequence=self.state
        self.action_sequence=[]
        self.reward_sequence=[]
        self.n_state=self.dim[0]*self.dim[1]*self.dim[2]
        self.n_action=self.dim[3]
        self.z_max=z_max







"""
3. Border walking environment ---------------------------------------------

This environment consists of an agent that can choose to walk left or right. It
gets rewarded for every step it does that changes the sign of the x coordinate.
The action is given as an integer index 0 or 1 (go left or go right)
"""



class Border_walking_env():
    
    def __init__(self):
    
        self.state=np.random.randint(-5,5)+0.5
        self.n_state=10
        self.n_action=2
        self.epoch=0
        self.max_epoch=10
        self.state_index=(self.state+4.5)
        self.state_sequence=self.state
        self.action_sequence=[]
        self.reward_sequence=[]
    
    
    def step(self,action):
        
        state_change=2*action-1
        new_state=np.clip(self.state+state_change,-4.5,4.5)
        new_state_index=(new_state+4.5)
        reward=np.abs(np.sign(new_state)-np.sign(self.state))
        
        self.last_transition=Transition(self.state_index,action,new_state_index,reward)
        self.epoch=self.epoch+1
        self.state=new_state
        self.state_index=(self.state+4.5)
        
        self.reward_sequence=np.append(self.reward_sequence,reward)
        self.state_sequence=np.append(self.state_sequence,self.state)
        self.action_sequence=np.append(self.action_sequence,action)
      
        if self.epoch==self.max_epoch:
            done=True
        else:
            done=False
        
        return reward, done
        
        
        
    def reset(self):
        self.state=np.random.randint(-5,5)+0.5
        self.n_state=10
        self.n_action=2
        self.epoch=0
        self.max_epoch=10
        self.state_index=(self.state+4.5)
        self.state_sequence=self.state
        self.action_sequence=[]
        self.reward_sequence=[]




"""
4. Border walking environment continuous---------------------------------------

This environment consists of an agent that can choose to walk left or right. It
gets rewarded for every step it does that changes the sign of the x coordinate.
The action is given as an integer index 0 or 1 (go left or go right)
"""



class Border_walking_env_continuous():
    
    def __init__(self):
    
        self.state=np.array([np.random.normal(0,3)])
        self.n_state=1
        self.n_action=2
        self.epoch=0
        self.max_epoch=10

        self.state_sequence=self.state
        self.action_sequence=[]
        self.reward_sequence=[]
    
    
    def step(self,action):
        
        state_change=2*action-1
        new_state=self.state+state_change
        reward=np.abs(np.sign(new_state)-np.sign(self.state))
        
        self.last_transition=Transition(self.state,action,new_state,reward)
        self.epoch=self.epoch+1
        self.state=new_state
        
        self.reward_sequence=np.append(self.reward_sequence,reward)
        self.state_sequence=np.append(self.state_sequence,self.state)
        self.action_sequence=np.append(self.action_sequence,action)
      
        if self.epoch==self.max_epoch:
            done=True
        else:
            done=False
        
        return reward, done
        
        
        
    def reset(self):
        self.state=np.array([np.random.normal(0,3)])
        self.n_state=3
        self.n_action=2
        self.epoch=0
        self.max_epoch=10

        self.state_sequence=self.state
        self.action_sequence=[]
        self.reward_sequence=[]




"""
5. Control landing environment continuous--------------------------------------

This environment consists of a model of a falling object that can act by 
accelerating to the left or the right. Each step it falls closer to the ground 
and is punished by its deviation on the x axis from a target value. The parameters
passed to it upon its creation are the the action dimension and the target x value.

INPUTS
    Name                 Interpretation                             Type
dims            Array of form [n_action]                        Array
x_target        Target coordinate                               Number
action          The action to be performed during step          Index
"""

class Control_landing_env_continuous():
    
    def __init__(self, dims, x_target):
    
        z_max=5
        n_a=dims[0]
        x=1*(np.random.normal(0,1))
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
        
        state_change=np.array([[-1],self.state[2],acceleration_value])       
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

        x=1*(np.random.normal(0,1))
        vx=0*(np.random.normal(0,1))
        
        self.state=np.vstack((self.max_epoch,x,vx))
        
        self.epoch=0
        self.state_sequence=self.state
        self.action_sequence=[]
        self.reward_sequence=[]








