"""
This file provides a set of reinforcement learning agents. These agents provide
some possibilities to test baseline algorithms on simple environments.

The following agents are implemented:
    Discrete Q learning 
    Least squares policy iteration
    Deep Q networks 
    Kernel reinforcement learning

"""


"""
    1. Definitions and imports ------------------------------------------------
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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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



"""
    Discrete Q learning ------------------------------------------------------
    
    The Q learning agent attempts to learn a table listing the values of performing
    action a when in state s, where both s and a are discrete variables.
    To this end, several methods are employed that can be called to train the
    agent when an environment is defined. 
    The methods are:
        self.reset()                resets the Q learning actor
        self.learn(data)            trains the actor with the data of one step
        self.batch_learn(memory)    trains the actor with a batch of data
        self.q_table                the current table listing state-action values
        self.best_action(state)     returns the best action given the current state
    The agent acts directly on state index variables and action index variables.
    For this, do the following:
        1. Definitions and imports
        2. Basic functions
        3. Learning functions
        4. Assemble solutions   
        
    INPUT
    Name                 Interpretation                             Type
    dims        List of dimensions for state and action             List of lists
                variables in form [[dim_state],[dim_action]]
    params      List of arrays containing options. Is of form       List of numbers
                [gamma, alpha] where gamma is discount and 
                alpha learning rate
    transition  Transition class named tuple                        Transition
"""





"""
    1. Definitions and imports
"""



class Discrete_Q_learner():
    
    def __init__(self,dims,params):
        
        self.dim=dims
        dim_state=dims[0]
        dim_action=dims[1]
        
        dim_q=dim_state+dim_action
        self.q_table=10*np.ones(dim_q)       
        self.e_trace=np.zeros(dim_q)
    
        self.gamma_val=params[0]
        self.alpha_val=params[1]    
    

    """
        2. Basic functions
    """

    def reset(self):
        dim_state=self.dim[0]
        dim_action=self.dim[1]
        
        dim_q=dim_state+dim_action
        self.q_table=10*np.ones(dim_q)        
        

    def best_action(self,state_index):
        q_vec=np.zeros(self.dim[1])
        
        for k in range(self.dim[1][0]):
            loc_index=copy.copy(state_index)
            loc_index=np.append(loc_index, k).astype(int)
            q_vec[k]=self.q_table[loc_index[0], loc_index[1]]
            
        return q_vec.argmax(), q_vec.max()

    """
        3. Learning functions
    """

    def learn(self,transition_data):
        state=transition_data.state
        next_state=transition_data.next_state
        action=transition_data.action
        reward=transition_data.reward
        
        index=np.append(state,action).astype(int)
        _,max_q=self.best_action(next_state)
               
        new_q_val=(1-self.alpha_val)*self.q_table[index[0],index[1]]+self.alpha_val*(reward+self.gamma_val*max_q)
        self.q_table[index[0],index[1]]=new_q_val
        
        
    # def batch_learn(self,memory):
        
        








"""
    Discrete SARSA(l) learning ------------------------------------------------
    
    The SARSA learning agent attempts to learn a table listing the values of performing
    action a when in state s, where both s and a are discrete variables.
    To this end, several methods are employed that can be called to train the
    agent when an environment is defined. 
    The methods are:
        self.reset()                resets the Q learning actor
        self.learn(data)            trains the actor with the data of one step
        self.batch_learn(memory)    trains the actor with a batch of data
        self.q_table                the current table listing state-action values
        self.best_action(state)     returns the best action given the current state
    The agent acts directly on state index variables and action index variables.
    Eligibility traces can be defined to speed up learning.
    For this, do the following:
        1. Definitions and imports
        2. Basic functions
        3. Learning functions
        4. Assemble solutions   
        
    INPUT
    Name                 Interpretation                             Type
    dims        List of dimensions for state and action             List of lists
                variables in form [[dim_state],[dim_action]]
    params      List of arrays containing options. Is of form       List of numbers
                [gamma,lambda, alpha] where gamma is discount and 
                lambda is eligibility decay, alpha learning rate
    transition  Transition class named tuple                        Transition
"""





"""
    1. Definitions and imports
"""


class Discrete_SARSA_learner():
    
    def __init__(self,dims,params):
        
        self.dim=dims
        dim_state=dims[0]
        dim_action=dims[1]
        
        dim_q=dim_state+dim_action
        self.q_table=10*np.ones(dim_q)       
        self.e_trace=np.zeros(dim_q)
    
        self.gamma_val=params[0]
        self.lambda_val=params[1]
        self.alpha_val=params[2]    
    


    """
        2. Basic functions
    """


    def reset(self):
        dim_state=self.dim[0]
        dim_action=self.dim[1]
        
        dim_q=dim_state+dim_action
        self.q_table=10*np.ones(dim_q)        
        self.e_trace=np.zeros(dim_q)
        

    def best_action(self,state_index):
        q_vec=np.zeros(self.dim[1])
        
        for k in range(self.dim[1][0]):
            loc_index=copy.copy(state_index)
            loc_index=np.append(loc_index, k).astype(int)
            q_vec[k]=self.q_table[loc_index[0], loc_index[1]]
            
        return q_vec.argmax(), q_vec.max()
    
    def eps_greedy_action(self,state_index,eps_thresh):
        q_vec=np.zeros(self.dim[1])
        
        for k in range(self.dim[1][0]):
            loc_index=copy.copy(state_index)
            loc_index=np.append(loc_index, k).astype(int)
            q_vec[k]=self.q_table[loc_index[0], loc_index[1]]
         
        eps_test=np.random.uniform()
        
        if eps_test>eps_thresh:
            
            action_index=q_vec.argmax()
            action_value=q_vec.max()
        else:
            action_index=np.random.randint(0,self.dim[1][0])
            action_value=q_vec[action_index]
        return action_index, action_value
        
    
    def reset_eligibility(self):
        dim_state=self.dim[0]
        dim_action=self.dim[1]
        
        dim_q=dim_state+dim_action
        self.e_trace=np.zeros(dim_q)



    """
        3. Learning functions
    """


    def learn(self,transition_data,eps_thresh):
        state=transition_data.state
        next_state=transition_data.next_state
        action=transition_data.action
        reward=transition_data.reward
        
        index=np.append(state,action).astype(int)
        _,max_q=self.best_action(next_state)
               
        _,q_val_next=self.eps_greedy_action(next_state, eps_thresh)
        delta = reward+self.gamma_val*q_val_next-self.q_table[index[0],index[1]]
        self.e_trace[index[0],index[1]]=self.e_trace[index[0],index[1]]+1
        
        self.q_table=self.q_table+self.alpha_val*delta*self.e_trace
        self.e_trace=self.gamma_val*self.lambda_val*self.e_trace
        
        # new_q_val=(1-self.alpha_val)*self.q_table[index[0],index[1]]+self.alpha_val*(reward+self.gamma_val*max_q)
        # self.q_table[index[0],index[1]]=new_q_val
        
        
    # def batch_learn(self,memory):
        
        


















"""
    Least squares policy iteration -------------------------------------------
"""



"""
    Deep Q networks ----------------------------------------------------------
"""


class DQN_learner(nn.Module):
    
    def __init__(self,dims,params):
        
        super(DQN_learner, self).__init__()
        
        self.n_state=dims[0]
        self.n_hidden=2*self.n_state
        self.n_action=dims[1]
        self.learning_rate=params[1]
        self.gamma=params[0]
        
        n_state=self.n_state
        n_hidden=self.n_hidden
        n_action=self.n_action
        
        self.fc1 = nn.Linear(n_state, n_hidden)
        self.fc2 = nn.Linear(n_hidden, n_hidden)
        self.fc3 = nn.Linear(n_hidden, n_hidden)
        self.fc4 = nn.Linear(n_hidden, n_hidden)
        self.fc5 = nn.Linear(n_hidden, n_action)



    # Define nn forward pass
    def forward(self, x):
        #x = x.to(device)
        x = torch.tanh(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        x = torch.tanh(self.fc3(x))
        x = torch.tanh(self.fc4(x))
        x = torch.tanh(self.fc5(x))
        return x

    """
        2. Basic functions
    """



    def best_action(self,state):

        state=torch.from_numpy(state)
        q_vec=self.forward(state.float())
        # q_vec=q_vec.detach().numpy()
        
        return q_vec.argmax(), q_vec.max()
        
    
    def eps_greedy_action(self,state,eps_thresh):
        
        state=torch.from_numpy(state)
        q_vec=self.forward(state.float())
        q_vec=q_vec.detach().numpy()         
        
        eps_test=np.random.uniform()
        
        if eps_test>eps_thresh:
            
            action_index=q_vec.argmax()
            action_value=q_vec.max()
        else:
            action_index=np.random.randint(0,self.n_action)
            action_value=q_vec[action_index]
        return np.array([action_index]), action_value
    


    """
        3. Learning functions
    """


    def learn(self,transition_data,eps_thresh,optimizer):
        #optimizer = optim.RMSprop(self.parameters())
        
        state=torch.from_numpy(transition_data.state)
        next_state=torch.from_numpy(transition_data.next_state)
        action=torch.from_numpy(transition_data.action)
        reward=torch.from_numpy(transition_data.reward)
        

        q_vec=self.forward(state.float())
        q_val=q_vec[action] 
                
        q_vec_next=self.forward(next_state.float())
        next_action,_ =self.best_action(next_state.detach().numpy())
        q_val_next=q_vec_next[next_action]   
        expected_q_vals = (q_val_next * self.gamma) + reward
        
        criterion = nn.SmoothL1Loss()
        loss = criterion(q_val.float(), expected_q_vals.float())
    
        # Optimize the model
        optimizer.zero_grad()
        loss.backward()
        for param in self.parameters():
            param.grad.data.clamp_(-1, 1)
        optimizer.step()
            
        
        
        

"""
    DQN with experience replay ------------------------------------------------
"""


class DQN_learner_exprep(nn.Module):
    
    def __init__(self,dims,params,batch_size):
        
        super(DQN_learner_exprep, self).__init__()
        
        self.n_state=dims[0]
        self.n_hidden=5*self.n_state
        self.n_action=dims[1]
        self.learning_rate=params[1]
        self.gamma=params[0]
        self.memory=ReplayMemory(1000)
        self.batch_size=batch_size
        self.device=device
        self.nr_observations=0
        
        n_state=self.n_state
        n_hidden=self.n_hidden
        n_action=self.n_action
        
        self.fc1 = nn.Linear(n_state, n_hidden)
        self.fc2 = nn.Linear(n_hidden, n_hidden)
        self.fc3 = nn.Linear(n_hidden, n_hidden)
        self.fc4 = nn.Linear(n_hidden, n_hidden)
        self.fc5 = nn.Linear(n_hidden, n_action)



    # Define nn forward pass
    def forward(self, x):
        #x = x.to(device)
        x = torch.tanh(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        x = torch.tanh(self.fc3(x))
        x = torch.tanh(self.fc4(x))
        x = torch.tanh(self.fc5(x))
        return x
    
    #self.target_net=


    """
        2. Basic functions
    """
    
    def reset(self):
        self.memory=ReplayMemory(1000)
        self.nr_observations=0


    def best_action(self,state):

        state=torch.from_numpy(state)
        q_vec=self.forward(state.float()).squeeze()
        q_vec=q_vec.detach().numpy()
        
        return np.array([q_vec.argmax()]), q_vec.max()
        
    
    def eps_greedy_action(self,state,eps_thresh):
        
        state=torch.from_numpy(state)
        q_vec=self.forward(state.float()).squeeze()
        q_vec=q_vec.detach().numpy()         
        
        eps_test=np.random.uniform()
        
        if eps_test>eps_thresh:
            
            action_index=q_vec.argmax()
            action_value=q_vec.max()
        else:
            action_index=np.random.randint(0,self.n_action)
            action_value=q_vec[action_index]
        return np.array([action_index]), action_value
    


    """
        3. Learning functions
    """



    def learn(self,transition_data,eps_thresh,optimizer):
             
        state=torch.from_numpy(transition_data.state)
        next_state=torch.from_numpy(transition_data.next_state)
        action=torch.from_numpy(transition_data.action)
        reward=torch.from_numpy(transition_data.reward)
        
        self.memory.push(state, action, next_state, reward)
        self.nr_observations=self.nr_observations+1

        if len(self.memory) < self.batch_size:
            return
        
        
        transition_batch = self.memory.sample(self.batch_size)
        batch = Transition(*zip(*transition_batch))
    

        state_batch = torch.cat(batch.state).reshape([self.batch_size,self.n_state]).float()
        action_batch = torch.cat(batch.action).reshape([self.batch_size,1])
        reward_batch = torch.cat(batch.reward).reshape([self.batch_size,1]).float()
        next_states_batch = torch.cat(batch.next_state).reshape([self.batch_size,self.n_state]).float()
      
        state_action_values = self.forward(state_batch).gather(1, action_batch)
        next_state_values= self.forward(next_states_batch).max(1)[0].reshape([100,1])
        # next_state_values= self.forward(next_states_batch).max(1)[0].detach().reshape([100,1]) # this is the original
        expected_state_action_values = (next_state_values * self.gamma) + reward_batch
    
        # Compute Huber loss
        criterion = nn.SmoothL1Loss()
        loss = criterion(state_action_values, expected_state_action_values)
    
        # Optimize the model
        optimizer.zero_grad()
        loss.backward()
        for param in self.parameters():
            param.grad.data.clamp_(-1, 1)
        optimizer.step()
         
        #print(loss)
    
        # if self.nr_observations % 200 == 0:
        # target_net.load_state_dict(self.state_dict())
            
        
        
        

"""
    Kernel reinforcement learning --------------------------------------------
"""








