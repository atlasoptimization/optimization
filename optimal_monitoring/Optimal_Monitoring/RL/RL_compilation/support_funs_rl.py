"""
The goal of this set of scripts and functions is to provide a set of support
functionalities for reinforcement learning that are available after import.
These include:
    1. e_greedy_action.py - e-greedy action choice
"""




"""
    1. Imports and Definitions -----------------------------------------------
"""

# i) Imports

import numpy as np
import random
import matplotlib.pyplot as plt
import torch

# ii) Definitions




"""
    e-greedy action choice ------------------------------------------------
    
    This function takes as input a Q function and several auxiliary parameters
    to derive a choice of action based on an epsilon greedy policy. For this,
    do the following:
        1. Definitions and imports
        2. eps greedy choice
    
    INPUTS
        Name                 Interpretation                             Type
    q_vec               Q vector detailing values of actions for        function
                        the current state
    dims                The dimensions of the environment (state        2d-vector
                        and action spaces) in form [n_state,n_action]                              
    step_nr             The number of steps already done                Integer
    eps_opts            Options for decay of random choice in vec-      3d-vector
                        tor [eps_start,eps_end,eps_decay_slowness]
                        
    OUTPUT
        Name                 Interpretation                             Type
    action_index        Index of action that was chose                  Integer   
"""



def e_greedy_action(q_vec, dims, step_nr, eps_opts=[0.9,0.01,300]):
    
    """
        1. Definitions and imports 
    """
    
    n_action=dims[1]
    sample = random.random()
    eps_threshold = eps_opts[1] + (eps_opts[0] - eps_opts[1]) * \
        np.exp(-1. * step_nr / eps_opts[2])
        
        
        
    """
        2. eps greedy choice
    """
    
    if sample > eps_threshold:
        with torch.no_grad():
            # t.max(1) will return largest column value of each row.
            # second column on max result is index of where max element was
            # found, so we pick action with the larger expected reward.
            return q_vec.max(0)[1]
    else:
        return torch.tensor([[random.randrange(n_action)]], dtype=torch.long)










