"""
This script tests some simple reinforcement learning algorithms. To this end,
environments from the rl_environments.py and agents from rl_agents.py are
invoked and executed.
"""



"""
    1. Definitions and imports -----------------------------------------------
"""


import sys

sys.path.insert(0, '../RL_compilation')
sys.path.insert(0, '../RL_tests')

import numpy as np
from itertools import count
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import rl_agents
import rl_environments
import support_funs_rl




"""
    2. Discrete Q learning for Control landing -------------------------------
"""


# # i) Set up the environments

# env_control_landing=rl_environments.Control_landing_env_discrete([10,6,4],0)
# Q_learner_landing=rl_agents.Discrete_Q_learner([[env_control_landing.n_state],[env_control_landing.n_action]],[0.9,0.1])


# # ii) Execute learning loop

# n_episodes=10000

# for k_episode in range(n_episodes):
    
#     print(k_episode)
#     env_control_landing.reset()
    
#     for t in count():
        
#         action,action_value=Q_learner_landing.best_action(env_control_landing.state_index)
#         reward,done=env_control_landing.step(action)
#         Q_learner_landing.learn(env_control_landing.last_transition)
             
#         if done:
#             break
    

# print('Complete')

# print(Q_learner_landing.q_table)

# print(env_control_landing.state_sequence)
# print(env_control_landing.action_sequence)
# print(env_control_landing.reward_sequence)


# # iii) Plot evaluation rounds

# n_episodes_eval=40
# z_trajectory_sample=np.zeros([env_control_landing.dim[0]-1,n_episodes_eval])
# x_trajectory_sample=np.zeros([env_control_landing.dim[0]-1,n_episodes_eval])
# vx_trajectory_sample=np.zeros([env_control_landing.dim[0]-1,n_episodes_eval])

# for k_episode in range(n_episodes_eval):
    
#     env_control_landing.reset()
    
#     for t in count():
        
#         z_trajectory_sample[t,k_episode]=env_control_landing.state[0]
#         x_trajectory_sample[t,k_episode]=env_control_landing.state[1]
#         vx_trajectory_sample[t,k_episode]=env_control_landing.state[2]
        
#         action,action_value=Q_learner_landing.best_action(env_control_landing.state_index)
#         reward,done=env_control_landing.step(action)
#         Q_learner_landing.learn(env_control_landing.last_transition)
        
#         if done:
#             break
        
# plt.plot(x_trajectory_sample)



"""
    2. Discrete Q learning for Border walking -------------------------------
"""


# # i) Set up the environments

# border_env=rl_environments.Border_walking_env()
# Q_learner_border=rl_agents.Discrete_Q_learner([[border_env.n_state],[border_env.n_action]],[0.9,0.1])


# # ii) Execute learning loop

# n_episodes=1000

# for k_episode in range(n_episodes):
    
#     print(k_episode)
#     border_env.reset()
    
#     for t in count():
        
#         action,action_value=Q_learner_border.best_action(border_env.state_index)
#         reward,done=border_env.step(action)
#         Q_learner_border.learn(border_env.last_transition)
             
#         if done:
#             break
    

# print('Complete')

# print(border_env.state_sequence)
# print(border_env.action_sequence)
# print(border_env.reward_sequence)
# print(Q_learner_border.q_table)


"""
    3. Discrete SARSA learning for Border walking -----------------------------
"""


# # i) Set up the environments

# border_env=rl_environments.Border_walking_env()
# SARSA_learner_border=rl_agents.Discrete_SARSA_learner([[border_env.n_state],[border_env.n_action]],[0.9,0.9,0.1])


# # ii) Execute learning loop

# n_episodes=1000

# for k_episode in range(n_episodes):
    
#     print(k_episode)
#     epsilon_thresh=1-k_episode/n_episodes
#     border_env.reset()
    
#     for t in count():
        
#         action,action_value=SARSA_learner_border.eps_greedy_action(border_env.state_index,epsilon_thresh)
#         reward,done=border_env.step(action)
#         SARSA_learner_border.learn(border_env.last_transition,epsilon_thresh)
             
#         if done:
#             SARSA_learner_border.reset_eligibility()
#             break
    

# print('Complete')

# print(border_env.state_sequence)
# print(border_env.action_sequence)
# print(border_env.reward_sequence)
# print(SARSA_learner_border.q_table)





"""
    4. Discrete SARSA learning for Control landing -----------------------------
"""


# # i) Set up the environments

# env_control_landing=rl_environments.Control_landing_env_discrete([10,6,2],0)
# SARSA_learner_landing=rl_agents.Discrete_SARSA_learner([[env_control_landing.n_state],[env_control_landing.n_action]],[0.9,0.9,0.1])


# # ii) Execute learning loop

# n_episodes=1000

# for k_episode in range(n_episodes):
    
#     print(k_episode)
#     epsilon_thresh=1-k_episode/n_episodes
#     env_control_landing.reset()
    
#     for t in count():
        
#         action,action_value=SARSA_learner_landing.eps_greedy_action(env_control_landing.state_index,epsilon_thresh)
#         reward,done=env_control_landing.step(action)
#         SARSA_learner_landing.learn(env_control_landing.last_transition,epsilon_thresh)
             
#         if done:
#             SARSA_learner_landing.reset_eligibility()
#             break
    

# print('Complete')

# print(SARSA_learner_landing.q_table)

# print(env_control_landing.state_sequence)
# print(env_control_landing.action_sequence)
# print(env_control_landing.reward_sequence)


# # iii) Plot evaluation rounds

# n_episodes_eval=40
# z_trajectory_sample=np.zeros([env_control_landing.dim[0]-1,n_episodes_eval])
# x_trajectory_sample=np.zeros([env_control_landing.dim[0]-1,n_episodes_eval])
# vx_trajectory_sample=np.zeros([env_control_landing.dim[0]-1,n_episodes_eval])

# for k_episode in range(n_episodes_eval):
    
#     env_control_landing.reset()
    
#     for t in count():
        
#         z_trajectory_sample[t,k_episode]=env_control_landing.state[0]
#         x_trajectory_sample[t,k_episode]=env_control_landing.state[1]
#         vx_trajectory_sample[t,k_episode]=env_control_landing.state[2]
        
#         action,action_value=SARSA_learner_landing.best_action(env_control_landing.state_index)
#         reward,done=env_control_landing.step(action)
#         SARSA_learner_landing.learn(env_control_landing.last_transition,0)
        
#         if done:
#             break

# plt.figure(1)        
# plt.plot(x_trajectory_sample)

# t_vals=np.linspace(-3,3,100)
# x_vals=np.array([np.linspace(5,5,100), np.linspace(-3,3,100),np.linspace(0,0,100)])
# # x_index=np.zeros([100,3])
# x_linind=np.zeros([100,1])


# for k in range(100):
    
#     x_index=env_control_landing.state_to_index_array(np.array([[x_vals[0,k]],[x_vals[1,k]],[x_vals[2,k]]]))
#     x_linind[k]=env_control_landing.array_to_linind(x_index)

# plt.figure(2)
# #index_x_change=np.ravel_multi_index(multi_index, (6,10,6) )


# plt.plot(t_vals,SARSA_learner_landing.q_table[x_linind.astype(int)].squeeze()[:,0])
# plt.plot(t_vals,SARSA_learner_landing.q_table[x_linind.astype(int)].squeeze()[:,1])



"""
    4. DQN for continuous Border walking --------------------------------------
"""



# # i) Set up the environments

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# border_env=rl_environments.Border_walking_env_continuous()
# DQN_learner_border=rl_agents.DQN_learner([border_env.n_state,border_env.n_action],[0.9,0.1]).to(device)

# optimizer = optim.RMSprop(DQN_learner_border.parameters())


# # ii) Execute learning loop


# n_episodes=100

# for k_episode in range(n_episodes):
    
#     print(k_episode)
#     epsilon_thresh=1-k_episode/n_episodes
#     border_env.reset()
    
#     for t in count():
        
#         action,action_value=DQN_learner_border.eps_greedy_action(border_env.state,epsilon_thresh)
#         reward,done=border_env.step(action)
#         DQN_learner_border.learn(border_env.last_transition,epsilon_thresh,optimizer)
             
#         if done:
#             break
    

# print('Complete')

# print(border_env.state_sequence)
# print(border_env.action_sequence)
# print(border_env.reward_sequence)

# t_vals=np.linspace(-3,3,100)
# x_vals=torch.from_numpy(np.linspace(-3,3,100))
# q=DQN_learner_border.forward(x_vals.reshape([100,1]).float())

# plt.plot(t_vals,q[:,0].detach())
# plt.plot(t_vals,q[:,1].detach())

# plt.show()



"""
    4. DQN with exprep for Border walking --------------------------------------
"""



# # i) Set up the environments

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# border_env=rl_environments.Border_walking_env_continuous()
# DQN_learner_border=rl_agents.DQN_learner_exprep([border_env.n_state,border_env.n_action],[0.9,0.01],100).to(device)

# optimizer = optim.RMSprop(DQN_learner_border.parameters())


# # ii) Execute learning loop


# n_episodes=100

# for k_episode in range(n_episodes):
    
#     print(k_episode)
#     epsilon_thresh=1-k_episode/n_episodes
#     border_env.reset()
    
#     for t in count():
        
#         action,action_value=DQN_learner_border.eps_greedy_action(border_env.state,epsilon_thresh)
#         reward,done=border_env.step(action)
#         DQN_learner_border.learn(border_env.last_transition,epsilon_thresh,optimizer)
             
#         if done:
#             break
    

# print('Complete')

# print(border_env.state_sequence)
# print(border_env.action_sequence)
# print(border_env.reward_sequence)

# t_vals=np.linspace(-3,3,100)
# x_vals=torch.from_numpy(np.linspace(-3,3,100))
# q=DQN_learner_border.forward(x_vals.reshape([100,1]).float())

# plt.plot(t_vals,q[:,0].detach())
# plt.plot(t_vals,q[:,1].detach())

# plt.show()


"""
    5. DQN with exprep for control landing ------------------------------------
"""



# # i) Set up the environments

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# landing_env=rl_environments.Control_landing_env_continuous(np.array([2]),0)
# DQN_learner_landing=rl_agents.DQN_learner_exprep([landing_env.dim[0],landing_env.dim[1]],[0.9,0.01],100).to(device)


# optimizer = optim.RMSprop(DQN_learner_landing.parameters())


# # ii) Execute learning loop


# n_episodes=1000

# for k_episode in range(n_episodes):
    
#     print(k_episode)
#     epsilon_thresh=1-k_episode/n_episodes
#     landing_env.reset()
    
#     for t in count():
        
#         action,action_value=DQN_learner_landing.eps_greedy_action(landing_env.state.T,epsilon_thresh)
#         reward,done=landing_env.step(action)
#         DQN_learner_landing.learn(landing_env.last_transition,epsilon_thresh,optimizer)
             
#         if done:
#             break
    

# print('Complete')

# print(landing_env.state_sequence)
# print(landing_env.action_sequence)
# print(landing_env.reward_sequence)



# # iii) Plot evaluation rounds

# n_episodes_eval=10
# z_trajectory_sample=np.zeros([landing_env.z_max,n_episodes_eval])
# x_trajectory_sample=np.zeros([landing_env.z_max,n_episodes_eval])
# vx_trajectory_sample=np.zeros([landing_env.z_max,n_episodes_eval])

# for k_episode in range(n_episodes_eval):
    
#     landing_env.reset()
    
#     for t in count():
        
#         z_trajectory_sample[t,k_episode]=landing_env.state[0]
#         x_trajectory_sample[t,k_episode]=landing_env.state[1]
#         vx_trajectory_sample[t,k_episode]=landing_env.state[2]
        
#         action,action_value=DQN_learner_landing.best_action(landing_env.state.T)
#         reward,done=landing_env.step(action)
#         DQN_learner_landing.learn(landing_env.last_transition,0,optimizer)
        
#         if done:
#             break
# plt.figure(1)      
# plt.plot(x_trajectory_sample)


# t_vals=np.linspace(-3,3,100)
# x_vals=torch.from_numpy(np.array([np.linspace(5,5,100), np.linspace(-3,3,100),np.linspace(0,0,100)]))
# q=DQN_learner_landing.forward(x_vals.reshape([100,3]).float())

# plt.figure(2)
# plt.plot(t_vals,q[:,0].detach())
# plt.plot(t_vals,q[:,1].detach())
# plt.show()














