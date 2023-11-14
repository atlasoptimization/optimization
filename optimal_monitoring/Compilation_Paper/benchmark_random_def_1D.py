"""
The goal of this script is to train a TD3 RL algorithm on the random deformation
task and compare the cumulative rewards to the ones gathered by alternative 
discretization strategies.
For this, do the following
    1. Definitions and imports
    2. Train with stable baselines
    3. Apply alternative methods
    4. Summarize and plot results
    
The script is meant solely for educational and illustrative purposes. Written by
Jemil Avers Butt, Atlas optimization GmbH, www.atlasoptimization.ch.

"""


"""
    1. Definitions and imports
"""


# i) Import basics and custom environment

import numpy as np
import time
import matplotlib.pyplot as plt
from scipy.optimize import basinhopping
import class_random_def_1D_env as def_1D


# ii) Import stable baselines

from stable_baselines3 import TD3
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.monitor import Monitor


# iii) Initialize and check

np.random.seed(0)
def_1D_env=def_1D.Env()
def_1D_env=Monitor(def_1D.Env())
def_1D_env.reset()
check_env(def_1D_env)



"""
    2. Train with stable baselines
"""


# i) Train a TD3 Model

# start_time=time.time()
# model = TD3("MlpPolicy", def_1D_env,verbose=1, seed=0)
# model.learn(total_timesteps=10000)
# end_time=time.time()

# model.save('./Saved_models/trained_benchmark_random_def_1D')
model=TD3.load('./Saved_models/trained_benchmark_random_def_1D')




"""
    3. Apply alternative methods
"""


# Note: All actions are in [-1,1] and get mapped to [0,1] by to the environment
# translating input actions from the symmetric box space [-1,1] to indices

# i) Grid based sampling

def grid_based_sampling(environment):
    action_index=environment.round_to_index(environment.epoch/(environment.max_epoch-1)-1)
    action=np.array(2*environment.x[action_index]-1)
    return action


# ii) Pseudo random sampling

def pseudo_random_sampling(environment):
    Halton_sequence=np.array([1/2, 1/4, 3/4, 1/8, 5/8, 3/8, 7/8])*2-np.ones([7])
    action=Halton_sequence[environment.epoch]
    return action


# iii) Random sampling

def random_sampling(environment):
    action=np.random.uniform(-1,1,[1])
    return action


# iv) Numerical integration

def quadrature_sampling(environment):
    Gauss_points=np.array([-0.861, -0.34, 0.34, 0.861])
    action=Gauss_points[environment.epoch]
    return action


# v) Experiment design based sampling

n_average=1000
fun_table=np.zeros([n_average,def_1D_env.n_disc_x])
for k in range(n_average):
    def_1D_env.reset()
    fun_table[k,:]=def_1D_env.def_fun

def loss_fun(x_vec):
    index_vec=np.zeros(def_1D_env.n_meas)
    for k in range(def_1D_env.n_meas):    
        index_vec[k]=def_1D_env.round_to_index(x_vec[k]*0.5+0.5)
    
    f_max=np.max(fun_table,axis=1)
    f_obs_mat=fun_table[:,index_vec.astype(int)]
    f_obs_max=np.max(f_obs_mat,axis=1)
    
    loss_vec=np.abs(f_obs_max-f_max)
    loss_val=np.mean(loss_vec)
    return loss_val

x_0 = np.array([-0.7,-0.3,0.3,0.7])
x_design = basinhopping(loss_fun, x_0, disp=True)

def experiment_design_sampling(environment):
    action=x_design.x[environment.epoch]
    return action



"""
    4. Summarize and plot results
"""


# i) Summarize results in table

n_episodes_table=1000
table=np.zeros([n_episodes_table,6])


# Grid based sampling results
for k in range(n_episodes_table):
    done=False
    obs = def_1D_env.reset()
    while done ==False:
        action = grid_based_sampling(def_1D_env)
        obs, reward, done, info = def_1D_env.step(action)

        if done:
            table[k,0]=reward
            break


# Pseudo random sampling results
for k in range(n_episodes_table):
    done=False
    obs = def_1D_env.reset()
    while done ==False:
        action = pseudo_random_sampling(def_1D_env)
        obs, reward, done, info = def_1D_env.step(action)

        if done:
            table[k,1]=reward
            break
        
# Random sampling results
for k in range(n_episodes_table):
    done=False
    obs = def_1D_env.reset()
    while done ==False:
        action = random_sampling(def_1D_env)
        obs, reward, done, info = def_1D_env.step(action)

        if done:
            table[k,2]=reward
            break
        
# Numerical integration sampling results
for k in range(n_episodes_table):
    done=False
    obs = def_1D_env.reset()
    while done ==False:
        action = quadrature_sampling(def_1D_env)
        obs, reward, done, info = def_1D_env.step(action)

        if done:
            table[k,3]=reward
            break
        
 # Experiment design sampling results
for k in range(n_episodes_table):
    done=False
    obs = def_1D_env.reset()
    while done ==False:
        action = experiment_design_sampling(def_1D_env)
        obs, reward, done, info = def_1D_env.step(action)

        if done:
            table[k,4]=reward
            break


# RL sampling results
for k in range(n_episodes_table):
    done=False
    obs = def_1D_env.reset()
    while done ==False:
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done, info = def_1D_env.step(action)

        if done:
            table[k,5]=reward
            break


# ii) Illustrate results

n_episodes=3

for k in range(n_episodes):
    done=False
    obs = def_1D_env.reset()
    while done ==False:
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done, info = def_1D_env.step(action)

        if done:
            def_1D_env.render(reward)
            # time.sleep(0.5)
            break

mean_summary=np.mean(table,axis=0)
std_summary=np.std(table,axis=0)

print(' Reward means of different methods')
print(mean_summary)
print(' Reward standard_deviations of different methods')
print(std_summary)
# print('Time for RL procedure = ', end_time-start_time ,'sec')


# iii) Plot training loss

plt.figure(2, dpi=300)
plt.plot(def_1D_env.get_episode_rewards(),color='k')
plt.title('Training reward')
plt.xlabel('Episode')
plt.ylabel('Reward')















