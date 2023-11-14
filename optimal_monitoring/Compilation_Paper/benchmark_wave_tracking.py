"""
The goal of this script is to train a TD3 RL algorithm on the deformation tracking
task and compare the cumulative rewards to the ones gathered by alternative 
discretization strategies.
For this, do the following
    1. Definitions and imports
    2. Train with stable baselines
    3. Apply alternative methods
    4. Summarize and plot results
"""


"""
    1. Definitions and imports
"""


# i) Import basics and custom environment

import numpy as np
import time
from scipy.optimize import basinhopping
import class_wave_tracking_env as def_track


# ii) Import stable baselines

from stable_baselines3 import TD3
from stable_baselines3.common.env_checker import check_env


# iii) Initialize and check

np.random.seed(0)
def_track_env=def_track.Env()
def_track_env.reset()
check_env(def_track_env)



"""
    2. Train with stable baselines
"""


# i) Train a TD3 Model

# start_time=time.time()
# model = TD3("MlpPolicy", def_track_env,verbose=1,seed=0)
# model.learn(total_timesteps=100000)
# end_time=time.time()

# model.save('./Saved_models/trained_benchmark_wave_tracking')
model=TD3.load('./Saved_models/trained_benchmark_wave_tracking')




"""
    3. Apply alternative methods
"""



# Note: All actions are in [-1,1] and get mapped to [0,1] by to the environment
# translating input actions from the symmetric box space [-1,1] to indices

# i) Grid based sampling - not sensible since only one measurement per epoch


# ii) Pseudo random sampling - not sensible since only one measurement per epoch


# iii) Random sampling

def random_sampling(environment):
    action=np.random.uniform(-1,1,[1])
    return action


# iv) Numerical integration - not sensible since only one measurement per epoch


# v) Experiment design based sampling

n_average=100
crest_table=np.zeros([n_average, def_track_env.t_max ])
for k in range(n_average):
    def_track_env.reset()
    crest_table[k,:]=(def_track_env.crest_vec).flatten()

def loss_fun(x_vec):
    loss_vec=np.zeros(n_average)
    for k in range(n_average):
        loss_vec[k]=np.linalg.norm(x_vec-crest_table[k,:].squeeze())
    loss_val=np.sum(loss_vec)
    return loss_val

x_0 = np.random.uniform(-1,1,def_track_env.t_max)
x_design = basinhopping(loss_fun, x_0, niter=5, disp=True)

def experiment_design_sampling(environment):
    action=x_design.x[environment.epoch]
    return action



"""
    4. Summarize and plot results
"""



# i) Summarize results in table

n_episodes_table=1000
table=np.zeros([n_episodes_table,6])

        
# Random sampling results
for k in range(n_episodes_table):
    done=False
    obs = def_track_env.reset()
    while done ==False:
        action = random_sampling(def_track_env)
        obs, reward, done, info = def_track_env.step(action)

        if done:
            table[k,2]=reward
            break
        
        
 # Experiment design sampling results
for k in range(n_episodes_table):
    done=False
    obs = def_track_env.reset()
    while done ==False:
        action = experiment_design_sampling(def_track_env)
        obs, reward, done, info = def_track_env.step(action)

        if done:
            table[k,4]=reward
            break


# RL sampling results
for k in range(n_episodes_table):
    done=False
    obs = def_track_env.reset()
    while done ==False:
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done, info = def_track_env.step(action)

        if done:
            table[k,5]=reward
            break


# ii) Illustrate results

n_episodes=1

for k in range(n_episodes):
    done=False
    obs = def_track_env.reset()
    while done ==False:
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done, info = def_track_env.step(action)

        if done:
            def_track_env.render(reward)
            break

mean_summary=np.mean(table,axis=0)
std_summary=np.std(table,axis=0)

print(' Reward means of different methods')
print(mean_summary)
print(' Reward standard_deviations of different methods')
print(std_summary)
# print('Time for RL procedure = ', end_time-start_time ,'sec')


















