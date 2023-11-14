"""
The goal of this script is to train a TD3 RL algorithm on the random deformation 
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
import matplotlib.pyplot as plt
from scipy.optimize import basinhopping
import class_random_def_2D_env as def_2D



# ii) Import stable baselines

from stable_baselines3 import TD3, PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.noise import OrnsteinUhlenbeckActionNoise


# iii) Initialize and check

np.random.seed(0)
def_2D_env=def_2D.Env()
def_2D_env=Monitor(def_2D_env)
def_2D_env.reset()
check_env(def_2D_env)



"""
    2. Train with stable baselines
"""


# i) Train a TD3 Model

start_time=time.time()
# action_noise=OrnsteinUhlenbeckActionNoise(np.zeros([2]), 0.1*np.ones([2]))
# model = TD3("MlpPolicy", def_2D_env,verbose=1, action_noise=action_noise, seed=0)
model = TD3("MlpPolicy", def_2D_env,verbose=1, seed=0)
# model = PPO("MlpPolicy", def_2D_env,verbose=1, seed=0)
model.learn(total_timesteps=200000)
end_time=time.time()

# model.save('./Saved_models/trained_benchmark_random_def_2D')
# model=TD3.load('./Saved_models/trained_benchmark_random_def_2D')



"""
    3. Apply alternative methods
"""


# Note: All actions are in [-1,1]x[-1,1] and get mapped to [0,1]x[0,1] by 
# the environment translating input actions from the symmetric box space 
# [-1,1]x[-1,1] to indices

# i) Grid based sampling

def grid_based_sampling(environment):
    grid_x1=np.kron(np.array([-1/3, 1/3, 1]),np.array([1, 1, 1]))
    grid_x2=np.kron(np.array([1, 1, 1]), np.array([-1/3, 1/3, 1]))
    grid=np.vstack((grid_x1, grid_x2))
    action=grid[:,environment.epoch]
    return action


# ii) Pseudo random sampling

def pseudo_random_sampling(environment):
    Halton_sequence=np.array([[1/2, 1/4, 3/4, 1/8, 5/8, 3/8, 7/8, 1/16, 9/17, 
                               3/16],[1/3, 2/3, 1/9, 4/9, 7/9, 2/9, 5/9, 8/9, 1/27, 10/27]])*2-np.ones([2,10])
    action=Halton_sequence[:, environment.epoch]
    return action


# iii) Random sampling

def random_sampling(environment):
    action=np.random.uniform(-1,1,[2])
    return action


# iv) Numerical integration

def quadrature_sampling(environment):
    Gauss_points_x1=np.kron(np.array([-0.77, 0, 0.77]),np.array([1, 1, 1]))
    Gauss_points_x2=np.kron(np.array([1, 1, 1]),np.array([-0.77, 0, 0.77]))
    Gauss_points=np.vstack((Gauss_points_x1, Gauss_points_x2))
    action=Gauss_points[:, environment.epoch]
    return action


# v) Experiment design based sampling

n_average=1000
fun_table=np.zeros([n_average,(def_2D_env.x_max[0]*def_2D_env.x_max[1]).astype(int).item()])
for k in range(n_average):
    def_2D_env.reset()
    fun_table[k,:]=def_2D_env.fun.flatten()

def loss_fun(x_vec):
    x_vec=np.reshape(x_vec, [2,9])
    index_vec=np.zeros([2,def_2D_env.n_meas])
    lin_ind_vec=np.zeros(def_2D_env.n_meas)
    for k in range(def_2D_env.n_meas):    
        index_vec[:,k]=def_2D_env.round_to_index(x_vec[:,k]*0.5+0.5)
        lin_ind_vec[k]=np.ravel_multi_index((index_vec[0,k].astype(int), index_vec[1,k].astype(int)),
                                            [def_2D_env.x_max[0].item(),def_2D_env.x_max[1].item()])
    
    f_max=np.max(fun_table,axis=1)
    f_obs_mat=fun_table[:,lin_ind_vec.astype(int)]
    f_obs_max=np.max(f_obs_mat,axis=1)
    
    loss_vec=np.abs(f_obs_max-f_max)
    loss_val=np.mean(loss_vec)
    return loss_val


x_0=np.zeros([2*def_2D_env.n_meas])
x_design = basinhopping(loss_fun, x_0, disp= True)
x_vec=np.reshape(x_design.x,[2,9])

def experiment_design_sampling(environment):
    action=x_vec[:,environment.epoch]
    return action



"""
    4. Summarize and plot results
"""



# i) Summarize results in table

n_episodes_table=10
table=np.zeros([n_episodes_table,6])


# Grid based sampling results
for k in range(n_episodes_table):
    done=False
    obs = def_2D_env.reset()
    while done ==False:
        action = grid_based_sampling(def_2D_env)
        obs, reward, done, info = def_2D_env.step(action)

        if done:
            table[k,0]=reward
            break


# Pseudo random sampling results
for k in range(n_episodes_table):
    done=False
    obs = def_2D_env.reset()
    while done ==False:
        action = pseudo_random_sampling(def_2D_env)
        obs, reward, done, info = def_2D_env.step(action)

        if done:
            table[k,1]=reward
            break
        
# Random sampling results
for k in range(n_episodes_table):
    done=False
    obs = def_2D_env.reset()
    while done ==False:
        action = random_sampling(def_2D_env)
        obs, reward, done, info = def_2D_env.step(action)

        if done:
            table[k,2]=reward
            break
        
# Numerical integration sampling results
for k in range(n_episodes_table):
    done=False
    obs = def_2D_env.reset()
    while done ==False:
        action = quadrature_sampling(def_2D_env)
        obs, reward, done, info = def_2D_env.step(action)

        if done:
            table[k,3]=reward
            break
        
 # Experiment design sampling results
for k in range(n_episodes_table):
    done=False
    obs = def_2D_env.reset()
    while done ==False:
        action = experiment_design_sampling(def_2D_env)
        obs, reward, done, info = def_2D_env.step(action)

        if done:
            table[k,4]=reward
            break


# RL sampling results
for k in range(n_episodes_table):
    done=False
    obs = def_2D_env.reset()
    while done ==False:
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done, info = def_2D_env.step(action)

        if done:
            table[k,5]=reward
            break


# ii) Illustrate results

n_episodes=1

for k in range(n_episodes):
    done=False
    obs = def_2D_env.reset()
    while done ==False:
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done, info = def_2D_env.step(action)

        if done:
            def_2D_env.render(reward)
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
plt.plot(np.convolve(def_2D_env.get_episode_rewards(),np.ones(100)/100,mode='valid'),color='k')
# plt.ylim([-1,0])
plt.title('Training reward')
plt.xlabel('Episode')
plt.ylabel('Reward')