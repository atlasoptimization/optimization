"""
The goal of this script is to load a model that can be used for optimizing 
total-station-based measurement sequences and provide the facalities to eable
interaction between the trained neural nets and the total stations.
For this, do the following:
    1. Definitions and imports
    2. Create environment: optimal reconstruction
    3. Train + save (Optional) + load the model 
    4. Create manual environment
    5. Interaction sequence: Measuring a plate
    6. Plots and illustrations
"""



"""
    1. Definitions and imports ------------------------------------------------
"""



# i) Import standard packages

import numpy as np
import matplotlib
import copy
import matplotlib.pyplot as plt
import time




# ii) Import optimization algorithms

import gym
from gym import spaces
import rl_support_funs as sf
from scipy.optimize import basinhopping
from stable_baselines3 import TD3
from stable_baselines3.common.env_checker import check_env



"""
    2. Create environment: optimal reconstruction ----------------------------
"""



class Env(gym.Env):
    metadata = {'render.modes': ['human']}



    # 2. Class intitalization -------------------------------------------------

        
    
    # i) Initialization
    
    def __init__(self):
        super(Env, self).__init__()
        
        
        # ii) Boundary definitions
            
        x1_max=50
        x2_max=40
        self.x_max=np.array([[x1_max],[x2_max]])
        
        self.n_meas=9
        self.n_state=3*self.n_meas
    
        self.max_epoch=9
        self.x1=np.linspace(0,1,x1_max)
        self.x2=np.linspace(0,1,x2_max)
        
        
        # iii) Observation and action spaces
        
        self.action_space = spaces.Box(low=-1, high=1,
                                        shape=(2,), dtype=np.float32)
        # Example for using image as input (channel-first; channel-last also works):
        self.observation_space = spaces.Box(low=-1, high=1,
                                        shape=(self.n_state,), dtype=np.float32)
        
        
        # iv) Initial properties
        
        self.epoch=0
        self.state=-1*np.ones(self.n_state)
        self.x_measured=np.empty([0,2])
        self.ind_measured=np.empty([0,2])
        self.f_measured=np.empty([0,1])
        self.x_meas_sequence=-1*np.ones([self.n_meas,2])
        self.f_meas_sequence=-1*np.ones(self.n_meas)
        
        rf, K_x1, K_x2 =sf.Simulate_random_field_fast(self.x1, self.x2, self.cov_fun, self.cov_fun)
        self.fun=rf
        self.K_x1, self.K_x2 = K_x1, K_x2
        self.K=np.kron(K_x1,K_x2)
        
        


    # 3. Auxiliary methods ---------------------------------------------------


    
    # i) Round continuous location to index
    
    def round_to_index(self,x):
        approx_index=x*self.x_max.squeeze()
        index=np.ceil(approx_index)-1
        if index[0]>=self.x_max[0]:
            index[0]=self.x_max[0]-1
        elif index[0]<=0:
            index[0]=0
        else:
            pass
            
        if index[1]>=self.x_max[1]:
            index[1]=self.x_max[1]-1
        elif index[1]<=0:
            index[1]=0
        else:
            pass
        
        return index.astype(int)

        
    # ii) Covariance function
        
    def cov_fun(self,t,s):
      cov_val = 0.1*np.exp(-np.abs((t-s)/0.3)**2)
      return cov_val
        


    """
        3. Step method
    """

    def step(self, action):
            
    
        # i) Perform action and update state
        
        meas_pos=(0.5)*action+0.5
        self.meas_pos=meas_pos
        
        meas_index=self.round_to_index(meas_pos)
        f_measured=self.fun[meas_index[0],meas_index[1]]
    
        self.f_measured=np.vstack((f_measured,self.f_measured))
        self.x_measured=np.vstack((np.array([self.x1[meas_index[0]], self.x2[meas_index[1]]]).squeeze(), self.x_measured))
        self.ind_measured=np.vstack((np.array([meas_index[0], meas_index[1]]).squeeze(), self.ind_measured)).astype(int)
        
        augmented_x_vec=np.vstack((self.x_measured, self.x_meas_sequence))
        augmented_f_vec=np.hstack((self.f_measured.squeeze(), self.f_meas_sequence))
        self.state=np.hstack((augmented_x_vec[0:self.n_meas,0], augmented_x_vec[0:self.n_meas,1], augmented_f_vec[0:self.n_meas]))
        
        
        # ii) Perform estimation
        
        n_full=self.fun.shape[0]*self.fun.shape[1]
        x=self.x_measured
        n_data=x.shape[0]
        lin_ind_vec=np.ravel_multi_index((self.ind_measured[:,0],self.ind_measured[:,1]), [self.fun.shape[0],self.fun.shape[1]])
        
        K_ij=np.zeros([n_data,n_data])
        K_t=np.zeros([n_full,n_data])
        
        for k in range(n_data):
            for l in range(n_data):
                K_ij[k,l]=self.K[lin_ind_vec[k],lin_ind_vec[l]]
        
        for k in range(n_data):
            K_t[:,k]=self.K[:,lin_ind_vec[k]]
            
        fun_hat=K_t@np.linalg.pinv(K_ij)@self.f_measured
        fun_hat=np.reshape(fun_hat,[self.x_max[0].item(),self.x_max[1].item()])
        
        # reward=-np.abs(np.max(np.abs(self.fun))-np.max(np.abs(self.f_measured)))
        
        reward=-np.mean(np.linalg.norm(self.fun.flatten()-fun_hat.flatten()))
        
        
        # iii) Update epoch, check if done
        
        self.epoch=self.epoch+1
        if self.epoch==self.max_epoch:
            done=True
        else:
            done=False
        
        info = {}

        return self.state, reward, done, info

        
            


     #  4. Reset method ------------------------------------------------------



    def reset(self):
        
        
        # i) Reinitialize by simulating again
        
        self.epoch=0
        self.state=-1*np.ones(self.n_state)
        self.x_measured=np.empty([0,2])
        self.ind_measured=np.empty([0,2])
        self.f_measured=np.empty([0,1])
        self.x_meas_sequence=-1*np.ones([self.n_meas,2])
        self.f_meas_sequence=-1*np.ones(self.n_meas)
        
        rf, K_x1, K_x2 =sf.Simulate_random_field_fast(self.x1, self.x2, self.cov_fun, self.cov_fun)
        self.fun=rf
        
        observation=self.state
        return observation
        


    #  5. Render method and close ---------------------------------------------

    
    
    # i) Render plot with measured locations
    
    def render(self, reward, mode='console'):
        
        plt.imshow(self.fun.T, extent=[0,1,1,0])
        plt.colorbar()
        plt.scatter(self.x_measured[:,0],self.x_measured[:,1])
        plt.title('Sequential spatial measurements')
        plt.xlabel('x1 axis')
        plt.ylabel('x2 axis')
        print(reward, self.x_measured,self.f_measured)
        print('Reward is ', reward) 
        print('Measured locations are', self.x_measured)
        print(' Measurements are', self.f_measured)
        
        
    # ii) Close method
    
    def close (self):
      pass




  

"""  
    3. Train + save (Optional) + load the model  ------------------------------
"""


# i) Initialize and check

np.random.seed(0)
def_2D_env=Env()
def_2D_env.reset()
check_env(def_2D_env)


# ii) Train a TD3 Model

start_time=time.time()
model = TD3("MlpPolicy", def_2D_env,verbose=1, seed=0)
model.learn(total_timesteps=100000)
end_time=time.time()

model.save("td3_plate_sampling")
del model


model = TD3.load("td3_plate_sampling")


# iii) Grid based sampling

def grid_based_sampling(environment):
    grid_x1=np.kron(np.array([-1/3, 1/3, 1]),np.array([1, 1, 1]))
    grid_x2=np.kron(np.array([1, 1, 1]), np.array([-1/3, 1/3, 1]))
    grid=np.vstack((grid_x1, grid_x2))
    action=grid[:,environment.epoch]
    return action


# iv) Summarize results in table

n_episodes_table=1000
table=np.zeros([n_episodes_table,2])


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
        
# RL sampling results
for k in range(n_episodes_table):
    done=False
    obs = def_2D_env.reset()
    while done ==False:
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done, info = def_2D_env.step(action)

        if done:
            table[k,1]=reward
            break


# v) Illustrate results

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



"""
    4. Create manual environment  --------------------------------------------
"""



"""
    5. Interaction sequence: Measuring a plate -------------------------------
"""



"""
    6. Plots and illustrations -----------------------------------------------
"""
