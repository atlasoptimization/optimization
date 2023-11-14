"""
The goal of this script is to provide the environment simulating a two-dimensional
random deformation. This is done by drawing realizations of a Gaussian process with 
squared exponential covariance based on tensored Karhunen Loewe decomposition. 
The environment consists of states indicating past measurements and actions 
indicating designated next measurements. Rewards are differences between 
measured and actual maximum deformations. 
For this, do the following:
    1. Definitions and imports
    2. Class intitalization
    3. Auxiliary methods
    3. Step method
    4. Reset method
    5. Render method
The class can be called upon in any script by evoking it e.g. via 
    import class_random_def_2D_env
    random_def_2D_env=class_random_def_2D_env.Env()
"""



"""
    1. Definitions and imports
"""


# i) Import standard packages

import numpy as np
import matplotlib
import copy
import matplotlib.pyplot as plt

import gym
from gym import spaces
import rl_support_funs as sf

# ii) Class constructor

class Env(gym.Env):
    metadata = {'render.modes': ['human']}



    """
        2. Class intitalization
    """
        
    
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
        self.state=-1*np.ones(self.n_state).astype('float32')
        self.x_measured=np.empty([0,2])
        self.f_measured=np.empty([0,1])
        self.x_meas_sequence=-1*np.ones([self.n_meas,2])
        self.f_meas_sequence=-1*np.ones(self.n_meas)
        
        rf, K_x1, K_x2 =sf.Simulate_random_field_fast(self.x1, self.x2, self.cov_fun, self.cov_fun)
        self.fun=rf
        


    """
        3. Auxiliary methods
    """

    
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
        
        augmented_x_vec=np.vstack((self.x_measured, self.x_meas_sequence))
        augmented_f_vec=np.hstack((self.f_measured.squeeze(), self.f_meas_sequence))
        self.state=np.hstack((augmented_x_vec[0:self.n_meas,0], augmented_x_vec[0:self.n_meas,1], augmented_f_vec[0:self.n_meas])).astype('float32')
        
        reward=-np.abs(np.max(self.fun)-np.max(self.f_measured))
        
        
        # ii) Update epoch, check if done
        
        self.epoch=self.epoch+1
        if self.epoch==self.max_epoch:
            done=True
        else:
            done=False
        
        info = {}

        return self.state, reward, done, info

        
            

    """
        4. Reset method
    """


    def reset(self):
        
        
        # i) Reinitialize by simulating again
        
        self.epoch=0
        self.state=-1*np.ones(self.n_state).astype('float32')
        self.x_measured=np.empty([0,2])
        self.f_measured=np.empty([0,1])
        self.x_meas_sequence=-1*np.ones([self.n_meas,2])
        self.f_meas_sequence=-1*np.ones(self.n_meas)
        
        rf, K_x1, K_x2 =sf.Simulate_random_field_fast(self.x1, self.x2, self.cov_fun, self.cov_fun)
        self.fun=rf
        
        observation=self.state
        return observation
        

    """
        5. Render method and close
    """
    
    
    # i) Render plot with measured locations
    
    def render(self, reward, mode='console'):
        
        plt.figure(1,dpi=300)
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





