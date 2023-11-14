"""
The goal of this script is to provide the environment simulating a random 
deformation of a beam fixed at the endpoints. This is done by drawing 
realizations of a Gaussian process with Brownian-Bridge type covariance. 
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
    import class_random_def_1D_env
    random_def_1D_env=class_random_def_1D_env.Env()
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
        
        self.n_meas=4
        self.max_epoch=self.n_meas
        self.n_state=self.n_meas*2
        
        self.n_disc_x=100
        self.x=np.linspace(0,1,self.n_disc_x)
        
        self.K=np.zeros([self.n_disc_x,self.n_disc_x])
        for k in range(self.n_disc_x):
            for l in range(self.n_disc_x):
                self.K[k,l]=self.cov_fun(self.x[k],self.x[l])
        
        
        # iii) Observation and action spaces
        
        self.action_space = spaces.Box(low=-1, high=1,
                                        shape=(1,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-1, high=1,
                                        shape=(self.n_state,), dtype=np.float32)
        
        
        # iv) Initial properties
        
        self.epoch=0
        self.def_fun=np.random.multivariate_normal(np.zeros([self.n_disc_x]), self.K)
        
        self.state=-np.ones([self.n_state]).astype('float32')
        self.x_measured=self.state[0:self.n_meas]
        self.f_measured=self.state[self.n_meas:self.n_state]


    """
        3. Auxiliary methods
    """

    
    # i) Round continuous location to index
    
    def round_to_index(self, location):
        location_index=np.floor(location*self.n_disc_x).astype(int).item()
        if location_index>=self.n_disc_x:
            location_index=self.n_disc_x-1
        elif location_index<=-1:
            location_index=0
                
        return location_index
    
    # ii) Covariance function
        # Brownian Bridge-type: Variance is 0 at endpoints, sinusoidals whose 
        # probabilities decrease with increasing frequency
        
    def cov_fun(self,s,t):
        cov_vec=np.zeros([10])
        for k in range(10):
            cov_vec[k]=(1/(k+1)**4)*(np.sin(1*np.pi*k*s)*np.sin(1*np.pi*k*t))
        
        cov_val=np.sum(cov_vec)
        return cov_val
    


    """
        3. Step method
    """

    def step(self, action):
        
    
        # i) Perform action and update state
        
        action_index=self.round_to_index(action*0.5+0.5)
        new_state=copy.copy(self.state)
        new_state[self.epoch]=self.x[action_index]
        new_state[self.epoch+self.n_meas]=self.def_fun[action_index]
        self.state=new_state.astype('float32')
        
        self.x_measured=self.state[0:self.n_meas]
        self.f_measured=self.state[self.n_meas:self.n_state]
        
        reward=-np.abs(np.max(self.def_fun)-np.max(self.f_measured))
        
        
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
        self.def_fun=np.random.multivariate_normal(np.zeros([self.n_disc_x]), self.K)
        
        self.state=-np.ones([self.n_state]).astype('float32')
        self.x_measured=self.state[0:self.n_meas]
        self.f_measured=self.state[self.n_meas:self.n_state]
        
        observation=self.state
        return observation
        

    """
        5. Render method and close
    """
    
    
    # i) Render plot with measured locations
    
    def render(self, reward, mode='console'):
        
        plt.figure(1,dpi=300)
        plt.plot(self.x,self.def_fun,linestyle='solid',color='0')
        plt.scatter(self.x_measured,self.f_measured)
        plt.title('Deformation functions and sample locations')
        plt.xlabel('x axis')
        plt.ylabel('y axis)')
        print('Reward is ', reward) 
        print('Measured locations are', self.x_measured)
        print(' Measurements are', self.f_measured)
        
        
    # ii) Close method
    
    def close (self):
      pass






























