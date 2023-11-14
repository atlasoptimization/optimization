"""
The goal of this script is to provide the environment simulating a spatial 
disturbance travelling along a line. Only scalar deformations are measured at
point in time, from these observations the next measurement is to be determined.
It is supposed to be close to the wave's crest.  The environment consists
of states indicating past measurements and actions indicating designated next
measurements. Rewards are differences between measured locations and the actual
location of the wave's crest.
For this, do the following:
    1. Definitions and imports
    2. Class intitalization
    3. Auxiliary methods
    3. Step method
    4. Reset method
    5. Render method
The class can be called upon in any script by evoking it e.g. via 
    import class_wave_tracking_env
    wave_tracking_env=class_wave_tracking_env.Env()
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
        
        x_max=50
        t_max=150
        self.x_max=np.array([[x_max],[t_max]])
        self.x_max=x_max
        self.t_max=t_max
        self.max_epoch=t_max
            
        self.n_meas=10
        self.n_state=self.n_meas*2
        
        self.x=np.linspace(0,1,self.x_max)
        self.t=np.linspace(0,1,self.t_max)
            
        
        # iii) Observation and action spaces
        
        self.action_space = spaces.Box(low=-1, high=1,
                                        shape=(1,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-1, high=1,
                                        shape=(self.n_state,), dtype=np.float32)
        
        
        # iv) Initial properties
        
        self.epoch=0
        self.t_wavegen=-np.random.uniform(0,1)
        self.v_rand=np.random.uniform(0.2,0.5)
        
        def_fun=np.zeros([self.t_max,self.x_max])
        crest_vec=np.zeros([self.t_max])
        
        for k in range(self.t_max):  
            x_crest=self.v_rand*(self.t[k]-self.t_wavegen)
            crest_vec[k]=x_crest
            
            for l in range(self.x_max):     
                def_fun[k,l]=np.sinc(4*(self.x[l]-x_crest))*np.exp(-(np.abs(self.x[l]-x_crest)/0.8)**2)
        
        self.def_fun=def_fun
        self.crest_vec=crest_vec
        
        self.state=-np.ones([self.n_state])
        
        self.x_measured=self.state[0:self.n_meas]
        self.x_meas_sequence=np.empty([0,2])
        self.f_measured=self.state[self.n_meas:self.n_state]
        self.f_meas_sequence=np.empty([0,1]) 



    """
        3. Auxiliary methods
    """

    
    # i) Round continuous location to index
    
    def round_to_index(self, location):
        location_index=np.floor(location*self.x_max).astype(int).item()
        if location_index>=self.x_max:
            location_index=self.x_max-1
        elif location_index<=-1:
            location_index=0
                
        return location_index
    


    """
        3. Step method
    """

    def step(self, action):
        
    
        # i) Perform action and update state
        
        action_index=self.round_to_index(action*0.5+0.5)
        self.x_measured=np.hstack((self.x_measured,self.x[action_index]))
        self.f_measured=np.hstack((self.f_measured,self.def_fun[self.epoch,action_index]))
        
        new_state=np.zeros(self.n_state)
        new_state[0:self.n_meas]=self.x_measured[self.epoch+1:(self.epoch+1+self.n_meas)]
        new_state[self.n_meas:self.n_state]=self.f_measured[self.epoch+1:(self.epoch+1+self.n_meas)]
        self.state=new_state
        
        self.f_meas_sequence=np.vstack((self.def_fun[self.epoch,action_index],self.f_meas_sequence))
        self.x_meas_sequence=np.vstack((np.array([self.x[action_index], self.t[self.epoch]]).squeeze(), self.x_meas_sequence))
        
        reward=-np.abs(self.crest_vec[self.epoch]-self.x[action_index])
        
        
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
               
        
        # i) Reinitialize with random Force and location 
        
        self.epoch=0
        self.t_wavegen=-np.random.uniform(0,1)
        self.v_rand=np.random.uniform(0.2,0.5)
        
        def_fun=np.zeros([self.t_max,self.x_max])
        crest_vec=np.zeros([self.t_max])
        
        for k in range(self.t_max):  
            x_crest=self.v_rand*(self.t[k]-self.t_wavegen)
            crest_vec[k]=x_crest
            
            for l in range(self.x_max):     
                def_fun[k,l]=np.sinc(4*(self.x[l]-x_crest))*np.exp(-(np.abs(self.x[l]-x_crest)/0.8)**2)
        
        self.def_fun=def_fun
        self.crest_vec=crest_vec
        
        self.state=-np.ones([self.n_state])
        
        self.x_measured=self.state[0:self.n_meas]
        self.x_meas_sequence=np.empty([0,2])
        self.f_measured=self.state[self.n_meas:self.n_state]
        self.f_meas_sequence=np.empty([0,1])  
        
        observation=self.state
        return observation
        
    

    """
        5. Render method and close
    """
    
    
    # i) Render plot with measured locations
    
    def render(self, reward, mode='console'):
        
        plt.figure(1,dpi=300)
        plt.imshow(self.def_fun, extent=[0,1,1,0])
        plt.colorbar()
        plt.scatter(self.x_meas_sequence[:,0],self.x_meas_sequence[:,1])
        plt.title('Disturbance propagation and sample locations')
        plt.xlabel('x axis')
        plt.ylabel('t axis)')
        print('Reward is ', reward) 
        print('Measured locations are', self.x_meas_sequence)
        print(' Measurements are', self.f_meas_sequence)
        
        
    # ii) Close method
    
    def close (self):
      pass
































