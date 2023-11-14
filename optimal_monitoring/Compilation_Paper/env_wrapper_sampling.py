"""
The goal of this script is to generate the 2d_deformation environment and wrap
it in such a way, that a person can try to guess the maximum.
For this, do the following:
    1. Definitions and imports
    2. Generate and illustrate data
    3. Sequential estimation by subject
"""



"""
    1. Definitions and imports -----------------------------------------------
"""



# i) Import basics and custom environment

import numpy as np
import time
import matplotlib.pyplot as plt
import class_random_def_2D_env as def_2D
from stable_baselines3.common.monitor import Monitor


# ii) Initialize and check

# np.random.seed(0)
def_2D_env=def_2D.Env()
def_2D_env=Monitor(def_2D_env)
def_2D_env.reset()



"""
    2. Generate and illustrate data ------------------------------------------
"""


# i) Illustrate random fields

n_illu=10
for k in range(n_illu):
    def_2D_env.reset()
    time.sleep(0.1)
    plt.figure(k, dpi=300)
    plt.imshow(def_2D_env.env.fun,extent=np.array([-1, 1, -1, 1]))
    plt.title('Example realization')
    plt.colorbar()
    plt.show()


"""
    3. Sequential estimation by subject --------------------------------------
"""

# i) initialize

n_test=5
reward_logger=np.zeros([n_test])


for k in range(n_test):
    print('Please enter your next measurement in the format x,y. x,y lie in the box [-1,1].')
    done=False
    obs = def_2D_env.reset()
    while done ==False:
        
        
        # ii) Let subject decide on locations
        
        action=input('[x,y]=')
        action=np.fromstring(action,sep=',')
        obs, reward, done, info = def_2D_env.step(action)
        print('Measured at: {}, \n observed: {}'.format(def_2D_env.env.x_measured*2-1, def_2D_env.env.f_measured))

        if done:
            reward_logger[k]=reward
            def_2D_env.render()
            print('Thank you! Starting next episode! Underlying truth in Plots section')
            print('There are {} episodes remaining'.format(n_test-k-1))
            break

print('The resulting rewards are {}'.format(reward_logger))











