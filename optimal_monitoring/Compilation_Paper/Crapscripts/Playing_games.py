"""
The goal of this script is to run a reinforcement learning agent on the Lunarlander
problem. This is meant to illustrate the progression of learning via an attractive video.
For this, do the following:
    1. Definitions and imports
    2. Set up the environment
    3. Set up the agent and train
    4. Run the agent
    5. Illustrate learning progression
    
The script is meant solely for educational and illustrative purposes. Written by
Jemil Avers Butt, Atlas optimization GmbH, www.atlasoptimization.ch.

"""


"""
    1. Definitions and imports
"""


# i) Imports basic packages

import numpy as np
import matplotlib.pyplot as plt
import time 


# ii) Import stable baselines and rl related

import gym
from stable_baselines3 import TD3, PPO
from stable_baselines3.common.env_checker import check_env



"""
    2. Set up the environment
"""


# i) Define environment

env=gym.make("LunarLander-v2")
observation=env.reset()

check_env(env)



"""
    3. Set up the agent and train
"""

# i) Train a PPO Model

start_time=time.time()
model = PPO("MlpPolicy", env,verbose=1, seed=0)
model.learn(total_timesteps=100000)
end_time=time.time()



"""
    4. Run the agent
"""


"""
    5. Illustrate learning progression
"""


# i) Run random policy

for _ in range(1000):
    env.render()
    action, _ = model.predict(observation)
    observation, reward, done, info=env.step(action)
    
    if done:
        observation=env.reset()
        
env.close()