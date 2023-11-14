# import gym
# import numpy as np

# from stable_baselines3 import DDPG
# from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise

# env = gym.make("Pendulum-v0")

# # The noise objects for DDPG
# n_actions = env.action_space.shape[-1]
# action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))

# model = DDPG("MlpPolicy", env, action_noise=action_noise, verbose=1)
# model.learn(total_timesteps=10000, log_interval=10)
# model.save("ddpg_pendulum")
# env = model.get_env()

# del model # remove to demonstrate saving and loading

# model = DDPG.load("ddpg_pendulum")

# obs = env.reset()
# while True:
#     action, _states = model.predict(obs)
#     obs, rewards, dones, info = env.step(action)
#     env.render()


# import gym

# from stable_baselines3 import DQN

# env = gym.make("CartPole-v0")

# model = DQN("MlpPolicy", env, verbose=1)
# model.learn(total_timesteps=25000, log_interval=4)
# model.save("dqn_cartpole")

# del model # remove to demonstrate saving and loading

# model = DQN.load("dqn_cartpole")

# obs = env.reset()
# while True:
#     action, _states = model.predict(obs, deterministic=True)
#     obs, reward, done, info = env.step(action)
#     env.render()
#     if done:
#       obs = env.reset()




# import gym

# from stable_baselines3 import A2C
# from stable_baselines3.common.env_util import make_vec_env

# # Parallel environments
# env = make_vec_env("CartPole-v1", n_envs=4)

# model = A2C("MlpPolicy", env, verbose=1)
# model.learn(total_timesteps=25000)
# model.save("a2c_cartpole")

# del model # remove to demonstrate saving and loading

# model = A2C.load("a2c_cartpole")

# obs = env.reset()
# while True:
#     action, _states = model.predict(obs)
#     obs, rewards, dones, info = env.step(action)
#     env.render()



# import gym

# from stable_baselines3 import PPO
# from stable_baselines3.common.env_util import make_vec_env

# # Parallel environments
# env = make_vec_env("CartPole-v1", n_envs=4)

# model = PPO("MlpPolicy", env, verbose=1)
# model.learn(total_timesteps=25000)
# model.save("ppo_cartpole")

# del model # remove to demonstrate saving and loading

# model = PPO.load("ppo_cartpole")

# obs = env.reset()
# while True:
#     action, _states = model.predict(obs)
#     obs, rewards, dones, info = env.step(action)
#     env.render()


# import gym
# import numpy as np

# from stable_baselines3 import SAC

# env = gym.make("Pendulum-v0")

# model = SAC("MlpPolicy", env, verbose=1)
# model.learn(total_timesteps=10000, log_interval=4)
# model.save("sac_pendulum")

# del model # remove to demonstrate saving and loading

# model = SAC.load("sac_pendulum")

# obs = env.reset()
# while True:
#     action, _states = model.predict(obs, deterministic=True)
#     obs, reward, done, info = env.step(action)
#     env.render()
#     if done:
#       obs = env.reset()


# import gym
# import numpy as np

# from stable_baselines3 import TD3
# from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise

# env = gym.make("Pendulum-v0")

# # The noise objects for TD3
# n_actions = env.action_space.shape[-1]
# action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))

# model = TD3("MlpPolicy", env, action_noise=action_noise, verbose=1)
# model.learn(total_timesteps=10000, log_interval=10)
# model.save("td3_pendulum")
# env = model.get_env()

# del model # remove to demonstrate saving and loading

# model = TD3.load("td3_pendulum")

# obs = env.reset()
# while True:
#     action, _states = model.predict(obs)
#     obs, rewards, dones, info = env.step(action)
#     env.render()




# import gym

# from stable_baselines3 import PPO
# from stable_baselines3.common.env_util import make_vec_env

# # Parallel environments
# env = make_vec_env("Pendulum-v0", n_envs=4)

# model = PPO("MlpPolicy", env, verbose=1)
# model.learn(total_timesteps=25000)
# model.save("ppo_cartpole")

# del model # remove to demonstrate saving and loading

# model = PPO.load("ppo_cartpole")

# obs = env.reset()
# while True:
#     action, _states = model.predict(obs)
#     obs, rewards, dones, info = env.step(action)
#     env.render()




import gym

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

# Parallel environments
env = make_vec_env("Acrobot-v1", n_envs=4)

model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=1000)
model.save("ppo_cartpole")

del model # remove to demonstrate saving and loading

model = PPO.load("ppo_cartpole")

obs = env.reset()
while True:
    action, _states = model.predict(obs)
    obs, rewards, dones, info = env.step(action)
    env.render()

# import gym

# from stable_baselines3 import A2C
# from stable_baselines3.common.env_util import make_vec_env

# # Parallel environments
# env = make_vec_env("Acrobot-v1", n_envs=4)

# model = A2C("MlpPolicy", env, verbose=1)
# model.learn(total_timesteps=100000)
# model.save("ppo_cartpole")

# del model # remove to demonstrate saving and loading

# model = A2C.load("ppo_cartpole")

# obs = env.reset()
# while True:
#     action, _states = model.predict(obs)
#     obs, rewards, dones, info = env.step(action)
#     env.render()

# import gym

# import stable_baselines3
# from stable_baselines3 import DQN

# policy=stable_baselines3.dqn.policies.DQNPolicy

# env = gym.make("CartPole-v0")

# model = DQN(policy, env, verbose=1)
# model.learn(total_timesteps=10000, log_interval=4)
# model.save("dqn_cartpole")

# del model # remove to demonstrate saving and loading

# model = DQN.load("dqn_cartpole")

# obs = env.reset()
# while True:
#     action, _states = model.predict(obs, deterministic=True)
#     obs, reward, done, info = env.step(action)
#     env.render()
#     if done:
#       obs = env.reset()





# import gym

# from stable_baselines3 import PPO
# from stable_baselines3.common.env_util import make_vec_env

# # Parallel environments
# env = make_vec_env("MountainCarContinuous-v0", n_envs=4)

# model = PPO("MlpPolicy", env, verbose=1, use_sde=True)
# model.learn(total_timesteps=1000)
# model.save("ppo_cartpole")

# del model # remove to demonstrate saving and loading

# model = PPO.load("ppo_cartpole")

# obs = env.reset()
# while True:
#     action, _states = model.predict(obs)
#     obs, rewards, dones, info = env.step(action)
#     env.render()



# import gym
# import numpy as np

# from stable_baselines3 import PPO, TD3
# from stable_baselines3.common.env_util import make_vec_env
# from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise



# # Parallel environments
# env = gym.make("MountainCarContinuous-v0")


# action_noise=OrnsteinUhlenbeckActionNoise(np.zeros(1), 0.1*np.ones(1))
# model = TD3("MlpPolicy", env, verbose=1, action_noise=action_noise)

# # model = PPO("MlpPolicy", env, verbose=1, use_sde=True)
# model.learn(total_timesteps=10000)


# obs = env.reset()
# while True:
#     action, _states = model.predict(obs)
#     obs, rewards, dones, info = env.step(action)
#     env.render()



# from stable_baselines3 import HerReplayBuffer, DDPG, DQN, SAC, TD3
# from stable_baselines3.her.goal_selection_strategy import GoalSelectionStrategy
# from stable_baselines3.common.envs import BitFlippingEnv
# from stable_baselines3.common.vec_env import DummyVecEnv

# model_class = DQN  # works also with SAC, DDPG and TD3
# N_BITS = 15

# env = BitFlippingEnv(n_bits=N_BITS, continuous=model_class in [DDPG, SAC, TD3], max_steps=N_BITS)

# # Available strategies (cf paper): future, final, episode
# goal_selection_strategy = 'future' # equivalent to GoalSelectionStrategy.FUTURE

# # If True the HER transitions will get sampled online
# online_sampling = True
# # Time limit for the episodes
# max_episode_length = N_BITS

# # Initialize the model
# model = model_class(
#     "MultiInputPolicy",
#     env,
#     replay_buffer_class=HerReplayBuffer,
#     # Parameters for HER
#     replay_buffer_kwargs=dict(
#         n_sampled_goal=4,
#         goal_selection_strategy=goal_selection_strategy,
#         online_sampling=online_sampling,
#         max_episode_length=max_episode_length,
#     ),
#     verbose=1,
# )

# # Train the model
# model.learn(1000)

# model.save("./her_bit_env")
# # Because it needs access to `env.compute_reward()`
# # HER must be loaded with the env
# model = model_class.load('./her_bit_env', env=env)

# obs = env.reset()
# for _ in range(100):
#     action, _ = model.predict(obs, deterministic=True)
#     obs, reward, done, _ = env.step(action)

#     if done:
#         obs = env.reset()















