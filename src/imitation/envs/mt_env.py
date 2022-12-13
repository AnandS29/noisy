# Make gym environment that wraps another environment and overrides the reward
# function.

# import gym
# import numpy as np

# from stable_baselines3 import PPO
# from stable_baselines3.common.env_util import make_vec_env

# import torch
# from torch import nn
# from torch.utils.data import TensorDataset, DataLoader

# import matplotlib.pyplot as plt

# class StatelessEnv(gym.Env):
#     def __init__(self, action_dim, reward_fn):
#         super(StatelessEnv, self).__init__()
#         self.action_space = gym.spaces.Box(low=0, high=1, shape=(action_dim,))
#         self.observation_space = gym.spaces.Box(low=0, high=0, shape=(1,), dtype=np.uint8) # No obs
#         self.reward_fn = reward_fn

#     def step(self, action):
#         reward = self.reward_fn(action)
#         reward = float(reward)
#         observation = self.observation_space.sample()
#         done = True
#         info = {}
#         return observation, reward, done, info

#     def reset(self):
#         observation = self.observation_space.sample()
#         return observation
        
#     def render(self):
#         raise NotImplementedError
#     # def close (self):
#     #     ...
        
# def register_fb_env(r_fn, action_dim):
#     gym.envs.register(
#         id='StatelessEnv-v0',
#         entry_point='__main__:StatelessEnv',
#         max_episode_steps=150,
#         kwargs={
#             'reward_fn' : r_fn,
#             'action_dim' : action_dim
#         }
#     )
    
# # Make gym environment that wraps another environment and overrides the reward
# # function.

# import gym
# from gym import spaces
# from gym.utils import seeding
# import numpy as np
# import metaworld
# import random

# class MetaEnv(gym.Env):
#     def __init__(self, task_names):
#         super(MetaEnv, self).__init__()
#         self.task_names = task_names
#         self.envs, self.tasks = self.create_envs()
#         self.is_done = [False] * len(self.envs)
#         self.action_space = self.envs[0].action_space
#         obs_space = self.envs[0].observation_space
#         low, high = obs_space.low, obs_space.high
#         low = np.concatenate([low, [0]])
#         high = np.concatenate([high, [len(self.envs)-1]])
#         self.observation_space = gym.spaces.Box(low=np.array(low), high=np.array(high))
#         self.active_task = 0

#     def create_envs(self):
#         mt10 = metaworld.MT10()
#         envs, tasks = [], []
#         for name, env_cls in mt10.train_classes.items():
#             if name in self.task_names:
#                 env = env_cls()
#                 task = random.choice([task for task in mt10.train_tasks
#                                         if task.env_name == name])
#                 env.set_task(task)
#                 envs.append(env)
#                 tasks.append(task)
#         return envs, tasks

#     def step(self, action):
#         obs, reward, done, info = self.envs[self.active_task].step(action)
#         obs = np.concatenate([obs, [self.active_task]])
#         info['task'] = self.tasks[self.active_task]
#         return obs, reward, done, info

#     def reset(self):
#         self.active_task = np.random.randint(len(self.envs))
#         obs = self.envs[self.active_task].reset()
#         obs = np.concatenate([obs, [self.active_task]])
#         return obs


# # reach-v2
# # push-v2
# # pick-place-v2
# # door-open-v2
# # drawer-open-v2
# # drawer-close-v2
# # button-press-topdown-v2
# # peg-insert-side-v2
# # window-open-v2
# # window-close-v2


# def register_meta_env(task_names):
#     gym.envs.register(
#         id='MetaEnv-v0',
#         entry_point='__main__:MetaEnv',
#         max_episode_steps=500,
#         kwargs={'task_names': task_names}
#     )

# class MTEnv(gym.Env):
#     def __init__(self):
#         super(MTEnv, self).__init__()
#         self.envs, self.tasks = self.create_envs()
#         self.is_done = [False] * len(self.envs)
#         self.action_space = gym.spaces.Box(low=-1, high=1, shape=(3,))
#         # print(self.action_space.sample())
#         self.observation_space = gym.spaces.Box(low=-1, high=1, shape=(12,))

#     def create_envs(self):
#         envs = [
#             gym.make("InvertedDoublePendulum-v2"),
#             gym.make("Hopper-v2"),
#             gym.make("Swimmer-v2")
#         ]
#         # envs = [
#         #     gym.make("InvertedDoublePendulumRewardWrapper-v0"),
#         #     gym.make("HopperRewardWrapper-v0"),
#         #     gym.make("SwimmerRewardWrapper-v0")
#         # ]
#         # for env in envs:
#         #     print(env.action_space)
#         #     print(env.observation_space)
#         tasks = ["pendulum", "hopper", "swimmer"]
#         return envs, tasks

#     def slice_action(self, action, task):
#         if task == "pendulum":
#             return action[:1]
#         elif task == "hopper":
#             return action[:3]
#         elif task == "swimmer":
#             return action[:2]
#         raise RuntimeError("Invalid task")

#     def step(self, action):
#         for i in range(len(self.envs)):
#             if self.is_done[i]:
#                 continue
#             action = self.slice_action(action, self.tasks[i])
#             obs, reward, done, info = self.envs[i].step(action)
#             # print(obs, reward, done, info)
#             if done:
#                 self.is_done[i] = True
#             all_done = np.all(self.is_done)
#             info["task"] = self.tasks[i]
#             info["task_done"] = done
#             obs = self.pad_obs(obs)
#             obs = np.concatenate([obs, [i]])
#             if done and not all_done:
#                 obs = self.obss[i+1]
#             return obs, reward, all_done, info
#         raise RuntimeError("No environment was stepped")

#     def reset(self):
#         self.obss = []
#         i = 0
#         for env in self.envs:
#             obs = self.pad_obs(env.reset())
#             obs = np.concatenate([obs, [i]])
#             self.obss.append(obs)
#             i += 1
#         self.is_done = [False] * len(self.envs)
#         return self.obss[0]

#     def pad_obs(self, obs):
#         return np.concatenate([obs, np.zeros(11 - len(obs))])

# def register_mt_env():
#     gym.envs.register(
#         id='MTEnv-v0',
#         entry_point='__main__:MTEnv',
#         max_episode_steps=3000,
#     )