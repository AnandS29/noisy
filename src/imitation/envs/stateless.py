import gym
import numpy as np

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader

import matplotlib.pyplot as plt

class StatelessEnv(gym.Env):
    def __init__(self, action_dim, reward_fn, info_fn):
        super(StatelessEnv, self).__init__()
        self.action_space = gym.spaces.Box(low=0, high=1, shape=(action_dim,))
        self.observation_space = gym.spaces.Box(low=0, high=0, shape=(1,), dtype=np.uint8) # No obs
        self.reward_fn = reward_fn
        self.info_fn = info_fn

    def step(self, action):
        reward = self.reward_fn(action)
        reward = float(reward)
        observation = self.observation_space.sample()
        done = True
        info = {}
        return observation, reward, done, self.info_fn(observation,action,reward,done,info)

    def reset(self):
        observation = self.observation_space.sample()
        return observation
        
    def render(self):
        raise NotImplementedError
    # def close (self):
    #     ...
        
def register_fb_env(r_fn, action_dim, info_fn=None):
    if info_fn is None:
        info_fn = lambda obs,action,reward,done,info: info
    gym.envs.register(
        id='StatelessEnv-v0',
        entry_point='__main__:StatelessEnv',
        max_episode_steps=150,
        kwargs={
            'reward_fn' : r_fn,
            'action_dim' : action_dim,
            'info_fn' : info_fn
        }
    )