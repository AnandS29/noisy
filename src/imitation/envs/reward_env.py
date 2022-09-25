# Make gym environment that wraps another environment and overrides the reward
# function.

import gym
from gym import spaces
from gym.utils import seeding
import numpy as np

class RewardWrapper(gym.Wrapper):
    def __init__(self, env, reward_fn):
        super().__init__(env)
        self.reward_fn = reward_fn
        self.env = env

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        return obs, self.reward_fn(obs, action, reward, done, info), done, info

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)

    def render(self, mode='human'):
        return self.env.render(mode)

    def close(self):
        return self.env.close()

    def seed(self, seed=None):
        return self.env.seed(seed)

class ReacherRewardWrapper(gym.Wrapper):
    def __init__(self):
        super().__init__(gym.make('Reacher-v2'))
        self.env = gym.make('Reacher-v2')

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        return obs, info["reward_dist"], done, info

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)

    def render(self, mode='human'):
        return self.env.render(mode)

    def close(self):
        return self.env.close()

    def seed(self, seed=None):
        return self.env.seed(seed)

def register_reward_env(env, r_fn):
    gym.envs.register(
        id='RewardWrapper-v0',
        entry_point='__main__:RewardWrapper',
        max_episode_steps=150,
        kwargs={
            'env' : env,
            'reward_fn' : r_fn
        }
    )

def register_reacher_reward_env():
    gym.envs.register(
        id='ReacherRewardWrapper-v0',
        entry_point='__main__:ReacherRewardWrapper',
        max_episode_steps=150,
    )