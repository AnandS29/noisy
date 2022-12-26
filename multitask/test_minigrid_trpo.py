from imitation.algorithms import preference_comparisons
from imitation.rewards.reward_nets import BasicRewardNet
from imitation.util.networks import RunningNorm
from imitation.util.util import make_vec_env
from imitation.policies.base import FeedForward32Policy, NormalizeFeaturesExtractor
import gym
from stable_baselines3 import PPO
from sb3_contrib import TRPO
from imitation.rewards.reward_wrapper import RewardVecEnvWrapper
from stable_baselines3 import PPO
from stable_baselines3.ppo import MlpPolicy
from stable_baselines3.common.evaluation import evaluate_policy
import minigrid

env = gym.make('MiniGrid-Empty-5x5-v0')
env.reset()

model = PPO(
    policy=MlpPolicy,
    env=env,
    seed=0,
    batch_size=64,
    ent_coef=0.0,
    learning_rate=0.001,
    n_epochs=10,
    n_steps=64,
)
# learner.load("model/ppo_reacher")

# model = TRPO("MlpPolicy", env, verbose=True)
model.learn(total_timesteps=2*(10**7))
# model.save("model/trpo_pendulum")

# obs = env.reset()
# for i in range(1000):
#     action, _states = model.predict(obs)
#     obs, rewards, dones, info = env.step(action)
#     env.render()

reward, _ = evaluate_policy(learner.policy, venv, 1000, render=False)
print(reward)