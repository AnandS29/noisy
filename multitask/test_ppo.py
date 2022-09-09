from imitation.algorithms import preference_comparisons
from imitation.rewards.reward_nets import BasicRewardNet
from imitation.util.networks import RunningNorm
from imitation.util.util import make_vec_env
from imitation.policies.base import FeedForward32Policy, NormalizeFeaturesExtractor
import gym
from stable_baselines3 import PPO
from imitation.rewards.reward_wrapper import RewardVecEnvWrapper
from stable_baselines3 import PPO
from stable_baselines3.ppo import MlpPolicy
from stable_baselines3.common.evaluation import evaluate_policy

venv = make_vec_env("Reacher-v2")

reward_net = BasicRewardNet(
    venv.observation_space, venv.action_space, normalize_input_layer=RunningNorm
)

fragmenter = preference_comparisons.RandomFragmenter(warning_threshold=0, seed=0)
gatherer = preference_comparisons.SyntheticGatherer(seed=0)
reward_trainer = preference_comparisons.CrossEntropyRewardTrainer(
    model=reward_net, epochs=3
)

agent = PPO(
    policy=FeedForward32Policy,
    policy_kwargs=dict(
        features_extractor_class=NormalizeFeaturesExtractor,
        features_extractor_kwargs=dict(normalize_class=RunningNorm),
    ),
    env=venv,
    seed=0,
    n_steps=2048 // venv.num_envs,
    batch_size=64,
    ent_coef=0.0,
    learning_rate=0.001,
    n_epochs=10,
)

trajectory_generator = preference_comparisons.AgentTrainer(
    algorithm=agent,
    reward_fn=reward_net,
    exploration_frac=0.0,
    seed=0,
)

pref_comparisons = preference_comparisons.PreferenceComparisons(
    trajectory_generator,
    reward_net,
    num_iterations=10,
    fragmenter=fragmenter,
    preference_gatherer=gatherer,
    reward_trainer=reward_trainer,
    fragment_length=50,
    transition_oversampling=1,
    initial_comparison_frac=0.1,
    allow_variable_horizon=False,
    seed=0,
    initial_epoch_multiplier=1,
)

pref_comparisons.train(
    total_timesteps=2*(10**7),  # For good performance this should be 1_000_000
    total_comparisons=5000,  # For good performance this should be 5_000
)

learned_reward_venv = RewardVecEnvWrapper(venv, reward_net.predict)

learner = PPO(
    policy=MlpPolicy,
    env=learned_reward_venv,
    seed=0,
    batch_size=64,
    ent_coef=0.0,
    learning_rate=0.001,
    n_epochs=10,
    n_steps=64,
)
learner.learn(2*(10**7))  # Note: set to 100000 to train a proficient expert
learner.save("model/ppo_reacher_pref")
reward, _ = evaluate_policy(learner.policy, venv, 10, render=True)
print(reward)