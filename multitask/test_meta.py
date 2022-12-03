import metaworld
import random
from imitation.envs.mt_env import *
from sb3_contrib import TRPO
from imitation.util.util import make_vec_env
from imitation.envs.stateless import *
from imitation.envs.reward_env import *
from imitation.envs.mt_env import *

# env_name = "StatelessEnv-v0"
# def r_fn(x):
#         return x
# env_kwargs = {"action_dim": 1, "r_fn": r_fn}
# register_fb_env(**env_kwargs)

# register_multi_base_env()
# register_mt_env()
# register_meta_env()
# env_name = "MetaEnv-v0"
# env_name = "MTEnv-v0"
# register_reacher_reward_env()
# env_name = "ReacherRewardWrapper-v0"


# venv = make_vec_env(env_name, n_envs=)
mt10 = metaworld.MT10()
envs, tasks = [], []
for name, env_cls in mt10.train_classes.items():
        env = env_cls()
        task_set = [task for task in mt10.train_tasks if task.env_name == name]
        task = random.choice(task_set)
        print(task.env_name, len(task_set))
        env.set_task(task)
        envs.append(env)
        tasks.append(task)
venv = envs[0]

# for j in range(500):
#         action = venv.action_space.sample()
#         obs, reward, done, info = venv.step(action)
#         print(info)
#         break
# learner = TRPO("MlpPolicy", venv)
# learner.learn(total_timesteps=1000)


# register_mt_env()
# env_name = "MTEnv-v0"
# env = make_vec_env(env_name, n_envs=1)

# print(env.action_space)
# print(env.observation_space)

# learner = TRPO("MlpPolicy", env, verbose=True)
# learner.learn(total_timesteps=10000)
# print(env.action_space)
# print(env.observation_space)

# for i in range(100):
#     env.reset()
#     done = False
#     iter = 0
#     while not done:
#         action = env.action_space.sample()
#         obs, reward, done, info = env.step(action)
#         # print(obs, reward, done, info)
#         print(iter, end=", ")
#         iter += 1

# mt10 = metaworld.MT10() # Construct the benchmark, sampling tasks

# training_envs = []
# for name, env_cls in mt10.train_classes.items():
#   env = env_cls()
#   task = random.choice([task for task in mt10.train_tasks
#                         if task.env_name == name])
#   env.set_task(task)
#   # print(task.env_name)
#   training_envs.append(env)
#   print(env.action_space)
#   print(env.observation_space)
#   break

# learner = TRPO("MlpPolicy", env, verbose=True)
# learner.learn(total_timesteps=10000)

# for env in training_envs:
#     learner = TRPO("MlpPolicy", env, verbose=True, n_steps=500)
#     learner.learn(total_timesteps=1000)
        # env.render()
# print("Training on %d environments" % len(training_envs))
# for env in training_envs:
#   obs = env.reset()  # Reset environment
#   print(env.action_space)
#   a = env.action_space.sample()  # Sample an action
#   obs, reward, done, info = env.step(a)  # Step the environoment with the sampled random action
#   print("Reward: %f" % reward)

# print("Done")
