import string
import os
import matplotlib.pyplot as plt
from imitation.algorithms import preference_comparisons
from imitation.rewards.reward_nets import BasicRewardNet
from imitation.util.networks import RunningNorm
from imitation.util.util import make_vec_env
from imitation.policies.base import FeedForward32Policy, NormalizeFeaturesExtractor
import gym
from stable_baselines3 import PPO
from imitation.rewards.reward_wrapper import RewardVecEnvWrapper
from stable_baselines3 import PPO
from stable_baselines3 import SAC
from stable_baselines3.ppo import MlpPolicy
from stable_baselines3.common.evaluation import evaluate_policy
from imitation.envs.stateless import *
from imitation.envs.reward_env import *
from imitation.envs.mt_env import *
from sb3_contrib import TRPO
import numpy as np
import argparse
import datetime
from pylab import figure, cm
import pdb
import pickle
import time
from stable_baselines3.common.callbacks import BaseCallback, EveryNTimesteps
import os

reacher_envs = ["reacher", "reacher_debug", "reacher2", "reacher3", "active_reacher_1", "active_reacher_2", "active_reacher_debug"]
task_map = {
    "reach": "reach-v2",
    "push": "push-v2",
    "pickplace": "pick-place-v2",
    "door_open": "door-open-v2",
    "drawer_open": "drawer-open-v2",
    "drawer_close": "drawer-close-v2",
    "button_press": "button-press-topdown-v2",
    "peg_insert_side": "peg-insert-side-v2",
    "window_open": "window-open-v2",
    "window_close": "window-close-v2",
}

def collect_trajectories(env, policy, num_episodes, render=False):
    trajectories = []
    for _ in range(num_episodes):
        trajectory = []
        obs = env.reset()
        done = False
        while not np.any(done):
            action, _ = policy.predict(obs)
            next_obs, reward, done, info = env.step(action)
            trajectory.append((obs, action, reward, next_obs, done, info))
            obs = next_obs
            if render:
                env.render()
        trajectories.append(trajectory)
    return trajectories


class TensorboardCallback(BaseCallback):
    """
    Custom callback for plotting additional values in tensorboard.
    """

    def __init__(self, verbose=0):
        super(TensorboardCallback, self).__init__(verbose)
    def _on_step(self) -> bool:
        if self.n_calls % 200000 == 0:
            # Log scalar value (here a random variable)
            trajectories = collect_trajectories(self.model.env, self.model.policy, 100)

            x_dist = np.array([np.sum([(np.abs(res[0][0,8])) for res in traj]) for traj in trajectories])
            y_dist = np.array([np.sum([(np.abs(res[0][0,9])) for res in traj]) for traj in trajectories])
            x_dist_last = np.array([np.abs(traj[-1][0][0,8]) for traj in trajectories])
            y_dist_last = np.array([np.abs(traj[-1][0][0,9]) for traj in trajectories])
            dist = np.array([
                np.sum([np.linalg.norm(res[0][0,8:10]) for res in traj]) 
            for traj in trajectories])
            dist_last = np.array([
                np.linalg.norm(traj[-1][0][0,8:10])
            for traj in trajectories])
            sq_dist = np.array([
                np.sum([np.linalg.norm(res[0][0,8:10])**2 for res in traj]) 
            for traj in trajectories])

            self.logger.record("x_dist_last", -np.mean(x_dist_last))
            self.logger.record("y_dist_last", -np.mean(y_dist_last))
            self.logger.record("dist_last", -np.mean(dist_last))
            
            self.logger.record("x_dist", -np.mean(x_dist))
            self.logger.record("y_dist", -np.mean(y_dist))
            self.logger.record("dist", -np.mean(dist))
            self.logger.record("sq dist", -np.mean(sq_dist))
        return True

reacher_callback = TensorboardCallback()

# python3 test_pref.py --env linear1d --pref --random --stats --noise --verbose --timesteps 25000 --fragment_length 1
# python3 test_pref.py --env linear1d --pref --random --stats --verbose --fragment_length 1 --timesteps 25000 --iterations 1 --parallel 10

## Reacher w/o noise: python3 test_pref.py --env reacher --pref --random --stats --verbose

parser = argparse.ArgumentParser()
parser.add_argument('--env', type=str, default="reacher")
parser.add_argument('--timesteps', type=int, default=2*(10**7))
parser.add_argument('--epochs_reward', type=int, default=3)
parser.add_argument('--epochs_agent', type=int, default=10)
parser.add_argument('--comparisons', type=int, default=300**2)
parser.add_argument('--algo', type=str, default="ppo")
parser.add_argument('--seed', type=int, default=None)
parser.add_argument('--pref', action='store_true')
parser.add_argument('--render', action='store_true')
parser.add_argument('--eval', action='store_true')
parser.add_argument('--verbose', action='store_true')
parser.add_argument('--stats', action='store_true')
parser.add_argument('--eval_episodes', type=int, default=10000)
parser.add_argument('--iterations', type=int, default=5)
parser.add_argument('--parallel', type=int, default=1)
parser.add_argument('--initial_comparison_frac', type=float, default=0.1)
parser.add_argument('--noise', type=float, default=0.0) # Add noise to reward function
parser.add_argument('--cpu', action='store_true')

args  = parser.parse_args()

noise_name = "noise_"+str(args.noise) if args.noise != 0 else "no_noise"
training_type = "pref" if args.pref else "rl"
filename = f"{args.env}_{args.algo}_{args.timesteps}_{training_type}_{noise_name}_{args.comparisons}"

def make_learner(env, algo, seed, fragment_length, name, verbose=False):
    time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    if algo == "ppo":
        learner = PPO(
            policy=FeedForward32Policy,
            policy_kwargs=dict(
                features_extractor_class=NormalizeFeaturesExtractor,
                features_extractor_kwargs=dict(normalize_class=RunningNorm),
            ),
            env=venv,
            seed=args.seed,
            n_steps=2048 // venv.num_envs,
            batch_size=64,
            ent_coef=0.0,
            learning_rate=0.001,
            n_epochs=args.epochs_agent,
            device="cuda",
            tensorboard_log=(f"./logs/{time}/{name}"),
            verbose=verbose,
        )
    elif algo == "trpo":
        learner = TRPO("MlpPolicy", env, verbose=args.verbose, tensorboard_log=f"./logs/{time}/{name}")
    elif algo == "sac":
        learner = SAC("MlpPolicy", env, verbose=args.verbose, tensorboard_log=f"./logs/{time}/{name}")
    else:
        raise ValueError("Unknown algo: {}".format(algo))
    return learner

np.random.seed(args.seed)

start = time.time()

print("Setting up environment...")
venv = None
splt = args.env.split("-")
if splt[0] == "multi":
    task_names = splt[1:]
    task_names = [task_map[task] for task in task_names]
    register_meta_env(task_names)
    env_name = "MetaEnv-v0"
    def noise_fn(obs, acts, rews, infos):
        return rews # Create task dependent noise
    frag_length = 100
elif args.env == "active_reacher_1":
    register_active_reacher_env(0.5, np.array([0.2, 0.0]))
    env_name = "ActiveReacherEnv-v0"
    def noise_fn(obs, acts, rews, infos):
        traj_len = obs.shape[0] - 1
        x_loc = obs[: traj_len, 4].reshape((traj_len,)) # x location of end effector
        def var(x):
            if x < 0:
                return args.noise
            return 0
        noise = np.array([np.random.normal(0, var(x)) for x in x_loc])
        noisy_rews = rews + noise
        # pdb.set_trace()
        return noisy_rews
    frag_length = 50
elif args.env == "active_reacher_2":
    register_active_reacher_env(0.5, np.array([-0.2, 0.0]))
    env_name = "ActiveReacherEnv-v0"
    def noise_fn(obs, acts, rews, infos):
        traj_len = obs.shape[0] - 1
        x_loc = obs[: traj_len, 4].reshape((traj_len,)) # x location of end effector
        def var(x):
            if x < 0:
                return args.noise
            return 0
        noise = np.array([np.random.normal(0, var(x)) for x in x_loc])
        noisy_rews = rews + noise
        # pdb.set_trace()
        return noisy_rews
    frag_length = 50
elif args.env == "active_reacher_debug":
    register_active_reacher_env(1, np.array([0.2, 0.0]),debug=True)
    env_name = "ActiveReacherEnv-v0"
    def noise_fn(obs, acts, rews, infos):
        traj_len = obs.shape[0] - 1
        x_loc = obs[: traj_len, 4].reshape((traj_len,)) # x location of end effector
        def var(x):
            if x < 0:
                return args.noise
            return 0
        noise = np.array([np.random.normal(0, var(x)) for x in x_loc])
        noisy_rews = rews + noise
        # pdb.set_trace()
        return noisy_rews
    frag_length = 50
elif args.env == "reacher":
    register_reacher_reward_env()
    # env_name = "Reacher-v2"
    env_name = "ReacherRewardWrapper-v0"
    def noise_fn(obs, acts, rews, infos):
        traj_len = obs.shape[0] - 1
        x_loc = obs[: traj_len, 9].reshape((traj_len,)) # y location of target - y location of end effector
        def var(x):
            if x < 0.5:
                return args.noise
            return 0
        noise = np.array([np.random.normal(0, var(np.abs(x))) for x in x_loc])
        noisy_rews = rews + noise
        # pdb.set_trace()
        return noisy_rews
    frag_length = 50
elif args.env == "reacher_debug":
    register_reacher_reward_env()
    # env_name = "Reacher-v2"
    env_name = "ReacherRewardWrapper-v0"
    def noise_fn(obs, acts, rews, infos):
        traj_len = obs.shape[0] - 1
        x_loc = obs[: traj_len, 9].reshape((traj_len,)) # y location of target - y location of end effector
        def var(x):
            if x < 0.5:
                return args.noise
            return 0
        noise = np.array([np.random.normal(0, var(np.abs(x))) for x in x_loc])
        noisy_rews = rews + noise
        # pdb.set_trace()
        return noisy_rews
    frag_length = 10
elif args.env == "reacher2":
    register_reacher_reward_env()
    # env_name = "Reacher-v2"
    env_name = "ReacherRewardWrapper-v0"
    def noise_fn(obs, acts, rews, infos):
        traj_len = obs.shape[0] - 1
        x_loc = obs[: traj_len, 8].reshape((traj_len,)) # x location of target - x location of end effector
        y_loc = obs[: traj_len, 9].reshape((traj_len,)) # y location of target - y location of end effector
        def var(x):
            if x < 0.5:
                return args.noise
            return 0
        noise_x = np.array([np.random.normal(0, var(np.abs(x))) for x in x_loc])
        noise_y = np.array([np.random.normal(0, var(np.abs(y))) for y in y_loc])
        noisy_rews = rews + noise_x + noise_y
        # pdb.set_trace()
        return noisy_rews
    frag_length = 50
elif args.env == "reacher3":
    register_reacher_reward_env()
    # env_name = "Reacher-v2"
    env_name = "ReacherRewardWrapper-v0"
    def noise_fn(obs, acts, rews, infos):
        traj_len = obs.shape[0] - 1
        x_loc = obs[: traj_len, 8].reshape((traj_len,)) # x location of target - x location of end effector
        y_loc = obs[: traj_len, 9].reshape((traj_len,)) # y location of target - y location of end effector
        def var(x):
            return args.noise
        noise_x = np.array([np.random.normal(0, var(np.abs(x))) for x in x_loc])
        noise_y = np.array([np.random.normal(0, var(np.abs(y))) for y in y_loc])
        noisy_rews = rews + noise_x + noise_y
        # pdb.set_trace()
        return noisy_rews
    frag_length = 50
elif args.env == "pendulum":
    env_name = "Pendulum-v1"
    frag_length = 100
elif args.env == "cartpole":
    env_name = "CartPole-v0"
elif args.env == "linear1d":
    env_name = "StatelessEnv-v0"
    def r_fn(x):
        return x
    env_kwargs = {"action_dim": 1, "r_fn": r_fn}
    register_fb_env(**env_kwargs)

    def noise_fn(obs, acts, rews, infos):
        # Change to include new noisy reward structure
        val = acts[0,0]
        noise = 0
        if val >= args.noise:
            if np.random.random() < 0.5:
                noise = val
            noise = -val
        return rews + noise
    frag_length = 1
elif args.env == "multi1d":
    env_name = "StatelessEnv-v0"
    act_dim = 3
    goal = np.array([0.2, 0.5, 0.8])
    def r_fn(x):
        return -np.linalg.norm(x-goal)**2
    env_kwargs = {"action_dim": act_dim, "r_fn": r_fn}
    register_fb_env(**env_kwargs)

    def noise_fn(obs, acts, rews, infos):
        # Change to include new noisy reward structure
        noise = 0
        for val in g:
            noise += np.random.normal(0, args.noise*val)
        return rews + noise
    frag_length = 1
elif args.env == "linear2d":
    env_name = "StatelessEnv-v0"
    def r_fn(x):
        return (x[0] + x[1])

    def noise_fn(obs, acts, rews, infos):
        return rews + np.random.normal(0, args.noise*(acts[0,0]**2))
    env_kwargs = {"action_dim": 2, "r_fn": r_fn}
    register_fb_env(**env_kwargs)
    frag_length = 1

if args.noise == 0:
    noise_fn = lambda obs, acts, rews, infos: rews

# print(make_vec_env.__code__.co_varnames)
venv = make_vec_env(env_name, n_envs=args.parallel)

if args.pref and not args.eval:

    reward_net = BasicRewardNet(
        venv.observation_space, venv.action_space, normalize_input_layer=RunningNorm
    )

    fragmenter = preference_comparisons.RandomFragmenter(warning_threshold=0, seed=0)
    gatherer = preference_comparisons.NoisyGatherer(seed=args.seed, noise_fn=noise_fn)
    preference_model = preference_comparisons.PreferenceModel(reward_net)
    reward_trainer = preference_comparisons.BasicRewardTrainer(
        model=reward_net,
        loss=preference_comparisons.CrossEntropyRewardLoss(preference_model),
        epochs=args.epochs_reward,
    )

    if args.algo == "ppo":
        agent = PPO(
            policy=FeedForward32Policy,
            policy_kwargs=dict(
                features_extractor_class=NormalizeFeaturesExtractor,
                features_extractor_kwargs=dict(normalize_class=RunningNorm),
            ),
            env=venv,
            seed=args.seed,
            n_steps=2048 // venv.num_envs,
            batch_size=64,
            ent_coef=0.0,
            learning_rate=0.001,
            n_epochs=args.epochs_agent,
            device="cuda",
        )
    elif args.algo == "trpo":
        agent = TRPO("MlpPolicy", venv, verbose=args.verbose)
    elif args.algo == "sac":
        agent = SAC("MlpPolicy", venv, verbose=args.verbose)
    else:
        raise ValueError("Invalid algo")

    trajectory_generator = preference_comparisons.AgentTrainer(
        algorithm=agent,
        reward_fn=reward_net,
        venv=venv,
        exploration_frac=0.0,
        seed=args.seed,
    )

    pref_comparisons = preference_comparisons.PreferenceComparisons(
        trajectory_generator,
        reward_net,
        num_iterations=args.iterations,
        fragmenter=fragmenter,
        preference_gatherer=gatherer,
        reward_trainer=reward_trainer,
        fragment_length=frag_length,
        transition_oversampling=1,
        initial_comparison_frac=args.initial_comparison_frac,
        allow_variable_horizon=False,
        seed=args.seed,
        initial_epoch_multiplier=1,
    )

    pref_comparisons.train(
        total_timesteps=args.timesteps,  # For good performance this should be 1_000_000
        total_comparisons=args.comparisons,  # For good performance this should be 5_000
    )

    learned_reward_venv = RewardVecEnvWrapper(venv, reward_net.predict)

    # Save the learned reward function using pickle
    with open(f"./rewards/reward_net_{filename}.pkl", "wb") as f:
        pickle.dump(reward_net, f)

    learner_env = learned_reward_venv
else:
    learner_env = venv

learner = make_learner(learner_env, args.algo, args.seed, frag_length, filename, args.verbose)

model_name = f"models/{filename}"
if not args.eval:
    print("Training agent...")
    if args.env in reacher_envs:
        learner.learn(args.timesteps, callback=reacher_callback)
    else:
        learner.learn(args.timesteps)
    learner.save(model_name)
else:
    print("Loading agent...")
    learner.load(model_name)
    # Load reward function
    if args.pref:
        with open(f"./rewards/reward_net_{filename}.pkl", "rb") as f:
            reward_net = pickle.load(f)
    else:
        reward_net = None

print("Evaluating agent...")
reward, _ = evaluate_policy(learner.policy, venv, args.eval_episodes, render=args.render)
print(f"Reward averaged over {args.eval_episodes} episodes: {reward}")

if args.stats:
    print("Saving stats...")
    try:
        print("Making directory...")
        os.mkdir("plots/"+args.env)
    except:
        print("Directory exists")
    filename = args.env + '/' + filename
    print(f"Filename: {filename}")
    if args.env == "linear1d":
        vals = []
        actions = np.arange(0, 1, 0.01)
        plt.title("Reward Function")
        for i in actions:
            val = reward_net.predict(np.array([[0]]), np.array([[i]]), np.array([[0]]), np.array([[True]]))
            vals.append(val)
        plt.figure()
        plt.plot(actions, vals)
        plt.xlabel("Action")
        plt.ylabel("Reward")
        plt.savefig(f"plots/{filename}_r_fn.png")
    if args.env == "multi1d":
        f = lambda x,y,z: reward_net.predict(np.array([[0]]), np.array([[x,y,z]]), np.array([[0]]), np.array([[True]]))
        def find_optimal(fn, ub):
            argmax = None
            max_val = None
            # Loop over n dimensions
            for a in np.arange(0,ub,0.01):
                for b in np.arange(0,ub,0.01):
                    for c in np.arange(0,ub,0.01):
                        val = fn(a,b,c)
                        if max_val is None or val >= max_val:
                            argmax = [a,b,c]
                            max_val = val
            return argmax
        plt.figure()
        plt.title("Optimal Values")
        vals = list(np.arange(0.01,1,0.01))
        opt = find_optimal(f, 1)
        plt.plot(ubs, [f(x,opt[1],opt[2]) for x in vals], label="x")
        plt.plot(ubs, [f(opt[0],y,opt[2]) for y in vals], label="y")
        plt.plot(ubs, [f(opt[0],opt[1],z) for z in vals], label="z")
        plt.legend()
        plt.savefig(f"plots/{filename}_opt_val.png")
    if args.env == "linear2d":
        plt.figure()
        plt.title("Reward Function")
        xs = np.arange(0, 1, 0.01)
        ys = np.arange(0, 1, 0.01)
        f = lambda x,y: reward_net.predict(np.array([[0]]), np.array([[x,y]]), np.array([[0]]), np.array([[True]]))
        z = np.array([[f(x,y) for x in xs] for y in ys])
        # print(z.shape)
        plt.imshow(z.reshape(z.shape[:2]), extent=[0,1,0,1], cmap=cm.jet, origin='lower')
        plt.colorbar()
        plt.savefig(f"plots/{filename}_r_fn.png")

        def find_optimal(fn, ub):
            argmax = None
            max_val = None
            for x in np.arange(0,ub,0.1):
                for y in np.arange(0,ub,0.1):
                    val = fn(x,y)
                    if max_val is None or val >= max_val:
                        argmax = [x,y]
                        max_val = val
            return argmax

        plt.figure()
        plt.title("Optimal Values")
        ubs = list(np.arange(0.1,10,0.1))
        vals = [find_optimal(f, ub) for ub in ubs]
        plt.plot(ubs, [v[0] for v in vals], label="x")
        plt.plot(ubs, [v[1] for v in vals], label="y")
        plt.plot(ubs, ubs, label="True opt")
        plt.legend()
        plt.savefig(f"plots/{filename}_opt_val.png")

    if args.env in reacher_envs: 
        trajs = collect_trajectories(venv, learner.policy, args.eval_episodes)

        # Plot state distribution
        plt.figure()
        plt.title("State Distribution")

        # pdb.set_trace()
        
        obss = {t:[traj[t][0] for traj in trajs] for t in range(50)}
        acts_temp = {t:[traj[t][1] for traj in trajs] for t in range(50)}
        x_obss = [[obs[0,8] for obs in obss[t]] for t in range(50)]
        y_obss = [[obs[0,9] for obs in obss[t]] for t in range(50)]
        x_obs_avg = [np.mean(np.abs(x)) for x in x_obss]
        y_obs_avg = [np.mean(np.abs(y)) for y in y_obss]
        x_obs_std = [np.std(x) for x in x_obss]
        y_obs_std = [np.std(y) for y in y_obss]

        plt.plot(x_obs_avg, label="abs x dist")
        plt.plot(y_obs_avg, label="abs y dist")
        # Plot std
        plt.fill_between(range(50), [x_obs_avg[i] - x_obs_std[i] for i in range(50)], [x_obs_avg[i] + x_obs_std[i] for i in range(50)], alpha=0.2)
        plt.fill_between(range(50), [y_obs_avg[i] - y_obs_std[i] for i in range(50)], [y_obs_avg[i] + y_obs_std[i] for i in range(50)], alpha=0.2)
        plt.xlabel("Time step")
        plt.ylabel("Distance")
        plt.legend()
        plt.savefig(f"plots/{filename}_state_dist.png") 
        
        if args.env == "active_reacher_1" or args.env == "active_reacher_2" or args.env == "active_reacher_debug":
            acts = []
            chosen_goals = []
            for traj in trajs:
                for t in range(50):
                    acts.append(traj[t][1])
                    chosen_goals.append(traj[t][-1][0]["chosen_goal"])
            
            # Plot reward distribution
            plt.figure()
            plt.title("Reward Distribution")
            # pdb.set_trace()

            for reward_type in ["pref_goal_bonus", "dist_to_goal", "chosen_to_pref", "action_reward"]:
                reward = {t:[traj[t][-1][0][reward_type] for traj in trajs] for t in range(50)}
                reward_avg = [np.mean(reward[t]) for t in range(50)]
                reward_std = [np.std(reward[t]) for t in range(50)]
                plt.plot(reward_avg, label=reward_type)
                plt.fill_between(range(50), [reward_avg[i] - reward_std[i] for i in range(50)], [reward_avg[i] + reward_std[i] for i in range(50)], alpha=0.2)
            plt.xlabel("Time step")
            plt.ylabel("Reward")
            plt.legend()
            plt.savefig(f"plots/{filename}_reward_dist.png")

            # # Set trace
            # dist_env_to_chosen = {t:[traj[t][-1][0]["is_goal_selected"] for traj in trajs] for t in range(50)}
            # print("dist_env_to_chosen", dist_env_to_chosen) # Problem with indexing of trajectories

            act_to_goal = lambda act: [act[0,1]*np.cos(act[0,0]), act[0,1]*np.sin(act[0,0])]
            # Plot histogram of actions
            plt.figure()
            plt.title("Chosen Goal Distribution")
            goals_x = [g[0] for g in chosen_goals]
            goals_y = [g[1] for g in chosen_goals]
            plt.hist2d(goals_x, goals_y, bins=20, range=[[-1,1],[-1,1]])
            plt.colorbar()
            plt.xlabel("x")
            plt.ylabel("y")
            plt.savefig(f"plots/{filename}_chosen_goal_dist.png")

            plt.figure()
            plt.title("Chosen Goal Distribution Scatter")
            goals_x = [g[0] for g in chosen_goals]
            goals_y = [g[1] for g in chosen_goals]
            plt.scatter(goals_x, goals_y, s=0.1)
            plt.xlabel("x")
            plt.ylabel("y")
            plt.savefig(f"plots/{filename}_scatter_chosen_goal_dist.png")
        else:
            # Plot reward distribution
            plt.figure()
            plt.title("Reward Distribution")
            # pdb.set_trace()
            rewards_dist = {t:[traj[t][-1][0]["reward_dist"] for traj in trajs] for t in range(50)}
            rewards_ctrl = {t:[traj[t][-1][0]["reward_ctrl"] for traj in trajs] for t in range(50)}
            rewards_dist_avg = [np.mean(rewards_dist[t]) for t in range(50)]
            rewards_ctrl_avg = [np.mean(rewards_ctrl[t]) for t in range(50)]
            rewards_dist_std = [np.std(rewards_dist[t]) for t in range(50)]
            rewards_ctrl_std = [np.std(rewards_ctrl[t]) for t in range(50)]
            plt.plot(rewards_dist_avg, label="dist")
            plt.plot(rewards_ctrl_avg, label="ctrl")
            # Plot std
            plt.fill_between(range(50), [rewards_dist_avg[i] - rewards_dist_std[i] for i in range(50)], [rewards_dist_avg[i] + rewards_dist_std[i] for i in range(50)], alpha=0.2)
            plt.fill_between(range(50), [rewards_ctrl_avg[i] - rewards_ctrl_std[i] for i in range(50)], [rewards_ctrl_avg[i] + rewards_ctrl_std[i] for i in range(50)], alpha=0.2)
            plt.xlabel("Time step")
            plt.ylabel("Reward")
            plt.legend()
            plt.savefig(f"plots/{filename}_reward_dist.png")

        # Plot reward function
        try:
            plt.figure()
            plt.title("Reward Function")
            xs = np.arange(-1, 1, 0.01)
            ys = np.arange(-1, 1, 0.01)
            f = lambda x,y: reward_net.predict(np.array([[0,0,0,0,0,0,0,0,x,y,0]]), np.array([[0,0]]), np.array([[0,0,0,0,0,0,0,0,x,y,0]]), np.array([[True]]))
            z = np.array([[f(x,y) for x in xs] for y in ys])[:,:,0]
            plt.imshow(z, extent=[-1,1,-1,1], cmap=cm.jet, origin='lower')
            plt.colorbar()
            plt.savefig(f"plots/{filename}_r_fn.png")
        except Exception as e:
            print(e)
 

        # Plot trajectory
        # plt.figure()
        # plt.title("Trajectory")
        # i = 0
        # for traj in trajs:
        #     plt.plot([res[0][0,8] for res in traj], [res[0][0,9] for res in traj])
        #     i += 1
        #     if i > 20:
        #         break
        # # plt.xlim(-1,1)
        # # plt.ylim(-1,1)
        # plt.savefig(f"plots/traj/{filename}_traj.png")

        # for traj in trajs:
        #     plt.figure()
        #     plt.title(f"Trajectory {i}")
        #     plt.plot([res[0][0,8] for res in traj], [res[0][0,9] for res in traj])
        #     plt.savefig(f"plots/traj/{filename}_traj_{i}.png")
        #     if i > 10:
        #         break
        #     i += 1
        # x obs over time
        
dur = time.time() - start
print(f"Time taken: {dur}")
   

        