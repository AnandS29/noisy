import string
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
from imitation.envs.stateless import *
from sb3_contrib import TRPO
import numpy as np
import argparse
import datetime
from pylab import figure, cm

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
parser.add_argument('--noise', action='store_true') # Add noise to reward function

args  = parser.parse_args()

noise_name = "noise" if args.noise else "no_noise"
training_type = "pref" if args.pref else "rl"
filename = f"{args.env}_{args.algo}_{training_type}_{noise_name}_{args.comparisons}"

def make_learner(env, algo, seed, fragment_length, verbose=False):
    if algo == "ppo":
        learner = PPO(
            policy=MlpPolicy,
            env=env,
            seed=seed,
            batch_size=64,
            ent_coef=0.0,
            learning_rate=0.001,
            n_epochs=10,
            n_steps=64,
            tensorboard_log="./logs/{}".format(datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")),
            verbose=verbose
        )
    elif algo == "trpo":
        learner = TRPO("MlpPolicy", env, verbose=verbose, tensorboard_log="./logs/{}".format(datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")))
    else:
        raise ValueError("Unknown algo: {}".format(algo))
    return learner

np.random.seed(args.seed)

print("Setting up environment...")
venv = None
if args.env == "reacher":
    env_name = "Reacher-v2"
    def noise_fn(obs, acts):
        print(obs.shape, acts.shape)
        raise NotImplementedError()
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

    def noise_fn(obs, acts):
        val = acts[0,0]
        if val >= 0.8:
            if np.random.random() < 0.5:
                return val
            return -val
        return 0
    frag_length = 1
elif args.env == "linear2d":
    env_name = "StatelessEnv-v0"
    def r_fn(x):
        return (x[0] + x[1])

    def noise_fn(obs, acts):
        return np.random.normal(0, 10*(acts[0,0]**2))
    env_kwargs = {"action_dim": 2, "r_fn": r_fn}
    register_fb_env(**env_kwargs)
    frag_length = 1

if args.noise is False:
    noise_fn = lambda a,b : 0

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
        n_epochs=args.epochs_agent
    )

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
    try:
        reward_net.save(f"./rewards/reward_net_{filename}.h5")
    except Exception as e:
        print(e)
        print("Could not save reward net")


    learner_env = learned_reward_venv
else:
    learner_env = venv

learner = make_learner(learner_env, args.algo, args.seed, frag_length, args.verbose)

model_name = f"models/{filename}"
if not args.eval:
    print("Training agent...")
    learner.learn(args.timesteps)
    learner.save(model_name)
else:
    print("Loading agent...")
    learner.load(model_name)

print("Evaluating agent...")
reward, _ = evaluate_policy(learner.policy, venv, args.eval_episodes, render=args.render)
print(f"Reward averaged over {args.eval_episodes} episodes: {reward}")

if args.stats:
    print("Saving stats...")
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
    if args.env == "linear2d":
        plt.figure()
        plt.title("Reward Function")
        xs = np.arange(0, 1, 0.01)
        ys = np.arange(0, 1, 0.01)
        f = lambda x,y: reward_net.predict(np.array([[0]]), np.array([[x,y]]), np.array([[0]]), np.array([[True]]))
        z = np.array([[f(x,y) for x in xs] for y in ys])
        plt.imshow(z, extent=[0,1,0,1], cmap=cm.jet, origin='lower')
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

        