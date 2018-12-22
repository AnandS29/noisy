"""
Utilities functions for manipulating AIRLTrainer.
"""

from yairl.airl import AIRLTrainer
from yairl.reward_net import BasicRewardNet
import yairl.util as util


def init_trainer(env_id, use_random_expert=True, **kwargs):
    """
    Build an AIRLTrainer, ready to be trained on a vectorized environment
    and either expert rollout data or random rollout data.

    env_id (str): The string id of a gym environment.
    use_random_expert (bool):
      If True, then use a blank (random) policy to generate rollouts.
      If False, then load an expert policy. Will crash if DNE.
    **kwargs -- Pass additional arguments to the AIRLTrainer constructor.
    """
    env = util.make_vec_env(env_id, 8)
    gen_policy = util.make_blank_policy(env, init_tensorboard=False)
    if use_random_expert:
        expert_policy = gen_policy
    else:
        expert_policy = util.load_expert_policy(env)
        if expert_policy is None:
            raise ValueError(env)

    rn = BasicRewardNet(env)
    trainer = AIRLTrainer(env, gen_policy, rn, expert_policies=expert_policy,
            **kwargs)
    return trainer