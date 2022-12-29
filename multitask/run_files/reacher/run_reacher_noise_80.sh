#!/bin/bash
cd /data/noisy
# mkdir -p /root/.mujoco \
#     && wget https://mujoco.org/download/mujoco210-linux-x86_64.tar.gz -O mujoco.tar.gz \
#     && tar -xf mujoco.tar.gz -C /root/.mujoco \
#     && rm mujoco.tar.gz
# export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/anandsranjan/.mujoco/mujoco210/bin

pip install sb3-contrib
pip install mujoco_py
pip install -e .
cd multitask

python3 test_pref.py --env reacher --stats --verbose --timesteps 10000000 --algo trpo --comparisons 100000 --eval_episodes 1000 --pref --noise 80
# python3 test_pref.py --env reacher --stats --verbose --timesteps 10000000 --algo trpo --comparisons 100000 --eval_episodes 1000 --pref --noise 50
# python3 test_pref.py --env reacher --stats --verbose --timesteps 10000000 --algo trpo --comparisons 100000 --eval_episodes 1000 --pref --noise 30
# python3 test_pref.py --env reacher --stats --verbose --timesteps 10000000 --algo trpo --comparisons 100000 --eval_episodes 1000 --pref --noise 10

# python3 test_pref.py --env reacher2 --stats --verbose --timesteps 10000000 --algo trpo --comparisons 100000 --eval_episodes 1000 --pref --noise 70
# python3 test_pref.py --env reacher2 --stats --verbose --timesteps 10000000 --algo trpo --comparisons 100000 --eval_episodes 1000 --pref --noise 50
# python3 test_pref.py --env reacher2 --stats --verbose --timesteps 10000000 --algo trpo --comparisons 100000 --eval_episodes 1000 --pref --noise 30
# python3 test_pref.py --env reacher2 --stats --verbose --timesteps 10000000 --algo trpo --comparisons 100000 --eval_episodes 1000 --pref --noise 10

# python3 test_pref.py --env reacher3 --stats --verbose --timesteps 10000000 --algo trpo --comparisons 100000 --eval_episodes 1000 --pref --noise 70
# python3 test_pref.py --env reacher3 --stats --verbose --timesteps 10000000 --algo trpo --comparisons 100000 --eval_episodes 1000 --pref --noise 50
# python3 test_pref.py --env reacher3 --stats --verbose --timesteps 10000000 --algo trpo --comparisons 100000 --eval_episodes 1000 --pref --noise 30
# python3 test_pref.py --env reacher3 --stats --verbose --timesteps 10000000 --algo trpo --comparisons 100000 --eval_episodes 1000 --pref --noise 10

# python3 test_pref.py --env reacher --stats --verbose --timesteps 1000 --algo ppo --comparisons 10 --eval_episodes 1000 --noise --pref