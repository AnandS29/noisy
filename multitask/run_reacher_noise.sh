# Reacher

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/anandsranjan/.mujoco/mujoco210/bin
# python3 test_pref.py --env reacher --stats --verbose --timesteps 20000000 --algo trpo --comparisons 10000 --eval_episodes 1000 --pref --noise 100
# python3 test_pref.py --env reacher --stats --verbose --timesteps 20000000 --algo trpo --comparisons 10000 --eval_episodes 1000 --pref --noise 90
# python3 test_pref.py --env reacher --stats --verbose --timesteps 20000000 --algo trpo --comparisons 10000 --eval_episodes 1000 --pref --noise 80
python3 test_pref.py --env reacher_debug --stats --verbose --timesteps 20000000 --algo trpo --comparisons 100000 --eval_episodes 1000 --pref --noise 30
python3 test_pref.py --env reacher --stats --verbose --timesteps 10000000 --algo trpo --comparisons 100000 --eval_episodes 1000 --pref --noise 100
# python3 test_pref.py --env reacher --stats --verbose --timesteps 10000000 --algo trpo --comparisons 100000 --eval_episodes 1000 --pref --noise 90
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