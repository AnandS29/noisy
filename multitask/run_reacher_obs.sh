# Reacher

python3 test_pref.py --env reacher_obs --stats --verbose --timesteps 20000000 --algo trpo --comparisons 100000 --eval_episodes 1000 --pref --noise 0
python3 test_pref.py --env reacher_obs --stats --verbose --timesteps 20000000 --algo trpo --comparisons 100000 --eval_episodes 1000 --pref --noise 20
python3 test_pref.py --env reacher_obs --stats --verbose --timesteps 20000000 --algo trpo --comparisons 100000 --eval_episodes 1000 --pref --noise 100

# python3 test_pref.py --env reacher --stats --verbose --timesteps 1000 --algo ppo --comparisons 10 --eval_episodes 1000 --noise --pref