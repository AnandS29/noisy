# Reacher

python3 test_pref.py --env reacher --stats --verbose --timesteps 10000000 --algo trpo --comparisons 100000 --eval_episodes 1000 --pref --noise
python3 test_pref.py --env reacher --stats --verbose --timesteps 10000000 --algo trpo --comparisons 100000 --eval_episodes 1000 --pref
python3 test_pref.py --env reacher2 --stats --verbose --timesteps 10000000 --algo trpo --comparisons 100000 --eval_episodes 1000 --pref --noise
python3 test_pref.py --env reacher3 --stats --verbose --timesteps 10000000 --algo trpo --comparisons 100000 --eval_episodes 1000 --pref --noise
python3 test_pref.py --env reacher --stats --verbose --timesteps 10000000 --algo trpo --comparisons 100000 --eval_episodes 1000

# python3 test_pref.py --env reacher --stats --verbose --timesteps 1000 --algo ppo --comparisons 10 --eval_episodes 1000 --noise --pref