# Reacher

python3 test_pref.py --env active_reacher_1 --stats --verbose --timesteps 20000000 --algo trpo --comparisons 10000 --eval_episodes 1000 --noise 0
python3 test_pref.py --env active_reacher_1 --stats --verbose --timesteps 20000000 --algo trpo --comparisons 10000 --eval_episodes 1000 --pref --noise 0
python3 test_pref.py --env active_reacher_1 --stats --verbose --timesteps 20000000 --algo trpo --comparisons 10000 --eval_episodes 1000 --pref --noise 100

python3 test_pref.py --env active_reacher_2 --stats --verbose --timesteps 20000000 --algo trpo --comparisons 10000 --eval_episodes 1000 --noise 0
python3 test_pref.py --env active_reacher_2 --stats --verbose --timesteps 20000000 --algo trpo --comparisons 10000 --eval_episodes 1000 --pref --noise 0
python3 test_pref.py --env active_reacher_2 --stats --verbose --timesteps 20000000 --algo trpo --comparisons 10000 --eval_episodes 1000 --pref --noise 100