# Reacher

python3 test_pref.py --eval --env active_reacher_1 --stats --verbose --timesteps 10000000 --algo trpo --comparisons 100000 --eval_episodes 1000 --noise 0
python3 test_pref.py --eval --env active_reacher_1 --stats --verbose --timesteps 10000000 --algo trpo --comparisons 100000 --eval_episodes 1000 --pref --noise 0
python3 test_pref.py --eval --env active_reacher_1 --stats --verbose --timesteps 10000000 --algo trpo --comparisons 100000 --eval_episodes 1000 --pref --noise 100
python3 test_pref.py --eval --env active_reacher_1 --stats --verbose --timesteps 10000000 --algo trpo --comparisons 100000 --eval_episodes 1000 --pref --noise 30

python3 test_pref.py --eval --env active_reacher_2 --stats --verbose --timesteps 10000000 --algo trpo --comparisons 100000 --eval_episodes 1000 --noise 0
python3 test_pref.py --eval --env active_reacher_2 --stats --verbose --timesteps 10000000 --algo trpo --comparisons 100000 --eval_episodes 1000 --pref --noise 0
python3 test_pref.py --eval --env active_reacher_2 --stats --verbose --timesteps 10000000 --algo trpo --comparisons 100000 --eval_episodes 1000 --pref --noise 100
python3 test_pref.py --eval --env active_reacher_2 --stats --verbose --timesteps 10000000 --algo trpo --comparisons 100000 --eval_episodes 1000 --pref --noise 30