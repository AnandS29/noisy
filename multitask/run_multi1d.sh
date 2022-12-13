#!/bin/bash
# Linear 1D
# Learning from preferences
# Save stats
# Noisy rewards
# 25k timesteps
# 90k comparisons

python3 test_pref.py --env multi1d --pref --stats --verbose --timesteps 50000 --comparisons 100000 --algo trpo --noise 5
python3 test_pref.py --env multi1d --pref --stats --verbose --timesteps 50000 --comparisons 100000 --algo trpo --noise 0
python3 test_pref.py --env multi1d --pref --stats --verbose --timesteps 50000 --comparisons 100000 --algo trpo --noise 20

# python3 test_pref.py --env linear1d --pref --stats --noise --verbose --timesteps 25000 --comparisons 90000 --algo trpo

# python3 test_pref.py --env linear1d --pref --stats --noise --verbose --timesteps 25000 --comparisons 100000 --algo trpo
# python3 test_pref.py --env linear1d --pref --stats --verbose --timesteps 25000 --comparisons 100000 --algo trpo
# python3 test_pref.py --env linear1d --pref --stats --verbose --timesteps 25000 --comparisons 90000 --initial_comparison_frac 0.99 --noise --algo trpo
# # python3 test_pref.py --env linear1d --pref --stats --verbose --timesteps 25000 --comparisons 90000 --initial_comparison_frac 0.99 --algo trpo
# python3 test_pref.py --env linear1d --pref --stats --verbose --timesteps 25000 --comparisons 10000 --algo trpo
# python3 test_pref.py --env linear1d --pref --stats --verbose --timesteps 25000 --comparisons 10000 --noise --algo trpo

