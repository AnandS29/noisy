#!/bin/bash
# Linear 1D
# Learning from preferences
# Save stats
# Noisy rewards
# 25k timesteps
# 90k comparisons

cd /data/noisy 
pip install sb3-contrib
pip install -e .
cd multitask
python3 test_pref.py --env linear1d --pref --stats --noise --verbose --timesteps 25000 --comparisons 90000 --algo trpo