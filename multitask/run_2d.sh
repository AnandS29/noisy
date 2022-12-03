# Linear 2D
# Learning from preferences
# Save stats
# Noisy rewards
# 25k timesteps
# 90k comparisons

python3 test_pref.py --env linear2d --pref --stats --noise --verbose --timesteps 1000 --comparisons 100 --algo trpo > out.txt
python3 test_pref.py --env linear2d --pref --stats --verbose --timesteps 1000 --comparisons 100 --algo trpo

python3 test_pref.py --env linear2d --pref --stats --noise --verbose --timesteps 25000 --comparisons 100000 --algo trpo
python3 test_pref.py --env linear2d --pref --stats --verbose --timesteps 25000 --comparisons 100000 --algo trpo

python3 test_pref.py --env linear2d --pref --stats --noise --verbose --timesteps 25000 --comparisons 90000 --initial_comparison_frac 0.99 --algo trpo
python3 test_pref.py --env linear2d --pref --stats --verbose --timesteps 25000 --comparisons 90000 --initial_comparison_frac 0.99 --algo trpo

python3 test_pref.py --env linear2d --pref --stats --noise --verbose --timesteps 25000 --comparisons 10000 --algo trpo
python3 test_pref.py --env linear2d --pref --stats --verbose --timesteps 25000 --comparisons 10000 --algo trpo