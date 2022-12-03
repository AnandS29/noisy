
# Linear 1D
# Learning from preferences
# Save stats
# Noisy rewards
# 25k timesteps
# 90k comparisons

# python3 test_pref.py --env multi --stats --noise --verbose --timesteps 30000000 --algo trpo
# python3 test_pref.py --env multi --pref --stats --noise --verbose --timesteps 30000000 --comparisons 10000 --algo trpo

python3 test_pref.py --env multi-reach --stats --noise --verbose --timesteps 30000000 --algo sac
python3 test_pref.py --env multi-push --stats --noise --verbose --timesteps 30000000 --algo sac
python3 test_pref.py --env multi-pick_place --stats --noise --verbose --timesteps 30000000 --algo sac
python3 test_pref.py --env multi-reach-push-pick_place --stats --noise --verbose --timesteps 30000000 --algo sac
# python3 test_pref.py --env multi-reach-push-pick_place --pref --stats --noise --verbose --timesteps 30000000 --comparisons 10000 --algo sac