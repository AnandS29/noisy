# Reacher
# RUN mkdir -p /root/.mujoco \
#     && wget https://mujoco.org/download/mujoco210-linux-x86_64.tar.gz -O mujoco.tar.gz \
#     && tar -xf mujoco.tar.gz -C /root/.mujoco \
#     && rm mujoco.tar.gz

# ENV LD_LIBRARY_PATH /root/.mujoco/mujoco210/bin:${LD_LIBRARY_PATH}
# ENV LD_LIBRARY_PATH /usr/local/nvidia/lib64:${LD_LIBRARY_PATH}

# mkdir -p /root/.mujoco \
#     && wget https://mujoco.org/download/mujoco210-linux-x86_64.tar.gz -O mujoco.tar.gz \
#     && tar -xf mujoco.tar.gz -C /root/.mujoco \
#     && rm mujoco.tar.gz
# export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/anandsranjan/.mujoco/mujoco210/bin
# python3 test_pref.py --env reacher --stats --verbose --timesteps 20000000 --algo trpo --comparisons 10000 --eval_episodes 1000 --pref --noise 100
# python3 test_pref.py --env reacher --stats --verbose --timesteps 20000000 --algo trpo --comparisons 10000 --eval_episodes 1000 --pref --noise 90
# python3 test_pref.py --env reacher --stats --verbose --timesteps 20000000 --algo trpo --comparisons 10000 --eval_episodes 1000 --pref --noise 80
# python3 test_pref.py --env reacher_debug --stats --verbose --timesteps 20000000 --algo trpo --comparisons 100000 --eval_episodes 1000 --pref --noise 30
python3 test_pref.py --env reacher --stats --verbose --timesteps 20000000 --algo trpo --comparisons 100000 --eval_episodes 1000 --pref --noise 100
python3 test_pref.py --env reacher --stats --verbose --timesteps 20000000 --algo trpo --comparisons 100000 --eval_episodes 1000 --pref --noise 90
python3 test_pref.py --env reacher --stats --verbose --timesteps 20000000 --algo trpo --comparisons 100000 --eval_episodes 1000 --pref --noise 30
python3 test_pref.py --env reacher --stats --verbose --timesteps 20000000 --algo trpo --comparisons 100000 --eval_episodes 1000 --pref --noise 10

python3 test_pref.py --env reacher2 --stats --verbose --timesteps 20000000 --algo trpo --comparisons 100000 --eval_episodes 1000 --pref --noise 70
python3 test_pref.py --env reacher2 --stats --verbose --timesteps 20000000 --algo trpo --comparisons 100000 --eval_episodes 1000 --pref --noise 50
python3 test_pref.py --env reacher2 --stats --verbose --timesteps 20000000 --algo trpo --comparisons 100000 --eval_episodes 1000 --pref --noise 30
python3 test_pref.py --env reacher2 --stats --verbose --timesteps 20000000 --algo trpo --comparisons 100000 --eval_episodes 1000 --pref --noise 10

python3 test_pref.py --env reacher3 --stats --verbose --timesteps 20000000 --algo trpo --comparisons 100000 --eval_episodes 1000 --pref --noise 70
python3 test_pref.py --env reacher3 --stats --verbose --timesteps 20000000 --algo trpo --comparisons 100000 --eval_episodes 1000 --pref --noise 50
python3 test_pref.py --env reacher3 --stats --verbose --timesteps 20000000 --algo trpo --comparisons 100000 --eval_episodes 1000 --pref --noise 30
python3 test_pref.py --env reacher3 --stats --verbose --timesteps 20000000 --algo trpo --comparisons 100000 --eval_episodes 1000 --pref --noise 10
# yes | ctl job run --name anand-run-reacher-80 --command "/data/noisy/multitask/run_files/reacher/run_reacher_noise_80.sh" --shared-host-dir /home/asiththaranjan --container anandsranjan/noisy:0.0.1 --cpu 16 --gpu 1 --memory 1000
# yes | ctl job run --name anand-run-reacher-90 --command "/data/noisy/multitask/run_files/reacher/run_reacher_noise_90.sh" --shared-host-dir /home/asiththaranjan --container anandsranjan/noisy:0.0.1 --cpu 16 --gpu 1 --memory 1000
# yes | ctl job run --name anand-run-reacher-100 --command "/data/noisy/multitask/run_files/reacher/run_reacher_noise_100.sh" --shared-host-dir /home/asiththaranjan --container anandsranjan/noisy:0.0.1 --cpu 16 --gpu 1 --memory 1000

# yes | ctl job run --name anand-run-reacher-100-200k --command "/data/noisy/multitask/run_files/reacher/run_reacher_noise_100_200k.sh" --shared-host-dir /home/asiththaranjan --container anandsranjan/noisy:0.0.1 --cpu 16 --gpu 1 --memory 1000

# yes | ctl job run --name anand-run-reacher2-80 --command "/data/noisy/multitask/run_files/reacher/run_reacher2_noise_80.sh" --shared-host-dir /home/asiththaranjan --container anandsranjan/noisy:0.0.1 --cpu 16 --gpu 1 --memory 1000
# yes | ctl job run --name anand-run-reacher2-90 --command "/data/noisy/multitask/run_files/reacher/run_reacher2_noise_90.sh" --shared-host-dir /home/asiththaranjan --container anandsranjan/noisy:0.0.1 --cpu 16 --gpu 1 --memory 1000
# yes | ctl job run --name anand-run-reacher2-100 --command "/data/noisy/multitask/run_files/reacher/run_reacher2_noise_100.sh" --shared-host-dir /home/asiththaranjan --container anandsranjan/noisy:0.0.1 --cpu 16 --gpu 1 --memory 1000

# yes | ctl job run --name anand-run-reacher3-80 --command "/data/noisy/multitask/run_files/reacher/run_reacher3_noise_80.sh" --shared-host-dir /home/asiththaranjan --container anandsranjan/noisy:0.0.1 --cpu 16 --gpu 1 --memory 1000
# yes | ctl job run --name anand-run-reacher3-90 --command "/data/noisy/multitask/run_files/reacher/run_reacher3_noise_90.sh" --shared-host-dir /home/asiththaranjan --container anandsranjan/noisy:0.0.1 --cpu 16 --gpu 1 --memory 1000
# yes | ctl job run --name anand-run-reacher3-100 --command "/data/noisy/multitask/run_files/reacher/run_reacher3_noise_100.sh" --shared-host-dir /home/asiththaranjan --container anandsranjan/noisy:0.0.1 --cpu 16 --gpu 1 --memory 1000

# python3 test_pref.py --env reacher --stats --verbose --timesteps 1000 --algo ppo --comparisons 10 --eval_episodes 1000 --noise --pref