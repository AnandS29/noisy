# Reacher

yes | ctl job run --name anand-run-reacher-80 --command "/data/noisy/multitask/run_files/reacher/run_reacher_noise_80.sh" --shared-host-dir /home/asiththaranjan --container anandsranjan/noisy:0.0.1 --cpu 16 --gpu 1 --memory 1000
yes | ctl job run --name anand-run-reacher-90 --command "/data/noisy/multitask/run_files/reacher/run_reacher_noise_90.sh" --shared-host-dir /home/asiththaranjan --container anandsranjan/noisy:0.0.1 --cpu 16 --gpu 1 --memory 1000
yes | ctl job run --name anand-run-reacher-100 --command "/data/noisy/multitask/run_files/reacher/run_reacher_noise_100.sh" --shared-host-dir /home/asiththaranjan --container anandsranjan/noisy:0.0.1 --cpu 16 --gpu 1 --memory 1000

yes | ctl job run --name anand-run-reacher-100-200k --command "/data/noisy/multitask/run_files/reacher/run_reacher_noise_100_200k.sh" --shared-host-dir /home/asiththaranjan --container anandsranjan/noisy:0.0.1 --cpu 16 --gpu 1 --memory 1000

yes | ctl job run --name anand-run-reacher2-80 --command "/data/noisy/multitask/run_files/reacher/run_reacher2_noise_80.sh" --shared-host-dir /home/asiththaranjan --container anandsranjan/noisy:0.0.1 --cpu 16 --gpu 1 --memory 1000
yes | ctl job run --name anand-run-reacher2-90 --command "/data/noisy/multitask/run_files/reacher/run_reacher2_noise_90.sh" --shared-host-dir /home/asiththaranjan --container anandsranjan/noisy:0.0.1 --cpu 16 --gpu 1 --memory 1000
yes | ctl job run --name anand-run-reacher2-100 --command "/data/noisy/multitask/run_files/reacher/run_reacher2_noise_100.sh" --shared-host-dir /home/asiththaranjan --container anandsranjan/noisy:0.0.1 --cpu 16 --gpu 1 --memory 1000

yes | ctl job run --name anand-run-reacher3-80 --command "/data/noisy/multitask/run_files/reacher/run_reacher3_noise_80.sh" --shared-host-dir /home/asiththaranjan --container anandsranjan/noisy:0.0.1 --cpu 16 --gpu 1 --memory 1000
yes | ctl job run --name anand-run-reacher3-90 --command "/data/noisy/multitask/run_files/reacher/run_reacher3_noise_90.sh" --shared-host-dir /home/asiththaranjan --container anandsranjan/noisy:0.0.1 --cpu 16 --gpu 1 --memory 1000
yes | ctl job run --name anand-run-reacher3-100 --command "/data/noisy/multitask/run_files/reacher/run_reacher3_noise_100.sh" --shared-host-dir /home/asiththaranjan --container anandsranjan/noisy:0.0.1 --cpu 16 --gpu 1 --memory 1000

# python3 test_pref.py --env reacher --stats --verbose --timesteps 1000 --algo ppo --comparisons 10 --eval_episodes 1000 --noise --pref