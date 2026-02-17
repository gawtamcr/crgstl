#!/usr/bin/env bash
#SBATCH --gres gpu:4
#SBATCH --mem 128GB
#SBATCH --cpus-per-task 12
#SBATCH --time 12:00:00
#SBATCH --constrain "eowyn|arwen|galadriel|balrog|smaug|shire|gondor" 
#SBATCH --output /Midgard/home/gawtam/crgstl/cluster_logs/%J_slurm.out
#SBATCH --error  /Midgard/home/gawtam/crgstl/cluster_logs/%J_slurm.err

echo "Starting job ${SLURM_JOB_ID} on ${SLURMD_NODENAME}"

PYTHON=/Midgard/home/gawtam/miniforge3/envs/stl/bin/python
cd /Midgard/home/gawtam/stl/src
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/Midgard/home/gawtam/miniforge3

wandb login
$PYTHON -B train_RL_with_BC_max_wandb.py

# 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