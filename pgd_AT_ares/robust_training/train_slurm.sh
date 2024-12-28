#!/bin/bash -e
#SBATCH --job-name swin_large
#SBATCH --partition=gpu_h100
#SBATCH --nodes 2
#SBATCH --gpus-per-node=4
#SBATCH --mem=100G
#SBATCH --cpus-per-task=32
#SBATCH --tasks-per-node 1
#SBATCH --time=5-00:00:00
#SBATCH --output=./slurm_log/my-experiment-%j.out
#SBATCH --error=./slurm_log/my-experiment-%j.err



export MASTER_ADDR=`/bin/hostname -s`
echo "MASTER_ADDR: "${MASTER_ADDR}

export MASTER_PORT=`netstat -tan | awk '$1 == "tcp" && $4 ~ /:/ { port=$4; sub(/^[^:]+:/, "", port); used[int(port)] = 1; } END { for (p = 11234; p <= 65535; ++p) if (! (p in used)) { print p; exit(0); }; exit(1); }'`
# export MASTER_PORT=21224


srun run_train.sh
