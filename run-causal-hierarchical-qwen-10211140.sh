#!/bin/bash
#SBATCH --time=16:00:00
#SBATCH --gres=gpu:h100:1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=32
#SBATCH --ntasks 1
#SBATCH --ntasks-per-node=1
#SBATCH --signal=INT@5
srun --export=ALL bash scripts/prob/train_darai_prob-causal-hierarchical_qwen-10211140.sh
