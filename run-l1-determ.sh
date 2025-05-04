#!/bin/bash
#SBATCH --time=16:00:00
#SBATCH --gpus-per-node=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=32
#SBATCH --ntasks 1
#SBATCH --ntasks-per-node=1


srun --export=ALL bash scripts/determ/train_bf_determ_l1.sh
