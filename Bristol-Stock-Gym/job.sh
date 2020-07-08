#!/bin/bash -login
#SBATCH --nodes=1
#SBATCH --gres=gpu:2
#SBATCH --partition gpu
#SBATCH --job-name=gpujob
#SBATCH --time=0:24:00
#SBATCH --account comsm0018
#SBATCH --mem=10GB
python BristolStockGym.py > log.txt
