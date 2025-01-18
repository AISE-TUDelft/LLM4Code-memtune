#!/bin/bash
#SBATCH --job-name=3b2g24b
#SBATCH --partition=gpu-a100
#SBATCH --time=25:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=80G
#SBATCH --gpus=2

# Deployment purposes
# This script is used to deploy run .py files on the cluster

# Set conda env:
unset CONDA_SHLVL
source "$(conda info --base)/etc/profile.d/conda.sh"

conda activate memenv
python3 train.py
conda deactivate