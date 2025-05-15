#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --job-name=lammas_train
#SBATCH --time=5-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --output=lammas_output.log
#SBATCH --error=lammas_error.log

# Activate the virtual environment
source ~/lammas/lammastreenimine/bin/activate

# Navigate to your scripts directory
cd ~

# Run the Python training script

python finetune_train.py --config config.yaml --data FT-13k_train.json
# python finetune_test.py
