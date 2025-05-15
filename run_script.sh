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

python final_finetune_train.py --config config.yaml --data 1004_final_finetune_train.json
# python final_finetune_test.py

# python pref_alignment_train.py --config config.yaml --data pref_alignment_train_all.json
# python grading_test_filtered.py
# python grading_train.py --config config.yaml --data grading_train.json
# python new_new_grading_test.py
# python new_new_grading_train.py --config config.yaml --data uus_uus_grading_train.json
