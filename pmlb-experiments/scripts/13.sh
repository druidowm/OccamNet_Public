#!/bin/sh
#SBATCH --gres=gpu:volta:1
#SBATCH --cpus-per-task=20
#SBATCH-o logs/13.out
#SBATCH --job-name=13

python -u run_sc_scale.py 13 
