#!/bin/bash
#SBATCH --job-name=3B_Inf
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --constraint=l40s
#SBATCH --cpus-per-task=2
#SBATCH --mem=30GB
#SBATCH -t 0-2:00:00
#SBATCH --output=slurm_logs/<jobname>_%A-%a.out 

bash script/run3B.sh