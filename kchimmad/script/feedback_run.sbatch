#!/bin/bash
#SBATCH --job-name=LLaMA1B_Feedback
#SBATCH --partition=gpu
#SBATCH --gres=gpu:2
#SBATCH --constraint=a100
#SBATCH --cpus-per-task=2
#SBATCH --mem=30GB
#SBATCH -t 2-00:00:00
#SBATCH --output=slurm_logs/<jobname>_%A-%a.out 

bash script/feedback_run.sh