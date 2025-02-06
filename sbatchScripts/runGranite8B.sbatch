#!/bin/bash
#SBATCH --job-name=Granite8B
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --constraint=a100
#SBATCH --cpus-per-task=2
#SBATCH --mem=30GB
#SBATCH -t 0-10:00:00
#SBATCH --output=slurm_logs/<jobname>_%A-%a.out 

bash sbatchScripts/runGranite8B.sh