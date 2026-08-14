#!/bin/bash
#SBATCH --job-name=synthcensus
#SBATCH --output=synthcensus.txt
#SBATCH --error=synthcensus_live.txt
#SBATCH --partition=public
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
# #SBATCH --mem=32G
#SBATCH --time=20:00:00

source synthcensus_env/bin/activate
python run.py
