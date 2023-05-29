#!/bin/bash

#SBATCH --partition=gpu          # Partition (job queue)
#SBATCH --requeue                 # Return job to the queue if preempted
#SBATCH --job-name=nflows_mixture  # Assign a short name to your job
#SBATCH --nodes=1                 # Number of nodes you require
#SBATCH --ntasks=1                # Total # of tasks across all nodes
#SBATCH --cpus-per-task=1         # Cores per task (>1 if multithread tasks)
#SBATCH --gres=gpu:1              # Number of GPUs per node
#SBATCH --mem=2000                # Real memory (RAM) required (MB)
#SBATCH --array=0-9               # Uncomment if you want to run multiple jobs
#SBATCH --time=02:00:00           # Total run time limit (HH:MM:SS)
#SBATCH --output=/scratch/rd804/m-anode/logs/output/slurm.%N.%a.out  # STDOUT output file
#SBATCH --error=/scratch/rd804/m-anode/logs/error/slurm.%N.%a.err   # STDERR output file (optional)

cd /scratch/rd804/m-anode/
#conda activate manode

python scripts/nflows_mixture.py --try $SLURM_ARRAY_TASK_ID


