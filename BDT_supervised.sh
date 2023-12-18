#!/bin/bash

#SBATCH --partition=main          # Partition (job queue)
#SBATCH --requeue                 # Return job to the queue if preempted
#SBATCH --job-name=BDT  # Assign a short name to your job
#SBATCH --nodes=1                 # Number of nodes you require
#SBATCH --ntasks=1                # Total # of tasks across all nodes
#SBATCH --cpus-per-task=1         # Cores per task (>1 if multithread tasks)
#SBATCH --array=0-9              # Uncomment for multiple jobs
#SBATCH --mem=2000                # Real memory (RAM) required (MB)
#SBATCH --time=4:00:00           # Total run time limit (HH:MM:SS)

source ~/.bashrc
conda activate manode

cd /scratch/rd804/m-anode/

#nsig=$1

python ./scripts/BDT_supervised.py --ensemble_size=20 \
        --resample \
        --seed=${SLURM_ARRAY_TASK_ID} \
        --n_sig=30000 \
        --wandb \
        --wandb_group='BDT' \
        --wandb_job_type='supervised' \
        --wandb_run_name='sample'_${SLURM_ARRAY_TASK_ID} \

