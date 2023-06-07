#!/bin/bash

#SBATCH --partition=gpu          # Partition (job queue)
#SBATCH --requeue                 # Return job to the queue if preempted
#SBATCH --job-name=nflows_mixture  # Assign a short name to your job
#SBATCH --nodes=1                 # Number of nodes you require
#SBATCH --ntasks=1                # Total # of tasks across all nodes
#SBATCH --cpus-per-task=1         # Cores per task (>1 if multithread tasks)
#SBATCH --gres=gpu:1              # Number of GPUs per node
#SBATCH --mem=2000                # Real memory (RAM) required (MB)
#SBATCH --time=08:00:00           # Total run time limit (HH:MM:SS)


cd /scratch/rd804/m-anode/

source ~/.bashrc
conda activate manode

try_=$1
group_name=$2
job_type=$3
sig=$4



nohup python scripts/m_anode.py --sig_train=${sig} --sig_test=10 \
        --mode_background='freeze' --epochs=100 \
        --wandb_group=${group_name} \
        --wandb_job_type=${job_type}'_'${sig} \
        --wandb_run_name='try_'${try_} \
        --data_loss_expr='true_likelihood'

        

