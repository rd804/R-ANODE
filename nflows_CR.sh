#!/bin/bash

#SBATCH --partition=gpu          # Partition (job queue)
#SBATCH --requeue                 # Return job to the queue if preempted
#SBATCH --job-name=nflows_mixture_SR  # Assign a short name to your job
#SBATCH --nodes=1                 # Number of nodes you require
#SBATCH --ntasks=1                # Total # of tasks across all nodes
#SBATCH --array=0-1              # Array rank
#SBATCH --cpus-per-task=1         # Cores per task (>1 if multithread tasks)
#SBATCH --gres=gpu:1              # Number of GPUs per node
#SBATCH --mem=8000                # Real memory (RAM) required (MB)
#SBATCH --time=05:00:00           # Total run time limit (HH:MM:SS)


cd /scratch/rd804/m-anode/


source ~/.bashrc
conda activate manode

#try_=$1
#group_name=$2
#job_type=$3
#n_sig=$4
try_='1'
group_name='nflows_lhc_co'
job_type='CR_bn_fixed'
n_sig=1000

python scripts/nflows_CR.py --try ${try_} \
    --epochs=100 --batch_size=256 --shuffle_split \
    --split=${SLURM_ARRAY_TASK_ID} --seed ${try_} \
    --wandb \
    --wandb_group ${group_name} --wandb_job_type ${job_type}_${n_sig} \
    --wandb_run_name try_${try_}_${SLURM_ARRAY_TASK_ID} \
    --split=${SLURM_ARRAY_TASK_ID} \
    --n_sig ${n_sig}


