#!/bin/bash

#SBATCH --partition=gpu          # Partition (job queue)
#SBATCH --requeue                 # Return job to the queue if preempted
#SBATCH --job-name=sigma_scan  # Assign a short name to your job
#SBATCH --nodes=1                 # Number of nodes you require
#SBATCH --ntasks=1                # Total # of tasks across all nodes
#SBATCH --cpus-per-task=1         # Cores per task (>1 if multithread tasks)
#SBATCH --array=0-19              # Uncomment for multiple jobs
#SBATCH --gres=gpu:1              # Number of GPUs per node
#SBATCH --mem=8000                # Real memory (RAM) required (MB)
#SBATCH --time=5:00:00           # Total run time limit (HH:MM:SS)


cd /scratch/rd804/m-anode/

source ~/.bashrc
conda activate manode


try_=$1
group_name=$2
job_type=$3
n_sig=$4



#python scripts/r_anode_lhc_co_mass_joint_untransformed.py --n_sig=${n_sig} \
python scripts/r_anode.py --n_sig=${n_sig} \
        --mini_batch=256 --mode_background='freeze' --epochs=300 \
        --shuffle_split --resample \
        --split=${SLURM_ARRAY_TASK_ID}  --validation_fraction=0.2 \
        --seed=${try_} \
        --wandb \
        --wandb_group=${group_name} \
        --wandb_job_type=${job_type}'_'${n_sig} \
        --wandb_run_name='try_'${try_}'_'${SLURM_ARRAY_TASK_ID} \
        --data_loss_expr='true_likelihood'

        

