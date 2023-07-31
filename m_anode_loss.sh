#!/bin/bash

#SBATCH --partition=gpu          # Partition (job queue)
#SBATCH --requeue                 # Return job to the queue if preempted
#SBATCH --job-name=nflows_mixture  # Assign a short name to your job
#SBATCH --nodes=1                 # Number of nodes you require
#SBATCH --ntasks=1                # Total # of tasks across all nodes
#SBATCH --cpus-per-task=1         # Cores per task (>1 if multithread tasks)
#SBATCH --array=0-2              # Uncomment for multiple jobs
#SBATCH --gres=gpu:1              # Number of GPUs per node
#SBATCH --mem=8000                # Real memory (RAM) required (MB)
#SBATCH --time=03:00:00           # Total run time limit (HH:MM:SS)


cd /scratch/rd804/m-anode/

source ~/.bashrc
conda activate manode

try_=$1
group_name=$2
job_type=$3
loss_type=$4
param=$5

sig=5


if [[ ${loss_type} == "capped_sigmoid" ]]
then
        nohup python scripts/m_anode.py --sig_train=${sig} --sig_test=10 --w=1.0 \
                --mode_background='freeze' --epochs=100 \
                --wandb_group=${group_name} \
                --wandb_job_type=${job_type}'_sig_'${sig}'_loss_'${loss_type}'_param_'${param} \
                --cap_sig=${param} \
                --wandb_run_name='try_'${try_} \
                --data_loss_expr=${loss_type} \

elif [[ ${loss_type} == "scaled_sigmoid" ]]
then
        nohup python scripts/m_anode.py --sig_train=${sig} --sig_test=10 --w=-4 \
                --mode_background='freeze' --epochs=100 \
                --wandb_group=${group_name} \
                --wandb_job_type=${job_type}'_sig_'${sig}'_loss_'${loss_type}'_param_'${param} \
                --scale_sig=${param} \
                --wandb_run_name='try_'${try_} \
                --data_loss_expr=${loss_type} \

elif [[ ${loss_type} == "with_self_weighted_KLD" ]]
then
        nohup python scripts/m_anode.py --sig_train=${sig} --sig_test=10 --w=-4 \
                --mode_background='freeze' --epochs=100 \
                --wandb_group=${group_name} \
                --wandb_job_type=${job_type}'_sig_'${sig}'_loss_'${loss_type}'_param_'${param} \
                --kld_w=${param} \
                --wandb_run_name='try_'${try_} \
                --data_loss_expr=${loss_type} \

elif [[ ${loss_type} == "true_likelihood" ]]
then
        nohup python scripts/m_anode.py --sig_train=${sig} --sig_test=10 --w=-4 \
                --mode_background='freeze' --epochs=100 --w_train=0\
                --wandb_group=${group_name} \
                --wandb_job_type=${job_type}'_sig_'${sig}'_loss_'${loss_type}'_param_'${param} \
                --wandb_run_name='try_'${try_} \
                --data_loss_expr=${loss_type} \

else
        python scripts/m_anode_fixed_w_resample.py --sig_train=${sig} --sig_test=10 \
                --mini_batch=2048 --mode_background='true' --epochs=10 \
                --gaussian_dim=2 --shuffle_split \
                --split=${SLURM_ARRAY_TASK_ID}   \
                --resample --seed=${try_} \
                --wandb_group=${group_name} \
                --wandb_job_type=${job_type}'_'${sig} \
                --wandb_run_name='try_'${try_}'_'${SLURM_ARRAY_TASK_ID} \
                --data_loss_expr=${loss_type} \


fi



        

