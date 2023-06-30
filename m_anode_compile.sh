#!/bin/bash

#SBATCH --partition=gpu          # Partition (job queue)
#SBATCH --requeue                 # Return job to the queue if preempted
#SBATCH --job-name=nflows_mixture  # Assign a short name to your job
#SBATCH --nodes=1                 # Number of nodes you require
#SBATCH --ntasks=1                # Total # of tasks across all nodes
#SBATCH --cpus-per-task=1         # Cores per task (>1 if multithread tasks)
#SBATCH --gres=gpu:1              # Number of GPUs per node
#SBATCH --mem=3000                # Real memory (RAM) required (MB)
#SBATCH --time=00:20:00           # Total run time limit (HH:MM:SS)


cd /scratch/rd804/m-anode/

source ~/.bashrc
conda activate manode

python scripts/m_anode_compile.py --scan_set='m_bw_fixed_resample_mb_2048_3'


        

