#!/usr/bin/env bash -l

all_sig=1
arr=1	
group="nflows_gaussian_mixture_1"
job_type="SR_mb_2048_tb_15_resample"

source ~/.bashrc
conda activate manode


while ((${#all_sig[@]}))
do
    all_sig=()

    for sig in 5 0.1 0.2 0.5 0.8 0.9 1 2 1.5 10
    do
        echo "sigma = ${sig}"
        arr=()
        for j in {0..9..1}
        do
            if [[ ! -f /scratch/rd804/m-anode/results/${group}/${job_type}_${sig}/try_${j}/best_val_loss_scores.npy ]]
            then
                arr+=("$j")
                all_sig+=("$j")

            fi

        done
        echo ${arr[@]}
        echo ${all_sig[@]}

        for try_ in ${arr[@]}
            do
                echo ${try_}
                sbatch -W --output=/scratch/rd804/m-anode/logs/output/${group}.${job_type}_${sig}.${try_}.out \
                --error=/scratch/rd804/m-anode/logs/error/${group}.${job_type}_${sig}.${try_}.err \
                --export=try_=${try_},group=${group},job_type=${job_type},sig=${sig} nflows_SR.sh ${try_} ${group} ${job_type} ${sig} &
                # get job id
            done
    done

    wait



done





