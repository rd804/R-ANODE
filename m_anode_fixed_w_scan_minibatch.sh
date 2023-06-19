#!/usr/bin/env bash -l

all_sig=1
arr=1	
group="nflows_gaussian_mixture_1"
job_type="m_wb_freeze_mini_scan"

source ~/.bashrc
conda activate manode


while ((${#all_sig[@]}))
do
    all_sig=()

#    for sig in 5 0.1 0.2 0.5 0.8 0.9 1 2 1.5 10
    for mini_batch in 1024 2048
    do
        echo "minibatch = ${mini_batch}"
        arr=()
        for j in {0..2..1}
        do
            if [[ ! -f /scratch/rd804/m-anode/results/${group}/${job_type}'_sig_1_mini_batch_'${mini_batch}/try_${j}/valloss.npy ]]
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
                sbatch -W --output=/scratch/rd804/m-anode/logs/output/${group}.${job_type}_${mini_batch}.${try_}.out \
                --error=/scratch/rd804/m-anode/logs/error/${group}.${job_type}_${mini_batch}.${try_}.err \
                --export=try_=${try_},group=${group},job_type=${job_type},mini_batch=${mini_batch} m_anode_fixed_w_minibatch.sh ${try_} ${group} ${job_type} ${mini_batch} &
                # get job id
            done
    done

    wait



done





