#!/usr/bin/env bash -l

all_sig=1
arr=1	
group="nflows_gaussian_mixture_2"
#group='test'
job_type="SR_shuffle_split_3"


source ~/.bashrc
conda activate manode


while ((${#all_sig[@]}))
do
    all_sig=()
    
   for sig in 0.1 0.3 0.4 0.5 0.6 0.7 0.8 1 1.5 2 5 10
   # for sig in 0.4 0.5 0.6 0.7
   # for sig in 1
    do
        arr=()
        for j in {0..5..1}
        do
            for split in {0..19..1}
            do
                if [[ ! -f /scratch/rd804/m-anode/results/${group}/${job_type}_${sig}/try_${j}_${split}/best_val_loss_scores.npy ]]
                then
                    arr+=("$j")
                    all_sig+=("$j")

                fi
            done

        done
        echo ${arr[@]}
        echo ${all_sig[@]}
        
        arr=( `for i in ${arr[@]}; do echo $i; done | sort -u` )
        echo ${arr[@]}

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

