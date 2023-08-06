#!/usr/bin/env bash -l

all_sig=1
arr=1	
group="nflows_lhc_co"
#group='test'
job_type="anode_CR_verify"


source ~/.bashrc
conda activate manode


while ((${#all_sig[@]}))
do
    all_sig=()
    
   for n_sig in 1000
   # for sig in 0.4 0.5 0.6 0.7
   # for sig in 1
    do
        arr=()
        for j in 1
        do
            for split in {0..1..1}
            do
                if [[ ! -f /scratch/rd804/m-anode/results/${group}/${job_type}_${n_sig}/try_${j}_${split}/best_val_loss_scores.npy ]]
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
                sbatch -W --output=/scratch/rd804/m-anode/logs/output/${group}.${job_type}_${n_sig}.${try_}.out \
                --error=/scratch/rd804/m-anode/logs/error/${group}.${job_type}_${n_sig}.${try_}.err \
                --export=try_=${try_},group=${group},job_type=${job_type},sig=${n_sig} nflows_CR.sh ${try_} ${group} ${job_type} ${n_sig} &
                # get job id
            done
    done

    wait



done

