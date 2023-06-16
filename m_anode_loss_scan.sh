#!/usr/bin/env bash -l

all_sig=1
arr=1	
group="nflows_gaussian_mixture_1"

declare -a loss=("capped_sigmoid" "scaled_sigmoid" "with_w_scaled_KLD" "with_w_weighted_KLD" "with_self_weighted_KLD")
#declare -a loss=("with_self_weighted_KLD")
declare -a cap_sig=(0.02 0.01)
declare -a scale_sig=(0.02 0.01)
declare -a kld_w=(0.1 0.01)

job_type="m_anode_loss_scan_tailbound_15"

source ~/.bashrc
conda activate manode


while ((${#all_sig[@]}))
do
    all_sig=()

#    for sig in 5 0.1 0.2 0.5 0.8 0.9 1 2 1.5 10
    for loss_ in ${loss[@]}
    do
        if [[ ${loss_} == "capped_sigmoid"  ]]
        then
            params=${cap_sig[@]}
        elif [[ ${loss_} == "scaled_sigmoid"  ]]
        then
            params=${scale_sig[@]}
        elif [[ ${loss_} == "with_self_weighted_KLD"  ]]
        then
            params=${kld_w[@]}
        else
            params=1
        fi

        for param in ${params[@]}
        do

            echo "loss = ${loss_} with param = ${param}"
            arr=()
            for j in {0..3..1}
            do
                if [[ ! -f /scratch/rd804/m-anode/results/${group}/${job_type}'_sig_1_loss_'${loss_}'_param_'${param}/try_${j}/valloss.npy ]]
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
                    sbatch -W --output=/scratch/rd804/m-anode/logs/output/${group}.${job_type}_${param}.${try_}.out \
                    --error=/scratch/rd804/m-anode/logs/error/${group}.${job_type}_${sig}.${try_}.err \
                    --export=try_=${try_},group=${group},job_type=${job_type},param=${param} m_anode_loss.sh ${try_} ${group} ${job_type} ${loss_} ${param} &
                    # get job id
                done
        done
    done

    wait



done





