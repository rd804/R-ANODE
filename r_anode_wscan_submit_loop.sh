#!/usr/bin/env bash -l

all_sig=1
arr=1	
group="nflows_lhco_m_data"
job_type="joint_wscan"

source ~/.bashrc
conda activate ranode


while ((${#all_sig[@]}))
do
    all_sig=()

   # for sig in 1
    for w_ in 0.0001 0.1 0.001 0.01
   # for sig in 0.4 0.5 0.6 0.7
    #for sig in 5
    #for sig in 0.1 0.2 0.8 0.9 5
    do
        echo "w = ${w_}"
        arr=()
        for j in {0..9..1}
        do
            for split in {0..19..1}
            do
            if [[ ! -f /scratch/rd804/m-anode/results/${group}/${job_type}_${w_}/try_${j}_${split}/valloss.npy ]]
            then
                arr+=("$j")
                all_sig+=("$j")

            fi
            done

        done
        echo ${all_sig[@]}

        arr=( `for i in ${arr[@]}; do echo $i; done | sort -u` )
        echo ${arr[@]}


        for try_ in ${arr[@]}
            do
                echo ${try_}
                sbatch -W --output=/scratch/rd804/m-anode/logs/output/${group}.${job_type}_${w_}.${try_}.out \
                --error=/scratch/rd804/m-anode/logs/error/${group}.${job_type}_${w_}.${try_}.err \
                --export=try_=${try_},group=${group},job_type=${job_type},sig=${w_} r_anode_wscan.sh ${try_} ${group} ${job_type} ${w_} &
                # get job id
            done
    done

    wait

done
