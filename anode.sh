#!/usr/bin/env bash -l

arr=1	
group="nflows_gaussian_mixture_1"
job_type="CR"

source ~/.bashrc
conda activate manode


while ((${#arr[@]}))
do
	arr=()
	for j in {0..9..1}
	do
		if [[ ! -f /scratch/rd804/m-anode/results/${group}/${job_type}/try_${j}/best_val_loss_scores.npy ]]
		then
			arr+=("$j")
		fi

	done
    echo ${arr[@]}

    for try_ in ${arr[@]}
        do
            echo ${try_}
            sbatch -W --output=/scratch/rd804/m-anode/logs/output/${group}.${job_type}.${try_}.out --error=/scratch/rd804/m-anode/logs/error/${group}.${job_type}.${try_}.err --export=try_=${try_},group=${group},job_type={job_type} nflows_CR.sh ${try_} ${group} ${job_type} &
            # get job id
        done

        wait



done

echo "All done"




