#!/bin/bash

declare -a loss=("true_likelihood" "expectation_likelihood")

declare -a clip_grad=(False 0.1)


for clip in ${clip_grad[@]}
do
    for loss_ in ${loss[@]}
    do

        if [[ loss_ == "true_likelihood" ]]
        then
            cuda_device="cuda:0"
        else
            cuda_device="cuda:1"
        fi

        python scripts/m_anode.py --sig_train=10 --sig_test=10 --mode_background='train' \
        --clip_grad=${clip} --epochs=30 --gpu=${cuda_device} --wandb_group='nflows_gaussian_mixture_1' \
        --wandb_job_type='m_anode_hyperparam_test' --wandb_run_name='${loss_}_clip_grad_${clip}' --loss=${loss_} &
    done
    wait
done