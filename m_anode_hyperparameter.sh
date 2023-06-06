#!/bin/bash

declare -a loss=("true_likelihood")

declare -a mode_back=("train" "freeze" "pretrained")


for mode in ${mode_back[@]}
do
    for loss_ in ${loss[@]}
    do

        if [[ ${mode} == "train" ]]
        then
            cuda_device="cuda:1"
        else
            cuda_device="cuda:2"
        fi

        nohup python scripts/m_anode.py --sig_train=10 --sig_test=10 \
        --mode_background=${mode} --epochs=100 --gpu=${cuda_device} --wandb_group='nflows_gaussian_mixture_1' \
        --wandb_job_type='m_anode_hyperparam_test' --wandb_run_name=${loss_}'_mode_background_'${mode} --data_loss_expr=${loss_} &> logs/m_anode_hyperparam_test_${loss_}_mode_background_${mode}.log &
    done
done