#!/bin/bash



python scripts/m_anode.py --sig_train=10 --sig_test=10 --mode_background='train' \
--clip_grad=0.1 --epochs=2 --gpu='cuda:0' --wandb_group='test' \
--wandb_job_type='m_anode' --wandb_run_name='try_0'
