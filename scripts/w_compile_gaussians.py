import numpy as np
from matplotlib import pyplot as plt


wandb_group = 'nflows_gaussian_mixture_2'
wandb_job_type = 'm_2d_w_train'


ntries = 4
nensembles = 6
sig_list = [1,2]

for sig in sig_list:
    w_sig = []
    for i in range(ntries):
        w_try = []
        for j in range(nensembles):
            wandb_run_name = f'try_{i}_{j}'
            wandb_job = f'{wandb_job_type}_{sig}_0.0001'
            path = f'./results/{wandb_group}/{wandb_job}/{wandb_run_name}'
            valloss = np.load(f'{path}/valloss.npy')
            lowest_epoch = np.argmin(valloss)

            w_ = np.load(f'{path}/w_{lowest_epoch}.npy')
            w_try.append(w_)
        w_try = np.array(w_try)
        w_sig.append(np.mean(w_try, axis=0))
    w_sig = np.array(w_sig)
    print( np.mean(w_sig, axis=0))
    print( np.std(w_sig, axis=0))
