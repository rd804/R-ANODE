import numpy as np
from matplotlib import pyplot as plt
import os

#wandb_group = 'nflows_lhco_w_train'
#wandb_job_type = 'joint_0.01'
wandb_group = 'nflows_lhco'
wandb_job_type = 'joint_random'


#ntries = 4
nensembles = 20
#sig_list = [1000,500,300,225,75]
#true_w = [0.006037,0.00299,0.00185,0.0014,0.00044]
#true_w_std = [0.00002,0.00004,0.00004,0.00004,0.00003]
sig_list = [1000,75]
true_w = [0.006037,0.00044]
true_w_std = [0.00002,0.00003]

w_mean = []
w_std = []

for sig in sig_list:
    w_sig = []
  #  w_std = []
    w_all = []
    for i in range(1):
        w_try = []
        for j in range(1,nensembles,1):
            wandb_run_name = f'try_{i}_{j}'
            wandb_job = f'{wandb_job_type}_{sig}'
            path = f'./results/{wandb_group}/{wandb_job}/{wandb_run_name}'

            if not os.path.exists(f'{path}/valloss.npy'):
                continue
            valloss = np.load(f'{path}/valloss.npy')
            lowest_epoch = np.argsort(valloss)[0:10]

            for epoch in lowest_epoch:
                w_ = np.load(f'{path}/w_{epoch}.npy')
                w_try.append(w_)
                w_all.append(w_)
        #w_try = np.array(w_try)
   #     print(w_try)
        w_sig.append(np.mean(w_try, axis=0))
       # w_std.append(np.std(w_try, axis=0))
    w_sig = np.array(w_sig)

    w_mean.append(np.mean(w_sig, axis=0))
    w_std.append(np.std(w_all, axis=0))
   # w_std = np.array(w_std)
   # w_std = np.std(w_all, axis=0)
   # print(w_sig)
   # print(w_std)
    #print(np.mean(w_sig, axis=0))
    #print(w_std)
   # print(w_sig)
#    print(w_std)
print(w_mean)
print(w_std)
  #  print( np.mean(w_sig, axis=0))
  #  print( np.std(w_sig, axis=0))
plt.plot(sig_list, w_mean, label='learned w',color='C0')
plt.errorbar(sig_list, w_mean, yerr=w_std, fmt='o',color='C0')
plt.plot(sig_list, true_w, label='true w',color='C1')
plt.errorbar(sig_list, true_w, yerr=true_w_std, fmt='o',color='C1') 
plt.xlabel(r'$N_{sig}$')
plt.ylabel(r'$w$')
plt.legend(loc='upper left', frameon=False)
plt.savefig('figures/w.pdf')
plt.show()