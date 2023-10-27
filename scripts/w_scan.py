import numpy as np
from sklearn.metrics import roc_curve
from scipy.interpolate import interp1d
from src.nflow_utils import *
import os
from matplotlib import colors
from src.utils import *
from src.nflow_utils import *
from src.generate_data_lhc import *
from src.utils import *
from src.flows import *

#nsig = 500
#true_w = 0.003
nsig = 1000
true_w = 0.006

wandb_group_true = "nflows_lhc_co_nsig_scan"
wandb_project_true = f"r_anode_RQS_affine_{nsig}"
wandb_job_type_true = "try"

data_dir = "data/lhc_co"

wandb_group_wscan = "nflows_lhc_co_w_scan"
wandb_project_wscan = f"r_anode_R_A_{nsig}"
wandb_job_type_wscan = "try"

wandb_group_BDT = "BDT"
wandb_project_BDT = "IAD_base_80"
wandb_job_type_BDT = "sample"
mask = np.load(f'results/{wandb_group_BDT}/{wandb_project_BDT}_{1000}/{wandb_job_type_BDT}_{0}/mask_test.npy')
labels = np.load(f'{data_dir}/x_test.npy')[mask][:,-1]

#weights = [0.001,0.002,'true_w',0.004,0.005]
weights = [0.0001,0.001,'true_w',0.009,0.1]
color = ['C0','C1','C2','C3','C4']

#colors = []

_median_sic_RANODE = []
_sic_min_RANODE = []
_sic_max_RANODE = []


for k,w in enumerate(weights):
    

    
    tprs_ranode = []
    fprs_ranode = []

    tpr_interp = np.linspace(0.01, 1, 1000)

    
    for i in range(10):
        if w != 'true_w':
            if i==0:
                ensemble_B = np.load(f'results/{wandb_group_wscan}/{wandb_project_wscan}_{w}/{wandb_job_type_wscan}_{i}_{0}/ensemble_B.npy')

            path_ensemble_S = f'results/{wandb_group_wscan}/{wandb_project_wscan}_{w}/{wandb_job_type_wscan}_{i}_'
        else:
            if i==0:
                ensemble_B = np.load(f'results/{wandb_group_true}/{wandb_project_true}/{wandb_job_type_true}_{i}_{0}/ensemble_B.npy')

            path_ensemble_S = f'results/{wandb_group_true}/{wandb_project_true}/{wandb_job_type_true}_{i}_'
            
        ensemble_S = [np.load(path_ensemble_S + f'{j}/ensemble_S.npy') for j in range(20) if os.path.exists(path_ensemble_S + f'{j}/ensemble_S.npy')]

        ensemble_S = np.array(ensemble_S)
        # print('ensemble_S shape', ensemble_S.shape)

        ensemble_S = np.mean(ensemble_S, axis=0)

        assert ensemble_B.shape == ensemble_S.shape

        ypred = np.log(ensemble_S+1e-32) - np.log(ensemble_B+1e-32)

        tpr_cut, fpr_cut = roc_interp(labels, ypred, tpr_interp)
        tprs_ranode.append(tpr_cut)
        fprs_ranode.append(fpr_cut)


    # print('tpr_cut shape', tpr_cut.shape)

    sic_median_ranode , sic_max_ranode, sic_min_ranode, tpr_cut_ranode = \
    ensembled_SIC(tprs_ranode, fprs_ranode)


    index_max = np.argmax(sic_median_ranode)
    _median_sic_RANODE.append(sic_median_ranode[index_max])
    _sic_min_RANODE.append(sic_min_ranode[index_max])
    _sic_max_RANODE.append(sic_max_ranode[index_max])



    plt.plot(tpr_cut_ranode, sic_median_ranode, label=f'w={w}',linestyle='-', color=color[k])
    plt.fill_between(tpr_cut_ranode, sic_min_ranode, sic_max_ranode, alpha=0.3, color=color[k])
    #plt.plot(tpr_cut, sic_median, label=f'Nsig={nsig}', linestyle='--')
    #plt.fill_between(tpr_cut, sic_min, sic_max, alpha=0.3)


#plt.plot(tpr_cut_BDT, tpr_cut_BDT**0.5, color='black',label='')
plt.xlabel('Signal efficiency')
plt.ylabel('SIC')
plt.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left')
plt.tight_layout()
#plt.colorbar()
plt.savefig(f'./figures/SIC_curves_wscan_{nsig}.png')
plt.close()


weight_array = np.array([i if i != 'true_w' else true_w for i in weights])
print(weight_array)
plt.plot(weight_array, _median_sic_RANODE, marker=".", label='R-ANODE')
plt.fill_between(weight_array, _sic_min_RANODE, _sic_max_RANODE, alpha=0.3)
plt.xlabel('weights')
plt.ylabel('max SIC')
plt.xscale('log')
plt.legend()
plt.savefig(f'./figures/SIC_vs_weight_{nsig}.png')
plt.close()



