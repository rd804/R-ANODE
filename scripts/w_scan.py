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
#wandb_project_true = f"ra_mass_{nsig}"
wandb_project_true = f'ra_mass_joint_un_clip_{nsig}'
wandb_job_type_true = "try"

data_dir = "data/lhc_co"

wandb_group_wscan = "nflows_lhc_co_w_scan"
#wandb_project_wscan = f"ra_mass_{nsig}"
wandb_project_wscan = f"ra_mass_joint_{nsig}"
wandb_job_type_wscan = "try"

wandb_group_BDT = "BDT"
wandb_project_BDT = "IAD_base_80"
wandb_job_type_BDT = "sample"

wandb_group_true_joint = "nflows_lhc_co_nsig_scan"
wandb_project_true_joint = f"ra_mass_joint_{nsig}"
wandb_job_type_true_joint = "try"

wandb_group_wscan_joint = "nflows_lhc_co_w_scan"
wandb_project_wscan_joint = f"ra_mass_{nsig}"
wandb_job_type_wscan_joint = "try"

mask = np.load(f'results/{wandb_group_BDT}/{wandb_project_BDT}_{1000}/{wandb_job_type_BDT}_{0}/mask_test.npy')
labels = np.load(f'{data_dir}/x_test.npy')[mask][:,-1]

#weights = [0.001,0.002,'true_w',0.004,0.005]
weights = [0.0001,0.001,'true_w',0.01,0.1]
color = ['C0','C1','C2','C3','C4']

#colors = []

_median_sic_RANODE = []
_sic_min_RANODE = []
_sic_max_RANODE = []


for k,w in enumerate(weights):
    
    print('w', w)
    
    tprs_ranode = []
    fprs_ranode = []

    tpr_interp = np.linspace(0.01, 1, 1000)

    
    for i in range(10):
        print('i', i)
        #if (w == 'true_w') & (i!=0):
           # continue

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
        #print('ensemble_S shape', ensemble_S.shape)
        #print('ensemble_B shape', ensemble_B.shape)

        assert ensemble_B.shape == ensemble_S.shape

        ypred = np.log(ensemble_S+1e-32) - np.log(ensemble_B+1e-32)

        tpr_cut, fpr_cut = roc_interp(labels, ypred, tpr_interp)
        tprs_ranode.append(tpr_cut)
        fprs_ranode.append(fpr_cut)
   # print('tprs_ranode shape', len(tprs_ranode))

    # print('tpr_cut shape', tpr_cut.shape)

    sic_median_ranode , sic_max_ranode, sic_min_ranode, tpr_cut_ranode = \
    ensembled_SIC(tprs_ranode, fprs_ranode)


    index_max = np.argmax(sic_median_ranode)
    _median_sic_RANODE.append(sic_median_ranode[index_max])
    _sic_min_RANODE.append(sic_min_ranode[index_max])
    _sic_max_RANODE.append(sic_max_ranode[index_max])


    if w != 'true_w':
        plt.plot(tpr_cut_ranode, sic_median_ranode, label=f'w={w}',linestyle='-', color=color[k])
        plt.fill_between(tpr_cut_ranode, sic_min_ranode, sic_max_ranode, alpha=0.3, color=color[k])

    else:
        plt.plot(tpr_cut_ranode, sic_median_ranode, label=f'w=0.006 (correct)',linestyle='-', color=color[k])
        plt.fill_between(tpr_cut_ranode, sic_min_ranode, sic_max_ranode, alpha=0.3, color=color[k])

    #plt.plot(tpr_cut, sic_median, label=f'Nsig={nsig}', linestyle='--')
    #plt.fill_between(tpr_cut, sic_min, sic_max, alpha=0.3)


#plt.plot(tpr_cut_BDT, tpr_cut_BDT**0.5, color='black',label='')
plt.xlabel('Signal efficiency',fontsize=15)
plt.ylabel('SIC',fontsize=15)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
#plt.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left')
plt.legend(loc='upper right',frameon=False,fontsize=15)
plt.tight_layout()
#plt.colorbar()
plt.savefig(f'./figures/SIC_curves_wscan_joint_{nsig}.pdf',dpi=200,bbox_inches='tight')
plt.close()


_median_sic_RANODE = np.array(_median_sic_RANODE)
_sic_min_RANODE = np.array(_sic_min_RANODE)
_sic_max_RANODE = np.array(_sic_max_RANODE)

weight_array = np.array([i if i != 'true_w' else true_w for i in weights])
mask = weight_array != true_w
print(weight_array)
plt.plot(weight_array, _median_sic_RANODE, marker=".", label='R-ANODE')
#plt.plot(weight_array[~mask], _median_sic_RANODE[~mask], marker="x", label='true w',markersize=10)
plt.fill_between(weight_array[mask], _sic_min_RANODE[mask], _sic_max_RANODE[mask], alpha=0.3)
plt.axvline(x=true_w, color='C1', linestyle='--', label=r'$w_{correct}$')
#plt.err
plt.xlabel(r'$w$',fontsize=15)
plt.ylabel('max SIC',fontsize=15)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.xscale('log')
plt.legend(frameon=False, fontsize=15,loc='lower right')
plt.savefig(f'./figures/SIC_vs_weight_joint_{nsig}.pdf',dpi=200,bbox_inches='tight')
plt.close()



