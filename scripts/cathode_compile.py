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

wandb_group = "BDT"
wandb_project = "IAD_base_80"
wandb_job_type = "sample"
data_dir = "data/lhc_co"

mask = np.load(f'results/{wandb_group}/{wandb_project}_{1000}/{wandb_job_type}_{0}/mask_test.npy')
labels = np.load(f'{data_dir}/x_test.npy')[mask][:,-1]

nsigs = [75,150,225,300,450,500,600,1000]
colors = []
_median_sic = []
_sic_min = []
_sic_max = []


for nsig in nsigs:
    tprs = []
    fprs = []
    tpr_interp = np.linspace(0.01, 1, 1000)

    for i in range(10):
        ypred = np.load(f'results/{wandb_group}/{wandb_project}_{nsig}/{wandb_job_type}_{i}/ypred.npy')

        #sic, tpr, max_sic = SIC_cut(labels, ypred)
        tpr_cut, fpr_cut = roc_interp(labels, ypred, tpr_interp)
        tprs.append(tpr_cut)
        fprs.append(fpr_cut)

    # print('tpr_cut shape', tpr_cut.shape)

    sic_median , sic_max, sic_min, tpr_cut = ensembled_SIC(tprs, fprs)

    index_max = np.argmax(sic_median)

    _median_sic.append(sic_median[index_max])
    _sic_min.append(sic_min[index_max])
    _sic_max.append(sic_max[index_max])


    plt.plot(tpr_cut, sic_median, label=f'Nsig={nsig}')
    plt.fill_between(tpr_cut, sic_min, sic_max, alpha=0.3)


plt.plot(tpr_cut, tpr_cut**0.5)
plt.xlabel('Signal efficiency')
plt.ylabel('SIC')
plt.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left')
plt.tight_layout()
#plt.colorbar()
plt.savefig('SIC_curves.png')
plt.close()

plt.plot(nsigs, _median_sic, marker=".")
plt.fill_between(nsigs, _sic_min, _sic_max, alpha=0.3)
plt.xlabel('Number of signal events')
plt.ylabel('SIC')
plt.savefig('SIC_vs_nsig.png')
plt.close()





   