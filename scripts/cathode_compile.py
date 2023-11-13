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
import pickle
from scipy.stats import rv_histogram


wandb_group_BDT = "BDT"
wandb_project_BDT = "supervised"
wandb_job_type_BDT = "sample"

wandb_group_BDT_50 = "BDT"
wandb_project_BDT_50 = "IAD_weighted_50"
#wandb_project_BDT_50 = "IAD_mass_50"
wandb_job_type_BDT_50 = "sample"

wandb_group_ranode = "nflows_lhc_co_nsig_scan"
wandb_project_ranode = "anode_SR_RQS"
wandb_job_type_ranode = "try"

wandb_group_ranode_mass = "nflows_lhc_co_nsig_scan"
#wandb_project_ranode_mass = "r_anode_R_A_50"
#wandb_project_ranode_mass = "ra_uncond"
wandb_job_type_ranode_mass = "try"
wandb_project_ranode_mass = "ra_mass"
#wandb_job_type_ranode_mass = "try"

wandb_group_ranode_uncond = "nflows_lhc_co_nsig_scan"
wandb_project_ranode_uncond = "ra_mass_joint_un_clip"
wandb_job_type_ranode_uncond = "try"


data_dir = "data/lhc_co"

mask = np.load(f'results/{wandb_group_BDT_50}/{wandb_project_BDT_50}_{1000}/{wandb_job_type_BDT_50}_{0}/mask_test.npy')
labels = np.load(f'{data_dir}/x_test.npy')[mask][:,-1]

#############################
CR_path = 'results/nflows_lhc_co/CR_bn_fixed_1000/try_1_0'
SR_data, CR_data , true_w, sigma = resample_split(data_dir, n_sig = 500, resample_seed = 1,resample = True)

with open(f'{CR_path}/pre_parameters.pkl', 'rb') as f:
    pre_parameters_CR = pickle.load(f)

pre_parameters_SR_ = preprocess_params_fit_all(SR_data)

pre_parameters_SR = pre_parameters_CR.copy()
for key in pre_parameters_CR.keys():
    pre_parameters_SR[key]= np.insert(pre_parameters_CR[key], 0, pre_parameters_SR_[key][0])
_x_test = np.load(f'{data_dir}/x_test.npy')
_, mask_CR = logit_transform(_x_test[:,1:-1], pre_parameters_CR['min'],
                                pre_parameters_CR['max'])
_, mask_SR = logit_transform(_x_test[:,:-1], pre_parameters_SR['min'],
                                pre_parameters_SR['max'])
mask_joint = mask_CR & mask_SR 
labels_joint = _x_test[mask_joint][:,-1]

mass = np.load(f'{data_dir}/true_mass.npy')
labels_m = np.load(f'{data_dir}/true_labels.npy')

bins = np.linspace(3.3, 3.7, 50)
#hist_sig = np.histogram(mass[labels==1], bins=bins, density=True)
#density_sig = rv_histogram(hist_sig)

hist_back = np.histogram(mass[labels_m==0], bins=bins, density=True)
density_back = rv_histogram(hist_back)
#############################


#nsigs = [75,150,225,300,450,500,600,1000]
nsigs = [1000,600,500,450,300,225,150,75]
sigmas = [2.166,1.294,1.076,0.9714,0.6499,0.4866,0.3247,0.1521]
#nsigs = [1000,600,450,300,225,150]
#sigmas = [2.166,1.294,0.9714,0.6499,0.4866,0.3247]
#nsigs = [1000,300]
#sigmas = [2.166,0.9714]
#sigmas = [0.9714,0.4866]
#nsigs = [300]
#sigmas = [0.6499]
color = ['C0','C1','C2','C3','C4','C5','C6','C7']

#colors = []
_median_sic_IAD = []
_sic_min_IAD = []
_sic_max_IAD = []

_median_sic_IAD_50 = []
_sic_min_IAD_50 = []
_sic_max_IAD_50 = []

_median_sic_ANODE = []
_sic_min_ANODE = []
_sic_max_ANODE = []

_median_sic_RANODE_mass = []
_sic_min_RANODE_mass = []
_sic_max_RANODE_mass = []

_median_sic_RANODE_uncond = []
_sic_min_RANODE_uncond = []
_sic_max_RANODE_uncond = []


for k,nsig in enumerate(nsigs):
    
    if k==0:
      tprs_BDT = []
      fprs_BDT = []
    
    tprs_BDT_50 = []
    fprs_BDT_50 = []
    
    tprs_anode = []
    fprs_anode = []

    tprs_ranode_mass = []
    fprs_ranode_mass = []

    tprs_ranode_uncond = []
    fprs_ranode_uncond = []

    tpr_interp = np.linspace(0.001, 1, 1000)

    # IAD
  #  for i in range(10):
   #     ypred = np.load(f'results/{wandb_group_BDT}/{wandb_project_BDT}_{nsig}/{wandb_job_type_BDT}_{i}/ypred.npy')

        #sic, tpr, max_sic = SIC_cut(labels, ypred)
    #    tpr_cut, fpr_cut = roc_interp(labels, ypred, tpr_interp)
     #   tprs_BDT.append(tpr_cut)
      #  fprs_BDT.append(fpr_cut)
    if k==0:
      for i in range(10):
          ypred = np.load(f'results/{wandb_group_BDT}/{wandb_project_BDT}/{wandb_job_type_BDT}_{i}/ypred.npy')

          #sic, tpr, max_sic = SIC_cut(labels, ypred)
          tpr_cut, fpr_cut = roc_interp(labels, ypred, tpr_interp)
          tprs_BDT.append(tpr_cut)
          fprs_BDT.append(fpr_cut)

        # IAD
    for i in range(10):
        ypred = np.load(f'results/{wandb_group_BDT_50}/{wandb_project_BDT_50}_{nsig}/{wandb_job_type_BDT_50}_{i}/ypred.npy')

        #sic, tpr, max_sic = SIC_cut(labels, ypred)
        tpr_cut, fpr_cut = roc_interp(labels, ypred, tpr_interp)
        tprs_BDT_50.append(tpr_cut)
        fprs_BDT_50.append(fpr_cut)
    
    for i in range(10):
        if i==0:
            ensemble_B = np.load(f'results/{wandb_group_ranode}/{wandb_project_ranode}_{nsig}/{wandb_job_type_ranode}_{i}_{0}/ensemble_B.npy')

        path_ensemble_S = f'results/{wandb_group_ranode}/{wandb_project_ranode}_{nsig}/{wandb_job_type_ranode}_{i}_'
        
        ensemble_S = [np.load(path_ensemble_S + f'{j}/ensemble_S.npy') for j in range(20) if os.path.exists(path_ensemble_S + f'{j}/ensemble_S.npy')]

        ensemble_S = np.array(ensemble_S)
        # print('ensemble_S shape', ensemble_S.shape)

        ensemble_S = np.mean(ensemble_S, axis=0)

        assert ensemble_B.shape == ensemble_S.shape
        ypred = ensemble_S - ensemble_B
        #ypred = np.log(ensemble_S+1e-32) - np.log(ensemble_B+1e-32)
        ypred = np.nan_to_num(ypred, nan=0, posinf=0, neginf=0)

        tpr_cut, fpr_cut = roc_interp(labels, ypred, tpr_interp)
        tprs_anode.append(tpr_cut)
        fprs_anode.append(fpr_cut)

    # Unconditional

    for i in range(10):
        if i==0:
          ensemble_B = np.load(f'results/{wandb_group_ranode_uncond}/{wandb_project_ranode_uncond}_{nsig}/{wandb_job_type_ranode_uncond}_{i}_{0}/ensemble_B.npy')
        
        path_ensemble_S = f'results/{wandb_group_ranode_uncond}/{wandb_project_ranode_uncond}_{nsig}/{wandb_job_type_ranode_uncond}_{i}_'

        #ensemble_S = [np.load(path_ensemble_S + f'{j}/ensemble_S.npy') for j in range(20) if os.path.exists(path_ensemble_S + f'{j}/ensemble_S.npy')]
        ensemble_S = [np.load(path_ensemble_S + f'{j}/ensemble_S_joint.npy') for j in range(20) if os.path.exists(path_ensemble_S + f'{j}/ensemble_S_joint.npy')]
        #ensemble_S = np.exp(np.array(ensemble_S))
        ensemble_S = np.array(ensemble_S)
       # print('ensemble_S shape', ensemble_S.shape)
        ensemble_S = np.mean(ensemble_S, axis=0)
        assert ensemble_B.shape == ensemble_S.shape

       # SR_data, CR_data , true_w, sigma = resample_split(data_dir, n_sig = nsig, resample_seed = 0,resample = True)

       # with open(f'{CR_path}/pre_parameters.pkl', 'rb') as f:
        #  pre_parameters_CR = pickle.load(f)

      #  pre_parameters_SR_ = preprocess_params_fit_all(SR_data)

      #  pre_parameters_SR = pre_parameters_CR.copy()
      #  for key in pre_parameters_CR.keys():
           # pre_parameters_SR[key]= np.insert(pre_parameters_CR[key], 0, pre_parameters_SR_[key][0])
        _x_test = np.load(f'{data_dir}/x_test.npy')
       # _, mask_CR = logit_transform(_x_test[:,1:-1], pre_parameters_CR['min'],
                             #           pre_parameters_CR['max'])
      #  _, mask_SR = logit_transform(_x_test[:,:-1], pre_parameters_SR['min'],
                       #                 pre_parameters_SR['max'])
      #  mask_joint = mask_CR & mask_SR 
        labels_joint = _x_test[mask][:,-1]
        mass_density = density_back.pdf(_x_test[mask][:,0])
      #  print('mass_density shape', mass_density.shape)
       # ypred = np.log(ensemble_S+1e-32) - np.log(ensemble_B+1e-32)
        ypred = np.log(ensemble_S+1e-32) - np.log(ensemble_B*mass_density+1e-32)       
        ypred = np.nan_to_num(ypred, nan=0, posinf=0, neginf=0)
       # ypred = ypred[mask_joint]
        ###############################
       # mass = _x_test[mask_joint][:,0]
       # mass_cut = (mass > 3.35) & (mass < 3.65)
       # labels_joint = labels_joint[mass_cut]
       # ypred = ypred[mass_cut]
########################################
        tpr_cut, fpr_cut = roc_interp(labels_joint, ypred, tpr_interp)
        tprs_ranode_uncond.append(tpr_cut)
        fprs_ranode_uncond.append(fpr_cut)

    for i in range(1):
       # if i==0:
        ensemble_B = np.load(f'results/{wandb_group_ranode_mass}/{wandb_project_ranode_mass}_{nsig}/{wandb_job_type_ranode_mass}_{i}_{0}/ensemble_B.npy')
        path_ensemble_S = f'results/{wandb_group_ranode_mass}/{wandb_project_ranode_mass}_{nsig}/{wandb_job_type_ranode_mass}_{i}_'
 
        ensemble_S = [np.load(path_ensemble_S + f'{j}/ensemble_S.npy') for j in range(20) if os.path.exists(path_ensemble_S + f'{j}/ensemble_S.npy')]
        ensemble_S = np.array(ensemble_S)
        # print('ensemble_S shape', ensemble_S.shape)
        ensemble_S = np.mean(ensemble_S, axis=0)
        assert ensemble_B.shape == ensemble_S.shape

        ypred = np.log(ensemble_S+1e-32) - np.log(ensemble_B+1e-32)

        tpr_cut, fpr_cut = roc_interp(labels, ypred, tpr_interp)
        tprs_ranode_mass.append(tpr_cut)
        fprs_ranode_mass.append(fpr_cut)

    # print('tpr_cut shape', tpr_cut.shape)
   # sic_median_BDT, tpr_cut_BDT = SIC_cut_(tprs_BDT,fprs_BDT)
    sic_median_BDT_50 , sic_max_BDT_50, sic_min_BDT_50, tpr_cut_BDT_50 = \
    ensembled_SIC(tprs_BDT_50, fprs_BDT_50)
    if k==0:
      sic_median_BDT , sic_max_BDT, sic_min_BDT, tpr_cut_BDT = \
      ensembled_SIC(tprs_BDT, fprs_BDT)
    sic_median_anode , sic_max_anode, sic_min_anode, tpr_cut_anode = \
    ensembled_SIC(tprs_anode, fprs_anode)
    sic_median_ranode_mass , sic_max_ranode_mass, sic_min_ranode_mass, tpr_cut_ranode_mass = \
    ensembled_SIC(tprs_ranode_mass, fprs_ranode_mass)
    sic_median_ranode_uncond , sic_max_ranode_uncond, sic_min_ranode_uncond, tpr_cut_ranode_uncond = \
    ensembled_SIC(tprs_ranode_uncond, fprs_ranode_uncond)

    if k==0:
      index_max = np.argmax(sic_median_BDT)
      _median_sic_IAD.append(sic_median_BDT[index_max])
      _sic_min_IAD.append(sic_min_BDT[index_max])
      _sic_max_IAD.append(sic_max_BDT[index_max])

    index_max = np.argmax(sic_median_anode)
    _median_sic_ANODE.append(sic_median_anode[index_max])
    _sic_min_ANODE.append(sic_min_anode[index_max])
    _sic_max_ANODE.append(sic_max_anode[index_max])

    index_max = np.argmax(sic_median_BDT_50)
    _median_sic_IAD_50.append(sic_median_BDT_50[index_max])
    _sic_min_IAD_50.append(sic_min_BDT_50[index_max])
    _sic_max_IAD_50.append(sic_max_BDT_50[index_max])

    index = np.argmax(sic_median_ranode_mass)
    _median_sic_RANODE_mass.append(sic_median_ranode_mass[index])
    _sic_min_RANODE_mass.append(sic_min_ranode_mass[index])
    _sic_max_RANODE_mass.append(sic_max_ranode_mass[index])

    index = np.argmax(sic_median_ranode_uncond)
    _median_sic_RANODE_uncond.append(sic_median_ranode_uncond[index])
    _sic_min_RANODE_uncond.append(sic_min_ranode_uncond[index])
    _sic_max_RANODE_uncond.append(sic_max_ranode_uncond[index])

    #if nsig == 1000:
    plt.plot(tpr_cut_BDT, sic_median_BDT, linestyle='-',color='C2', label='supervised')
    plt.fill_between(tpr_cut_BDT, sic_min_BDT, sic_max_BDT, alpha=0.3, color='C2')

    plt.plot(tpr_cut_BDT_50, sic_median_BDT_50, linestyle='-',color='C0', label='IAD-BDT')
    plt.fill_between(tpr_cut_BDT_50, sic_min_BDT_50, sic_max_BDT_50, alpha=0.3, color='C0')

    plt.plot(tpr_cut_anode, sic_median_anode,linestyle='-', color='C1', label='ANODE')
    plt.fill_between(tpr_cut_anode, sic_min_anode, sic_max_anode, alpha=0.3, color='C1')

   # plt.plot(tpr_cut_ranode_mass, sic_median_ranode_mass, color='C3', linestyle='-', label='R-ANODE conditional')
    plt.plot(tpr_cut_ranode_uncond, sic_median_ranode_uncond, color='C4', linestyle='-',label='R-ANODE')
    plt.fill_between(tpr_cut_ranode_uncond, sic_min_ranode_uncond, sic_max_ranode_uncond, alpha=0.3, color='C4')
    #plt.plot(tpr_cut_BDT, sic_median_BDT, color='C2', linestyle='-',label='IAD-BDT')
    plt.xlabel('Signal efficiency',fontsize=15)
    plt.ylabel('SIC',fontsize=15)
    plt.title(f'$Nsig={nsig}$', fontsize=15)
    plt.legend(loc='upper right', frameon=False, fontsize=12)
    #plt.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left')
    plt.tight_layout()
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    #plt.colorbar()
    plt.savefig(f'./figures/presentation/SIC_curves_mass_supervised_2_{nsig}.pdf',dpi=200)
    plt.close()

    #plt.plot(tpr_cut, sic_median, label=f'Nsig={nsig}', linestyle='--')
    #plt.fill_between(tpr_cut, sic_min, sic_max, alpha=0.3)


#plt.plot(tpr_cut_BDT, tpr_cut_BDT**0.5, color='black',label='')
#plt.plot([],[], ls = '-', label='ANODE',color='black')
#plt.plot([],[], ls = '--', label='IAD-BDT',color='black')
#plt.plot([],[], ls = '-.', label='R-ANODE conditional',color='black')
#plt.plot([],[], ls = ':', label='R-ANODE joint',color='black')

#plt.xlabel('Signal efficiency')
#plt.ylabel('SIC')
#plt.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left')
#plt.tight_layout()
#plt.colorbar()
#plt.savefig('./figures/presentation/SIC_curves_mass_1000.pdf',dpi=200)
#plt.close()


_median_sic_IAD = np.array(_median_sic_IAD)
_sic_max_IAD = np.array(_sic_max_IAD)
_sic_min_IAD = np.array(_sic_min_IAD)

#print(_sic_max_IAD - _sic_min_IAD)

_median_sic_IAD_50 = np.array(_median_sic_IAD_50)
_sic_max_IAD_50 = np.array(_sic_max_IAD_50)
_sic_min_IAD_50 = np.array(_sic_min_IAD_50)

#print(_sic_max_IAD_50 - _sic_min_IAD_50)

_median_sic_ANODE = np.array(_median_sic_ANODE)
_sic_max_ANODE = np.array(_sic_max_ANODE)
_sic_min_ANODE = np.array(_sic_min_ANODE)

_median_sic_RANODE_mass = np.array(_median_sic_RANODE_mass)
_sic_max_RANODE_mass = np.array(_sic_max_RANODE_mass)
_sic_min_RANODE_mass = np.array(_sic_min_RANODE_mass)

_median_sic_RANODE_uncond = np.array(_median_sic_RANODE_uncond)
_sic_max_RANODE_uncond = np.array(_sic_max_RANODE_uncond)
_sic_min_RANODE_uncond = np.array(_sic_min_RANODE_uncond)


plt.plot(nsigs, _median_sic_IAD_50, marker=".", label='IAD-BDT', color='C0')
#plt.errorbar(nsigs, _median_sic_IAD_50, yerr=(_median_sic_IAD_50 - _sic_max_IAD_50,
 #                                             _sic_min_IAD_50 - _median_sic_IAD_50 ), fmt='none', capsize=3, color='C0')
plt.fill_between(nsigs, _sic_min_IAD_50, _sic_max_IAD_50, alpha=0.3, color='C0')

plt.plot(nsigs, _median_sic_ANODE, marker=".", label='ANODE', color='C1')
plt.fill_between(nsigs, _sic_min_ANODE, _sic_max_ANODE, alpha=0.3, color='C1')
#plt.errorbar(nsigs, _median_sic_RANODE, yerr=( _median_sic_RANODE - _sic_max_RANODE,
 #                                             _sic_min_RANODE - _median_sic_RANODE ), \
  #           fmt='none', capsize=3, color='C1')
#plt.fill_between(nsigs, _sic_min_RANODE, _sic_max_RANODE, alpha=0.3)

#plt.plot(nsigs, _median_sic_IAD, marker=".", label='IAD-BDT 80',color='C2')
#plt.errorbar(nsigs, _median_sic_IAD, yerr=(_median_sic_IAD-_sic_max_IAD,
 #                                          _sic_min_IAD-_median_sic_IAD), fmt='none', capsize=3, color='C2')
#plt.fill_between(nsigs, _sic_min_IAD, _sic_max_IAD, alpha=0.3)

#plt.plot(nsigs, _median_sic_RANODE_mass, marker=".", label='R-ANODE conditional', color='C3')
#plt.errorbar(nsigs, _median_sic_RANODE_mass, yerr=(_median_sic_RANODE_mass-_sic_max_RANODE_mass,
 #                                                  _sic_min_RANODE_mass-_median_sic_RANODE_mass), fmt='none', capsize=3, color='C3')
plt.plot(nsigs, _median_sic_RANODE_uncond, marker=".", label='R-ANODE ', color='C4')
plt.fill_between(nsigs, _sic_min_RANODE_uncond, _sic_max_RANODE_uncond, alpha=0.3, color='C4')
plt.xlabel('Number of signal events',fontsize=15)
plt.ylabel('max SIC',fontsize=15)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.legend(frameon=False, fontsize=12,loc='lower right')
plt.savefig('./figures/presentation/SIC_vs_nsig_rxm_2.pdf',dpi=200)
plt.close()


sigma = np.array(sigmas)
#_median_sic_IAD = np.array(_median_sic_IAD) * sigma
#_sic_max_IAD = np.array(_sic_max_IAD) * sigma
#_sic_min_IAD = np.array(_sic_min_IAD) * sigma

_median_sic_IAD_50 = np.array(_median_sic_IAD_50) * sigma
_sic_min_IAD_50 = np.array(_sic_min_IAD_50) * sigma
_sic_max_IAD_50 = np.array(_sic_max_IAD_50) * sigma

_median_sic_ANODE = np.array(_median_sic_ANODE) * sigma
_sic_min_ANODE = np.array(_sic_min_ANODE) * sigma
_sic_max_ANODE = np.array(_sic_max_ANODE) * sigma

_median_sic_RANODE_mass = np.array(_median_sic_RANODE_mass) * sigma
_sic_min_RANODE_mass = np.array(_sic_min_RANODE_mass) * sigma
_sic_max_RANODE_mass = np.array(_sic_max_RANODE_mass) * sigma

_median_sic_RANODE_uncond = np.array(_median_sic_RANODE_uncond) * sigma
_sic_min_RANODE_uncond = np.array(_sic_min_RANODE_uncond) * sigma
_sic_max_RANODE_uncond = np.array(_sic_max_RANODE_uncond) * sigma

plt.plot(nsigs, _median_sic_IAD_50, marker=".", label='IAD-BDT')
plt.fill_between(nsigs, _sic_min_IAD_50, _sic_max_IAD_50, alpha=0.3)
plt.plot(nsigs, _median_sic_ANODE, marker=".", label='ANODE')
plt.fill_between(nsigs, _sic_min_ANODE, _sic_max_ANODE, alpha=0.3)
#plt.plot(nsigs, _median_sic_IAD, marker=".", label='IAD-BDT 80')
#plt.fill_between(nsigs, _sic_min_IAD, _sic_max_IAD, alpha=0.3)
plt.plot(nsigs, _median_sic_RANODE_uncond, marker=".", label='R-ANODE',color='C4')
plt.fill_between(nsigs, _sic_min_RANODE_uncond, _sic_max_RANODE_uncond, alpha=0.3,color='C4')
plt.plot(nsigs, sigmas, marker=".", label='sigma',color = 'C6')
#plt.plot(nsigs, _median_sic_RANODE_mass, marker=".", label='R-ANODE condtional')

plt.axhline(5, ls='--', color='black')
plt.axhline(3, ls='--', color='black')
plt.xlabel('Number of signal events',fontsize=15)
plt.ylabel('significance',fontsize=15)
plt.legend(fontsize=12, frameon=False)
plt.xlim(75,400)
plt.ylim(0,15)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.savefig('./figures/presentation/Significance_vs_nsig_rxm_2.pdf',dpi=200)
plt.close()



   
