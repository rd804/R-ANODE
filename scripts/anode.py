import numpy as np
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm
from src.nflow_utils import *
import os
from src.utils import *
from nflows import transforms, distributions, flows
import torch
import torch.nn.functional as F
from nflows.distributions import uniform
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import argparse
import wandb
import pickle

parser = argparse.ArgumentParser()
parser.add_argument('--sig_train', default=10)
parser.add_argument('--try_', type=int, default=0)
parser.add_argument('--wandb_group', type=str, default='nflows_gaussian_mixture_1')
parser.add_argument('--wandb_job_type', type=str, default='SR')
parser.add_argument('--wandb_run_name', type=str, default='try_')


#TODO: load best_models, get log probs, and then do logp - logq
#TODO: remove outliers and check SIC curve
#TODO: smaller nflow for CR region

args = parser.parse_args()



job_name = args.wandb_job_type+'_'+str(args.sig_train)

#initialize wandb

#wandb.init(project="m-anode", config=args,
 #          group=args.wandb_group, job_type=job_name)

run_name = args.wandb_run_name+str(args.try_)
#wandb.run.name = name

with open('data/data.pkl', 'rb') as f:
     data = pickle.load(f)

with open('data/true_w.pkl', 'rb') as f:
    true_w = pickle.load(f)

back_mean = 0
sig_mean = 3
sig_simga = 0.5
back_sigma = 3



sig_train = args.sig_train

x_test = data[str(sig_train)]['val']['data']
label_test = data[str(sig_train)]['val']['label']







# plot SIC curve 



CR = []
for i in range(10):
    CR_file = f'results/{args.wandb_group}/CR/try_{i}/best_val_loss_scores.npy'
    CR.append(np.load(CR_file))

    plt.hist(CR[i],bins=100, histtype='step')
    plt.hist(CR[i][label_test==0],bins=100,label='background',histtype='step')
    plt.hist(CR[i][label_test==1],bins=100,label='signal',histtype='step')
    plt.legend()
    plt.savefig(f'results/{args.wandb_group}/CR/try_{i}/CR_hist.png')
    plt.close()



CR = np.array(CR)
CR = np.mean(CR,axis=0)

plt.hist(CR,bins=100,histtype='step')
plt.hist(CR[label_test==0],bins=100,label='background',histtype='step')
plt.hist(CR[label_test==1],bins=100,label='signal',histtype='step')
plt.legend()
plt.savefig('avg_CR_hist.png')
plt.close()
# For each sigma, generate all SIC curves,
# and get the average max SIC curve for each sigma
# get average AUC for each sigma
# get average SIC_01 for each sigma
# get average SIC_001 for each sigma
summary = {}
sig_train_list = [5, 0.1, 0.2, 0.5, 0.8, 0.9, 1, 2, 1.5, 10]
_true_max_SIC = []
_true_AUC = []
_true_SIC_01 = []
_true_SIC_001 = []

max_SIC_avg = []
AUC_avg = []
SIC_01_avg = []
SIC_001_avg = []

max_SIC_std = []
AUC_std = []
SIC_01_std = []
SIC_001_std = []


summary = {}


for sig_train in sig_train_list:
    _max_SIC = []
    _AUC = []
    _SIC_01 = []
    _SIC_001 = []




    w1 = true_w[str(sig_train)][0]
    w2 = true_w[str(sig_train)][1]
    true_likelihood = p_data(x_test,[sig_mean, back_mean],[sig_simga**2,back_sigma**2],[w1,w2])/p_back(x_test,back_mean,back_sigma**2)
    sic_true , tpr_true , auc_true = SIC(label_test, true_likelihood)
    sic_true_01 = SIC_fpr(label_test, true_likelihood, 0.1)
    sic_true_001 = SIC_fpr(label_test, true_likelihood, 0.01)

    _true_max_SIC.append(np.max(sic_true))
    _true_AUC.append(auc_true)
    _true_SIC_01.append(sic_true_01)
    _true_SIC_001.append(sic_true_001)


    for try_ in range(10):
        SR_file = f'results/{args.wandb_group}/SR_{sig_train}/try_{try_}'
        SR_score_file = f'{SR_file}/best_val_loss_scores.npy'
        SR = np.load(SR_score_file)

        plt.hist(SR,bins=100,histtype='step')
        plt.hist(SR[label_test==0],bins=100, label='background',histtype='step')
        plt.hist(SR[label_test==1],bins=100, label='signal',histtype='step')
        plt.legend()
        plt.savefig(f'{SR_file}/SR_hist.png')
        plt.close()

        SR = SR[CR>0]
        label_test_ = label_test[CR>0]

        CR_ = CR[CR>0]
        assert len(CR_) == len(SR)

        likelihood_score = SR/CR_

        plt.hist(likelihood_score,bins=100,histtype='step')
        plt.hist(likelihood_score[label_test_==0],bins=100, label='background',histtype='step')
        plt.hist(likelihood_score[label_test_==1],bins=100, label='signal',histtype='step')
        plt.legend()
        plt.yscale('log')
        plt.savefig(f'{SR_file}/likelihood_hist.png')
        plt.close()

     
        sic , tpr , auc = SIC(label_test_, likelihood_score)

        _max_SIC.append(np.max(sic))
        _AUC.append(auc)

        sic_01 = SIC_fpr(label_test_, likelihood_score, 0.1)
        sic_001 = SIC_fpr(label_test_, likelihood_score, 0.01)

        _SIC_01.append(sic_01)
        _SIC_001.append(sic_001)

        figure = plt.figure()

        plt.plot(tpr,sic,label='SIC: AUC {:.3f}'.format(auc))
        plt.plot(tpr_true,sic_true,label='true: AUC {:.3f}'.format(auc_true))
        plt.xlabel('TPR')
        plt.ylabel('SIC')
        plt.title(f'SR: {sig_train}, try: {try_}')
        plt.legend()
        plt.savefig(f'{SR_file}/SIC.png')
        plt.close()

    max_SIC_avg.append(np.mean(_max_SIC))
    max_SIC_std.append(np.std(_max_SIC))

    AUC_avg.append(np.mean(_AUC))
    AUC_std.append(np.std(_AUC))

    SIC_01_avg.append(np.mean(_SIC_01))
    SIC_01_std.append(np.std(_SIC_01))

    SIC_001_avg.append(np.mean(_SIC_001))
    SIC_001_std.append(np.std(_SIC_001))

plt.figure()
plt.errorbar(sig_train_list,max_SIC_avg,yerr=max_SIC_std,label='max SIC',fmt='o')
plt.plot(sig_train_list,_true_max_SIC,'o',label='true max SIC')
plt.xlabel('sig_train')
plt.ylabel('max SIC')
plt.legend()
plt.xscale('log')
plt.savefig('anode_max_SIC.png')
plt.close()

plt.figure()
plt.errorbar(sig_train_list,AUC_avg,yerr=AUC_std,label='AUC',fmt='o')
plt.plot(sig_train_list,_true_AUC,'o',label='true AUC')
plt.xlabel('sig_train')
plt.ylabel('AUC')
plt.legend()
plt.xscale('log')
plt.savefig('anode_AUC.png')
plt.close()

plt.figure()
plt.errorbar(sig_train_list,SIC_01_avg,yerr=SIC_01_std,label='SIC_01',fmt='o')
plt.plot(sig_train_list,_true_SIC_01,'o',label='true SIC_01')
plt.xlabel('sig_train')
plt.ylabel('SIC_01')
plt.legend()
plt.xscale('log')
plt.savefig('anode_SIC_01.png')
plt.close()

plt.figure()
plt.errorbar(sig_train_list,SIC_001_avg,yerr=SIC_001_std,label='SIC_001',fmt='o')
plt.plot(sig_train_list,_true_SIC_001,'o',label='true SIC_001')
plt.xlabel('sig_train')
plt.ylabel('SIC_001')
plt.legend()
plt.xscale('log')
plt.savefig('anode_SIC_001.png')
plt.close()


summary['sig_train'] = sig_train_list
summary['max_SIC_avg'] = max_SIC_avg
summary['max_SIC_std'] = max_SIC_std
summary['AUC_avg'] = AUC_avg
summary['AUC_std'] = AUC_std
summary['SIC_01_avg'] = SIC_01_avg
summary['SIC_01_std'] = SIC_01_std
summary['SIC_001_avg'] = SIC_001_avg
summary['SIC_001_std'] = SIC_001_std


    


