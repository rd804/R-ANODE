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
import wandb
import pickle
import os
import argparse

np.seterr(divide='ignore', invalid='ignore')

parser = argparse.ArgumentParser()
parser.add_argument('--scan_set', nargs='+', type=str, default='r_anode_RQS')
parser.add_argument('--anode', action='store_true', help='if anode is to be plotted as well')
parser.add_argument('--cathode', action='store_true', help='if cathode is to be plotted as well')
parser.add_argument('--anode_set', type=str, default='SR_fixed')
parser.add_argument('--data_dir', type=str, default='data/lhc_co')
parser.add_argument('--config_file', type=str, default='scripts/DE_MAF_model.yml', help='config file')
parser.add_argument('--CR_path', type=str, default='results/nflows_lhc_co/CR_bn_fixed_1000/try_1_0', help='CR data path')


parser.add_argument('--ensemble', action='store_true', default=True)
parser.add_argument('--wandb', action='store_true')
# pass a list in argparse

# note ANODE results come with tailbound=10


args = parser.parse_args()
args.scan_set = [args.scan_set]
tries = 10
splits = 20


#TODO: load best_models, get log probs, and then do logp - logq
#TODO: remove outliers and check SIC curve
#TODO: smaller nflow for CR region


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


import wandb

group_name = 'nflows_lhc_co'

group_name_r_anode = 'nflows_lhc_co_nsig_scan'

if args.wandb:
    wandb.init(project='r_anode',group='summary', job_type=f'summary_r_anode',
            config={'device':device})

    wandb.run.name = f'R_ANODE_{args.scan_set[0]}'



_x_test = np.load(f'{args.data_dir}/x_test.npy')

# load pre_parameters of CR: 
with open(f'{args.CR_path}/pre_parameters.pkl', 'rb') as f:
    pre_parameters_CR = pickle.load(f)

pre_parameters_CR_tensor = pre_parameters_CR.copy()
for key in pre_parameters_CR_tensor.keys():
    pre_parameters_CR_tensor[key] = torch.tensor(pre_parameters_CR_tensor[key]).to(device)

SR_file = f'results/{group_name}/{args.anode_set}_1000/try_1_1'

with open(f'{SR_file}/pre_parameters.pkl', 'rb') as f:
    pre_parameters_SR = pickle.load(f)




_, mask_CR = logit_transform(_x_test[:,1:-1], pre_parameters_CR['min'],
                                pre_parameters_CR['max'])

_, mask_SR = logit_transform(_x_test[:,1:-1], pre_parameters_SR['min'],
                                pre_parameters_SR['max'])

mask = mask_CR & mask_SR



x_test_masked = _x_test[mask]
x_test_CR = preprocess_params_transform(x_test_masked, pre_parameters_CR)
test_tensor_CR = torch.from_numpy(x_test_CR).float().to(device)

label_test = x_test_masked[:,-1]



# get CR region for ANODE
val_losses = np.load(f'{args.CR_path}/valloss_list.npy')
best_epoch = np.argsort(val_losses)[0:10]

model_B = DensityEstimator(args.config_file, eval_mode=True, device=device)
background_log_p = []

for epoch in best_epoch:
  #  model_B = DensityEstimator(args.config_file, eval_mode=True, load_path=f"{args.CR_path}/my_ANODE_model_epoch_{epoch}.par", device=device)
    model_B.model.load_state_dict(torch.load(f'{args.CR_path}/model_CR_{epoch}.pt'))

    model_B.model.eval()
    with torch.no_grad():
        log_p = evaluate_log_prob(model_B.model, test_tensor_CR, pre_parameters_CR).cpu().detach().numpy()
        background_log_p.append(log_p)

background_log_p = np.array(background_log_p)
background_exp_p = np.exp(background_log_p)
background_log_p = np.mean(background_exp_p, axis=0)
mean_log_CR = np.log(background_log_p+1e-32)

#sig_train_list = [0.1, 0.3, 0.5, 0.8, 1, 1.5, 2, 5, 10]
#n_sig_train_list = [1000, 750, 500, 250, 100, 75]
#sig_train_list = [2.17, 1.6, 1.1, 0.54 , 0.21, 0.16]
#n_sig_train_list = [1000, 750, 500, 250, 100]
#sig_train_list = [2.17, 1.6, 1.1, 0.54 , 0.21]
#n_sig_train_list = [1000, 750]
#sig_train_list = [2.17, 1.6]
n_sig_train_list = [1000, 600, 450 , 300]
sig_train_list = [2.17, 1.35, 1, 0.67]
sic_ad = [15.3, 12.5, 7, 1.25, 1.0, 1.0, 1.0]
sic_ad_std = [2.5, 3, 5, 0.25, 0.25, 0.25, 0.25]
sig_ad = [2.17, 1.35, 1.0, 0.68, 0.54, 0.34, 0.17]




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


ANODE_file = f'results/nflows_lhc_co/'

# get SR region for ANODE

if args.anode:
    for sig_train in n_sig_train_list:
        _max_SIC = []
        _AUC = []
        _SIC_01 = []
        _SIC_001 = []
    #  print(f'sig_train: {sig_train}')

        for try_ in range(tries):

            SR_array = []

            for split in range(splits):
                model_SR = DensityEstimator(args.config_file, eval_mode=True, device=device)
                SR_file = f'results/{group_name}/{args.anode_set}_{sig_train}/try_{try_+1}_{split}'
                
                with open(f'{SR_file}/pre_parameters.pkl', 'rb') as f:
                    pre_parameters_SR = pickle.load(f)
                
                x_test_SR = preprocess_params_transform(x_test_masked, pre_parameters_SR)
                test_tensor_S = torch.from_numpy(x_test_SR).float().to(device)

                pre_parameters_SR_tensor = pre_parameters_SR.copy()
                for key in pre_parameters_SR_tensor.keys():
                    pre_parameters_SR_tensor[key] = torch.tensor(pre_parameters_SR_tensor[key]).to(device)



                if not os.path.exists(f'{SR_file}/valloss_list.npy'):
                        continue   
                        
                valloss = np.load(f'{SR_file}/valloss_list.npy')

                if args.ensemble:
                    best_epochs = np.argsort(valloss)[0:10]


                    SR_ = []
                    for epoch in best_epochs:
                        best_model_file = f'{SR_file}/model_SR_{epoch}.pt'      

                        model_SR.model.load_state_dict(torch.load(best_model_file, map_location=device))
                        model_SR.model.eval()
                        model_SR.model.to(device)

                        with torch.no_grad():
                            SR = evaluate_log_prob(model_SR.model, test_tensor_S, pre_parameters_SR).cpu().detach().numpy()
                            SR_.append(SR)
                    
                    SR_ = np.array(SR_)
                    SR = np.exp(SR_)
                    SR = np.mean(SR,axis=0)
                #  SR = np.log(SR+1e-32)
                else:
                    pass

                SR_array.append(SR)

            
            SR_array = np.array(SR_array)
            SR = np.mean(SR_array,axis=0)
            SR = np.log(SR+1e-32)

            likelihood_score = (SR - mean_log_CR)
            sic , tpr , auc = SIC(label_test, likelihood_score)

            _max_SIC.append(np.max(sic))
            _AUC.append(auc)

            sic_01 = SIC_fpr(label_test, likelihood_score, 0.1)
            sic_001 = SIC_fpr(label_test, likelihood_score, 0.01)

            _SIC_01.append(sic_01)
            _SIC_001.append(sic_001)


        max_SIC_avg.append(np.mean(_max_SIC))
        max_SIC_std.append(np.std(_max_SIC))

        AUC_avg.append(np.mean(_AUC))
        AUC_std.append(np.std(_AUC))

        SIC_01_avg.append(np.mean(_SIC_01))
        SIC_01_std.append(np.std(_SIC_01))

        SIC_001_avg.append(np.mean(_SIC_001))
        SIC_001_std.append(np.std(_SIC_001))


        print(f'max_SIC_avg: {max_SIC_avg}')


# Get the likelihood ratio from m_anode   

summary_m = {}


for scan in args.scan_set:

    max_SIC_avg_m = []
    AUC_avg_m = []
    SIC_01_avg_m = []
    SIC_001_avg_m = []

    max_SIC_std_m = []
    AUC_std_m = []
    SIC_01_std_m = []
    SIC_001_std_m = []

    summary_m[scan] = {}
    summary_m[scan]['sig_train'] = n_sig_train_list

    for sig_train in n_sig_train_list:

        _max_SIC = []
        _AUC = []
        _SIC_01 = []
        _SIC_001 = []
        print(f'sig_train: {sig_train}')



        for try_ in range(tries):

            print(f'try: {try_}')

        #  print(f'try: {try_}')
 

            SR_shuffle = []

            for split in range(splits):
                print(f'split: {split}')

                model_S = flows_model_RQS(device=device)
               # model_S = DensityEstimator(args.config_file, eval_mode=True, device=device)
                SR_file = f'results/{group_name_r_anode}/{scan}_{sig_train}/try_{try_}_{split}'
                if not os.path.exists(f'{SR_file}/valloss.npy'):
                    continue

                
                with open(f'{args.CR_path}/pre_parameters.pkl', 'rb') as f:
                    pre_parameters_SR = pickle.load(f)

                x_test_SR = preprocess_params_transform(x_test_masked, pre_parameters_SR)
                test_tensor_S = torch.from_numpy(x_test_SR).float().to(device)

                pre_parameters_SR_tensor = pre_parameters_SR.copy()
                for key in pre_parameters_SR_tensor.keys():
                    pre_parameters_SR_tensor[key] = torch.tensor(pre_parameters_SR_tensor[key]).to(device)




                
                valloss = np.load(f'{SR_file}/valloss.npy')
                testloader = torch.utils.data.DataLoader(test_tensor_S, batch_size=200000, shuffle=False)

                if args.ensemble:
                    best_epoch = np.argsort(valloss)[0:10]
                    SR_epoch = []
                    for epoch in best_epoch:
                        
                        best_model_file = f'{SR_file}/model_S_{epoch}.pt'
                        model_S.load_state_dict(torch.load(best_model_file, map_location=device))
                        model_S.eval()
                        model_S.to(device)

                        with torch.no_grad():
                            log_S_data = []
                            for i, data in enumerate(testloader):
                                log_S_data.extend(model_S.log_prob(data[:,1:-1],context=data[:,0].reshape(-1,1)).cpu().detach().numpy().tolist())
                            # SR = evaluate_log_prob(model_S.model, test_tensor_S, pre_parameters_SR).cpu().detach().numpy()
                            SR_epoch.append(log_S_data)

                SR_ = np.array(SR_epoch)
                SR = np.exp(SR_)
                SR = np.mean(SR,axis=0)
                SR_shuffle.append(SR)

            SR_shuffle = np.array(SR_shuffle)
            SR = np.mean(SR_shuffle,axis=0)
            SR = np.log(SR+1e-32)            


            likelihood_score = (SR - mean_log_CR)
            likelihood_score = np.nan_to_num(likelihood_score, nan=0, posinf=0, neginf=0)
            sic , tpr , auc = SIC(label_test, likelihood_score)

            _max_SIC.append(np.max(sic))
            _AUC.append(auc)

            sic_01 = SIC_fpr(label_test, likelihood_score, 0.1)
            sic_001 = SIC_fpr(label_test, likelihood_score, 0.01)

            _SIC_01.append(sic_01)
            _SIC_001.append(sic_001)

            print(f'max_SIC: {np.max(sic)}')

        max_SIC_avg_m.append(np.mean(_max_SIC))
        max_SIC_std_m.append(np.std(_max_SIC))

        AUC_avg_m.append(np.mean(_AUC))
        AUC_std_m.append(np.std(_AUC))

        SIC_01_avg_m.append(np.mean(_SIC_01))
        SIC_01_std_m.append(np.std(_SIC_01))

        SIC_001_avg_m.append(np.mean(_SIC_001))
        SIC_001_std_m.append(np.std(_SIC_001))

    summary_m[scan]['max_SIC_avg'] = max_SIC_avg_m
    summary_m[scan]['max_SIC_std'] = max_SIC_std_m

    summary_m[scan]['AUC_avg'] = AUC_avg_m
    summary_m[scan]['AUC_std'] = AUC_std_m

    summary_m[scan]['SIC_01_avg'] = SIC_01_avg_m
    summary_m[scan]['SIC_01_std'] = SIC_01_std_m

    summary_m[scan]['SIC_001_avg'] = SIC_001_avg_m
    summary_m[scan]['SIC_001_std'] = SIC_001_std_m

    print(f'max_SIC_avg: {max_SIC_avg_m}')
    print(f'max_SIC_std: {max_SIC_std_m}')


figure = plt.figure()

#sorted = np.argsort(sig_train_list)

# fill between for std
ax1 = figure.add_subplot(111)
for scan in args.scan_set:
    ax1.errorbar(n_sig_train_list,summary_m[scan]['max_SIC_avg'],yerr=summary_m[scan]['max_SIC_std'],label=f'{scan}',fmt='o')

if args.anode:
    ax1.errorbar(n_sig_train_list,max_SIC_avg,yerr=max_SIC_std,label='ANODE',fmt='o')
ax1.set_xlabel('n_sig_train')
ax1.set_ylabel('SIC')
#ax1.set_xscale('log')
ax1.set_xticks(n_sig_train_list)
ax1.set_xticklabels(n_sig_train_list)
ax1.set_xlim(80,1020)

ax1.legend(loc = 'lower right', frameon=False)

ax2 = ax1.twiny()

ax2.set_xlabel('n_sig_train')
ax2.set_xlim(80,1020)
ax2.set_xticks(n_sig_train_list)
ax2.set_xticklabels(sig_train_list)
ax2.set_xlabel('sig_train')
if args.wandb:
    wandb.log({'SIC': wandb.Image(figure)})
 
plt.close()

figure = plt.figure()
ax1 = figure.add_subplot(111)

for scan in args.scan_set:
    ax1.errorbar(n_sig_train_list,summary_m[scan]['AUC_avg'],yerr=summary_m[scan]['AUC_std'],label=f'{scan}',fmt='o')
if args.anode:
    ax1.errorbar(n_sig_train_list,AUC_avg,yerr=AUC_std,label='ANODE',fmt='o')
ax1.set_xlabel('n_sig_train')
ax1.set_ylabel('AUC')
ax1.set_xlim(80,1020)
#ax1.set_xscale('log')
ax1.set_xticks(n_sig_train_list)
ax1.set_xticklabels(n_sig_train_list)

ax1.legend(loc = 'lower right', frameon=False)
ax2 = ax1.twiny()
ax2.set_xlim(80,1020)
ax2.set_xlabel('n_sig_train')
ax2.set_xticks(n_sig_train_list)
ax2.set_xticklabels(sig_train_list)
ax2.set_xlabel('sig_train')
if args.wandb:
    wandb.log({'AUC': wandb.Image(figure)})
 
plt.close()

figure = plt.figure()
ax1 = figure.add_subplot(111)
for scan in args.scan_set:
    ax1.errorbar(n_sig_train_list,summary_m[scan]['SIC_01_avg'],yerr=summary_m[scan]['SIC_01_std'],label=f'{scan}',fmt='o')
if args.anode:
    ax1.errorbar(n_sig_train_list,SIC_01_avg,yerr=SIC_01_std,label='ANODE',fmt='o')
ax1.set_xlabel('n_sig_train')
ax1.set_ylabel('SIC_01')
ax1.set_xlim(80,1020)

#ax1.set_xscale('log')
ax1.set_xticks(n_sig_train_list)
ax1.set_xticklabels(n_sig_train_list)

ax1.legend(loc = 'lower right', frameon=False)
ax2 = ax1.twiny()
ax2.set_xlabel('n_sig_train')
ax2.set_xlim(80,1020)

ax2.set_xticks(n_sig_train_list)
ax2.set_xticklabels(sig_train_list)
ax2.set_xlabel('sig_train')

if args.wandb:
    wandb.log({'SIC_01': wandb.Image(figure)})
 
plt.close()

figure = plt.figure()
ax1 = figure.add_subplot(111)
for scan in args.scan_set:
    ax1.errorbar(n_sig_train_list,summary_m[scan]['SIC_001_avg'],yerr=summary_m[scan]['SIC_001_std'],label=f'{scan}',fmt='o')
if args.anode:
    ax1.errorbar(n_sig_train_list,SIC_001_avg,yerr=SIC_001_std,label='ANODE',fmt='o')
ax1.set_xlabel('n_sig_train')
ax1.set_ylabel('SIC_001')
ax1.set_xlim(80,1020)

#ax1.set_xscale('log')
ax1.set_xticks(n_sig_train_list)
ax1.set_xticklabels(n_sig_train_list)

ax1.legend(loc = 'lower right', frameon=False)
ax2 = ax1.twiny()
ax2.set_xlabel('n_sig_train')
ax2.set_xlim(80,1020)

ax2.set_xticks(n_sig_train_list)
ax2.set_xticklabels(sig_train_list)
ax2.set_xlabel('sig_train')

if args.wandb:
    wandb.log({'SIC_001': wandb.Image(figure)})
 
plt.close()




figure = plt.figure()

#sorted = np.argsort(sig_train_list)

# fill between for std
ax1 = figure.add_subplot(111)
for scan in args.scan_set:
    ax1.errorbar(sig_train_list,summary_m[scan]['max_SIC_avg'],yerr=summary_m[scan]['max_SIC_std'],label=f'{scan}',fmt='o', capsize=5)

if args.anode:
    ax1.errorbar(sig_train_list,max_SIC_avg,yerr=max_SIC_std,label='ANODE',fmt='o')
if args.cathode:
    ax1.errorbar(sig_ad,sic_ad,yerr=sic_ad_std,label='ideal AD',fmt='o', capsize=5)

ax1.set_xlabel('sigma_train')
ax1.set_ylabel('SIC')
#ax1.set_xscale('log')
ax1.set_xlim(0,2.3)
ax1.set_xticks(sig_train_list)
ax1.set_xticklabels(sig_train_list)

ax1.legend(loc = 'lower right', frameon=False)

ax2 = ax1.twiny()

ax2.set_xlabel('n_sig_train')
ax2.set_xlim(0,2.3)
ax2.set_xticks(sig_train_list)
ax2.set_xticklabels(n_sig_train_list)
if args.wandb:
    wandb.log({'SIC_cathode': wandb.Image(figure)})
plt.close()


if args.wandb:
    wandb.finish()
