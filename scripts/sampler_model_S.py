import numpy as np
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm
from src.nflow_utils import *
import os
from src.utils import *
from src.nflow_utils import *
from src.generate_data_lhc import *
from src.utils import *
from src.flows import *
import glob

from nflows import transforms, distributions, flows
import torch
import torch.nn.functional as F
from nflows.distributions import uniform
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split, ShuffleSplit
import argparse
import pickle
import wandb
import sys

wandb_job_type = "try"
data_dir = "data/lhc_co"
plot = False

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
CR_path = 'results/nflows_lhc_co/CR_bn_fixed_1000/try_1_0'
with open(f'{CR_path}/pre_parameters.pkl', 'rb') as f:
    pre_parameters = pickle.load(f)

# convert to torch tensors
for key in pre_parameters.keys():
    pre_parameters[key] = torch.tensor(pre_parameters[key]).float().to(device)

# Load model_B
config_file = f'scripts/DE_MAF_model.yml'
val_losses = np.load(f'{CR_path}/valloss_list.npy')
best_epochs = np.argsort(val_losses)[0] 
model_B = DensityEstimator(config_file, eval_mode=True, load_path=f"{CR_path}/model_CR_{best_epochs}.pt", device=device)

x_test = np.load(f'{data_dir}/x_test.npy')
label = x_test[:,-1]


#nsigs = [1000,600,500,450,300,225,150]
sigmas = [2.166,1.294,1.076,0.9714,0.6499,0.4866,0.3247]
color = ['C0','C1','C2','C3','C4','C5','C6']

model_S = flows_model_RQS(device=device)

# import kernel density estimator
from sklearn.neighbors import KernelDensity
SR_data, CR_data , true_w, sigma = resample_split(data_dir, n_sig = 1000, resample_seed = 0,resample = True)

#mass_SR = x_test[:,0][label==1]
mass_SR = SR_data[:,0]
#mass_CR = CR_data[:,0]

# fit KDE to mass_SR
kde_SR = KernelDensity(kernel='gaussian', bandwidth=0.01).fit(mass_SR.reshape(-1,1))
#kde_CR = KernelDensity(kernel='gaussian', bandwidth=0.01).fit(mass_CR.reshape(-1,1))

# generate samples from KDE


#sample_dict['CR'] = {}
for try_ in range(1):
    print('try', try_)
    sample_dict = {}
    log_prob_dict = {}

    sample_dict['SR'] = {}
    log_prob_dict['S'] = {}
    log_prob_dict['B'] = {}

    #for w_ in [0.006, 0.1,0.001,0.0001,0.009]:
    for w_ in [0.006]:
        if w_ != 0.006:
            wandb_group = "nflows_lhc_co_w_scan"
        else:
            wandb_group = "nflows_lhc_co_nsig_scan"

        for nsig in [1000]:
            if w_ != 0.006:
                wandb_project = f"r_anode_R_A_{nsig}_{w_}"
            else:
              #  wandb_project = f"r_anode_RQS_affine_{nsig}"
                wandb_project = f"ra_mass_{nsig}"
            
            sample_S = []
            log_prob_S = []
            log_prob_B = []
        # sample_C = []
            

            for shuffles in range(20):
                path = f'results/{wandb_group}/{wandb_project}/{wandb_job_type}_{try_}_{shuffles}'
                list_of_files = glob.glob(f'{path}/model_S_*.pt') # * means all if need specific format then *.csv

            #   print(shuffles)
                for model_path in list_of_files:
                    mass = kde_SR.sample(1000).reshape(-1,1)
                    mass_SR_samples = torch.tensor(mass).float().to(device)
                # mass_CR_samples = torch.tensor(kde_CR.sample(1000).reshape(-1,1)).float().to(device)
                    

                    model_S.load_state_dict(torch.load(model_path, map_location=device))

                    model_S.eval()
                    x_samples_SR,log_prob_S_SR = model_S.sample_and_log_prob(1, mass_SR_samples)
                # x_samples_CR = model_S.sample(1, mass_CR_samples).detach().cpu().reshape(-1,4)
                    x_samples_SR = x_samples_SR.detach().cpu().reshape(-1,4)
                    log_prob_S_SR = log_prob_S_SR.detach().cpu().reshape(-1,1)
                    log_prob_B_SR = model_B.model.log_probs(inputs=x_samples_SR, cond_inputs=mass_SR_samples).detach().cpu().reshape(-1,1)

                    x_samples_SR = inverse_standardize(x_samples_SR, pre_parameters["mean"], pre_parameters["std"])
                    x_samples_SR = inverse_logit_transform(x_samples_SR, pre_parameters["min"], pre_parameters["max"])

                    x_samples_SR = np.hstack((mass, x_samples_SR))
                    x_samples_SR = x_samples_SR[~np.isnan(x_samples_SR).any(axis=1)]

                # x_samples_CR = inverse_standardize(x_samples_CR, pre_parameters["mean"], pre_parameters["std"])
                # x_samples_CR = inverse_logit_transform(x_samples_CR, pre_parameters["min"], pre_parameters["max"])
                # x_samples_CR = x_samples_CR[~np.isnan(x_samples_CR).any(axis=1)]
                # sample_C.append(x_samples_CR)
                    sample_S.append(x_samples_SR)
                    log_prob_S.append(log_prob_S_SR)
                    log_prob_B.append(log_prob_B_SR)

          #  x_samples_SR = np.array(sample_S)
          #  log_prob_S_SR = np.array(log_prob_S)
          #  log_prob_B_SR = np.array(log_prob_B)
            x_samples_SR = np.concatenate(sample_S)
            log_prob_S_SR = np.concatenate(log_prob_S)
            log_prob_B_SR = np.concatenate(log_prob_B)
        # x_samples_CR = np.concatenate(sample_C)
            print(x_samples_SR.shape)
            print(log_prob_S_SR.shape)
            print(log_prob_B_SR.shape)
            sample_dict['SR'][f'{w_}_{nsig}'] = x_samples_SR
            log_prob_dict['S'][f'{w_}_{nsig}'] = log_prob_S_SR
            log_prob_dict['B'][f'{w_}_{nsig}'] = log_prob_B_SR

    with open(f'figures/sample_dict_SR_mass_true_m_1000_{try_}.pkl', 'wb') as f:
        pickle.dump(sample_dict, f)
    
    with open(f'figures/log_prob_dict_SR_mass_true_m_1000_{try_}.pkl', 'wb') as f:
        pickle.dump(log_prob_dict, f)
      #  sample_dict['CR'][f'{w_}_{nsig}'] = x_samples_CR




if plot:
    for i in range(4):
        for key in sample_dict['SR'].keys():
            plt.hist(sample_dict['SR'][key][:,i], bins = 100, histtype='step', label = f'feature {i}, {key}',density = True)
    # plt.hist(x_samples_SR[:,i], bins = 100, histtype='step', label = f'feature sample {i}',density = True)
        plt.hist(x_test[:,i+1][label==1], bins = 100, label = f'signal {i}',density = True, color = 'k')
        plt.legend()
        plt.savefig(f'figures/feature_SR{i}.png')
        plt.close()

        #for key in sample_dict['CR'].keys():
        #   plt.hist(sample_dict['CR'][key][:,i], bins = 100, histtype='step', label = f'feature {i}, {key}',density = True)
        plt.hist(x_test[:,i+1][label==1], bins = 100, histtype='step', label = f'signal {i}',density = True)
        plt.hist(x_test[:,i+1][label==0], bins = 100, histtype='step', label = f'background {i}',density = True)
        plt.legend()
        plt.savefig(f'figures/feature_CR{i}.png')
        plt.close()




