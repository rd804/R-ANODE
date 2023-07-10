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

parser = argparse.ArgumentParser()
parser.add_argument('--scan_set_CR', nargs='+', type=str, default='10')
parser.add_argument('--scan_set_SR', nargs='+', type=str, default='10')
parser.add_argument('--mode_background', type=str, default='false')
# pass a list in argparse

# note ANODE results come with tailbound=10


args = parser.parse_args()


print(args.scan_set_CR)
print(args.scan_set_SR)

#TODO: load best_models, get log probs, and then do logp - logq
#TODO: remove outliers and check SIC curve
#TODO: smaller nflow for CR region


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


import wandb

group_name = 'nflows_gaussian_mixture_1'

with open('data/data.pkl', 'rb') as f:
     data = pickle.load(f)

with open('data/background.pkl', 'rb') as f:
     background = pickle.load(f)

print(background.shape)

with open('data/true_w.pkl', 'rb') as f:
    true_w = pickle.load(f)



back_mean = 0
sig_mean = 3
sig_simga = 0.5
back_sigma = 3


wandb.init(project='m-anode',group=group_name, job_type=f'summary',
           config={'device':device, 'back_mean':back_mean,
                   'sig_mean':sig_mean,'sig_simga':sig_simga, 
                    'back_sigma':back_sigma, 'test_set':'10',
                    'test_sigma':100})

wandb.run.name = f'ANODE'



x_test = data['10']['val']['data']
label_test = data['10']['val']['label']
test_tensor = torch.from_numpy(x_test.astype('float32').reshape((-1,1))).to(device)


# get CR region for ANODE

CR = {}



for runs in args.scan_set_CR:
	if runs=='CR':
		model = define_model(nfeatures=1,nhidden=2,hidden_size=20,embedding=None,dropout=0,nembedding=0,device=device,tailbound=10)
	else:
		model = define_model(nfeatures=1,nhidden=2,hidden_size=20,embedding=None,dropout=0,nembedding=0,device=device,tailbound=15)


	CR_array = []
	for try_ in range(10):
		CR_file = f'results/nflows_gaussian_mixture_1/{runs}/try_{try_}'
		best_model_file = f'{CR_file}/model_CR_best.pt'

		model.load_state_dict(torch.load(best_model_file))
		model.eval()
		model.to(device)

		with torch.no_grad():
			CR_ = model.log_prob(test_tensor).cpu().detach().numpy()

		CR_p = np.exp(CR_)
		CR_array.append(CR_p)

	CR_array = np.array(CR_array)
	mean_CR = np.mean(CR_array,axis=0)
		# replace 0 with a small number
	mean_CR[mean_CR==0] = 1e-10

	mean_log_CR = np.log(mean_CR)
	CR[runs] = mean_log_CR	


sig_train_list = [0.1, 0.2, 0.5, 0.8, 0.9, 1, 1.5, 2, 5 , 10]
	
_true_max_SIC = []
_true_AUC = []
_true_SIC_01 = []
_true_SIC_001 = []
	
for sig_train in sig_train_list:

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



ANODE_file = f'results/nflows_gaussian_mixture_1'

SR = {}
# get SR region for ANODE
for runs in args.scan_set_SR:

	SR[runs] = {}

	if runs=='SR':
		model = define_model(nfeatures=1,nhidden=2,hidden_size=20,embedding=None,dropout=0,nembedding=0,device=device,tailbound=10)
	else:
		model = define_model(nfeatures=1,nhidden=2,hidden_size=20,embedding=None,dropout=0,nembedding=0,device=device,tailbound=15)

	for sig_train in sig_train_list:
		SR[runs][str(sig_train)] = {}

		_max_SIC = []
		_AUC = []
		_SIC_01 = []
		_SIC_001 = []

		for try_ in range(10):
			SR_file = f'results/nflows_gaussian_mixture_1/{runs}_{sig_train}/try_{try_}'

			if not os.path.exists(f'{SR_file}/valloss_list.npy'):
				continue   

			valloss = np.load(f'{SR_file}/valloss_list.npy')
			best_epoch = np.argmin(valloss)
			best_model_file = f'{SR_file}/model_SR_{best_epoch}.pt'      



			model.load_state_dict(torch.load(best_model_file))
			model.eval()
			model.to(device)

			with torch.no_grad():
				SR_ = model.log_prob(test_tensor).cpu().detach().numpy()

			SR[runs][str(sig_train)][str(try_)] = SR_


summary = {}
for CR_runs in args.scan_set_CR:
	summary[CR_runs] = {}
	mean_log_CR = CR[CR_runs]

	for SR_runs in args.scan_set_SR:
		summary[CR_runs][SR_runs] = {}
		summary[CR_runs][SR_runs]['max_SIC'] = []
		summary[CR_runs][SR_runs]['AUC'] = []
		summary[CR_runs][SR_runs]['SIC_01'] = []
		summary[CR_runs][SR_runs]['SIC_001'] = []

		summary[CR_runs][SR_runs]['max_SIC_std'] = []
		summary[CR_runs][SR_runs]['AUC_std'] = []
		summary[CR_runs][SR_runs]['SIC_01_std'] = []
		summary[CR_runs][SR_runs]['SIC_001_std'] = []

		
		for sig_train in sig_train_list:
			
			_max_SIC = []
			_AUC = []
			_SIC_01 = []
			_SIC_001 = []


			for try_ in range(10):
				SR_score = SR[SR_runs][str(sig_train)][str(try_)]
				likelihood_score = (SR_score - mean_log_CR)
				sic_01 = SIC_fpr(label_test, likelihood_score, 0.1)
				sic_001 = SIC_fpr(label_test, likelihood_score, 0.01)

				sic , tpr , auc = SIC(label_test, likelihood_score)

				_max_SIC.append(np.max(sic))
				_AUC.append(auc)
				_SIC_01.append(sic_01)
				_SIC_001.append(sic_001)
			
			summary[CR_runs][SR_runs]['max_SIC'].append(np.mean(_max_SIC))
			summary[CR_runs][SR_runs]['AUC'].append(np.mean(_AUC))
			summary[CR_runs][SR_runs]['SIC_01'].append(np.mean(_SIC_01))
			summary[CR_runs][SR_runs]['SIC_001'].append(np.mean(_SIC_001))

			summary[CR_runs][SR_runs]['max_SIC_std'].append(np.std(_max_SIC))
			summary[CR_runs][SR_runs]['AUC_std'].append(np.std(_AUC))
			summary[CR_runs][SR_runs]['SIC_01_std'].append(np.std(_SIC_01))
			summary[CR_runs][SR_runs]['SIC_001_std'].append(np.std(_SIC_001))





figure = plt.figure()
for CR_run in args.scan_set_CR:
    for SR_run in args.scan_set_SR:
    	plt.errorbar(sig_train_list,summary[CR_runs][SR_runs]['max_SIC'],yerr=summary[CR_runs][SR_runs]['max_SIC_std'],label=f'CR:{CR_run} SR:{SR_run}',fmt='o')
plt.plot(sig_train_list,_true_max_SIC,'o',label='true max SIC', color='C2')
plt.xlabel('sig_train')
plt.ylabel('max SIC')
plt.xscale('log')
plt.legend(loc = 'lower right', frameon=False)
wandb.log({'max_SIC': wandb.Image(figure)})
plt.show()

figure = plt.figure()
for CR_run in args.scan_set_CR:
	for SR_run in args.scan_set_SR:
		plt.errorbar(sig_train_list,summary[CR_runs][SR_runs]['AUC'],yerr=summary[CR_runs][SR_runs]['AUC_std'],label=f'CR:{CR_run} SR:{SR_run}',fmt='o')
plt.plot(sig_train_list,_true_AUC,'o',label='true AUC', color='C2')
plt.xlabel('sig_train')
plt.ylabel('AUC')
plt.legend()
plt.xscale('log')
plt.legend(loc = 'lower right', frameon=False)
wandb.log({'AUC': wandb.Image(figure)})
plt.show()

figure = plt.figure()
for CR_run in args.scan_set_CR:
	for SR_run in args.scan_set_SR:
		plt.errorbar(sig_train_list,summary[CR_runs][SR_runs]['SIC_01'],yerr=summary[CR_runs][SR_runs]['SIC_01_std'],label=f'CR:{CR_run} SR:{SR_run}',fmt='o')
plt.plot(sig_train_list,_true_SIC_01,'o',label='true')
plt.xlabel('sig_train')
plt.ylabel('SIC_01')
plt.legend()
plt.xscale('log')
plt.legend(loc = 'lower right', frameon=False)
wandb.log({'SIC_01': wandb.Image(figure)})
plt.show()

figure = plt.figure()
for CR_run in args.scan_set_CR:
	for SR_run in args.scan_set_SR:
		plt.errorbar(sig_train_list,summary[CR_runs][SR_runs]['SIC_001'],yerr=summary[CR_runs][SR_runs]['SIC_001_std'],label=f'CR:{CR_run} SR:{SR_run}',fmt='o')
plt.plot(sig_train_list,_true_SIC_001,'o',label='true')
plt.xlabel('sig_train')
plt.ylabel('SIC_001')
plt.legend()
plt.xscale('log')
plt.legend(loc = 'lower right', frameon=False)
wandb.log({'SIC_001': wandb.Image(figure)})
plt.show()

wandb.finish()

