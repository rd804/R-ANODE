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
parser.add_argument('--scan_set', nargs='+', type=str, default='10')
parser.add_argument('--anode_set', type=str, default='SR')
parser.add_argument('--mode_background', type=str, default='false')
parser.add_argument('--gaussian_dim',type=int,default=1)
parser.add_argument('--ensemble', action='store_true')
# pass a list in argparse

# note ANODE results come with tailbound=10


args = parser.parse_args()


#TODO: load best_models, get log probs, and then do logp - logq
#TODO: remove outliers and check SIC curve
#TODO: smaller nflow for CR region


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


import wandb

group_name = 'nflows_gaussian_mixture_1'

if args.gaussian_dim == 1:
    with open('data/data.pkl', 'rb') as f:
        data = pickle.load(f)

    with open('data/background.pkl', 'rb') as f:
        background = pickle.load(f)

    print(background.shape)

    with open('data/true_w.pkl', 'rb') as f:
        true_w = pickle.load(f)

    back_mean = 0
    sig_mean = 3
    sig_sigma = 0.5
    back_sigma = 3

else:
    with open(f'data/data_{args.gaussian_dim}d.pkl', 'rb') as f:
        data = pickle.load(f)

    with open(f'data/background_{args.gaussian_dim}d.pkl', 'rb') as f:
        background = pickle.load(f)

    print(background.shape)

    with open(f'data/true_w_{args.gaussian_dim}d.pkl', 'rb') as f:
        true_w = pickle.load(f)

    back_mean = 0
    sig_mean = 2
    sig_sigma = 0.25
    back_sigma = 3


wandb.init(project='m-anode',group=group_name, job_type=f'summary',
           config={'device':device, 'back_mean':back_mean,
                   'sig_mean':sig_mean,'sig_sigma':sig_sigma, 
                    'back_sigma':back_sigma, 'test_set':'10',
                    'test_sigma':100, 'mode_background':args.mode_background,
                    'gaussian_dim':args.gaussian_dim})

wandb.run.name = f'M_ANODE_{args.scan_set}'



x_test = data['10']['val']['data']
label_test = data['10']['val']['label']
test_tensor = torch.from_numpy(x_test.astype('float32').reshape((-1,args.gaussian_dim))).to(device)



# get CR region for ANODE
if args.mode_background != 'true':
    model = define_model(nfeatures=1,nhidden=2,hidden_size=20,embedding=None,dropout=0,nembedding=0,device=device,
                        tailbound=10)

    CR_array = []
    for try_ in range(10):
        CR_file = f'results/nflows_gaussian_mixture_1/CR/try_{try_}'
        best_model_file = f'{CR_file}/model_CR_best.pt'

        model.load_state_dict(torch.load(best_model_file))
        model.eval()
        model.to(device)

        with torch.no_grad():
            CR = model.log_prob(test_tensor).cpu().detach().numpy()

        CR_p = np.exp(CR)
        CR_array.append(CR_p)

    CR_array = np.array(CR_array)
    mean_CR = np.mean(CR_array,axis=0)
    # replace 0 with a small number
    mean_CR[mean_CR==0] = 1e-10

    mean_log_CR = np.log(mean_CR)
else:
    model_B = gaussian_prob(back_mean,back_sigma, dim = args.gaussian_dim)

    mean_log_CR = model_B.log_prob(torch.from_numpy(x_test)).numpy().flatten()


sig_train_list = [0.1, 0.2, 0.5, 0.8, 0.9, 1, 1.5, 2, 5 , 10]


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


ANODE_file = f'results/nflows_gaussian_mixture_1'

# get SR region for ANODE

for sig_train in sig_train_list:
    _max_SIC = []
    _AUC = []
    _SIC_01 = []
    _SIC_001 = []
  #  print(f'sig_train: {sig_train}')



    w1 = true_w[str(sig_train)][0]
    w2 = true_w[str(sig_train)][1]
    true_likelihood = p_data(x_test,[sig_mean, back_mean],[sig_sigma,back_sigma],
                [w1, w2], dim=args.gaussian_dim)/p_back(x_test,back_mean, back_sigma, dim=args.gaussian_dim)

    sic_true , tpr_true , auc_true = SIC(label_test, true_likelihood)
    sic_true_01 = SIC_fpr(label_test, true_likelihood, 0.1)
    sic_true_001 = SIC_fpr(label_test, true_likelihood, 0.01)

    _true_max_SIC.append(np.max(sic_true))
    _true_AUC.append(auc_true)
    _true_SIC_01.append(sic_true_01)
    _true_SIC_001.append(sic_true_001)


    for try_ in range(10):

        model = flows_for_gaussian(gaussian_dim = args.gaussian_dim, num_transforms = 2, num_blocks = 3, 
                       hidden_features = 32, device = device)

        SR_file = f'results/nflows_gaussian_mixture_1/{args.anode_set}_{sig_train}/try_{try_}'

        if not os.path.exists(f'{SR_file}/valloss_list.npy'):
                continue   
                 
        valloss = np.load(f'{SR_file}/valloss_list.npy')

        if args.ensemble:
            best_epochs = np.argsort(valloss)[0:10]


            SR_ = []
            for epoch in best_epochs:
                best_model_file = f'{SR_file}/model_SR_{epoch}.pt'      



                model.load_state_dict(torch.load(best_model_file))
                model.eval()
                model.to(device)

                with torch.no_grad():
                    SR = model.log_prob(test_tensor).cpu().detach().numpy()
                    SR_.append(SR)
            
            SR_ = np.array(SR_)
            SR = np.exp(SR_)
            SR = np.mean(SR,axis=0)
            SR = np.log(SR+1e-32)
        else:
            pass


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
    summary_m[scan]['sig_train'] = sig_train_list

    for sig_train in sig_train_list:

        _max_SIC = []
        _AUC = []
        _SIC_01 = []
        _SIC_001 = []
        print(f'sig_train: {sig_train}')



        for try_ in range(10):

        #  print(f'try: {try_}')
            model = flows_for_gaussian(gaussian_dim = args.gaussian_dim, num_transforms = 2, num_blocks = 3, 
                       hidden_features = 32, device = device)            
            
            SR_file = f'results/nflows_gaussian_mixture_1/{scan}_{sig_train}/try_{try_}'
            #SR_file = f'results/nflows_gaussian_mixture_1/{scan}_{sig_train}/try_{try_}'
            #SR_file = f'results/nflows_gaussian_mixture_1/m_anode_true_likelihood_b_freeze_{sig_train}/try_{try_}'

            if not os.path.exists(f'{SR_file}/valloss.npy'):
                continue
            
            valloss = np.load(f'{SR_file}/valloss.npy')


            if args.ensemble:
                best_epoch = np.argsort(valloss)[0:10]
                SR_ = []
                for epoch in best_epoch:
                    best_model_file = f'{SR_file}/model_S_{epoch}.pt'
                    model.load_state_dict(torch.load(best_model_file))
                    model.eval()
                    model.to(device)

                    with torch.no_grad():
                        SR = model.log_prob(test_tensor).cpu().detach().numpy()
                        SR_.append(SR)

            SR_ = np.array(SR_)
            SR = np.exp(SR_)
            SR = np.mean(SR,axis=0)
            SR = np.log(SR+1e-32)            


            likelihood_score = (SR - mean_log_CR)
            sic , tpr , auc = SIC(label_test, likelihood_score)

            _max_SIC.append(np.max(sic))
            _AUC.append(auc)

            sic_01 = SIC_fpr(label_test, likelihood_score, 0.1)
            sic_001 = SIC_fpr(label_test, likelihood_score, 0.01)

            _SIC_01.append(sic_01)
            _SIC_001.append(sic_001)

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




figure = plt.figure()

#sorted = np.argsort(sig_train_list)

# fill between for std

for scan in args.scan_set:
    plt.errorbar(sig_train_list,summary_m[scan]['max_SIC_avg'],yerr=summary_m[scan]['max_SIC_std'],label=f'{scan}',fmt='o')

plt.errorbar(sig_train_list,max_SIC_avg,yerr=max_SIC_std,label='ANODE',fmt='o')
plt.plot(sig_train_list,_true_max_SIC,'o',label='true max SIC', color='C2')
plt.xlabel('sig_train')
plt.ylabel('max SIC')
plt.xscale('log')
plt.legend(loc = 'lower right', frameon=False)
wandb.log({'max_SIC': wandb.Image(figure)})
#plt.close()
plt.show()

figure = plt.figure()

for scan in args.scan_set:
    plt.errorbar(sig_train_list,summary_m[scan]['AUC_avg'],yerr=summary_m[scan]['AUC_std'],label=f'{scan}',fmt='o')
plt.errorbar(sig_train_list,AUC_avg,yerr=AUC_std,label='ANODE',fmt='o')
plt.plot(sig_train_list,_true_AUC,'o',label='true AUC', color='C2')
plt.xlabel('sig_train')
plt.ylabel('AUC')
plt.legend()
plt.xscale('log')
plt.legend(loc = 'lower right', frameon=False)
wandb.log({'AUC': wandb.Image(figure)})
#plt.close()
plt.show()

figure = plt.figure()
for scan in args.scan_set:
    plt.errorbar(sig_train_list,summary_m[scan]['SIC_01_avg'],yerr=summary_m[scan]['SIC_01_std'],label=f'{scan}',fmt='o')
plt.errorbar(sig_train_list,SIC_01_avg,yerr=SIC_01_std,label='ANODE',fmt='o')
plt.plot(sig_train_list,_true_SIC_01,'o',label='true')
plt.xlabel('sig_train')
plt.ylabel('SIC_01')
plt.legend()
plt.xscale('log')
plt.legend(loc = 'lower right', frameon=False)
wandb.log({'SIC_01': wandb.Image(figure)})
plt.show()

figure = plt.figure()
for scan in args.scan_set:
    plt.errorbar(sig_train_list,summary_m[scan]['SIC_001_avg'],yerr=summary_m[scan]['SIC_001_std'],label=f'{scan}',fmt='o')
plt.errorbar(sig_train_list,SIC_001_avg,yerr=SIC_001_std,label='ANODE',fmt='o')
plt.plot(sig_train_list,_true_SIC_001,'o',label='true')
plt.xlabel('sig_train')
plt.ylabel('SIC_001')
plt.legend()
plt.xscale('log')
plt.legend(loc = 'lower right', frameon=False)
wandb.log({'SIC_001': wandb.Image(figure)})
plt.show()

wandb.finish()

