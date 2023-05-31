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



#TODO: load best_models, get log probs, and then do logp - logq
#TODO: remove outliers and check SIC curve
#TODO: smaller nflow for CR region

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

sig_train = 5

group_name = 'nflows_gaussian_mixture_1'
job_name = f'SR_{sig_train}'

with open('data/data.pkl', 'rb') as f:
     data = pickle.load(f)

with open('data/true_w.pkl', 'rb') as f:
    true_w = pickle.load(f)

back_mean = 0
sig_mean = 3
sig_simga = 0.5
back_sigma = 3

x_train = data[str(sig_train)]['train']['data']
x_train = shuffle(x_train, random_state=10)

x_train , x_val = train_test_split(x_train, test_size=0.5, random_state=22)


x_test = data[str(sig_train)]['val']['data']
label_test = data[str(sig_train)]['val']['label']


bins = np.linspace(min(x_test), max(x_test), 100)


model=define_model(nfeatures=1,nhidden=2,hidden_size=20,embedding=None,dropout=0,nembedding=0,device=device)
print(model)

CR = []
for i in range(10):
    try_no = 'try_'+str(i)
    print(i)
    # load best model
    model_file = f'results/{group_name}/CR/{try_no}/model_CR_best.pt'
    model.load_state_dict(torch.load(model_file))
    model.eval()
    model.to(device)
    density_estimator = np.exp(model.log_prob(torch.tensor(bins).float().to(device)).detach().cpu().numpy())

    plt.step(bins,density_estimator,label='CR')
    plt.hist(x_test,bins=bins,label='test',density=True,histtype='step')
    plt.hist(x_train,bins=bins,label='train',density=True,histtype='step')
    plt.legend()
    plt.savefig(f'results/{group_name}/CR/{try_no}/nflows_density.png')
    plt.close()

    CR.append(density_estimator)

CR = np.array(CR)
CR_mean = np.mean(CR, axis=0)

plt.step(bins,CR_mean,label='CR')
plt.hist(x_test,bins=bins,label='test',density=True,histtype='step')
plt.hist(x_train,bins=bins,label='train',density=True,histtype='step')
plt.legend()
plt.savefig(f'results/{group_name}/CR/average_density.png')
plt.close()