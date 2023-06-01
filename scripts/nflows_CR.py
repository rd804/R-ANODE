# %%
import numpy as np
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm
from src.nflow_utils import *
import os

# %%
from nflows import transforms, distributions, flows
import torch
import torch.nn.functional as F
from nflows.distributions import uniform
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import argparse
import wandb

# %%
import pickle

# %%
#os.environ["CUDA_VISIBLE_DEVICES"]='2'

parser = argparse.ArgumentParser()
parser.add_argument('--epochs', type=int, default=100)
parser.add_argument('--try_', type=int, default=0)
parser.add_argument('--sig_train', type=int, default=10)
parser.add_argument('--wandb_group', type=str, default='test')
parser.add_argument('--wandb_job_type', type=str, default='CR')
parser.add_argument('--wandb_run_name', type=str, default='try_')


args = parser.parse_args()


CUDA = True
device = torch.device("cuda:0" if CUDA else "cpu")


# initialize wandb
wandb.init(project="m-anode", config=args,
           group=args.wandb_group, job_type=args.wandb_job_type)

name = args.wandb_run_name+str(args.try_)
wandb.run.name = name




save_path = 'results/'+args.wandb_group+'/'\
            +args.wandb_job_type+'/'+name+'/'

if not os.path.exists(save_path):
    os.makedirs(save_path)

kwargs = {'num_workers': 4, 'pin_memory': True} if CUDA else {}
kwargs = {}

print(device)


# load data
with open('data/data.pkl', 'rb') as f:
     data = pickle.load(f)

back_mean = 0
sig_mean = 3
sig_simga = 0.5
back_sigma = 3

wandb.config.update({'back_mean': back_mean, 'sig_mean': sig_mean,
                        'sig_sigma': sig_simga, 'back_sigma': back_sigma})

# load background
with open('data/background.pkl', 'rb') as f:
    background = pickle.load(f)

sig_train = args.sig_train
# load test set
x_test = data[str(sig_train)]['val']['data']


# split background
background_train , background_val = train_test_split(background, test_size=0.5, random_state=22)

# load to torch dataloaders
background_train_tensor = torch.from_numpy(background_train.astype('float32').reshape((-1,1)))
background_val_tensor = torch.from_numpy(background_val.astype('float32').reshape((-1,1)))
test_tensor = torch.from_numpy(x_test.astype('float32').reshape((-1,1))).to(device)

batch_size = 256
background_train_loader = torch.utils.data.DataLoader(background_train_tensor, batch_size=batch_size, shuffle=True)
background_val_loader = torch.utils.data.DataLoader(background_val_tensor, batch_size=batch_size*6, shuffle=False)
test_loader = torch.utils.data.DataLoader(test_tensor, batch_size=batch_size*6, shuffle=False)


# define normalizing flow model
model_background=define_model(nfeatures=1,nhidden=2,hidden_size=5,
                              embedding=None,dropout=0,nembedding=0
                              ,device=device)


valloss_list_background=[]
trainloss_list_background=[]
optimizer = torch.optim.Adam(model_background.parameters(),lr=1e-4) #,lr=1e-4)#, lr=1e-4)

for epoch in range(args.epochs):
#     print('\n Epoch: {}'.format(epoch))
    trainloss= train(model_background,optimizer,background_train_loader ,noise_data=0.,noise_context=0.0
                     ,device=device)
    valloss=val(model_background,background_val_loader, device=device)
#     valloss=val(model_background,background_val_loader)
#    # print('epoch '+str(epoch)+' val loss: ',valloss)
#     # save model each epoch
    torch.save(model_background.state_dict(), save_path+'model_CR_'+str(epoch)+'.pt')

    valloss_list_background.append(valloss)
    trainloss_list_background.append(trainloss)
    wandb.log({'train_loss_background': trainloss, 'val_loss_background': valloss,
               'epoch': epoch})
# print('done')


# save loss lists
valloss_list_background=np.array(valloss_list_background)
trainloss_list_background=np.array(trainloss_list_background)

np.save(save_path+'trainloss_list_background.npy', trainloss_list_background)
np.save(save_path+'valloss_list_background.npy', valloss_list_background)

# load best model
min_epoch=np.argmin(valloss_list_background)
print('min epoch CR: ',min_epoch)

model_background.load_state_dict(torch.load(save_path+'model_CR_'+str(min_epoch)+'.pt'))
torch.save(model_background.state_dict(), save_path+'model_CR_best.pt')

bins = np.linspace(min(x_test), max(x_test), 50)

model_background.eval()
with torch.no_grad():
        samples_background=model_background.sample(50000)
        density_background = np.exp(model_background.log_prob(torch.from_numpy(bins.astype('float32').reshape((-1,1))).to(device)).cpu().detach().numpy())

samples_background=samples_background.cpu().detach().numpy().reshape((-1))

# plot density estimation
figure=plt.figure()
_=plt.hist(samples_background,bins=50, density=True, histtype='step', label='nflow sample')
_=plt.hist(background_val,bins=50, density=True, histtype='step', label='valdata')
_=plt.hist(background_train,bins=50, density=True, histtype='step', label='traindata')
_=plt.plot(bins,density_background,label='nflow density')
plt.legend(loc='upper right')
plt.savefig(save_path+'nflow_CR.png')

wandb.log({'nflow_CR': wandb.Image(figure)})
plt.close()

# save scores
model_background.eval()
with torch.no_grad():
    log_p = model_background.log_prob(test_tensor).cpu().detach().numpy()
    p = np.exp(log_p)

np.save(save_path+'best_val_loss_scores.npy', p)



wandb.finish()




