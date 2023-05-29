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
# import train_test_split
from sklearn.model_selection import train_test_split
import argparse
import wandb

# %%
import pickle

# %%
#os.environ["CUDA_VISIBLE_DEVICES"]='2'

parser = argparse.ArgumentParser()
parser.add_argument('--sig_train', type=int, default=10)
parser.add_argument('--try_', type=int, default=0)


args = parser.parse_args()


CUDA = True
device = torch.device("cuda:0" if CUDA else "cpu")


# initialize wandb
wandb.init(project="m-anode", config=args)
wandb.group = 'test'
wandb.job_type = 'sig_train_'+str(args.sig_train)
wandb.run.name = 'try_'+str(args.try_)



kwargs = {'num_workers': 4, 'pin_memory': True} if CUDA else {}
kwargs = {}

print(device)
# %%

# %%
with open('data/data.pkl', 'rb') as f:
     data = pickle.load(f)

back_mean = 0
sig_mean = 3
sig_simga = 0.5
back_sigma = 3


with open('data/background.pkl', 'rb') as f:
     background = pickle.load(f)


# # fit train data
best_parameters = {}
sig_train = args.sig_train
# sig_train = 10

# best_parameters[str(sig_train)] = {}
x_train = data[str(sig_train)]['train']['data']
x_train = shuffle(x_train, random_state=10)

x_train , x_val = train_test_split(x_train, test_size=0.5, random_state=22)
x_test = data[str(sig_train)]['val']['data']
labels_test = data[str(sig_train)]['val']['labels']

# # %%
# plt.hist(x_train, bins=100, alpha=0.5, label='train', density=True, histtype='step')
# plt.hist(x_val, bins=100, alpha=0.5, label='val', density=True, histtype='step')
# plt.show()

# # %%
traintensor = torch.from_numpy(x_train.astype('float32').reshape((-1,1)))
# #traindataset = torch.utils.data.TensorDataset(traintensor)

valtensor = torch.from_numpy(x_val.astype('float32').reshape((-1,1)))
# #valdataset = torch.utils.data.TensorDataset(valtensor)

testtensor = torch.from_numpy(x_test.astype('float32').reshape((-1,1)))

# # %%
# # Use the standard pytorch DataLoader
batch_size = 256
trainloader = torch.utils.data.DataLoader(traintensor, batch_size=batch_size, shuffle=True)

test_batch_size=batch_size*5
valloader = torch.utils.data.DataLoader(valtensor, batch_size=test_batch_size, shuffle=False)
testloader = torch.utils.data.DataLoader(testtensor, batch_size=test_batch_size, shuffle=False)


# # %%
model=define_model(nfeatures=1,nhidden=2,hidden_size=20,embedding=None,dropout=0,nembedding=0)

# # %%
# # train
if not os.path.exists('results/'+wandb.group+'/'+
                      wandb.job_type+'/'+wandb.run.name+'/'):
    os.makedirs('results/'+wandb.group+'/'+
                wandb.job_type+'/'+wandb.run.name+'/')
    
save_path = 'results/'+wandb.group+'/'+wandb.job_type+'/'+wandb.run.name+'/'

trainloss_list=[]
valloss_list=[]

optimizer = torch.optim.Adam(model.parameters(),lr=1e-4) #,lr=1e-4)#, lr=1e-4)

for epoch in range(100):
  #   print('\n Epoch: {}'.format(epoch))
    trainloss= train(model,optimizer,trainloader,noise_data=0.,noise_context=0.0)
    valloss=val(model,valloader)
#     print('epoch '+str(epoch)+' val loss: ',valloss)
#     # save model each epoch
    torch.save(model.state_dict(), save_path+'model_SR_'+str(epoch)+'.pt')
#     torch.save(model.state_dict(), 'models/model_'+str(sig_train)+'/model_SR_'+str(epoch)+'.pt')

    valloss_list.append(valloss)
    trainloss_list.append(trainloss)
    wandb.log({'train_loss': trainloss, 'val_loss': valloss, 'epoch_SR': epoch})

trainloss_list=np.array(trainloss_list)
valloss_list=np.array(valloss_list)
np.save(save_path+'trainloss_list.npy', trainloss_list)
np.save(save_path+'valloss_list.npy', valloss_list)

# # %%
valloss_list=np.array(valloss_list)
min_epoch=np.argmin(valloss_list)
print('min epoch SR: ',min_epoch)

# # %%
# plt.plot(valloss_list)
# plt.show()

# # %%
model.load_state_dict(torch.load('models/model_'+str(sig_train)+'/model_SR_'+str(min_epoch)+'.pt'))
torch.save(model.state_dict(), save_path+'model_SR_best.pt')
model.eval()
with torch.no_grad():
     samples=model.sample(50000)
samples=samples.cpu().detach().numpy().reshape((-1))

# # %%
figure=plt.figure()
_=plt.hist(samples,bins=50, density=True, histtype='step', label='nflow')
_=plt.hist(x_val,bins=50, density=True, histtype='step', label='valdata')
_=plt.hist(x_train,bins=50, density=True, histtype='step', label='traindata')
plt.legend(loc='upper right')
plt.savefig(save_path+'nflow_SR.png')

wandb.log({'nflow_SR': wandb.Image(figure)})
plt.close()




# # %%
background_train , background_val = train_test_split(background, test_size=0.5, random_state=22)

# # %%
background_train_tensor = torch.from_numpy(background_train.astype('float32').reshape((-1,1)))
background_val_tensor = torch.from_numpy(background_val.astype('float32').reshape((-1,1)))

# # %%


# # %%
batch_size = 256
background_train_loader = torch.utils.data.DataLoader(background_train_tensor, batch_size=batch_size, shuffle=True)
background_val_loader = torch.utils.data.DataLoader(background_val_tensor, batch_size=batch_size*6, shuffle=False)


# # %%
model_background=define_model(nfeatures=1,nhidden=2,hidden_size=20,embedding=None,dropout=0,nembedding=0)

# # %%
# # train
# if not os.path.exists('models/model_'+str(sig_train)):
#     os.makedirs('models/model_'+str(sig_train))

valloss_list_background=[]
trainloss_list_background=[]
optimizer = torch.optim.Adam(model_background.parameters(),lr=1e-4) #,lr=1e-4)#, lr=1e-4)

for epoch in range(100):
#     print('\n Epoch: {}'.format(epoch))
    trainloss= train(model_background,optimizer,background_train_loader ,noise_data=0.,noise_context=0.0)
    valloss=val(model_background,background_val_loader)
#     valloss=val(model_background,background_val_loader)
#    # print('epoch '+str(epoch)+' val loss: ',valloss)
#     # save model each epoch
    torch.save(model_background.state_dict(), save_path+'model_CR_'+str(epoch)+'.pt')

    valloss_list_background.append(valloss)
    trainloss_list_background.append(trainloss)
    wandb.log({'train_loss_background': trainloss, 'val_loss_background': valloss,
               'epoch_CR': epoch})
# print('done')


# # %%
valloss_list_background=np.array(valloss_list_background)
trainloss_list_background=np.array(trainloss_list_background)

np.save(save_path+'trainloss_list_background.npy', trainloss_list_background)
np.save(save_path+'valloss_list_background.npy', valloss_list_background)

min_epoch=np.argmin(valloss_list_background)
print('min epoch CR: ',min_epoch)

model_background.load_state_dict(torch.load(save_path+'model_CR_'+str(min_epoch)+'.pt'))
torch.save(model_background.state_dict(), save_path+'model_CR_best.pt')
model_background.eval()
with torch.no_grad():
        samples_background=model_background.sample(50000)
samples_background=samples_background.cpu().detach().numpy().reshape((-1))

figure=plt.figure()
_=plt.hist(samples_background,bins=50, density=True, histtype='step', label='nflow')
_=plt.hist(background_val,bins=50, density=True, histtype='step', label='valdata')
_=plt.hist(background_train,bins=50, density=True, histtype='step', label='traindata')
plt.legend(loc='upper right')
plt.savefig(save_path+'nflow_CR.png')

wandb.log({'nflow_CR': wandb.Image(figure)})
plt.close()



# # %%
# plt.plot(valloss_list_background)
# plt.show()

# # %%


# # %%



