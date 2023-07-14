import numpy as np
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm
from src.nflow_utils import *
import os

from nflows import transforms, distributions, flows
import torch
import torch.nn.functional as F
from nflows.distributions import uniform
from sklearn.utils import shuffle
# import train_test_split
from sklearn.model_selection import train_test_split
import argparse
import wandb
import pickle

#os.environ["CUDA_VISIBLE_DEVICES"]='2'

parser = argparse.ArgumentParser()
parser.add_argument('--sig_train', default=10)
parser.add_argument('--try_', type=int, default=0)
parser.add_argument('--epochs', type=int, default=2)
parser.add_argument('--batch_size', type=int, default=256)

parser.add_argument('--resample', action='store_true', help='if data is to resampled')
parser.add_argument('--seed', type=int, default=22, help='seed')
parser.add_argument('--gaussian_dim',  default=1, type=int,help='dimension of gaussian')

parser.add_argument('--wandb_group', type=str, default='test')
parser.add_argument('--wandb_job_type', type=str, default='SR')
parser.add_argument('--wandb_run_name', type=str, default='try_')


args = parser.parse_args()


CUDA = True
device = torch.device("cuda:0" if CUDA else "cpu")

job_name = args.wandb_job_type

# initialize wandb
wandb.init(project="m-anode", config=args,
           group=args.wandb_group, job_type=job_name)

name = args.wandb_run_name+str(args.try_)
wandb.run.name = name



kwargs = {'num_workers': 4, 'pin_memory': True} if CUDA else {}
kwargs = {}

print(device)

if args.gaussian_dim == 1:
    with open('data/data.pkl', 'rb') as f:
        data = pickle.load(f)


    with open('data/background.pkl', 'rb') as f:
        background = pickle.load(f)



    sig_train = args.sig_train
    back_mean = 0
    sig_mean = 3
    sig_sigma = 0.5
    back_sigma = 3

else:
    with open(f'data/data_{args.gaussian_dim}d.pkl', 'rb') as f:
        data = pickle.load(f)

    with open(f'data/background_{args.gaussian_dim}d.pkl', 'rb') as f:
        background = pickle.load(f)

    with open(f'data/true_w_{args.gaussian_dim}d.pkl', 'rb') as f:
        true_w = pickle.load(f)

    sig_train = args.sig_train
    back_mean = 0
    sig_mean = 2
    sig_sigma = 0.25
    back_sigma = 3

    print(true_w[str(sig_train)])




# best_parameters[str(sig_train)] = {}
if not args.resample:
    x_train = data[str(sig_train)]['train']['data']
    x_train = shuffle(x_train, random_state=10)

else:

    sig_train = args.sig_train

    n_back = 200000
    n_sig = int(np.sqrt(n_back) * float(sig_train))

    # set seed
    if args.gaussian_dim == 1:
        np.random.seed(args.seed)
        x_back = np.random.normal(back_mean, back_sigma, n_back)
        np.random.seed(args.seed)
        x_sig = np.random.normal(sig_mean, sig_sigma, n_sig)
    else:
        np.random.seed(args.seed)
        x_back = np.random.normal(back_mean, back_sigma, (n_back, args.gaussian_dim))
        np.random.seed(args.seed)
        x_sig = np.random.normal(sig_mean, sig_sigma, (n_sig, args.gaussian_dim))

    x_train = np.concatenate((x_back, x_sig), axis=0)
    x_train = shuffle(x_train, random_state=args.seed)

    print('resampled data with seed %d' % args.seed)
    print('amount of background: %d' % n_back)
    print('amount of signal: %d' % n_sig)

x_train , x_val = train_test_split(x_train, test_size=0.5, random_state=args.seed)

x_test = data['10']['val']['data']
label_test = data['10']['val']['label']



traintensor = torch.from_numpy(x_train.astype('float32').reshape((-1,args.gaussian_dim)))
valtensor = torch.from_numpy(x_val.astype('float32').reshape((-1,args.gaussian_dim)))
testtensor = torch.from_numpy(x_test.astype('float32').reshape((-1,args.gaussian_dim)))


# Use the standard pytorch DataLoader
batch_size = args.batch_size
trainloader = torch.utils.data.DataLoader(traintensor, batch_size=batch_size, shuffle=True)

test_batch_size=batch_size*5
valloader = torch.utils.data.DataLoader(valtensor, batch_size=test_batch_size, shuffle=False)
testloader = torch.utils.data.DataLoader(testtensor, batch_size=test_batch_size, shuffle=False)


model = flows_for_gaussian(gaussian_dim = args.gaussian_dim, num_transforms = 2, num_blocks = 3, 
                       hidden_features = 32, device = device)



# # %%
# define savepath
save_path = 'results/'+args.wandb_group+'/'\
            +job_name+'/'+name+'/'

if not os.path.exists(save_path):
    os.makedirs(save_path)
    

trainloss_list=[]
valloss_list=[]

optimizer = torch.optim.Adam(model.parameters()) #,lr=1e-4)#, lr=1e-4)
##############
# train model
for epoch in range(args.epochs):
    trainloss= train(model,optimizer,trainloader,noise_data=0.,noise_context=0.0,device=device)
    valloss=val(model,valloader,device=device)

    torch.save(model.state_dict(), save_path+'model_SR_'+str(epoch)+'.pt')

    valloss_list.append(valloss)
    trainloss_list.append(trainloss)
    wandb.log({'train_loss': trainloss, 'val_loss': valloss, 'epoch': epoch})

trainloss_list=np.array(trainloss_list)
valloss_list=np.array(valloss_list)
np.save(save_path+'trainloss_list.npy', trainloss_list)
np.save(save_path+'valloss_list.npy', valloss_list)

##################
#################

#########
# load best model
valloss_list=np.array(valloss_list)
min_epoch=np.argmin(valloss_list)
print('min epoch SR: ',min_epoch)

model.load_state_dict(torch.load(save_path+'model_SR_'+str(min_epoch)+'.pt'))
torch.save(model.state_dict(), save_path+'model_SR_best.pt')


# check density estimation
model.eval()
with torch.no_grad():
    x_samples = model.sample(10000).cpu().detach().numpy()


if args.gaussian_dim == 2:
    figure=plt.figure()
    #if dims > 1:
    plt.hist2d(x_samples[:,0],x_samples[:,1],bins=100, density=True, label='nflow for 2d')

    wandb.log({f'nflow_SR': wandb.Image(figure)})
    plt.close()

# compute scores
model.eval()
with torch.no_grad():
    log_p = model.log_prob(testtensor.to(device)).cpu().detach().numpy()
    data_p = np.exp(log_p)



back_p = p_back(x_test,back_mean, back_sigma, dim=args.gaussian_dim)

likelihood = data_p / back_p

true_likelihoods = {}
true_likelihoods[str(sig_train)] = {}


w1 = true_w[str(sig_train)][0]
w2 = true_w[str(sig_train)][1]



true_likelihoods[str(sig_train)] = p_data(x_test,[sig_mean, back_mean],[sig_sigma,back_sigma],
                [w1, w2], dim=args.gaussian_dim)/p_back(x_test,back_mean, back_sigma, dim=args.gaussian_dim)





sic_true , tpr_true , auc_true = SIC(label_test, true_likelihoods[str(sig_train)])
sic_score , tpr_score , auc_score = SIC(label_test, likelihood)

figure = plt.figure()

plt.plot(tpr_true, sic_true, label='true')
plt.plot(tpr_score, sic_score, label='score')

plt.legend(loc='lower right')
wandb.log({'SIC': wandb.Image(figure)})
plt.savefig(f'results/{args.wandb_group}/{args.wandb_job_type}/{wandb.run.name}/SIC.png')
plt.show()

wandb.log({'AUC': auc_score, 'max SIC': np.max(sic_score)})

# save scores
np.save(save_path+'best_val_loss_scores.npy', data_p)



wandb.finish()
    
