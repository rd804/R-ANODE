import numpy as np
import matplotlib.pyplot as plt
from src.nflow_utils import *
from src.generate_data_lhc import *
from src.utils import *
from src.flows import *
import os

from nflows import transforms, distributions, flows
import torch
import torch.nn.functional as F
from sklearn.utils import shuffle
# import train_test_split
from sklearn.model_selection import train_test_split, ShuffleSplit
import argparse
import wandb
import pickle
import sys

#os.environ["CUDA_VISIBLE_DEVICES"]='2'

parser = argparse.ArgumentParser()
parser.add_argument('--n_sig',type=int , default=1000)
parser.add_argument('--try_', type=int, default=0)
parser.add_argument('--epochs', type=int, default=3)
parser.add_argument('--batch_size', type=int, default=256)

parser.add_argument('--resample', action='store_true', help='if data is to resampled')
parser.add_argument('--seed', type=int, default=22, help='seed')
parser.add_argument('--shuffle_split', action='store_true', help='if shuffle split is used')
parser.add_argument('--split', type=int, default=1, help='split number')
parser.add_argument('--data_dir', type=str, default='data/lhc_co', help='data directory')
parser.add_argument('--config_file', type=str, default='scripts/DE_MAF_model.yml', help='config file')
parser.add_argument('--CR_path', type=str, default='results/nflows_lhc_co/manuel_flows/training_1', help='CR data path')
parser.add_argument('--ensemble', action='store_true', default=True ,help='if ensemble is used')


parser.add_argument('--wandb', action='store_true', help='if wandb is used')
parser.add_argument('--wandb_group', type=str, default='debugging_anode_SR')
parser.add_argument('--wandb_job_type', type=str, default='SR')
parser.add_argument('--wandb_run_name', type=str, default='try_')


args = parser.parse_args()
save_path = 'results/'+args.wandb_group+'/'\
            +args.wandb_job_type+'/'+args.wandb_run_name+'/'

if os.path.exists(f'{save_path}best_val_loss_scores.npy'):
    print(f'already done {args.wandb_run_name}')
    sys.exit()

CUDA = True
device = torch.device("cuda:0" if CUDA else "cpu")

job_name = args.wandb_job_type

# initialize wandb for logging
if args.wandb:
    wandb.init(project="r_anode", config=args,
            group=args.wandb_group, job_type=job_name)

    name = args.wandb_run_name
    wandb.run.name = name


print(device)

SR_data, CR_data , true_w, sigma = resample_split(args.data_dir, n_sig = args.n_sig, resample_seed = args.seed,resample = args.resample)

print('x_train shape', SR_data.shape)
print('true_w', true_w)
print('sigma', sigma)

if args.wandb:
    wandb.config.update({'true_w': true_w, 'sigma': sigma})


pre_parameters = preprocess_params_fit(SR_data)
x_train = preprocess_params_transform(SR_data, pre_parameters) 


if not args.shuffle_split:    
    data_train, data_val = train_test_split(x_train, test_size=0.1, random_state=args.seed)
else:
    ss_data = ShuffleSplit(n_splits=20, test_size=0.5, random_state=22)

    print(f'doing a shuffle split with split number {args.split}')

    for i, (train_index, test_index) in enumerate(ss_data.split(x_train)):
        if i == args.split:
            data_train, data_val = x_train[train_index], x_train[test_index]
            break



_x_test = np.load(f'{args.data_dir}/x_test.npy')
x_test = preprocess_params_transform(_x_test, pre_parameters)


traintensor = torch.from_numpy(data_train.astype('float32')).to(device)
valtensor = torch.from_numpy(data_val.astype('float32')).to(device)
testtensor = torch.from_numpy(x_test.astype('float32')).to(device)

print('X_train shape', traintensor.shape)
print('X_val shape', valtensor.shape)
print('X_test shape', testtensor.shape)

for key in pre_parameters.keys():
    pre_parameters[key] = torch.from_numpy(pre_parameters[key].astype('float32')).to(device)

train_tensor = torch.utils.data.TensorDataset(traintensor)
val_tensor = torch.utils.data.TensorDataset(valtensor)
test_tensor = torch.utils.data.TensorDataset(testtensor)


# Use the standard pytorch DataLoader
batch_size = args.batch_size
trainloader = torch.utils.data.DataLoader(train_tensor, batch_size=batch_size, shuffle=True)

test_batch_size=batch_size*5
valloader = torch.utils.data.DataLoader(val_tensor, batch_size=test_batch_size, shuffle=False)
testloader = torch.utils.data.DataLoader(test_tensor, batch_size=test_batch_size, shuffle=False)




# # %%
# define savepath
save_path = 'results/'+args.wandb_group+'/'\
            +job_name+'/'+args.wandb_run_name+'/'


if not os.path.exists(save_path):
    os.makedirs(save_path)
    

trainloss_list=[]
valloss_list=[]


model = DensityEstimator(args.config_file, eval_mode=False, device=device)

##############
# train model
for epoch in range(args.epochs):
    trainloss=anode(model.model,trainloader,model.optimizer,pre_parameters ,device=device, mode='train')
    valloss=anode(model.model,valloader,model.optimizer,pre_parameters, device=device, mode='val')

    torch.save(model.model.state_dict(), save_path+'model_SR_'+str(epoch)+'.pt')

    valloss_list.append(valloss)
    trainloss_list.append(trainloss)

    print('epoch: ', epoch, 'trainloss: ', trainloss, 'valloss: ', valloss)

    if np.isnan(trainloss) or np.isnan(valloss):
        print(' nan loss ')
        if args.wandb:
            wandb.finish()
        sys.exit()

    if args.wandb:
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

#min_epoch = args.epochs-1
model.model.load_state_dict(torch.load(save_path+'model_SR_'+str(min_epoch)+'.pt'))
torch.save(model.model.state_dict(), save_path+'model_SR_best.pt')


# check density estimation
model.model.eval()

test_data = inverse_transform(testtensor, pre_parameters).cpu().detach().numpy()
label_test = test_data[:,-1]

train_data = inverse_transform(traintensor, pre_parameters).cpu().detach().numpy()
val_data = inverse_transform(valtensor, pre_parameters).cpu().detach().numpy()


model.model.eval()

x_samples_train = generate_transformed_samples(model.model, traintensor, pre_parameters, device=device).cpu().detach().numpy()
x_samples_val = generate_transformed_samples(model.model, valtensor, pre_parameters, device=device).cpu().detach().numpy()

x_samples = np.vstack((x_samples_train, x_samples_val))
all_data = np.vstack((train_data, val_data))

for i in range(5):
    figure=plt.figure()
    #if dims > 1:
    plt.hist(all_data[:,i],bins=100, density=True, label=f'data for {i}', histtype='step')
    plt.hist(x_samples[:,i],bins=100, density=True, label=f'nflow sample for {i}', histtype='step')
    plt.title(f'NFlow vs data for {i}')
    plt.legend(loc='upper right')
    plt.savefig(f'{save_path}/nflow_{i}.png')
    if args.wandb:
        wandb.log({f'nflow_SR': wandb.Image(figure)})

    plt.close()

# compute scores

if not args.ensemble:
    best_epoch = np.argsort(valloss_list)[0]
else:
    best_epoch = np.argsort(valloss_list)[0:10]

data_log_p = []
for epoch in best_epoch:
    model.model.load_state_dict(torch.load(save_path+'model_SR_'+str(epoch)+'.pt'))
    model.model.eval()
    with torch.no_grad():
        log_p = evaluate_log_prob(model.model, testtensor, pre_parameters).cpu().detach().numpy()
        data_log_p.append(log_p)

data_log_p = np.array(data_log_p)
data_log_p = np.mean(data_log_p, axis=0)
   # data_p = np.exp(log_p)

print('data_p', data_log_p.shape)
print('data_p', data_log_p[:10])

#likelihood = data_p / back_p

# load CR model

val_losses = np.load(f'{args.CR_path}/my_ANODE_model_val_losses.npy')

if not args.ensemble:
    best_epoch = np.argsort(val_losses)[0]
else:
    best_epoch = np.argsort(val_losses)[0:10]


background_log_p = []
for epoch in best_epoch:
    model_B = DensityEstimator(args.config_file, eval_mode=True, load_path=f"{args.CR_path}/my_ANODE_model_epoch_{epoch}.par", device=device)

    model_B.model.eval()
    with torch.no_grad():
        log_p = evaluate_log_prob(model_B.model, testtensor, pre_parameters).cpu().detach().numpy()
        background_log_p.append(log_p)

background_log_p = np.array(background_log_p)
background_log_p = np.mean(background_log_p, axis=0)

likelihood_ = (data_log_p - background_log_p)
likelihood = np.nan_to_num(likelihood_, nan=0, posinf=0, neginf=0)        

sic_score , tpr_score , auc_score = SIC(label_test, likelihood)

figure = plt.figure()

plt.plot(tpr_score, sic_score, label='score')
plt.plot(tpr_score, tpr_score**0.5, label='random')
plt.xlabel('signal efficiency')
plt.ylabel('SIC')

plt.legend(loc='lower right')
if args.wandb:
    wandb.log({'SIC': wandb.Image(figure)})
    wandb.log({'AUC': auc_score, 'max SIC': np.max(sic_score)})

plt.savefig(f'results/{args.wandb_group}/{args.wandb_job_type}/{args.wandb_run_name}/SIC.png')
plt.close()


# save scores
np.save(save_path+'best_val_loss_scores.npy', data_log_p)


if args.wandb:
    wandb.finish()
    
