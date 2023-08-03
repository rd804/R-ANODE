import numpy as np
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm
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

#os.environ["CUDA_VISIBLE_DEVICES"]='2'

parser = argparse.ArgumentParser()
parser.add_argument('--n_sig', default=1000)
parser.add_argument('--try_', type=int, default=0)
parser.add_argument('--epochs', type=int, default=2)
parser.add_argument('--batch_size', type=int, default=256)

parser.add_argument('--resample', action='store_true', help='if data is to resampled')
parser.add_argument('--seed', type=int, default=22, help='seed')
parser.add_argument('--shuffle_split', action='store_true', help='if shuffle split is used')
parser.add_argument('--split', type=int, default=1, help='split number')
parser.add_argument('--data_dir', type=str, default='data/lhc_co', help='data directory')


parser.add_argument('--wandb', action='store_true', help='if wandb is used')
parser.add_argument('--wandb_group', type=str, default='test')
parser.add_argument('--wandb_job_type', type=str, default='SR')
parser.add_argument('--wandb_run_name', type=str, default='try_')


args = parser.parse_args()


CUDA = False
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

pre_parameters = preprocess_params_fit(SR_data)
x_train = preprocess_params_transform(SR_data, pre_parameters) 

if not args.shuffle_split:    
    data_train, data_val = train_test_split(x_train, test_size=0.1, random_state=args.seed)
else:
    ss_data = ShuffleSplit(n_splits=20, test_size=0.1, random_state=22)

    print(f'doing a shuffle split with split number {args.split}')

    for i, (train_index, test_index) in enumerate(ss_data.split(x_train)):
        if i == args.split:
            data_train, data_val = x_train[train_index], x_train[test_index]
            break



_x_test = np.load(f'{args.data_dir}/x_test.npy')
x_test = preprocess_params_transform(_x_test, pre_parameters)



traintensor = torch.from_numpy(data_train.astype('float32'))
valtensor = torch.from_numpy(data_val.astype('float32'))
testtensor = torch.from_numpy(x_test.astype('float32'))

print('X_train shape', traintensor.shape)
print('X_val shape', valtensor.shape)
print('X_test shape', testtensor.shape)

# Use the standard pytorch DataLoader
batch_size = args.batch_size
trainloader = torch.utils.data.DataLoader(traintensor, batch_size=batch_size, shuffle=True)

test_batch_size=batch_size*5
valloader = torch.utils.data.DataLoader(valtensor, batch_size=test_batch_size, shuffle=False)
testloader = torch.utils.data.DataLoader(testtensor, batch_size=test_batch_size, shuffle=False)



# # %%
# define savepath
save_path = 'results/'+args.wandb_group+'/'\
            +job_name+'/'+args.wandb_run_name+'/'

if not os.path.exists(save_path):
    os.makedirs(save_path)
    

trainloss_list=[]
valloss_list=[]

config_file = "./scripts/DE_MAF_model.yml"
model = DensityEstimator(config_file, eval_mode=False, device=device)

##############
# train model
for epoch in range(args.epochs):
    trainloss=anode(model,trainloader, pre_parameters ,device=device, mode='train')
    valloss=anode(model,valloader, pre_parameters, device=device, mode='val')

    torch.save(model.model.state_dict(), save_path+'model_SR_'+str(epoch)+'.pt')

    valloss_list.append(valloss)
    trainloss_list.append(trainloss)

    print('epoch: ', epoch, 'trainloss: ', trainloss, 'valloss: ', valloss)

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

mass_test = testtensor[:,0].reshape(-1,1).to(device)
with torch.no_grad():
    x_samples = model.model.sample(10000, cond_inputs=mass_test ).cpu().detach().numpy()
    x_samples = inverse_transform(x_samples, pre_parameters)


for i in range(5):
    figure=plt.figure()
    #if dims > 1:
    plt.hist(testtensor[:,i],bins=100, density=True, label=f'data for {i}')
    plt.hist(x_samples[:,i],bins=100, density=True, label=f'nflow sample for {i}')
    plt.legend(loc='upper right')
    plt.savefig(f'results/{args.wandb_group}/{args.wandb_job_type}/{wandb.run.name}/nflow_{i}.png')
    if args.wandb:
        wandb.log({f'nflow_SR': wandb.Image(figure)})

    plt.close()

# compute scores
model.eval()
with torch.no_grad():
    log_p = evaluate_log_prob(model, testtensor, pre_parameters).cpu().detach().numpy()

    data_p = np.exp(log_p)


print('data_p', data_p.shape)
print('data_p', data_p[:10])

likelihood = data_p / back_p





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
    
