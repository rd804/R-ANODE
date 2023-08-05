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


parser = argparse.ArgumentParser()
parser.add_argument('--n_sig',  default=1000, help='signal train')
parser.add_argument('--mode_background', type=str, default='freeze', help='train, freeze, pretrained')


parser.add_argument('--epochs', type=int, default=3, help='epochs')
parser.add_argument('--batch_size', type=int, default = 256, help = 'batch size')
parser.add_argument('--mini_batch', type=int, default=256, help='mini batch size')
parser.add_argument('--gpu', type=str, default='cuda:0', help='gpu to train on')
parser.add_argument('--data_loss_expr', type=str, default='true_likelihood', help='loss for SR region')


parser.add_argument('--resample', action='store_true', help='if data is to resampled')
parser.add_argument('--seed', type=int, default=22, help='seed')
parser.add_argument('--shuffle_split', action='store_true', help='if shuffle split is used')
parser.add_argument('--split', type=int, default=1, help='split number')
parser.add_argument('--data_dir', type=str, default='data/lhc_co', help='data directory')
parser.add_argument('--config_file', type=str, default='scripts/DE_MAF_model.yml', help='config file')
parser.add_argument('--CR_path', type=str, default='results/nflows_lhc_co/manuel_flows/training_1', help='CR data path')


parser.add_argument('--wandb', action='store_true', help='if wandb is used' )
parser.add_argument('--wandb_group', type=str, default='debugging_r_anode')
parser.add_argument('--wandb_job_type', type=str, default='lhc_co')
parser.add_argument('--wandb_run_name', type=str, default='try_0')





args = parser.parse_args()

if os.path.exists(f'results/{args.wandb_group}/{args.wandb_job_type}/{args.wandb_run_name}/valloss.npy'):
    print('already done')
    sys.exit()

if args.wandb:
    wandb.init(project="r_anode", config=args,
                group=args.wandb_group, job_type=args.wandb_job_type)

    wandb.run.name = args.wandb_run_name

# print wandb group


CUDA = False
device = torch.device(args.gpu if CUDA else "cpu")

SR_data, CR_data , true_w, sigma = resample_split(args.data_dir, n_sig = args.n_sig, resample_seed = args.seed,resample = args.resample)

print('x_train shape', SR_data.shape)
print('true_w', true_w)
print('sigma', sigma)

if args.wandb:
    wandb.config.update({'true_w': true_w, 'sigma': sigma})


pre_parameters = preprocess_params_fit(SR_data)
x_train = preprocess_params_transform(SR_data, pre_parameters) 

# print(pre_parameters)

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

for key in pre_parameters.keys():
    pre_parameters[key] = torch.from_numpy(pre_parameters[key].astype('float32')).to(device)



# Use the standard pytorch DataLoader
batch_size = args.batch_size
trainloader = torch.utils.data.DataLoader(traintensor, batch_size=batch_size, shuffle=True)

test_batch_size=batch_size*5
valloader = torch.utils.data.DataLoader(valtensor, batch_size=test_batch_size, shuffle=False)
testloader = torch.utils.data.DataLoader(testtensor, batch_size=test_batch_size, shuffle=False)


model_S = DensityEstimator(args.config_file, eval_mode=False, device=device)


if args.mode_background == 'train':
    pass

elif args.mode_background == 'freeze':
    val_losses = np.load(f'{args.CR_path}/my_ANODE_model_val_losses.npy')
    best_epoch = np.argmin(val_losses)
    model_B = DensityEstimator(args.config_file, eval_mode=True, load_path=f"{args.CR_path}/my_ANODE_model_epoch_{best_epoch}.par", device=device)

elif args.mode_background == 'pretrained':
    val_losses = np.load(f'{args.CR_path}/my_ANODE_model_val_losses.npy')
    best_epoch = np.argmin(val_losses)
    model_B = DensityEstimator(args.config_file, eval_mode=False, load_path=f"{args.CR_path}/my_ANODE_model_epoch_{best_epoch}.par", device=device)




valloss = []
trainloss = []


if not os.path.exists(f'results/{args.wandb_group}/{args.wandb_job_type}/{args.wandb_run_name}'):
    os.makedirs(f'results/{args.wandb_group}/{args.wandb_job_type}/{args.wandb_run_name}')


for epoch in range(args.epochs):

    train_loss = r_anode(model_S.model,model_B.model,true_w,model_S.optimizer,trainloader, 
                         pre_parameters, device=device, mode='train',\
                          data_loss_expr=args.data_loss_expr)
    val_loss = r_anode(model_S.model,model_B.model,true_w,model_S.optimizer,valloader, 
                       pre_parameters, device=device, mode='val',\
                        data_loss_expr=args.data_loss_expr)


    ##################################
    ##############################
    # Save model and weights

    torch.save(model_S.model.state_dict(), f'results/{args.wandb_group}/{args.wandb_job_type}/{args.wandb_run_name}/model_S_{epoch}.pt')

    
    if args.mode_background == 'train' or args.mode_background == 'pretrained':
        torch.save(model_B.model.state_dict(), f'results/{args.wandb_group}/{args.wandb_job_type}/{args.wandb_run_name}/model_B_{epoch}.pt')

    if np.isnan(train_loss) or np.isnan(val_loss):
        print(' nan loss ')
        if args.wandb:
            wandb.finish()
        break

    if args.wandb:
        wandb.log({'train_loss': train_loss, 'val_loss': val_loss, \
               'true_w': true_w})

    print('epoch: ', epoch, 'trainloss: ', train_loss, 'valloss: ', val_loss)
    valloss.append(val_loss)
    trainloss.append(train_loss)



if ~np.isnan(train_loss) or ~np.isnan(val_loss):

  
    args.ensemble = True
    # Load best model
    if not args.ensemble:
        index = np.argmin(valloss).flatten()[0]

        model_S.model.load_state_dict(torch.load(f'results/{args.wandb_group}/{args.wandb_job_type}/{args.wandb_run_name}/model_S_{index}.pt'))
        
        model_S.model.eval()
        log_S = evaluate_log_prob(model_S.model, testtensor, pre_parameters).cpu().detach().numpy()
        #S = np.exp(log_S)


    else:
        log_S = []
        sorted_index = np.argsort(valloss).flatten()[0:10]
        for index in sorted_index:

            model_S.model.load_state_dict(torch.load(f'results/{args.wandb_group}/{args.wandb_job_type}/{args.wandb_run_name}/model_S_{index}.pt'))
            
            model_S.model.eval()
            log_S_ = evaluate_log_prob(model_S.model, testtensor, pre_parameters).cpu().detach().numpy()
            log_S.append(log_S_)

        log_S = np.array(log_S)
        S = np.exp(log_S)
        S = np.mean(S, axis=0)
        log_S = np.log(S + 1e-32)


    if args.mode_background == 'train' or args.mode_background == 'pretrained':
        model_B.model.load_state_dict(torch.load(f'results/{args.wandb_group}/{args.wandb_job_type}/{args.wandb_run_name}/model_B_{index}.pt'))
    
    test_data = inverse_transform(testtensor, pre_parameters).cpu().detach().numpy()
    label_test = test_data[:,-1]

    model_S.model.eval()

    x_samples = generate_transformed_samples(model_S.model, testtensor[testtensor[:,-1]==1], pre_parameters, device=device)

    for i in range(5):
        figure=plt.figure()
        #if dims > 1:
        plt.hist(test_data[:,i][label_test==1],bins=100, density=True, label=f'data for {i}', histtype='step')
        plt.hist(x_samples[:,i],bins=100, density=True, label=f'nflow sample for {i}', histtype='step')
        plt.legend(loc='upper right')
        plt.title(f'Nflow vs S for {i}')
        plt.savefig(f'results/{args.wandb_group}/{args.wandb_job_type}/{args.wandb_run_name}/nflow_S_{i}.png')
        if args.wandb:
            wandb.log({f'nflow_S': wandb.Image(figure)})

        plt.close()

    model_B.model.eval()
    x_samples = generate_transformed_samples(model_B.model, testtensor[testtensor[:,-1]==0], pre_parameters, device=device)


    for i in range(5):
        figure=plt.figure()
        #if dims > 1:
        plt.hist(test_data[:,i][label_test==0],bins=100, density=True, label=f'data for {i}', histtype='step')
        plt.hist(x_samples[:,i],bins=100, density=True, label=f'nflow sample for {i}', histtype='step')
        plt.legend(loc='upper right')
        plt.title(f'Nflow vs B for {i}')
        plt.savefig(f'results/{args.wandb_group}/{args.wandb_job_type}/{args.wandb_run_name}/nflow_B_{i}.png')
        if args.wandb:
            wandb.log({f'nflow_B': wandb.Image(figure)})

        plt.close()


    score_likelihoods = {}


    model_B.model.eval()
    with torch.no_grad():
        log_B = evaluate_log_prob(model_B.model, testtensor, pre_parameters).cpu().detach().numpy()

    assert len(log_S) == len(log_B)

    likelihood_ = log_S - log_B
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


    # check density estimation


    np.save(f'results/{args.wandb_group}/{args.wandb_job_type}/{args.wandb_run_name}/valloss.npy', valloss)
    np.save(f'results/{args.wandb_group}/{args.wandb_job_type}/{args.wandb_run_name}/trainloss.npy', trainloss)

    if args.wandb:
        wandb.finish()






