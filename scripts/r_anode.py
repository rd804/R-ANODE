import numpy as np
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm
#from src.nflow_utils import *
import os
import sys
sys.path.append('/scratch/rd804/m-anode/')
from src.utils import *
from src.nflow_utils import *
from src.generate_data_lhc import *
from src.utils import *
from src.flows import *
from scipy.stats import rv_histogram

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
parser.add_argument('--n_sig',type=int , default=1000, help='signal train')
parser.add_argument('--mode_background', type=str, default='freeze', help='train, freeze, pretrained')


parser.add_argument('--epochs', type=int, default=2, help='epochs')
parser.add_argument('--batch_size', type=int, default = 256, help = 'batch size')
parser.add_argument('--mini_batch', type=int, default=256, help='mini batch size')
parser.add_argument('--gpu', type=str, default='cuda:0', help='gpu to train on')
parser.add_argument('--data_loss_expr', type=str, default='true_likelihood', help='loss for SR region')
parser.add_argument('--w_scan', action='store_true',  help='if true w is used')
parser.add_argument('--w', type=float, default=0.1, help='weight for loss function')
parser.add_argument('--w_train', action='store_true',  help='if true w is used')
parser.add_argument('--no_signal_fit',action='store_true', help='compare to no signal fit')

parser.add_argument('--validation_fraction', type=float, default=0.2, help='validation fraction')
parser.add_argument('--resample', action='store_true', help='if data is to resampled')
parser.add_argument('--seed', type=int, default=22, help='seed')
parser.add_argument('--shuffle_split', action='store_true', help='if shuffle split is used')
parser.add_argument('--split', type=int, default=1, help='split number')
parser.add_argument('--data_dir', type=str, default='data/lhc_co', help='data directory')
parser.add_argument('--config_file', type=str, default='scripts/DE_MAF_model.yml', help='config file')
#parser.add_argument('--S_config_file', type=str, default='scripts/DE_MAF_model.yml', help='config file')
parser.add_argument('--random_w', action='store_true', help='if random w is used for initialization')
parser.add_argument('--CR_path', type=str, default='results/nflows_lhc_co/CR_bn_fixed_1000/try_1_0', help='CR data path')
#parser.add_argument('--CR_path', type=str, default='results/nflows_lhc_co/CR_RQS_4_1000/try_1_2', help='CR data path')
parser.add_argument('--ensemble', action='store_true',default = True ,help='if ensemble is used')

parser.add_argument('--wandb', action='store_true', help='if wandb is used' )
parser.add_argument('--wandb_group', type=str, default='debugging_r_anode')
parser.add_argument('--wandb_job_type', type=str, default='lhc_co')
parser.add_argument('--wandb_run_name', type=str, default='joint_test_4')
#

args = parser.parse_args()

#args.wandb = True
if os.path.exists(f'results/{args.wandb_group}/{args.wandb_job_type}/{args.wandb_run_name}/valloss.npy'):
    print('already done')
    sys.exit()

if not os.path.exists(f'results/{args.wandb_group}/{args.wandb_job_type}/{args.wandb_run_name}'):
    os.makedirs(f'results/{args.wandb_group}/{args.wandb_job_type}/{args.wandb_run_name}')


if args.wandb:
    wandb.init(project="r_anode", config=args,
                group=args.wandb_group, job_type=args.wandb_job_type)

    wandb.run.name = args.wandb_run_name

# print wandb group




CUDA = True
device = torch.device(args.gpu if CUDA else "cpu")

##########################
# We use same preprocessing in SR as in CR, to simplify the addition of 
# probabilties. However the script allows you to have pre_parameters_CR and pre_parameters_SR.
# for the case where you want to have different preprocessing for SR and CR.
# In this case you need to change the addition of probabilities in the training function.

# load pre_parameters of CR 
with open(f'{args.CR_path}/pre_parameters.pkl', 'rb') as f:
    pre_parameters_CR = pickle.load(f)

#pre_parameters_SR_ = preprocess_params_fit_all(SR_data)

pre_parameters_SR = pre_parameters_CR.copy()
#for key in pre_parameters_CR.keys():
    #pre_parameters_SR[key]= np.insert(pre_parameters_CR[key], 0, pre_parameters_SR_[key][0])
    
print('pre_parameters_CR', pre_parameters_CR)
print('pre_parameters_SR', pre_parameters_SR)


# _, mask_SR = logit_transform(SR_data[:,1:-1], pre_parameters_SR['min'],
                           #     pre_parameters_SR['max'])
#mask = mask_CR & mask_SR

if args.no_signal_fit:
    #background = np.load(f'{args.data_dir}/extrabkg_train_val.npy')
    SR_data, CR_data , true_w, sigma = resample_split(args.data_dir, n_sig = args.n_sig, resample_seed = args.seed,resample = args.resample)

    background = SR_data[SR_data[:,-1]==0]

    mass = background[:,0]
    labels = background[:,-1]
    bins = np.linspace(3.3, 3.7, 50)
    hist_back = np.histogram(mass[labels==0], bins=bins, density=True)
    density_back = rv_histogram(hist_back)

    _, mask = logit_transform(background[:,1:-1], pre_parameters_CR['min'],
                                pre_parameters_CR['max'])
    x_train = background[mask]
    print('true background fit ... .. ... .. ...')
   # print('x_train shape', x_train.shape)
    x_train_S = preprocess_params_transform(x_train, pre_parameters_SR)
    x_train_B = preprocess_params_transform(x_train, pre_parameters_CR)
else:
    SR_data, CR_data , true_w, sigma = resample_split(args.data_dir, n_sig = args.n_sig, resample_seed = args.seed,resample = args.resample)

    ########################################################################
    # fit histogram to data

    mass = SR_data[:,0]
    #labels = SR_data[:,-1]
    bins = np.linspace(3.3, 3.7, 50)
    
    hist_back = np.histogram(mass, bins=bins, density=True)

    # hist_back = np.histogram(mass[labels==0], bins=bins, density=True)
    density_back = rv_histogram(hist_back)
    #########################################################################

    print('x_train shape', SR_data.shape)
    print('true_w', true_w)
    print('sigma', sigma)

    if args.wandb:
        wandb.config.update({'true_w': true_w, 'sigma': sigma})


    _, mask = logit_transform(SR_data[:,1:-1], pre_parameters_CR['min'],
                             pre_parameters_CR['max'])
    x_train = SR_data[mask]
  #  print('x_train shape', x_train.shape)
    # have two seperate transforms of the data for model_S and model_B (if required)
    x_train_S = preprocess_params_transform(x_train, pre_parameters_SR) 
    x_train_B = preprocess_params_transform(x_train, pre_parameters_CR)

print('x_train_S shape', x_train_S.shape)

# create masked test data
_x_test = np.load(f'{args.data_dir}/x_test.npy')
_, mask_CR = logit_transform(_x_test[:,1:-1], pre_parameters_CR['min'],
                                pre_parameters_CR['max'])
_, mask_SR = logit_transform(_x_test[:,1:-1], pre_parameters_SR['min'],
                                pre_parameters_SR['max'])
mask = mask_CR & mask_SR
x_test = _x_test[mask]
x_test_B = preprocess_params_transform(x_test, pre_parameters_CR)
x_test_S = preprocess_params_transform(x_test, pre_parameters_SR)


if not args.shuffle_split: 
    # pass   
    data_train_S, data_val_S = train_test_split(x_train_S, test_size=0.2, random_state=args.seed)
    data_train_B, data_val_B = train_test_split(x_train_B, test_size=0.2, random_state=args.seed)

else:
    ss_data = ShuffleSplit(n_splits=20, test_size=args.validation_fraction, random_state=22)

    print(f'doing a shuffle split with split number {args.split}')

    for i, (train_index, test_index) in enumerate(ss_data.split(x_train)):
        if i == args.split:
            data_train_S, data_val_S = x_train_S[train_index], x_train_S[test_index]
            data_train_B, data_val_B = x_train_B[train_index], x_train_B[test_index]
            break



traintensor_S = torch.from_numpy(data_train_S.astype('float32')).to(device)
traintensor_B = torch.from_numpy(data_train_B.astype('float32')).to(device)
traintensor_S[:,0]-=3.5

valtensor_S = torch.from_numpy(data_val_S.astype('float32')).to(device)
valtensor_B = torch.from_numpy(data_val_B.astype('float32')).to(device)
valtensor_S[:,0]-=3.5

testtensor_S = torch.from_numpy(x_test_S.astype('float32')).to(device)
testtensor_B = torch.from_numpy(x_test_B.astype('float32')).to(device)
testtensor_S[:,0]-=3.5

print('X_train shape', traintensor_S.shape)
print('X_val shape', valtensor_S.shape)
print('X_test shape', testtensor_S.shape)

pre_parameters_CR_tensor = pre_parameters_CR.copy()
pre_parameters_SR_tensor = pre_parameters_SR.copy()

for key in pre_parameters_CR_tensor.keys():
    pre_parameters_CR_tensor[key] = torch.from_numpy(pre_parameters_CR_tensor[key].astype('float32')).to(device)

for key in pre_parameters_SR_tensor.keys():
    pre_parameters_SR_tensor[key] = torch.from_numpy(pre_parameters_SR_tensor[key].astype('float32')).to(device)

if args.mode_background == 'train':
    pass

elif args.mode_background == 'freeze':
    val_losses = np.load(f'{args.CR_path}/valloss_list.npy')
    best_epochs = np.argsort(val_losses)[0:10]

elif args.mode_background == 'pretrained':
    val_losses = np.load(f'{args.CR_path}/my_ANODE_model_val_losses.npy')
    best_epoch = np.argmin(val_losses)
    model_B = DensityEstimator(args.config_file, eval_mode=False, load_path=f"{args.CR_path}/my_ANODE_model_epoch_{best_epoch}.par", device=device)


#######################################################
# load CR model and evaluate on train and val data
log_B_val = []

for i in best_epochs:
    model_B = DensityEstimator(args.config_file, eval_mode=True, load_path=f"{args.CR_path}/model_CR_{i}.pt", device=device)
    log_B_ = model_B.model.log_probs(inputs=valtensor_B[:,1:-1], cond_inputs=valtensor_B[:,0].reshape(-1,1))
    log_B_val.append(log_B_.detach().cpu().numpy())

log_B_val = np.array(log_B_val)
B_val = np.exp(log_B_val)
B_val = np.mean(B_val, axis=0)
log_B_val = np.log(B_val + 1e-32)


log_B_train = []
for i in best_epochs:
    model_B = DensityEstimator(args.config_file, eval_mode=True, load_path=f"{args.CR_path}/model_CR_{i}.pt", device=device)
    log_B_ = model_B.model.log_probs(inputs=traintensor_B[:,1:-1], cond_inputs=traintensor_B[:,0].reshape(-1,1))
    log_B_train.append(log_B_.detach().cpu().numpy())

log_B_train = np.array(log_B_train)
B_train = np.exp(log_B_train)
B_train = np.mean(B_train, axis=0)
log_B_train = np.log(B_train + 1e-32)

log_B_train_tensor = torch.from_numpy(log_B_train.astype('float32')).to(device)
log_B_val_tensor = torch.from_numpy(log_B_val.astype('float32')).to(device)

############################################

train_mass_prob_back = torch.from_numpy(density_back.pdf(traintensor_B[:,0].cpu().detach().numpy())).to(device)
val_mass_prob_back = torch.from_numpy(density_back.pdf(valtensor_B[:,0].cpu().detach().numpy())).to(device)
test_mass_prob_back = torch.from_numpy(density_back.pdf(testtensor_B[:,0].cpu().detach().numpy())).to(device)

train_tensor = torch.utils.data.TensorDataset(traintensor_S, log_B_train_tensor, train_mass_prob_back)
val_tensor = torch.utils.data.TensorDataset(valtensor_S, log_B_val_tensor, val_mass_prob_back)


batch_size = args.batch_size
trainloader = torch.utils.data.DataLoader(train_tensor, batch_size=batch_size, shuffle=True)

test_batch_size=batch_size*5
valloader = torch.utils.data.DataLoader(val_tensor, batch_size=test_batch_size, shuffle=False)

# initiate model
model_S = flows_model_RQS(device=device, num_features=5, context_features=None)
#print(model_S)



valloss = []
trainloss = []

pre_parameters = {}

pre_parameters['CR'] = pre_parameters_CR_tensor
pre_parameters['SR'] = pre_parameters_SR_tensor

# needed for sampling
#train_data = inverse_transform(traintensor_B, pre_parameters['CR']).cpu().detach().numpy()
#val_data = inverse_transform(valtensor_B, pre_parameters['CR']).cpu().detach().numpy()
#all_data = np.vstack((train_data, val_data))
if not args.w_scan:
    if not args.w_train:
        w_ = true_w
    else:
        if not args.random_w:
            w_ = torch.tensor(inverse_sigmoid(args.w), requires_grad=True, device=device)
        else:
            np.random.seed(args.split)
            w_ = torch.tensor(inverse_sigmoid(np.random.uniform(0.01,0.001)), requires_grad=True, device=device)
else:
    w_ = args.w

epochs = args.epochs

if args.w_train:
    optimizer = torch.optim.AdamW(list(model_S.parameters()) + [w_] ,lr=3e-4)
else:
    optimizer = torch.optim.AdamW(model_S.parameters(),lr=3e-4)
#scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=0.01, steps_per_epoch=len(trainloader), epochs=epochs, anneal_strategy='linear')
#scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=1e-4, max_lr=1e-2,cycle_momentum=False, step_size_up = len(trainloader)*30)


for epoch in range(args.epochs):

    train_loss = r_anode_mass_joint_untransformed(model_S,model_B.model,w_,optimizer ,trainloader, 
                         pre_parameters, device=device, mode='train',\
                          data_loss_expr=args.data_loss_expr, w_train=args.w_train)
    val_loss = r_anode_mass_joint_untransformed(model_S,model_B.model,w_,optimizer, valloader, 
                       pre_parameters, device=device, mode='val',\
                        data_loss_expr=args.data_loss_expr,w_train=args.w_train)


    ##################################
    ##############################
    # Save model and weights

    torch.save(model_S.state_dict(), f'results/{args.wandb_group}/{args.wandb_job_type}/{args.wandb_run_name}/model_S_{epoch}.pt')
    if args.w_train:
        np.save(f'results/{args.wandb_group}/{args.wandb_job_type}/{args.wandb_run_name}/w_{epoch}.npy', torch.sigmoid(w_).item())

    
    if args.mode_background == 'train' or args.mode_background == 'pretrained':
        torch.save(model_B.model.state_dict(), f'results/{args.wandb_group}/{args.wandb_job_type}/{args.wandb_run_name}/model_B_{epoch}.pt')

    # generate logs
    if args.wandb:
            if args.w_train:
                if not args.no_signal_fit:
                    wandb.log({'train_loss': train_loss, 'val_loss': val_loss, 
                           'w': torch.sigmoid(w_).item(), 'true_w': true_w})
                else:
                    wandb.log({'train_loss': train_loss, 'val_loss': val_loss, 
                           'w': torch.sigmoid(w_).item()})   
            else:
                wandb.log({'train_loss': train_loss, 'val_loss': val_loss,                 
                        'true_w': true_w, 'w': w_})


    if (np.isnan(train_loss) or np.isnan(val_loss)):
        print(' nan loss ')
       # if args.wandb:
        #    wandb.finish()
        #break

    

    print('epoch: ', epoch, 'trainloss: ', train_loss, 'valloss: ', val_loss)
    valloss.append(val_loss)
    trainloss.append(train_loss)

    # generate samples every 10 epochs (to keep track of the training)
    if epoch % 50 == 0:
        if ~(np.isnan(train_loss) or np.isnan(val_loss)):
            model_S.eval()
            with torch.no_grad():
                x_samples = model_S.sample(len(train_tensor))
            x_samples = x_samples.reshape(-1,5)
            x_samples = torch.hstack((x_samples, torch.ones((len(x_samples),1)).to(device)))
            x_samples = inverse_transform(x_samples, pre_parameters['SR']).cpu().detach().numpy()
            x_samples = x_samples[~np.isnan(x_samples).any(axis=1)]
            x_samples[:,0]+=3.5
            print('x_samples shape', x_samples.shape)
           # print('all_data shape', all_data.shape)

            figure = plt.figure(figsize=(5,5))
            bins_0 = np.linspace(3.3, 3.7, 50)
            for i in range(0,5):
                plt.subplot(3,2,i+1)
                #if dims > 1:
                if i == 0:
                    plt.hist(x_test[:,i][x_test[:,-1]==1],bins=bins_0, density=True, label=f'sig', histtype='step')
                    plt.hist(x_test[:,i][x_test[:,-1]==0],bins=bins_0, density=True, label=f'back', histtype='step')
                    plt.hist(x_samples[:,i],bins=bins_0, density=True, label=f'sample', histtype='step')
                    # plt.legend(loc='upper right')
                else:
                    plt.hist(x_test[:,i][x_test[:,-1]==1],bins=50, density=True, label=f'sig', histtype='step')
                    plt.hist(x_test[:,i][x_test[:,-1]==0],bins=50, density=True, label=f'back', histtype='step')
                    plt.hist(x_samples[:,i],bins=50, density=True, label=f'sample', histtype='step')
            
            if args.wandb:
                wandb.log({f'nflow_S': wandb.Image(figure)})
            
            plt.savefig(f'results/{args.wandb_group}/{args.wandb_job_type}/{args.wandb_run_name}/nflow_S_{epoch}.png')
            plt.close()


############################################
# Load best models and ensemble signal
if not args.ensemble:
    index = np.argmin(valloss).flatten()[0]

    model_S.load_state_dict(torch.load(f'results/{args.wandb_group}/{args.wandb_job_type}/{args.wandb_run_name}/model_S_{index}.pt'))
    
    model_S.eval()
    log_S = evaluate_log_prob(model_S.model, testtensor_S, 
                                pre_parameters['SR'], transform = True).cpu().detach().numpy()
    #S = np.exp(log_S)


else:
    log_S = []
    sorted_index = np.argsort(valloss).flatten()[0:10]
    for index in sorted_index:

        model_S.load_state_dict(torch.load(f'results/{args.wandb_group}/{args.wandb_job_type}/{args.wandb_run_name}/model_S_{index}.pt'))
        
        model_S.eval()
        log_S_ = []
       # for i, data in enumerate(testloader):
        with torch.no_grad():
                #log_S_.extend(model_S.log_prob(data[:,:-1]).cpu().detach().numpy().tolist())
            #log_S_ = evaluate_log_prob_mass(model_S, testtensor_S,preprocessing_params=pre_parameters['SR'], transform=True, mode='test').cpu().detach().numpy()
            log_S_ = model_S.log_prob(testtensor_S[:,:-1]).cpu().detach().numpy()                                  
           
        #log_S_ = evaluate_log_prob(model_S.model, testtensor_S, 
                                    #pre_parameters['SR'], transform = True).cpu().detach().numpy()


        log_S.append(log_S_)

    log_S = np.array(log_S)
    S = np.exp(log_S)
    S = np.mean(S, axis=0)
    log_S = np.log(S + 1e-32)




if args.mode_background == 'train' or args.mode_background == 'pretrained':
    model_B.model.load_state_dict(torch.load(f'results/{args.wandb_group}/{args.wandb_job_type}/{args.wandb_run_name}/model_B_{index}.pt'))


model_S.eval()

# save lowest 10 epochs and delete the rest (otherwise it takes too much space)
not_lowest_10 = np.argsort(valloss)[10:]
file_list = ['model_S_{}.pt'.format(i) for i in not_lowest_10]
print(f'Deleting models not in lowest 10 epochs')
for file_ in file_list:
    os.remove(f'results/{args.wandb_group}/{args.wandb_job_type}/{args.wandb_run_name}/{file_}')



# load CR model and evaluate on test_data
val_losses = np.load(f'{args.CR_path}/valloss_list.npy')

if not args.ensemble:
    best_epoch = np.argsort(val_losses)[0]
else:
    best_epoch = np.argsort(val_losses)[0:10]


model_B = DensityEstimator(args.config_file, eval_mode=True, device=device)

log_B = []
for epoch in best_epoch:

    model_B.model.load_state_dict(torch.load(f'{args.CR_path}/model_CR_{epoch}.pt'))

    model_B.model.eval()

    with torch.no_grad():
        log_p = evaluate_log_prob(model_B.model, testtensor_B, pre_parameters['CR'], 
                                  transform=False).cpu().detach().numpy()
       # log_p = model_B.log_prob(traintensor_B[:,1:-1],
        #                      context=traintensor_B[:,0].reshape(-1,1)).cpu().detach().numpy()
        log_B.append(log_p)

log_B = np.array(log_B)
B = np.exp(log_B)
B = np.mean(B, axis=0)
log_B = np.log(B + 1e-32)

np.save(f'results/{args.wandb_group}/{args.wandb_job_type}/{args.wandb_run_name}/ensemble_B.npy', B)
np.save(f'results/{args.wandb_group}/{args.wandb_job_type}/{args.wandb_run_name}/ensemble_S_joint.npy', S)

############################################

#########################################
# save signal samples
samples_all = []
for index in sorted_index:

    model_S.load_state_dict(torch.load(f'results/{args.wandb_group}/{args.wandb_job_type}/{args.wandb_run_name}/model_S_{index}.pt'))
        
    model_S.eval()
    with torch.no_grad():
        samples = model_S.sample(10000)
        samples = samples.reshape(-1,5)
        samples = torch.hstack((samples, torch.ones((len(samples),1)).to(device)))
        samples = inverse_transform(samples, pre_parameters['SR']).cpu().detach().numpy()
        samples_all.append(samples)


samples_all = np.array(samples_all)
samples_all = np.concatenate(samples_all, axis=0)
samples_all+=3.5
np.save(f'results/{args.wandb_group}/{args.wandb_job_type}/{args.wandb_run_name}/samples.npy', samples_all)

############################################
############################################

# compute likelihood
hist_sig = np.histogram(samples_all[:,0], bins=bins, density=True)
density_sig = rv_histogram(hist_sig)

log_test_mass = np.log(density_sig.pdf(x_test[:,0])+10e-32)
likelihood_ = log_S - log_B - log_test_mass

np.save(f'results/{args.wandb_group}/{args.wandb_job_type}/{args.wandb_run_name}/ensemble_S.npy', np.exp(log_S - log_test_mass))

likelihood = np.nan_to_num(likelihood_, nan=0, posinf=0, neginf=0)        


# compute SIC
sic_score , tpr_score , auc_score = SIC(x_test[:,-1], likelihood)

figure = plt.figure(figsize=(5,5))
plt.plot(tpr_score, sic_score)
plt.xlabel('tpr')
plt.ylabel('SIC')
plt.savefig(f'results/{args.wandb_group}/{args.wandb_job_type}/{args.wandb_run_name}/SIC.png')
if args.wandb:
    wandb.log({'SIC': wandb.Image(figure)})
plt.close()

if args.wandb:
    wandb.log({'max_SIC': np.amax(sic_score), 'AUC': auc_score})


############################################
# save losses
    
np.save(f'results/{args.wandb_group}/{args.wandb_job_type}/{args.wandb_run_name}/valloss.npy', valloss)
np.save(f'results/{args.wandb_group}/{args.wandb_job_type}/{args.wandb_run_name}/trainloss.npy', trainloss)

if args.wandb:
    wandb.finish()






