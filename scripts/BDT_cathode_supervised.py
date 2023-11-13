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

# import sklearn sample_weights
from sklearn.utils.class_weight import compute_sample_weight

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
from sklearn.ensemble import HistGradientBoostingClassifier

# [1000,600,450,300,225,150,75]

parser = argparse.ArgumentParser()
parser.add_argument('--n_sig',type=int , default=75, help='signal train')
parser.add_argument('--mode_background', type=str, default='freeze', help='train, freeze, pretrained')

parser.add_argument('--epochs', type=int, default=2, help='epochs')
parser.add_argument('--batch_size', type=int, default = 256, help = 'batch size')
parser.add_argument('--mini_batch', type=int, default=256, help='mini batch size')
parser.add_argument('--gpu', type=str, default='cuda:0', help='gpu to train on')

parser.add_argument('--resample', action='store_true', help='if data is to resampled')
parser.add_argument('--seed', type=int, default=22, help='seed')
parser.add_argument('--shuffle_split', action='store_true', help='if shuffle split is used')
#parser.add_argument('--split', type=int, default=1, help='split number')
parser.add_argument('--data_dir', type=str, default='data/lhc_co', help='data directory')

parser.add_argument('--CR_path', type=str, default='results/nflows_lhc_co/CR_RQS_4_1000/try_1_2', help='CR data path')
parser.add_argument('--ensemble', action='store_true',default = True ,help='if ensemble is used')
parser.add_argument('--ensemble_size', type=int, default=50, help='ensemble size')

parser.add_argument('--wandb', action='store_true', help='if wandb is used' )
parser.add_argument('--wandb_group', type=str, default='debugging_r_anode')
parser.add_argument('--wandb_job_type', type=str, default='lhc_co')
parser.add_argument('--wandb_run_name', type=str, default='try_1')




args = parser.parse_args()

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

# load data
SR_data, CR_data , true_w, sigma = resample_split(args.data_dir, n_sig = args.n_sig, resample_seed = args.seed,resample = args.resample)

print('x_train shape', SR_data.shape)
print('true_w', true_w)
print('sigma', sigma)

if args.wandb:
    wandb.config.update({'true_w': true_w, 'sigma': sigma})



with open(f'{args.CR_path}/pre_parameters.pkl', 'rb') as f:
    pre_parameters = pickle.load(f)
background = np.load(f'{args.data_dir}/extrabkg_train_val.npy')
print(f'background shape: {background.shape}')

_, mask = logit_transform(SR_data[:,1:-1], pre_parameters['min'],
                             pre_parameters['max'])
x_train = SR_data[mask]

_, mask = logit_transform(background[:,1:-1], pre_parameters['min'],
                                pre_parameters['max'])
background = background[mask]

x_train_S = preprocess_params_transform(x_train, pre_parameters) 
background_S = preprocess_params_transform(background, pre_parameters)

labels = np.concatenate([x_train_S[:,-1], np.zeros(len(background_S))])
#labels = np.concatenate([np.ones(len(x_train_S)), np.zeros(len(background_S))])
#data = np.concatenate([x_train_S, background_S])[:,1:-1]
data = np.concatenate([x_train_S, background_S])[:,1:-1]

# sample weights using sklearn



# create masked test data
_x_test = np.load(f'{args.data_dir}/x_test.npy')
_, mask_test = logit_transform(_x_test[:,1:-1], pre_parameters['min'],
                                pre_parameters['max'])


sample_weights = compute_sample_weight(class_weight='balanced', y=labels)

x_test = _x_test[mask_test]
label_test = x_test[:,-1]

#x_test_S = preprocess_params_transform(x_test, pre_parameters)[:,1:-1]
x_test_S = preprocess_params_transform(x_test, pre_parameters)[:,1:-1]
print('x_test shape', x_test_S.shape)
print('x_train shape', x_train_S.shape)

from sklearn.preprocessing import StandardScaler
data[:,0]-=3.5
x_test_S[:,0]-=3.5

scaler = StandardScaler()
scaler.fit(data[:,0].reshape(-1,1))
mass_transformed = scaler.transform(data[:,0].reshape(-1,1))
data[:,0] = mass_transformed[:,0]
mass_transformed_test = scaler.transform(x_test_S[:,0].reshape(-1,1))
x_test_S[:,0] = mass_transformed_test[:,0]


ypred = []
for _shuffle in range(args.ensemble_size):
    clf = HistGradientBoostingClassifier(
        validation_fraction=0.5,max_iter=200, random_state=_shuffle,verbose=0)
    
    clf.fit(data, labels, sample_weight=sample_weights)
    predict = clf.predict_proba(x_test_S)[:,1]
    ypred.append(predict)

    pickle.dump(clf, open(f'results/{args.wandb_group}/{args.wandb_job_type}/{args.wandb_run_name}/clf_{_shuffle}.pkl', 'wb'))


ypred = np.array(ypred)
ypred = np.mean(ypred, axis=0)

sic_score , tpr_score , max_sic = SIC_cut(label_test, ypred)

np.save(f'results/{args.wandb_group}/{args.wandb_job_type}/{args.wandb_run_name}/ypred.npy', ypred)
np.save(f'results/{args.wandb_group}/{args.wandb_job_type}/{args.wandb_run_name}/mask_test.npy', mask_test)


figure = plt.figure()

plt.plot(tpr_score, sic_score, label='score')
plt.plot(tpr_score, tpr_score**0.5, label='random')
plt.xlabel('signal efficiency')
plt.ylabel('SIC')

plt.legend(loc='lower right')
if args.wandb:
    wandb.log({'SIC': wandb.Image(figure)})
    wandb.log({'max SIC': max_sic})

plt.savefig(f'results/{args.wandb_group}/{args.wandb_job_type}/{args.wandb_run_name}/SIC.png')
#plt.savefig('SIC.png')

plt.close()



    



