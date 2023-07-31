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
import argparse
import pickle
import wandb
from sklearn.model_selection import ShuffleSplit


parser = argparse.ArgumentParser()
parser.add_argument('--sig_train',  default=10, help='signal train')
parser.add_argument('--sig_test', default=10, help='signal test')
parser.add_argument('--mode_background', type=str, default='train', help='train, freeze, pretrained')

parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
parser.add_argument('--epochs', type=int, default=20, help='epochs')
parser.add_argument('--mini_batch', type=int, default=128, help='mini batch size')
parser.add_argument('--gpu', type=str, default='cuda:0', help='gpu to train on')
parser.add_argument('--data_loss_expr', type=str, default='true_likelihood', help='loss for SR region')

parser.add_argument('--ensemble', action = 'store_true', help='ensemble')
parser.add_argument('--resample', action = 'store_true', help='resample')
parser.add_argument('--true_w', action = 'store_true', help='true_w')
parser.add_argument('--cross_validate', action = 'store_true', help='cross_validate')

                    
parser.add_argument('--seed', type=int, default=22, help='seed')
parser.add_argument('--gaussian_dim', type=int, default=1)

parser.add_argument('--w', type=float, default=0.0, help='initial for SR region')

parser.add_argument('--wandb_group', type=str, default='test')
parser.add_argument('--wandb_job_type', type=str, default='CR')
parser.add_argument('--wandb_run_name', type=str, default='try_0')




args = parser.parse_args()

wandb.init(project="m-anode", config=args,
              group=args.wandb_group, job_type=args.wandb_job_type)

wandb.run.name = args.wandb_run_name

# print wandb group


CUDA = True
device = torch.device(args.gpu if CUDA else "cpu")

kwargs = {'num_workers': 4, 'pin_memory': True} if CUDA else {}
kwargs = {}


# load data
if args.gaussian_dim == 2:
    back_mean = 0
    sig_mean = 2
    sig_sigma = 0.25
    back_sigma = 3


sig_train = args.sig_train

n_back = 10_000
n_sig = int(np.sqrt(n_back) * float(sig_train))

true_w = n_sig/(n_sig + n_back)

# generate random sample of training data
x_back = np.random.normal(back_mean, back_sigma, (n_back, args.gaussian_dim))
x_sig = np.random.normal(sig_mean, sig_sigma, (n_sig, args.gaussian_dim))
x_train = np.concatenate((x_back, x_sig), axis=0)
x_train = shuffle(x_train)
_y_train = np.ones(len(x_train))


print('amount of background: %d' % n_back)
print('amount of signal: %d' % n_sig)
print(f'total amount of data: {len(x_train)}')

# shuffle data
_X_train, _y_train  = shuffle(x_train, _y_train)

# split data into train and validation
X_train , X_val = train_test_split(_X_train, test_size=0.1, random_state=22)
y_train , y_val = train_test_split(_y_train, test_size=0.1, random_state=22)


#shuffle_split = ShuffleSplit(n_splits=10, test_size=0.1, random_state=0)

#_data_train = {}
#_data_val = {}

#for split, train, val in enumerate(shuffle_split.split(x_train)):
 #   data_train = x_train[train]
  #  data_val = x_train[val]

   # _data_train[split] = data_train
    #_data_val[split] = data_val


test_signal = np.random.normal(sig_mean, sig_sigma, (1000000, 2))
test_background = np.random.normal(back_mean, back_sigma, (1000000, 2))

x_test = np.concatenate((test_signal, test_background), axis=0)
label_test = np.concatenate((np.ones(len(test_signal)), np.zeros(len(test_background))), axis=0)


# define test data
#x_test = data[str(args.sig_test)]['val']['data']
#label_test = data[str(args.sig_test)]['val']['label']

# define minibatch
batch_size = args.mini_batch


X_train = torch.from_numpy(X_train).float()
y_train = torch.from_numpy(y_train.reshape(-1,1))
X_val = torch.from_numpy(X_val).float()
y_val = torch.from_numpy(y_val.reshape(-1,1))

print('X_train shape', X_train.shape)
print('X_val shape', X_val.shape)

# define dataloaders
traindataset = torch.utils.data.TensorDataset(X_train, y_train)
trainloader = torch.utils.data.DataLoader(traindataset, batch_size=batch_size, shuffle=True)

valdataset = torch.utils.data.TensorDataset(X_val, y_val)
valloader = torch.utils.data.DataLoader(valdataset, batch_size=batch_size*5, shuffle=False)

testtensor = torch.from_numpy(x_test).float()

# models for background and signal
model_S = flows_for_gaussian(gaussian_dim = 2, num_transforms = 2, num_blocks = 3, 
                       hidden_features = 32, device = device)
model_B = gaussian_prob(back_mean, back_sigma, dim = 2)
w = torch.tensor(inverse_sigmoid(true_w), requires_grad=False, device=device)


# define optimizer
model_B.eval()
optimizer = torch.optim.Adam(list(model_S.parameters()), lr=args.lr)


valloss = []
trainloss = []


if not os.path.exists(f'results/{args.wandb_group}/{args.wandb_job_type}/{wandb.run.name}'):
    os.makedirs(f'results/{args.wandb_group}/{args.wandb_job_type}/{wandb.run.name}')


for epoch in range(args.epochs):

    # define train and val loss
    train_loss = m_anode_test(model_S,model_B,w,optimizer,trainloader,device=device, mode='train',\
                         data_loss_expr=args.data_loss_expr)
    val_loss = m_anode_test(model_S,model_B,w,optimizer,valloader,device=device, mode='val',\
                       data_loss_expr=args.data_loss_expr)

    w_ = torch.sigmoid(w).item()

    torch.save(model_S.state_dict(), f'results/{args.wandb_group}/{args.wandb_job_type}/{wandb.run.name}/model_S_{epoch}.pt')
    np.save(f'results/{args.wandb_group}/{args.wandb_job_type}/{wandb.run.name}/w_{epoch}.npy', w_)

    wandb.log({'train_loss': train_loss, 'val_loss': val_loss, 'w': w_ , \
               'true_w': true_w})

    valloss.append(val_loss)
    trainloss.append(train_loss)



np.save(f'results/{args.wandb_group}/{args.wandb_job_type}/{wandb.run.name}/valloss.npy', valloss)
np.save(f'results/{args.wandb_group}/{args.wandb_job_type}/{wandb.run.name}/trainloss.npy', trainloss)

# load the last model
sorted_index = args.epochs - 1 
model_S.load_state_dict(torch.load(f'results/{args.wandb_group}/{args.wandb_job_type}/{wandb.run.name}/model_S_{sorted_index}.pt'))
model_S.eval()
log_S = model_S.log_prob(testtensor.to(device)).cpu().detach().numpy()


w_ = np.load(f'results/{args.wandb_group}/{args.wandb_job_type}/{wandb.run.name}/w_{sorted_index}.npy')


w1 = true_w
w2 = 1-true_w

true_likelihoods = p_data(x_test,[sig_mean, back_mean],[sig_sigma,back_sigma],
            [w1, w2], dim=args.gaussian_dim)/p_back(x_test,back_mean, back_sigma, dim=args.gaussian_dim)


score_likelihoods = {}

model_B.eval()
with torch.no_grad():
    log_B = model_B.log_prob(torch.from_numpy(x_test).float()).numpy()

assert log_S.shape == log_B.shape
likelihood_ = log_S - log_B

likelihood_ = np.nan_to_num(likelihood_, nan=0, posinf=0, neginf=0)
score_likelihoods = likelihood_
        

sic_true , tpr_true , auc_true = SIC(label_test, true_likelihoods)
sic_score , tpr_score , auc_score = SIC(label_test, score_likelihoods)

figure = plt.figure()

plt.plot(tpr_true, sic_true, label='true')
plt.plot(tpr_score, sic_score, label='score')
plt.legend(loc='lower right')
wandb.log({'SIC': wandb.Image(figure)})
plt.savefig(f'results/{args.wandb_group}/{args.wandb_job_type}/{wandb.run.name}/SIC.png')
plt.show()

wandb.log({'AUC': auc_score, 'max SIC': np.max(sic_score)})



wandb.finish()

