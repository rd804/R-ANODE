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


parser = argparse.ArgumentParser()
parser.add_argument('--sig_train',  default=10, help='signal train')
parser.add_argument('--sig_test', default=10, help='signal test')
parser.add_argument('--mode_background', type=str, default='train', help='train, freeze, pretrained')

parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
parser.add_argument('--clip_grad', default=False, help='clip grad')
parser.add_argument('--epochs', type=int, default=20, help='epochs')
parser.add_argument('--mini_batch', type=int, default=128, help='mini batch size')
parser.add_argument('--gpu', type=str, default='cuda:0', help='gpu to train on')
parser.add_argument('--data_loss_expr', type=str, default='true_likelihood', help='loss for SR region')

parser.add_argument('--w_train', action='store_true', help='train w if true, else fix w to initial value value')
parser.add_argument('--true_w', action='store_true', help='use true w, as initial value for w')
parser.add_argument('--resample', action='store_true', help='if data is to resampled')

parser.add_argument('--seed', type=int, default=10, help='seed')


parser.add_argument('--w', type=float, default=0.0, help='initial for SR region')
parser.add_argument('--cap_sig', type=float, default=1.0, help='capping the maximum value of sigmoid function for w: (cap_sig)/(1+exp(-x))')
parser.add_argument('--scale_sig', type=float, default=1.0, help='scaling the sigmoid function for w: 1/(1+exp(-scale_sig * w))')
parser.add_argument('--kld_w', type=float, default=1.0)

parser.add_argument('--wandb_group', type=str, default='test')
parser.add_argument('--wandb_job_type', type=str, default='CR')
parser.add_argument('--wandb_run_name', type=str, default='try_0')




args = parser.parse_args()

wandb.init(project="m-anode", config=args,
              group=args.wandb_group, job_type=args.wandb_job_type)

wandb.run.name = args.wandb_run_name
wandb.config['tailbound'] = 15

# print wandb group


CUDA = True
device = torch.device(args.gpu if CUDA else "cpu")

kwargs = {'num_workers': 4, 'pin_memory': True} if CUDA else {}
kwargs = {}


back_mean = 0
sig_mean = 3
sig_simga = 0.5
back_sigma = 3

with open('data/true_w.pkl', 'rb') as f:
    true_w = pickle.load(f)

with open('data/background.pkl', 'rb') as f:
    background = pickle.load(f)

with open('data/data.pkl', 'rb') as f:
    data = pickle.load(f)


if not args.resample:

    sig_train = args.sig_train
    x_train = data[str(sig_train)]['train']['data']

else:
    # set seed
    sig_train = args.sig_train

    n_back = 200000
    n_sig = int(np.sqrt(n_back) * float(sig_train))

    # set seed
    np.random.seed(args.seed)
    x_back = np.random.normal(back_mean, back_sigma, n_back)
    np.random.seed(args.seed)
    x_sig = np.random.normal(sig_mean, sig_simga, n_sig)

    x_train = np.concatenate((x_back, x_sig), axis=0)
    x_train = shuffle(x_train, random_state=args.seed)

    print('resampled data with seed %d' % args.seed)
    print('amount of background: %d' % n_back)
    print('amount of signal: %d' % n_sig)




    # sample traindata

    


_X_train = np.concatenate((x_train, background), axis=0)
_y_train = np.concatenate((np.ones(len(x_train)), np.zeros(len(background))), axis=0)






_X_train, _y_train  = shuffle(_X_train, _y_train, random_state=10)


X_train , X_val = train_test_split(_X_train, test_size=0.5, random_state=22)
y_train , y_val = train_test_split(_y_train, test_size=0.5, random_state=22)

x_test = data[str(args.sig_test)]['val']['data']
label_test = data[args.sig_test]['val']['label']


batch_size = args.mini_batch

X_train = torch.from_numpy(X_train.reshape(-1,1)).float()
y_train = torch.from_numpy(y_train.reshape(-1,1)).float()

X_val = torch.from_numpy(X_val.reshape(-1,1)).float()
y_val = torch.from_numpy(y_val.reshape(-1,1)).float()

traindataset = torch.utils.data.TensorDataset(X_train, y_train)
trainloader = torch.utils.data.DataLoader(traindataset, batch_size=batch_size, shuffle=True)

valdataset = torch.utils.data.TensorDataset(X_val, y_val)
valloader = torch.utils.data.DataLoader(valdataset, batch_size=batch_size*5, shuffle=False)

testtensor = torch.from_numpy(x_test.reshape(-1,1)).float()

model_S=define_model(nfeatures=1,nhidden=2,hidden_size=20,embedding=None,dropout=0,nembedding=0, device=device)
model_B=define_model(nfeatures=1,nhidden=2,hidden_size=20,embedding=None,dropout=0,nembedding=0, device=device)

if args.w_train:
    w = torch.tensor(inverse_sigmoid(args.w), requires_grad=False, device=device)

else:
    if args.true_w:
        w = torch.tensor(inverse_sigmoid(true_w[str(sig_train)][0]), requires_grad=False, device=device)
    else:
        w = torch.tensor(inverse_sigmoid(args.w), requires_grad=False, device=device)


if args.mode_background == 'train':
    optimizer = torch.optim.Adam(list(model_S.parameters()) + list(model_B.parameters()) + [w], lr=args.lr)
elif args.mode_background == 'freeze':
    valloss = np.load('results/nflows_gaussian_mixture_1/CR/try_0/valloss_list_background.npy')

    index = np.argmin(valloss).flatten()[0]

    model_B.load_state_dict(torch.load(f'results/nflows_gaussian_mixture_1/CR/try_0/model_CR_{index}.pt'))
    model_B.eval()
    if args.w_train:
        optimizer = torch.optim.Adam(list(model_S.parameters()) + [w], lr=args.lr)
    else:
        optimizer = torch.optim.Adam(list(model_S.parameters()), lr=args.lr)

elif args.mode_background == 'pretrained':
    valloss = np.load('results/nflows_gaussian_mixture_1/CR/try_0/valloss_list_background.npy')

    index = np.argmin(valloss).flatten()[0]

    model_B.load_state_dict(torch.load(f'results/nflows_gaussian_mixture_1/CR/try_0/model_CR_{index}.pt'))


    if args.w_train:
        optimizer = torch.optim.Adam(list(model_S.parameters()) + list(model_B.parameters()) + [w], lr=args.lr)
    else:
        optimizer = torch.optim.Adam(list(model_S.parameters()) + list(model_B.parameters()), lr=args.lr)





valloss = []
trainloss = []


if not os.path.exists(f'results/{args.wandb_group}/{args.wandb_job_type}/{wandb.run.name}'):
    os.makedirs(f'results/{args.wandb_group}/{args.wandb_job_type}/{wandb.run.name}')


for epoch in range(args.epochs):

    train_loss = m_anode(model_S,model_B,w,optimizer,trainloader,noise_data=0,noise_context=0, device=device, mode='train',\
                         mode_background=args.mode_background, clip_grad=args.clip_grad, data_loss_expr=args.data_loss_expr,
                         w_train=args.w_train, cap_sig=args.cap_sig, scale_sig=args.scale_sig, kld_w=args.kld_w)
    val_loss = m_anode(model_S,model_B,w,optimizer,valloader,noise_data=0,noise_context=0, device=device, mode='val',\
                       mode_background=args.mode_background, clip_grad=args.clip_grad, data_loss_expr=args.data_loss_expr,
                       w_train=args.w_train, cap_sig=args.cap_sig, scale_sig=args.scale_sig, kld_w=args.kld_w)

    if args.data_loss_expr == 'capped_sigmoid':
        w_ = capped_sigmoid(w, args.cap_sig).item()
    elif args.data_loss_expr == 'scaled_sigmoid':
        w_ = scaled_sigmoid(w, args.scale_sig).item()
    else:
        w_ = torch.sigmoid(w).item()


    ##################################
    ##############################
    # Save model and weights

    torch.save(model_S.state_dict(), f'results/{args.wandb_group}/{args.wandb_job_type}/{wandb.run.name}/model_S_{epoch}.pt')
    np.save(f'results/{args.wandb_group}/{args.wandb_job_type}/{wandb.run.name}/w_{epoch}.npy', w_)

    if args.mode_background == 'train' or args.mode_background == 'pretrained':
        torch.save(model_B.state_dict(), f'results/{args.wandb_group}/{args.wandb_job_type}/{wandb.run.name}/model_B_{epoch}.pt')

    
    


    if np.isnan(train_loss) or np.isnan(val_loss):
        print(' nan loss ')
        wandb.finish()
        break

    wandb.log({'train_loss': train_loss, 'val_loss': val_loss, 'w': w_ , \
               'true_w': true_w[str(sig_train)][0]})

    valloss.append(val_loss)
    trainloss.append(train_loss)



if ~np.isnan(train_loss) or ~np.isnan(val_loss):
    np.save(f'results/{args.wandb_group}/{args.wandb_job_type}/{wandb.run.name}/valloss.npy', valloss)
    np.save(f'results/{args.wandb_group}/{args.wandb_job_type}/{wandb.run.name}/trainloss.npy', trainloss)

    # Load best model
    index = np.argmin(valloss).flatten()[0]
    model_S.load_state_dict(torch.load(f'results/{args.wandb_group}/{args.wandb_job_type}/{wandb.run.name}/model_S_{index}.pt'))
    w_ = np.load(f'results/{args.wandb_group}/{args.wandb_job_type}/{wandb.run.name}/w_{index}.npy')

    if args.mode_background == 'train' or args.mode_background == 'pretrained':
        model_B.load_state_dict(torch.load(f'results/{args.wandb_group}/{args.wandb_job_type}/{wandb.run.name}/model_B_{index}.pt'))


    true_likelihoods = {}
    true_likelihoods[str(sig_train)] = {}


    w1 = true_w[str(sig_train)][0]
    w2 = true_w[str(sig_train)][1]

    true_likelihoods[str(sig_train)] = p_data(x_test,[sig_mean, back_mean],[sig_simga**2,back_sigma**2],[w1,w2])/p_back(x_test,back_mean,back_sigma**2)


    score_likelihoods = {}
    score_likelihoods[str(sig_train)] = {}

    wandb.log({'Best learned weight': w_})

    model_S.eval()
    model_B.eval()
    with torch.no_grad():
        log_S = model_S.log_prob(testtensor.to(device)).cpu().detach().numpy()
        log_B = model_B.log_prob(testtensor.to(device)).cpu().detach().numpy()

        data  = w_ * np.exp(log_S) + (1-w_) * np.exp(log_B)
        back = np.exp(log_B)

        likelihood_ = data/back

        likelihood_ = np.nan_to_num(likelihood_, nan=0, posinf=0, neginf=0)

        score_likelihoods[str(sig_train)] = likelihood_
        

    sic_true , tpr_true , auc_true = SIC(label_test, true_likelihoods[str(sig_train)])
    sic_score , tpr_score , auc_score = SIC(label_test, score_likelihoods[str(sig_train)])

    figure = plt.figure()

    plt.plot(tpr_true, sic_true, label='true')
    plt.plot(tpr_score, sic_score, label='score')

    plt.legend(loc='lower right')
    wandb.log({'SIC': wandb.Image(figure)})
    plt.savefig(f'results/{args.wandb_group}/{args.wandb_job_type}/{wandb.run.name}/SIC.png')
    plt.show()

    wandb.log({'AUC': auc_score, 'max SIC': np.max(sic_score)})

    figure = plt.figure()
    bins = np.linspace(min(X_train), max(X_train), 100).flatten()
    Background = model_B.log_prob(torch.from_numpy(bins.reshape(-1,1)).float().to(device)).exp().detach().cpu().numpy()
    Signal = model_S.log_prob(torch.from_numpy(bins.reshape(-1,1)).float().to(device)).exp().detach().cpu().numpy()

    Data = w_ * Signal + (1-w_) * Background

    plt.plot(bins, Background, label='model B')
    plt.plot(bins, Signal, label='model S')
    plt.plot(bins, Data, label='w * S + (1-w) * B with w=%.5f' % w_)
    plt.hist(X_train[y_train==0], bins=bins, label='back' , density=True, histtype='step')
    plt.hist(X_train[y_train==1], bins=bins, label='data', density=True, histtype='step')
    plt.legend(frameon=False)
    wandb.log({'model': wandb.Image(figure)})
    plt.savefig(f'results/{args.wandb_group}/{args.wandb_job_type}/{wandb.run.name}/model.png')
    plt.show()

    wandb.finish()






