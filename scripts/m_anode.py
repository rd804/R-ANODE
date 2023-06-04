
# %%
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
parser.add_argument('--sig_train',  default=10)
parser.add_argument('--sig_test', default=10)
parser.add_argument('--mode_background', type=str, default='train')
parser.add_argument('--clip_grad', type=float, default=0.1)
parser.add_argument('--epochs', type=int, default=20)
parser.add_argument('--data_loss_expr', type=str, default='true_likelihood')
parser.add_argument('--w', type=float, default=0.0)
parser.add_argument('--gpu', type=str, default='cuda:0')
parser.add_argument('--wandb_group', type=str, default='test')
parser.add_argument('--wandb_job_type', type=str, default='CR')
parser.add_argument('--wandb_run_name', type=str, default='try_0')




args = parser.parse_args()

wandb.init(project="m-anode", config=args,
              group=args.wandb_group, job_type=args.wandb_job_type)

wandb.run.name = args.wandb_run_name


CUDA = True
device = torch.device(args.gpu if CUDA else "cpu")

kwargs = {'num_workers': 4, 'pin_memory': True} if CUDA else {}
kwargs = {}


with open('data/data.pkl', 'rb') as f:
    data = pickle.load(f)

back_mean = 0
sig_mean = 3
sig_simga = 0.5
back_sigma = 3

with open('data/true_w.pkl', 'rb') as f:
    true_w = pickle.load(f)

with open('data/background.pkl', 'rb') as f:
    background = pickle.load(f)


# fit train data
best_parameters = {}
run = 0

sig_train = 10

best_parameters[str(sig_train)] = {}

# Load train data
x_train = data[str(sig_train)]['train']['data']

_X_train = np.concatenate((x_train, background), axis=0)
_y_train = np.concatenate((np.ones(len(x_train)), np.zeros(len(background))), axis=0)






_X_train, _y_train  = shuffle(_X_train, _y_train, random_state=10)


X_train , X_val = train_test_split(_X_train, test_size=0.5, random_state=22)
y_train , y_val = train_test_split(_y_train, test_size=0.5, random_state=22)


x_test = data[str(args.sig_test)]['val']['data']
label_test = data[args.sig_test]['val']['label']


batch_size = 128

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

w = torch.tensor(args.w, requires_grad=True, device=device)

optimizer = torch.optim.Adam(list(model_S.parameters()) + list(model_B.parameters()) + [w])


valloss = []
trainloss = []

if not os.path.exists('models'):
    os.makedirs('models')

if not os.path.exists('results/' + wandb.run.name):
    os.makedirs('results/' + wandb.run.name)


for epoch in range(args.epochs):

    train_loss = m_anode(model_S,model_B,w,optimizer,trainloader,noise_data=0,noise_context=0, device=device, mode='train',\
                         mode_background=args.mode_background, clip_grad=args.clip_grad, data_loss_expr=args.data_loss_expr)
    val_loss = m_anode(model_S,model_B,w,optimizer,valloader,noise_data=0,noise_context=0, device=device, mode='val',\
                       mode_background=args.mode_background, clip_grad=args.clip_grad, data_loss_expr=args.data_loss_expr)


    print('epoch: %d, train_loss: %.3f, val_loss: %.3f' % (epoch, train_loss, val_loss))

    # save model
    

    if np.isnan(train_loss) or np.isnan(val_loss):
        print(' nan loss ')
        wandb.finish()
        break

    wandb.log({'train_loss': train_loss, 'val_loss': val_loss})

    valloss.append(val_loss)
    trainloss.append(train_loss)


print(np.isnan(train_loss), np.isnan(val_loss))


if ~np.isnan(train_loss) or ~np.isnan(val_loss):

    torch.save(model_S.state_dict(), 'results/' + wandb.run.name + '/model_S.pt')
    torch.save(model_B.state_dict(), 'results/' + wandb.run.name + '/model_B.pt')
    w_ = torch.sigmoid(w).item()
    np.save('results/' + wandb.run.name + '/w.npy', w_)

    with torch.no_grad():
        samples_S=model_S.sample(50000)
        samples_B=model_B.sample(50000)
    samples_S=samples_S.cpu().detach().numpy().reshape((-1))
    samples_B=samples_B.cpu().detach().numpy().reshape((-1))


    bins = np.linspace(-10,10,50)
    figure = plt.figure()
    plt.hist(samples_S, bins = bins, label='S', histtype='step')
    plt.hist(samples_B, bins = bins, label='B', histtype='step')
    plt.legend()

    wandb.log({'samples': wandb.Image(figure)})
    plt.show()



    true_likelihoods = {}
    true_likelihoods[str(sig_train)] = {}


    w1 = true_w[str(sig_train)][0]
    w2 = true_w[str(sig_train)][1]

    true_likelihoods[str(sig_train)] = p_data(x_test,[sig_mean, back_mean],[sig_simga**2,back_sigma**2],[w1,w2])/p_back(x_test,back_mean,back_sigma**2)


    score_likelihoods = {}
    score_likelihoods[str(sig_train)] = {}
    w_ = torch.sigmoid(w).item()

    wandb.log({'Learned weight': w_})

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
    wandb.log({'SIC': wandb.Image(plt)})
    plt.show()

    wandb.finish()




