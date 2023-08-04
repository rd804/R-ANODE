import numpy as np
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm
from src.utils import *
from nflows import transforms, distributions, flows
import torch
import torch.nn.functional as F
from nflows.distributions import uniform
from nflows.distributions.base import Distribution
from nflows.utils import torchutils
from torch import nn
import nflows
#from torch import distributions




class Net(nn.Module):
    def __init__(self, n_features, n_hidden, n_output, n_layers=2, activation=nn.ReLU()):
        super(Net, self).__init__()
        self.n_features = n_features
        self.n_hidden = n_hidden
        self.n_output = n_output
        self.n_layers = n_layers
        self.activation = activation

        self.layers = nn.ModuleList([nn.Linear(n_features, n_hidden)])
        for i in range(n_layers-1):
            self.layers.append(nn.Linear(n_hidden, n_hidden))
        self.layers.append(nn.Linear(n_hidden, n_output))

    def forward(self, x):
        for layer in self.layers[:-1]:
            x = self.activation(layer(x))
        x = self.layers[-1](x)
        x = nn.Sigmoid()(x)
        return x




def flows_for_gaussian(gaussian_dim = 2, num_transforms = 2, num_blocks = 3, 
                       hidden_features = 32, device = 'cpu'):

    base_dist = nflows.distributions.normal.StandardNormal(shape=[gaussian_dim])

    list_transforms = []
    for _ in range(num_transforms):
        list_transforms.append(
            nflows.transforms.permutations.RandomPermutation(2)
        )
        list_transforms.append(
            nflows.transforms.autoregressive.MaskedAffineAutoregressiveTransform(
                features=gaussian_dim, 
                hidden_features=hidden_features,
                num_blocks=num_blocks,
                activation=torch.nn.functional.relu
            )
        )

    transform = nflows.transforms.base.CompositeTransform(list_transforms)

    flow = nflows.flows.base.Flow(transform, base_dist).to(device)

    return flow







def m_anode(model_S,model_B,w,optimizer,data_loader,noise_data=0,noise_context=0, device='cpu', mode='train',mode_background='train', clip_grad=False,
            data_loss_expr = 'true_likelihood', w_train = True, cap_sig=1.0, scale_sig=1.0, kld_w = 1.0):
    

    if mode == 'train':
        model_S.train()

        w.requires_grad = w_train
        
            
        if mode_background == 'train' or mode_background == 'pretrained':
            model_B.train()
            print('training both models')
        else:
            model_B.eval()

    else:
        model_S.eval()
        model_B.eval()
        w.requires_grad = False

    total_loss = 0

    for batch_idx, data_ in enumerate(data_loader):
        data, label = data_
        data = data.to(device)
        label = label.to(device)
        label  = label.reshape(-1,)

        if mode == 'train':
            optimizer.zero_grad()

        #############################################
        ##############################################
        # define data losses for SR
        ##############################################
        ##############################################



        # define data losses for signal and background
        if data_loss_expr == 'expectation_likelihood':
            data_loss = torch.sigmoid(w) * model_S.log_prob(data[label==1]) + (1-torch.sigmoid(w)) * model_B.log_prob(data[label==1])
        
        elif data_loss_expr == 'true_likelihood':
          #  print(model_S.log_prob(data[label==1,:]).shape)
           # print(model_B.log_prob(data[label==1,:]).shape)
            
            if batch_idx==0:
                assert model_S.log_prob(data[label==1,:]).shape == model_B.log_prob(data[label==1,:]).shape
            data_p = torch.sigmoid(w) * torch.exp(model_S.log_prob(data[label==1])) + (1-torch.sigmoid(w)) * torch.exp(model_B.log_prob(data[label==1]))
            data_loss = torch.log(data_p + 1e-32)

        elif data_loss_expr == 'capped_sigmoid':
            data_p = capped_sigmoid(w,cap_sig) * torch.exp(model_S.log_prob(data[label==1])) + (1-capped_sigmoid(w,cap_sig)) * torch.exp(model_B.log_prob(data[label==1]))
            data_loss = torch.log(data_p + 1e-32)

        elif data_loss_expr == 'scaled_sigmoid':
            data_p = scaled_sigmoid(w,scale_sig) * torch.exp(model_S.log_prob(data[label==1])) + (1-scaled_sigmoid(w,scale_sig)) * torch.exp(model_B.log_prob(data[label==1]))
            data_loss = torch.log(data_p + 1e-32)

        elif data_loss_expr == 'with_w_scaled_KLD':
            p_s = torch.exp(model_S.log_prob(data[label==1]))
            p_b = torch.exp(model_B.log_prob(data[label==1]))
            w_ = torch.sigmoid(w)

            data_p = w_ * p_s + (1-w_) * p_b
            data_loss = torch.log(data_p + 1e-32)

            data_loss += ( p_s * w_ ) * (torch.log(w_ * p_s) - \
                                         torch.log( (1-w_) * p_b))

            

        elif data_loss_expr == 'with_w_weighted_KLD':

            p_s = torch.exp(model_S.log_prob(data[label==1]))
            p_b = torch.exp(model_B.log_prob(data[label==1]))
            w_ = torch.sigmoid(w)

            data_p = w_ * p_s + (1-w_) * p_b
            data_loss = torch.log(data_p + 1e-32)

            data_loss += ( p_s * w_ ) * (torch.log(p_s) - \
                                         torch.log(p_b))

        elif data_loss_expr == 'with_self_weighted_KLD':
            
            p_s = torch.exp(model_S.log_prob(data[label==1]))
            p_b = torch.exp(model_B.log_prob(data[label==1]))
            w_ = torch.sigmoid(w)

            data_p = w_ * p_s + (1-w_) * p_b

            data_loss = (1-kld_w) * torch.log(data_p + 1e-32)

            data_loss += kld_w * ( p_s * (torch.log(p_s) - \
                                            torch.log(p_b)))

        elif data_loss_expr == 'minimize_w':
            data_p = torch.sigmoid(w) * torch.exp(model_S.log_prob(data[label==1])) + (1-torch.sigmoid(w)) * torch.exp(model_B.log_prob(data[label==1]))
            data_loss = torch.log(data_p + 1e-32)

            data_loss += -w

        elif data_loss_expr == 'model_w':
            if mode == 'train':
                w.train()
            else:
                w.eval()
            
            log_B = model_B.log_prob(data[label==1])

            _input = torch.cat([data[label==1], log_B.reshape(-1,1)], dim=1)
           # print('input shape: ', _input.shape)

            w_ = w(_input)
            data_p = w_ * torch.exp(model_S.log_prob(data[label==1])) \
                + (1-w_) * torch.exp(model_B.log_prob(data[label==1]))

          #  print('w: ', w)
            data_loss = torch.log(data_p + 1e-32)
            # model(w) instead of torch.sigmoid(w)
            

        else:
            raise ValueError('data_loss must be either expectation_likelihood , true_likelihood, capped_sigmoid, scaled_sigmoid, with_w_scaled_KLD, with_w_weighted_KLD, with_self_weighted_KLD')
        
        #############################################
        ##############################################
        
        background_loss = model_B.log_prob(data[label==0])
       # print('background_loss_shape: ', background_loss.shape)
        
        loss = -data_loss.sum() - background_loss.sum() 

        if np.isnan(loss.item()):
            break

        total_loss += loss.item()



        if mode == 'train':
            if clip_grad:
                torch.nn.utils.clip_grad_norm_(model_S.parameters(), clip_grad)
                if w_train:
                    torch.nn.utils.clip_grad_norm_([w], clip_grad)


                if mode_background == 'train' or mode_background == 'pretrained':
                    torch.nn.utils.clip_grad_norm_(model_B.parameters(), clip_grad)
            loss.backward()
            optimizer.step()


 

    return total_loss, w_


def m_anode_test(model_S,model_B,w,optimizer,data_loader, device='cpu', 
                 mode='train', data_loss_expr = 'true_likelihood'):
    

    if mode == 'train':
        model_S.train()
        model_B.eval()

    else:
        model_S.eval()
        model_B.eval()
        w.requires_grad = False

    total_loss = 0

    for batch_idx, data_ in enumerate(data_loader):
        data, label = data_
        data = data.to(device)
        label = label.to(device)
        label  = label.reshape(-1,)

        if mode == 'train':
            optimizer.zero_grad()

        if data_loss_expr == 'true_likelihood':
            if batch_idx==0:
                assert model_S.log_prob(data[label==1,:]).shape == model_B.log_prob(data[label==1,:]).shape
                print(torch.sigmoid(w).item())

            data_p = torch.sigmoid(w) * torch.exp(model_S.log_prob(data[label==1])) + (1-torch.sigmoid(w)) * torch.exp(model_B.log_prob(data[label==1]))
            data_loss = torch.log(data_p + 1e-32)

        else:
            raise ValueError('data_loss must be either expectation_likelihood , true_likelihood, capped_sigmoid, scaled_sigmoid, with_w_scaled_KLD, with_w_weighted_KLD, with_self_weighted_KLD')
        
        #############################################
        ##############################################
            
        loss = -data_loss.sum()
        total_loss += loss.item()



        if mode == 'train':
            loss.backward()
            optimizer.step()


 

    return total_loss




def anode(model,train_loader, optimizer, params, device='cpu', mode='train'):
    
    if mode == 'train':
        model.train()
    else:
        model.eval()
    
    total_loss = 0


    for batch_idx, data in enumerate(train_loader):
        if batch_idx == 5:
            break

        data = data.to(device)
        #params = params.to(device)

        if mode == 'train':
            optimizer.zero_grad()
        
        loss = - evaluate_log_prob(model, data, params).mean()
        total_loss += loss.item()

        if mode == 'train':
            loss.backward()        
            optimizer.step()

    total_loss /= len(train_loader)

    return total_loss
    

def evaluate_log_prob(model, data, preprocessing_params):
    logit_prob = model.log_probs(data[:, 1:-1], data[:,0].reshape(-1,1))
    log_prob = logit_prob.flatten() + torch.sum(
    torch.log(
        2 * (1 + torch.cosh(data[:, 1:-1] * preprocessing_params["std"] + preprocessing_params["mean"]))
        / (preprocessing_params["std"] * (preprocessing_params["max"] - preprocessing_params["min"]))
    ), axis=1
) # type: ignore
    return log_prob



