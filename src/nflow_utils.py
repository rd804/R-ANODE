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
import src.flows as fnn
#from torch import distributions



def r_anode(model_S,model_B,w,optimizer, scheduler, data_loader, params, device='cpu', 
                 mode='train', data_loss_expr = 'true_likelihood'):
    
    n_nans = 0
    if mode == 'train':
        model_S.train()
        model_B.eval()

    else:
        model_S.eval()
        model_B.eval()

    total_loss = 0

    params_CR = params['CR']
    params_SR = params['SR']

    for batch_idx, data in enumerate(data_loader):

        data_SR = data[0].to(device)
        data_CR = data[1].to(device)

        if mode == 'train':
            optimizer.zero_grad()

        if data_loss_expr == 'true_likelihood':
         #   model_S_log_prob = model_S.log_prob(data_SR[:,1:-1])
            model_S_log_prob = model_S.log_prob(data_SR[:,1:-1],context=data_SR[:,0].reshape(-1,1))
       #     model_S_log_prob = evaluate_log_prob(model_S, data_SR, params_SR,
        #                                         transform=False)
            model_B_log_prob = evaluate_log_prob(model_B, data_CR, params_CR,
                                                 transform=False)
            if batch_idx==0:
                assert model_S_log_prob.shape == model_B_log_prob.shape
                print(f'value of w: {w}')    
            
            
            
            data_p = w * torch.exp(model_S_log_prob) + (1-w) * torch.exp(model_B_log_prob)
            data_loss = torch.log(data_p + 1e-32)

        else:
            raise ValueError('only true_likelihood is implemented')
        #############################################
        ##############################################
        
        # remove data_loss with nan values
        n_nans += sum(torch.isnan(data_loss)).item()
        data_loss = data_loss[~torch.isnan(data_loss)]


        loss = -data_loss.mean()
        total_loss += loss.item()



        if mode == 'train':
            loss.backward()
            optimizer.step()
            scheduler.step()

    total_loss /= len(data_loader)

   # if mode == 'train':
    # set batch norm layers to eval mode
    # what dafaq is this doing?
  #      print('setting batch norm layers to eval mode')
   #     has_batch_norm = False
   #     for module in model_S.modules():
    #        if isinstance(module, fnn.BatchNormFlow):
     #           has_batch_norm = True
       #         module.momentum = 0
        # forward pass to update batch norm statistics
      #  if has_batch_norm:
       #     with torch.no_grad():
            ## NOTE this is not yet fully understood but it crucial to work with BN
        #        model_S(data_loader.dataset.tensors[1][:,1:-1].to(data[0].device),
         #           data_loader.dataset.tensors[1][:,0].to(data[0].device).reshape(-1,1).float())

          #  for module in model_S.modules():
           #     if isinstance(module, fnn.BatchNormFlow):
            #        module.momentum = 1
    if n_nans > 0:
        print('---------------------------------------------------')
        print('---------------------------------------------------')
        print('---------------------------------------------------')
        print(f'WARNING: {n_nans} nans in data_loss in mode {mode}')
        print('---------------------------------------------------')
        print('---------------------------------------------------')
        print(f'max model_S_log_prob: {torch.max(model_S_log_prob)}')
        print(f'min model_S_log_prob: {torch.min(model_S_log_prob)}')
        print(f'max model_B_log_prob: {torch.max(model_B_log_prob)}')
        print(f'min model_B_log_prob: {torch.min(model_B_log_prob)}')


    return total_loss




def anode(model,train_loader, optimizer, params, device='cpu', mode='train'):
    
    if mode == 'train':
        model.train()
    else:
        model.eval()
    
    total_loss = 0


    for batch_idx, data in enumerate(train_loader):


        data = data[0].to(device)
        #params = params.to(device)

        if mode == 'train':
            optimizer.zero_grad()
        
        loss = - evaluate_log_prob(model, data, params).mean()
        total_loss += loss.item()

        if mode == 'train':
            loss.backward()        
            optimizer.step()

    total_loss /= len(train_loader)

    if mode == 'train':
        # set batch norm layers to eval mode
        # what dafaq is this doing?
        print('setting batch norm layers to eval mode')
        has_batch_norm = False
        for module in model.modules():
            if isinstance(module, fnn.BatchNormFlow):
                has_batch_norm = True
                module.momentum = 0
        # forward pass to update batch norm statistics
        if has_batch_norm:
            with torch.no_grad():
            ## NOTE this is not yet fully understood but it crucial to work with BN
                model(train_loader.dataset.tensors[0][:,1:-1].to(data[0].device),
                    train_loader.dataset.tensors[0][:,0].to(data[0].device).reshape(-1,1).float())

            for module in model.modules():
                if isinstance(module, fnn.BatchNormFlow):
                    module.momentum = 1

    return total_loss
    

def evaluate_log_prob(model, data, preprocessing_params, transform=False):
    logit_prob = model.log_probs(data[:, 1:-1], data[:,0].reshape(-1,1))
    
    if transform:
        log_prob = logit_prob.flatten() + torch.sum(
        torch.log(
            2 * (1 + torch.cosh(data[:, 1:-1] * preprocessing_params["std"] + preprocessing_params["mean"]))
            / (preprocessing_params["std"] * (preprocessing_params["max"] - preprocessing_params["min"]))
        +1e-32), axis=1
    ) # type: ignore
    else:
        log_prob = logit_prob.flatten()
    return log_prob

def flows_for_gaussian(gaussian_dim = 2, num_transforms = 2, num_blocks = 3, 
                       hidden_features = 32, device = 'cpu'):

    base_dist = nflows.distributions.normal.StandardNormal(shape=[gaussian_dim])

    list_transforms = []
    for _ in range(num_transforms):
        list_transforms.append(
            nflows.transforms.permutations.RandomPermutation(gaussian_dim)
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


