import numpy as np
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm

from nflows import transforms, distributions, flows
import torch
import torch.nn.functional as F
from nflows.distributions import uniform


CUDA = True
device = torch.device("cuda:0" if CUDA else "cpu")

kwargs = {'num_workers': 4, 'pin_memory': True} if CUDA else {}
kwargs = {}


def define_model(nhidden=1,hidden_size=200,nblocks=8,nbins=8,embedding=None,dropout=0.05,nembedding=20,nfeatures=2):

    init_id=True    
    
    flow_params_RQS = {'num_blocks':nhidden, # num of hidden layers per block
                       'use_residual_blocks':False,
                       'use_batch_norm':False,
                       'dropout_probability':dropout,
                       'activation':getattr(F, 'relu'),
                       'random_mask':False,
                       'num_bins':nbins,
                       'tails':'linear',
                       'tail_bound':10,
                       'min_bin_width': 1e-6,
                       'min_bin_height': 1e-6,
                       'min_derivative': 1e-6}
    flow_blocks = []
    for i in range(nblocks):
        flow_blocks.append(
            transforms.MaskedPiecewiseRationalQuadraticAutoregressiveTransform(
                **flow_params_RQS,
                features=nfeatures,
#                context_features=ncontext,
                hidden_features=hidden_size
            ))
        if init_id:
            torch.nn.init.zeros_(flow_blocks[-1].autoregressive_net.final_layer.weight)
            torch.nn.init.constant_(flow_blocks[-1].autoregressive_net.final_layer.bias,
                                    np.log(np.exp(1 - 1e-6) - 1))

        if i%2 == 0:
            flow_blocks.append(transforms.ReversePermutation(nfeatures))
        else:
            flow_blocks.append(transforms.RandomPermutation(nfeatures))

    del flow_blocks[-1]
    flow_transform = transforms.CompositeTransform(flow_blocks)

    flow_base_distribution = distributions.StandardNormal(shape=[nfeatures])
    flow = flows.Flow(transform=flow_transform, distribution=flow_base_distribution)

    model = flow.to(device)
    #print(model)
    
    return model

# training

def train(model,optimizer,train_loader,noise_data=0,noise_context=0):
    
    model.train()
    train_loss = 0

 #   pbar = tqdm(total=len(train_loader.dataset),leave=True)
    for batch_idx, data in enumerate(train_loader):
#        print(data)
#        data+=noise_data*torch.normal(mean=torch.zeros(data.shape),std=torch.ones(data.shape))
        data = data.to(device)

#        cond_data = cond_data.float()
#        cond_data+=noise_context*torch.normal(mean=torch.zeros(cond_data.shape),std=torch.ones(cond_data.shape))
#        cond_data = cond_data.to(device)
        
        optimizer.zero_grad()
        # print(data, cond_data)
#        loss = -model(data, cond_data).mean()
        loss = -model.log_prob(data).mean()
        train_loss += loss.item()
        loss.backward()
        
    
#        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.01)
        
        optimizer.step()

    train_loss=train_loss.cpu().detach().numpy()
    train_loss=train_loss/len(train_loader.dataset)

    return train_loss
    

# validation

def val(model,val_loader):

    valloss=0
    model.eval()
    with torch.no_grad():

        for batch_idx, data in enumerate(val_loader):
            data = data.to(device)
            valloss+=-model.log_prob(data).sum()
#            print(valloss)
#            cond_data = cond_data.float()
#            cond_data = cond_data.to(device)
#            valloss+=-model(data, cond_data).sum()

    valloss=valloss.cpu().detach().numpy()
    valloss=valloss/len(val_loader.dataset)
    
    return valloss



