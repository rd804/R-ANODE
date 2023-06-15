import numpy as np
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm
from src.utils import *
from nflows import transforms, distributions, flows
import torch
import torch.nn.functional as F
from nflows.distributions import uniform




def define_model(nhidden=1,hidden_size=200,nblocks=8,nbins=8,embedding=None,dropout=0.05,nembedding=20,nfeatures=2,
                 device='cpu'):

    init_id=True    
    
    flow_params_RQS = {'num_blocks':nhidden, # num of hidden layers per block
                       'use_residual_blocks':False,
                       'use_batch_norm':False,
                       'dropout_probability':dropout,
                       'activation':getattr(F, 'relu'),
                       'random_mask':False,
                       'num_bins':nbins,
                       'tails':'linear',
                       'tail_bound':20,
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


        else:
            raise ValueError('data_loss must be either expectation_likelihood , true_likelihood, capped_sigmoid, scaled_sigmoid, with_w_scaled_KLD, with_w_weighted_KLD, with_self_weighted_KLD')
        
        #############################################
        ##############################################
        
        
        background_loss = -model_B.log_prob(data[label==0])
        loss = -data_loss.sum() + background_loss.sum() 

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


 

    return total_loss


def m_anode_EM(model_S,model_B,w,optimizer,data_loader,noise_data=0,noise_context=0, device='cpu', mode='train',mode_background='train', clip_grad=False,
            data_loss_expr = 'true_likelihood', w_train = True ):
    
    '''EM algorithm for model_S and model_B'''

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

    # Initialize the weights
    

    for batch_idx, data_ in enumerate(data_loader):
        # E-step
        data, label = data_
        data = data.to(device)
        label = label.to(device)
        label  = label.reshape(-1,)


        with torch.no_grad():
            p_x_b = torch.exp(model_B.log_prob(data))
            p_x_s = torch.exp(model_S.log_prob(data))



            p_b_x = p_x_b * (1-w) / (p_x_b * (1-w) + p_x_s * w)
            p_s_x = 1 - p_b_x
           
           
            p_b = p_b_x.mean()
            p_s = 1.0 - p_b

            print('p_b: ', p_b.item(), 'p_s: ', p_s.item())

            # w = torch.nan_to_num(p_s, nan=0.0, posinf=0.0, neginf=0.0)

        if mode == 'train':
            optimizer.zero_grad()



        # M-step
        # define data losses for signal and background
        if data_loss_expr == 'expectation_likelihood':
            data_loss = (p_s_x[label==1] * model_S.log_prob(data[label==1])).sum() + (p_b_x[label==1] * model_B.log_prob(data[label==1])).sum()
        else:
            pass

        loss = -data_loss

        print('w: ', w.item(), 'loss: ', loss.item())

        if np.isnan(loss.item()):
            break

        total_loss += loss.item()

    return total_loss, w



# training

def train(model,optimizer,train_loader,noise_data=0,noise_context=0, device='cpu'):
    
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

    train_loss=train_loss
    train_loss=train_loss/len(train_loader.dataset)

    return train_loss
    

# validation

def val(model,val_loader,device='cpu'):

    valloss=0
    model.eval()
    with torch.no_grad():

        for batch_idx, data in enumerate(val_loader):
            data = data.to(device)
            valloss+=-model.log_prob(data).mean()
#            print(valloss)
#            cond_data = cond_data.float()
#            cond_data = cond_data.to(device)
#            valloss+=-model(data, cond_data).sum()

    valloss=valloss.cpu().detach().numpy()
    valloss=valloss/len(val_loader.dataset)
    
    return valloss



