import numpy as np
import matplotlib.pyplot as plt
import torch
import nflows
from nflows import flows
from nflows import transforms
from nflows import distributions

from scipy.stats import norm


back_mean = 0
back_sigma = 3
sig_mean = 2
sig_sigma = 0.25
nbg=10000
nsig=100
ndim=2




background_eval=np.random.normal(size=ndim*1000000,loc=back_mean,scale=back_sigma).reshape((1000000,ndim))
signal_eval=np.random.normal(size=ndim*1000000,loc=sig_mean,scale=sig_sigma).reshape((1000000,ndim))

# training

def train(model,optimizer,train_loader):

    model.train()
    train_loss = 0

  #  pbar = tqdm(total=len(train_loader.dataset),leave=True)
    for batch_idx, x in enumerate(train_loader):
        x = x.cuda()
        optimizer.zero_grad()
        loss=-model.log_prob(x).mean()
        loss.backward()
        optimizer.step()
   #     pbar.update(x.size(0))
        train_loss += float(loss.item())
    #    pbar.set_description('Train, Log likelihood in nats: {:.6f}'.format(
     #       train_loss / (batch_idx + 1)))
    return train_loss

        
# validation

def val(model,val_loader):
    model.eval()
    val_loss = 0

    for batch_idx, x in enumerate(val_loader):
        x = x.cuda()

        loss=-model.log_prob(x).mean()

        val_loss += float(loss.item())

   # val_loss=val_loss/ (batch_idx + 1)

    return val_loss

def learnedlogR(eval_loader,flow):
    logRlist=np.empty((0))
    for x in eval_loader:
        Pbg=norm.pdf(x[:,0],loc=back_mean,scale=back_sigma)*norm.pdf(x[:,1],loc=back_mean,scale=back_sigma)
    
        logPdata=flow.log_prob(x.cuda()).detach().cpu().numpy()
        logPbg=np.log(Pbg)
        logRlist=np.concatenate((logRlist,logPdata-logPbg))
    return logRlist



for nsig in [10,50,100,200,500,1000]:
    for itrain in range(10):
        print("nsig,itrain ",nsig,itrain)
        
        # make sure to resample each iteration
        background=np.random.normal(size=ndim*nbg,loc=back_mean,scale=back_sigma).reshape((nbg,ndim))
        signal=np.random.normal(size=ndim*nsig,loc=sig_mean,scale=sig_sigma).reshape((nsig,ndim))
        data=np.concatenate((background,signal))
        np.random.shuffle(data)
        
        # Use the standard pytorch DataLoader
        batch_size = 256
        trainloader = torch.utils.data.DataLoader(torch.from_numpy(data[:9000].astype('float32')), batch_size=batch_size, shuffle=True)

        val_batch_size=batch_size*10
        valloader = torch.utils.data.DataLoader(torch.from_numpy(data[9000:].astype('float32')), batch_size=val_batch_size, shuffle=False)


        flow_sig = nflows.flows.base.Flow(transform_sig, base_dist_sig).cuda()
        valloss_list=[]
        optimizer_sig = torch.optim.Adam(flow_sig.parameters())

        for epoch in range(50):
         #   print('\n Epoch: {}'.format(epoch))
            train_Psigonly(flow_sig,optimizer_sig,trainloader)
            valloss=val_Psigonly(flow_sig,valloader)
          #  print('epoch '+str(epoch)+' val loss: ',valloss)
            valloss_list.append(valloss)

        torch.save(flow_sig.state_dict(),"./model/Psigonly_nsig"+str(nsig)+'_itrain'+str(itrain)+".par")
        
        
        flow1 = nflows.flows.base.Flow(transform, base_dist).cuda()
        valloss_list=[]
        optimizer = torch.optim.Adam(flow1.parameters())

        for epoch in range(50):
        #    print('\n Epoch: {}'.format(epoch))
            train(flow1,optimizer,trainloader)
            valloss=val(flow1,valloader)
        #    print('epoch '+str(epoch)+' val loss: ',valloss)
            valloss_list.append(valloss)

        torch.save(flow1.state_dict(),"./model/Pdata_nsig"+str(nsig)+'_itrain'+str(itrain)+".par")

        
        
        
        
        