
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def L2_loss(a,b, dim=0, eps=1e-8):
    squared_diff = torch.pow((a - b),2)
    value = torch.sqrt(torch.sum(squared_diff,dim=dim)+eps)
    return torch.max(value,torch.tensor(eps))

class WeightedRanking():
    def __init__(self,verbose=False):
        pass
        self.verbose = verbose

    def __call__(self,x,y):

        y = F.softmin(y,dim=1)

        sort_x = torch.argsort(x,descending=True)
        sort_y = torch.argsort(y,dim=-1,descending=True)
       
        idx_sort_x = torch.argsort(sort_x)
        idx_sort_y = torch.argsort(sort_y)

        # re-weight scale
        weights = torch.exp(-torch.arange(0,25,1))
        # re-weight from vec to matrix
        weigths_=torch.zeros(x.shape)
        for i,w in enumerate(sort_y):
            weigths_[i,w]=weights

        idx_dif = (idx_sort_x - idx_sort_y).clip(min=0)

        value = (idx_dif-x).clip(min=0) * weigths_

        if self.verbose == True:
            sortgt_np = sort_y.detach().cpu().numpy()
            print(sortgt_np[0])

            sortx_np = sort_x.detach().cpu().numpy()
            print(sortx_np[0])

            idx_dif_np = value.detach().cpu().numpy()
            print(np.round(idx_dif_np[0],3))

        value = torch.sum(value,dim=-1)
        return torch.mean(value)

class PairWiseMSE():
    def __init__(self,alpha=0.5,verbose=False):
        pass
        self.verbose = verbose
        self.alpha = alpha

    def __call__(self,xc,yc):

        y = F.softmin(yc,dim=1)
        
        sort_x = torch.argsort(xc,descending=True)
        sort_y = torch.argsort(y,descending=True)
        
        #a = F.mse_loss(x,y)
        a = L2_loss(xc,y,dim=1)
        #print(a[0])

        x_ =sort_x.unsqueeze(dim=-1)
        xx = torch.abs(torch.multiply(x_,torch.transpose(-x_,2,1)))
        loss_values = 0
        #for sortx,sorty in zip(sort_x,sort_y):
        #    #L2_loss()
        #    value = 0
        #    for sxi,syi in zip(sortx,sorty):
        #        for sxj,syj in zip(sortx,sorty):
        #            value = value + torch.abs((torch.pow((sxi-sxj),2)-torch.pow((syi-syj),2)))
        #    loss_values = loss_values + (value/sortx.shape[0])
        
        #print(xx[0])
        #b = loss_values/sort_x.shape[0]
        y_ =sort_y.unsqueeze(dim=-1)
        yy = torch.abs(torch.multiply(y_,torch.transpose(-y_,2,1)))
        #print(yy[0])
        
        mul = ((xx)-(yy)).clip(min=0)
        b = torch.sum(torch.sum(mul,dim=-1),dim=-1)
        #sort_x = torch.argsort(x,descending=True)
        #sort_y = torch.argsort(y,dim=-1,descending=True)
       
        value = (1-self.alpha)*a + (self.alpha * b)

        if self.verbose == True:
            
            
            print("\nGT")
            #print(yc[0].detach().cpu().numpy())

            #print(y[0].detach().cpu().numpy())

            sortgt_np = sort_y.detach().cpu().numpy()
            print(sortgt_np[0])

            #print("\nPred")

            #print(xc[0].detach().cpu().numpy())

            sortx_np = sort_x.detach().cpu().numpy()
            print(sortx_np[0])

        #    idx_dif_np = value.detach().cpu().numpy()
        #    print(np.round(idx_dif_np[0],3))

        value = torch.sum(value,dim=-1)
        return torch.mean(value)