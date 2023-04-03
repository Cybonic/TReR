


import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import numpy as np

from utils import retrieve_eval

from torch.utils.data import DataLoader
from dataloaders.rankingdata import RankingDataset

import RERANKING
import loss 

from base_trainer import ReRankingTrainer


class AttentionRanking(torch.nn.Module):
    """ Multi-layer perception.
        [B, Cin, N] -> [B, Cout, N] or
        [B, Cin] -> [B, Cout]
    """
    def __init__(self,cand=35,feat_size = 256):
        super().__init__()
        
        self.Win =  nn.Parameter(torch.zeros(256,25))
        self.Wout =  nn.Parameter(torch.zeros(25,1))
        
        nn.init.normal_(self.Win.data, mean=0, std=0.1)
        nn.init.normal_(self.Wout.data, mean=0, std=0.1)

        self.att = torch.nn.MultiheadAttention(256,2)
        self.row_norm = nn.LogSoftmax(dim=2)
        self.column_norm = nn.LogSoftmax(dim=1)


    def forward(self,k):
        out, attn_output_weights = self.att(k,k,k)
        out = torch.matmul(out,self.Win)
        out = self.column_norm(self.row_norm(out))
        if self.training:
           return out.float()
        
        outmax = torch.argmax(out,dim=-1)
        
        #print(out[0][0].detach().cpu().numpy())
        #print(outmax[0][0].detach().cpu().numpy())
        return outmax.float()
    
# =====================================================
class CERanking():
    def __init__(self,verbose=False):
        pass
        self.verbose = verbose
        self.bce = nn.NLLLoss()

    def __call__(self,x,y):
        value = 0
        target = torch.argmax(y,dim=-1)
        #for bxx,byy in zip(x,y):
            #for xx,yy in zip(bxx,byy):
                #print(xx.detach().cpu().numpy())
                #print(yy.detach().cpu().numpy())
        value = self.bce(x,target)
        #value = value + self.bce(x,target)
        #value = value/(y.shape[0]*y.shape[1])
       
        return value
    
  
class AttentionTrainer(ReRankingTrainer):
  def __init__(self,**args):
    super(AttentionTrainer,self).__init__(**args)
    

  def train_epoch(self,epoch,loader):
    #tain_report_terminal= 10
    self.model.train()
    loss_buffer = []
    for batch,gt in loader:
      self.optimizer.zero_grad()
      batch = batch.to(self.device)
      gt = gt.to(self.device)

      out = self.model(batch)
      print("\n\n")
      print(np.argmax(out[0].detach().cpu().numpy(),axis=-1))

      print(np.argmax(gt[0].detach().cpu().numpy(),axis=1))

      loss_value = self.loss(out,gt)
      loss_value.backward()
      self.optimizer.step()
      loss_buffer.append(loss_value.cpu().detach().numpy().item())
    return(loss_buffer) 


  def predict(self,testloader,test_base_loop):
      self.model.eval()
      re_rank_idx = []
      for x,_ in testloader:
        #x = 1-x # .cuda()
        values = self.model(x)
        #out = values.detach().cpu().numpy()
        #print()
        #values = torch.argsort(values,descending=True)
        
        values = values.detach().cpu().numpy().astype(np.uint8)
        re_rank_idx.extend(values)
      #re_rank_idx = np.argsort(scores)
      rerank_loops = np.array([loops[l] for loops,l in zip(test_base_loop,re_rank_idx)])
      return(rerank_loops)



# LOAD TTRAINING DATA

root = '/home/tiago/Dropbox/RAS-publication/predictions/paper/kitti'
model_name = 'ORCHNet_pointnet'
sequence = '00'
train_data = RankingDataset(root,model_name,sequence)
trainloader = DataLoader(train_data,batch_size = len(train_data),shuffle=False)

# LOAD TEST DATA
sequence = '00'
test_data = RankingDataset(root,model_name,sequence)
testloader = DataLoader(test_data,batch_size = len(test_data)) #

#===== RE-RANKING ========
#loss = nn.MSELoss()
model = AttentionRanking(25,256)
loss_fun = CERanking() 

rerank = AttentionTrainer(loss = loss_fun, model = model,lr= 0.01,epochs = 200,lr_step=20,val_report=1,tain_report_terminal=1,device='cpu')

rerank.Train(trainloader,testloader)




