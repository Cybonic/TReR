


import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import numpy as np

from utils import retrieve_eval

from torch.utils.data import DataLoader
from dataloaders.rankingdata import RankingDataset,RankingMSE

import RERANKING
import loss 

from base_trainer import ReRankingTrainer


class AttentionRanking(torch.nn.Module):
    def __init__(self,cand=35,feat_size = 256):
        super().__init__()
        
        self.Win =  nn.Parameter(torch.zeros(256,cand))
        self.Wout =  nn.Parameter(torch.zeros(cand,1))
        
        nn.init.normal_(self.Win.data, mean=0, std=0.1)
        nn.init.normal_(self.Wout.data, mean=0, std=0.1)

        self.att = torch.nn.MultiheadAttention(256,1)
        self.classifier = torch.nn.Conv1d(256, 1, 1)

    def __str__(self):
      return "AttentionRanking"
    
    def forward(self,k):
        out, attn_output_weights = self.att(k,k,k)
        out = torch.transpose(out,dim0=2,dim1=1)
        out = self.classifier(out).squeeze()
        if self.training:
           return out.float()
        return out.float()

class MaskRanking(torch.nn.Module):
    def __init__(self,cand=35,feat_size = 256):
      super().__init__()
        
      self.Win =  nn.Parameter(torch.zeros(4,256,cand))
      self.Wout =  nn.Parameter(torch.zeros(cand,1))
        
      nn.init.normal_(self.Win.data, mean=0, std=0.1)
      nn.init.normal_(self.Wout.data, mean=0, std=0.1)

      self.att = torch.nn.MultiheadAttention(256,1)
      self.classifier = torch.nn.Conv1d(256, 1, 1)

      fc = [ nn.LazyLinear(cand),
             nn.BatchNorm1d(cand, momentum=0.01),
             #nn.LazyLinear(outdim),
             nn.ReLU()]
    
      self.fc = nn.Sequential(*fc)

    def __str__(self):
      return "MaskRanking"
    
    def forward(self,k):
      #out = []
      #for w in  self.Win:
      #  out.append(torch.matmul(k,w))
      
      #ou_stack = torch.stack(out,dim=1)
      #out = torch.mean(ou_stack,dim=1)
      out, attn_output_weights = self.att(k,k,k)
      out = torch.transpose(out,dim0=2,dim1=1)
      #out = torch.matmul(out,self.Wout).squeeze()
      out = self.classifier(out).squeeze()
      out = self.fc(out)
      if self.training:
          return out.float()
      return out.float()
    

# =====================================================


class MSERanking():
    def __init__(self):
        self.bce = nn.MSELoss()
    def __call__(self,x,y):
        x = F.softmax(x,dim=-1)
        y = F.softmax(y,dim=-1)
        error = self.bce(x,y)
        return error.float()
      
  
class AttentionTrainer(ReRankingTrainer):
  def __init__(self,**args):
    super(AttentionTrainer,self).__init__(**args)
    

  def train_epoch(self,epoch,loader):
    #tain_report_terminal= 10
    self.model.train()
    loss_buffer = []
    for batch,gt in loader:
      #batch = batch.to(self.device)
      self.optimizer.zero_grad()
      batch = batch.to(self.device)
      gt = gt.to(self.device)
      #std = 0.001*epoch
      #noise = torch.randn(batch.size()) * std + 0
      
      #batch = batch + noise.to(self.device)
      #noise =  torch.random.normal(0, 0.1, size=keys.shape)
      out = self.model(batch)
      #print("\n\n")
      #a = torch.argsort(out[0],dim=-1,descending=True)
      #print(a.detach().cpu().numpy())

      #b = torch.argsort(gt[0],dim=-1,descending=True)
      #print(b.detach().cpu().numpy())

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
        x = x.to(self.device)
        values = self.model(x)

        values = torch.argsort(values,dim=-1,descending=True)
        
        values = values.detach().cpu().numpy().astype(np.uint8)
        re_rank_idx.extend(values)
      #re_rank_idx = np.argsort(scores)
      rerank_loops = np.array([loops[l] for loops,l in zip(test_base_loop,re_rank_idx)])
      return(rerank_loops)


def load_cross_data(root,model_name,seq_train,seq_test):
  train_data = RankingMSE(root,model_name,seq_train)
  trainloader = DataLoader(train_data,batch_size = len(train_data),shuffle=True)
  # LOAD TEST DATA
  test_data = RankingMSE(root,model_name,seq_test)
  testloader = DataLoader(test_data,batch_size = len(test_data)) #
  return trainloader,testloader,test_data.get_max_top_cand()

def load_data(root,model_name,seq,train_percentage):
  train_data = RankingMSE(root,model_name,seq)

  train_size = int(len(train_data)*train_percentage)
  test_size = len(train_data) - train_size

  train, test =torch.utils.data.random_split(train_data,[train_size,test_size])
  
  trainloader = DataLoader(train,batch_size = len(train),shuffle=True)
  testloader = DataLoader(test,batch_size = len(test)) #

  return trainloader,testloader,train_data.get_max_top_cand()
# LOAD TTRAINING DATA

root = '/home/tiago/Dropbox/RAS-publication/predictions/paper/kitti'

train_size = 0.2
device = 'cuda:0'
#device = 'cpu'

#sequences = ['00','02','05','06','08']
sequences = ['00']
for j in range(1):
  # 'SPoC_pointnet', 'GeM_pointnet' ,
  #train_size = float(j)/10
  for model_name in ['VLAD_pointnet', 'ORCHNet_pointnet']:
    for seq in sequences:
      #seq = '02'
      trainloader,testloader,max_top_cand = load_data(root,model_name,seq,train_size)
      #trainloader,testloader,max_top_cand  = load_cross_data(root,model_name,seq,seq)

      #===== RE-RANKING ========
      model = MaskRanking(max_top_cand,256)
      loss_fun = MSERanking()

      root_save = os.path.join('tests','loss_log_softmax',model_name,seq)
      if not os.path.isdir(root_save):
        os.makedirs(root_save)

      # experiment = os.path.join(root_save,f'{str(model)}-{str(train_size)}')
      experiment = os.path.join(root_save,f'{str(model)}')
      rerank = AttentionTrainer(experiment=experiment,loss = loss_fun, model = model,lr= 0.01,epochs = 200,lr_step=5000,val_report=1,tain_report_terminal=1,device=device,max_top_cand = max_top_cand)

      rerank.Train(trainloader,testloader)




