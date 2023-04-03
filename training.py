


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


# =====================================================

class ReRankingTrainer(nn.Module):
  def __init__(self,model,loss):
    super(ReRankingTrainer,self).__init__()
    
    self.model = model
    #self.norm = nn.ReLU()
    self.norm = nn.LayerNorm(25)

    
    self.loss = loss 
    #self.loss = nn.BCEWithLogitsLoss()
    self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=0.01)
    self.val_period = 1
  
  def train_epoch(self,epoch,loader):
    #tain_report_terminal= 10
    self.model.train()

    loss_buffer = []
    for batch,gt in loader:
      self.optimizer.zero_grad()
      out = self.model(batch)
      loss_value = self.loss(out,gt)
      #loss = self.loss(out, sortgt_)
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
        values = torch.argsort(values,descending=True)
        
        values = values.detach().cpu().numpy()
        re_rank_idx.extend(values)
      #re_rank_idx = np.argsort(scores)
      rerank_loops = np.array([loops[l] for loops,l in zip(test_base_loop,re_rank_idx)])
      return(rerank_loops)


  def trainer(self,trainloader,testloader):
  
    val_report = 1
    tain_report_terminal= 100
    epochs     = 2000
    old_perfm  = 0
    loss_log   = []
    lr_step = 1000

    # TRAIN PERFORMANCE
    base_loop,base_sim =  trainloader.dataset.get_base_loops()
    targets =train_data.get_targets()  
    base_perfm = retrieve_eval(base_loop,targets,top=1)

    # TEST PERFORMANCE
    test_base_loop,test_base_sim =  testloader.dataset.get_base_loops()
    test_targets =test_data.get_targets()  
    test_base_perfm = retrieve_eval(test_base_loop,test_targets,top=1)

    for epoch in range(epochs):
      
      loss = self.train_epoch(epoch,trainloader)
      value = np.round(np.mean(loss),3)
      loss_log.append(value)

      if epoch%lr_step == 0 and epoch >0:
        self.optimizer.param_groups[0]['lr'] = self.optimizer.param_groups[0]['lr']/10
      
      if epoch%tain_report_terminal == 0:
        print('T ({}) | Loss {:.10f}'.format(epoch,np.mean(loss_log)))
      
      # Val 
      if epoch % val_report == 0:
        rerank_loops = self.predict(testloader,test_base_loop)
        rerank_perfm = retrieve_eval(rerank_loops,targets,top=1)
        
        delta = rerank_perfm['recall'] - test_base_perfm['recall']
        print(f"\nReRank Recall: {round(rerank_perfm['recall'],5)} | delta: {round(delta,5)} ")

        if rerank_perfm['recall']>old_perfm:
          old_perfm = rerank_perfm['recall']
          print("\nBest performance\n")
          #save_log = [ed_sim,ed_loop,rerank_loops,scores,target_ord]
    
    print("\n ******************Best*********************\n")
    print(f"BaseLine  {test_base_perfm['recall']}")
    print(f"Reranking {old_perfm}")




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
# Get metric distance (ground truth)
re_model = RERANKING.FC(25)
#re_model = RERANKING.Attention(25)
loss_f = loss.PairWiseMSE(0,verbose=True)
#loss = nn.MSELoss()
rerank = ReRankingTrainer(re_model,loss_f)

rerank.trainer(trainloader,testloader)




