


import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import numpy as np

from utils import retrieve_eval

from torch.utils.data import DataLoader
from dataloaders.distancedata import RankingDataset

import RERANKING 


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
    rerank.train()

    loss_buffer = []
    for batch,gt in loader:
      self.optimizer.zero_grad()
      #batch = self.norm(batch)
      # batch = 1-batch
      #print("\nBatch")
      batch_np = batch.detach().cpu().numpy()
      #print(np.round(batch_np[0],3))
      sortout_ = torch.argsort(batch)
      #print(sortout_[0])
      #sortgt_ = torch.argsort(gt,dim=1,descending=False).float()
      #sortgt_np = sortgt_.detach().cpu().numpy()
      #print("\nSort")

      
      gt = gt
      out = self.model(batch)
      #out_max = torch.max(out,dim=1)[0].unsqueeze(-1)
      #out = torch.div(out,out_max)
      outnp = out.detach().cpu().numpy()
      
      #print("\nOut")
      #print(np.round(outnp[0],3))
     
      sortout_ = torch.argsort(out,descending=True)
      #print(sortout_[0])
      #pred =torch.argmax(out,dim=1).float()
      #out = self.norm(out)
      # gt  = 1/gt
      #print(outnp[0])
      #gt_np_ = gt.detach().cpu().numpy()
      #print(gt_np_[0])
      #print("\nGT")
      #gt_np = gt.detach().cpu().numpy()
      #print(np.round(gt_np[0],3))

      gt = F.softmin(gt,dim=1)
      #print("\nsoftmin")
      gt_np = gt.detach().cpu().numpy()
      #print(np.round(gt_np[0],3))
      
      sortgt_ = torch.argsort(gt,dim=-1,descending=True)
      sortgt_np = sortgt_.detach().cpu().numpy()
      #print("\nSort")
      #print(sortgt_np[0])
      loss = 0
      #for bgt, bout, bsort,bsortout in zip(gt,out,sortgt_,sortout_):
        
        #idx = (bsortout - bsort).clip(min=0)
        #print(idx.detach().cpu().numpy())
      xx = torch.argsort(sortout_)
      xx_ = torch.argsort(sortgt_)

      weights = torch.exp(-torch.arange(0,25,1))
      #weights = torch.repeat_interleave(weights,sortgt_np.shape[0],dim=0)
      #weights.repeat(sortgt_np.shape[0])
      weigths_=torch.zeros(out.shape)
      for i,w in enumerate(sortgt_):
        weigths_[i,w]=weights
      #weigths_[sortgt_]= weights
      #print(weigths_.detach().cpu().numpy()[0])
      #print(xx_.detach().cpu().numpy()[0])
      idx_dif = xx - xx_.clip(min=0) - out 
      #value = torch.div(idx_dif,out.clip(min=0.001)).clip(min=0)
      value = idx_dif.clip(min=0) * weigths_
      #print(value.detach().cpu().numpy()[0])
      #print(np.round(value.detach().cpu().numpy(),3))
      local_loss = 0
      #for i in range(25):
      #  value = torch.abs(xx[bsort[i]]-i)
      #  value = (value - bout[bsort[i]]).clip(min=0)
      #  local_loss = local_loss +value
        #value = ((bsort-bsortout)-bout[bsortout]).clip(min=0)
        #print(np.round(value.detach().cpu().numpy(),3))
      value = torch.sum(value,dim=-1)
      loss =torch.mean(value)
        #loss_value.append(value)
      # loss =loss/gt.shape[0]
 
      #loss = self.loss(out, sortgt_)
      loss.backward()
      self.optimizer.step()
      loss_buffer.append(loss.cpu().detach().numpy().item())
    return(loss_buffer) 


  def predict(self,testloader,test_base_loop):
      self.model.eval()
      re_rank_idx = []
      for x,_ in testloader:
        x = 1-x # .cuda()
        values = self.model(x)
        out = values.detach().cpu().numpy()
        values = torch.argsort(values,axis=1,descending=False)
        
        values = values.detach().cpu().numpy()
        re_rank_idx.extend(values)
      #re_rank_idx = np.argsort(scores)
      rerank_loops = np.array([loops[l] for loops,l in zip(test_base_loop,re_rank_idx)])
      return(rerank_loops)


  def trainer(self,trainloader,testloader):
  
    val_report = 5
    tain_report_terminal= 1
    epochs     = 10000
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
sequence = '02'
train_data = RankingDataset(root,model_name,sequence)
trainloader = DataLoader(train_data,batch_size = len(train_data),shuffle=False)

# LOAD TEST DATA
sequence = '02'
test_data = RankingDataset(root,model_name,sequence)
testloader = DataLoader(test_data,batch_size = len(test_data)) #

#===== RE-RANKING ========
# Get metric distance (ground truth)
re_model = RERANKING.FC(25)
#re_model = RERANKING.Attention(25)
loss = nn.MSELoss()
rerank = ReRankingTrainer(re_model,loss)

rerank.trainer(trainloader,testloader)




