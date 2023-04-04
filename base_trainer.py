


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
  def __init__(self,model,loss,lr = 0.01,epochs = 100,lr_step=100,val_report=1,tain_report_terminal=1,device='cuda',**args):
    super(ReRankingTrainer,self).__init__()
    
    self.device = device
    self.val_report = val_report
    self.tain_report_terminal= tain_report_terminal
    self.epochs     = epochs
    self.lr_step = lr_step
    self.model = model.to(device)
    self.loss  = loss
    self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr)
    self.top_cand = [1,25]


  def Train(self,trainloader,testloader):
    old_perfm  = 0
    loss_log   = []

    try:
      base_loop,base_sim =  trainloader.dataset.dataset.get_base_loops()
      targets =trainloader.dataset.dataset.get_targets()
      test_base_loop,test_base_sim =  testloader.dataset.dataset.get_base_loops()
      test_targets = testloader.dataset.dataset.get_targets()  
    except:
      base_loop,base_sim =  trainloader.dataset.get_base_loops()
      targets =trainloader.dataset.get_targets()
      test_base_loop,test_base_sim =  testloader.dataset.get_base_loops()
      test_targets = testloader.dataset.get_targets() 

    # TRAIN PERFORMANCE
    base_perfm = retrieve_eval(base_loop,targets,top=2)

    # TEST PERFORMANCE
    test_perf_record = {}
    for i in self.top_cand:
      rerank_perfm = retrieve_eval(test_base_loop,test_targets,top=i)
      test_perf_record[i]=rerank_perfm

    # test_base_perfm = retrieve_eval(test_base_loop,test_targets,top=2)

    best_perf_record = []
    for epoch in range(self.epochs):
      
      loss  = self.train_epoch(epoch,trainloader)
      value = np.round(np.mean(loss),3)
      loss_log.append(value)

      if epoch%self.lr_step == 0 and epoch >0:
        self.optimizer.param_groups[0]['lr'] = self.optimizer.param_groups[0]['lr']/10
      
      if epoch%self.tain_report_terminal == 0:
        print('T ({}) | Loss {:.10f}'.format(epoch,np.mean(loss_log)))
      
      # Val 
      if epoch % self.val_report == 0:
        rerank_loops = self.predict(testloader,test_base_loop)
        
        perf_record = {}
        for i in self.top_cand:
          rerank_perfm = retrieve_eval(rerank_loops,targets,top=i)
          perf_record[i]=rerank_perfm
        
        delta = perf_record[1]['recall'] - test_perf_record[1]['recall']
        print(f"\nReRank Recall: {round(rerank_perfm['recall'],5)} | delta: {round(delta,5)} ")

        if perf_record[1]['recall']>old_perfm:
          old_perfm = perf_record[1]['recall']
          print("\nBest performance\n")
          best_perf_record = perf_record
          #save_log = [ed_sim,ed_loop,rerank_loops,scores,target_ord]
    
    print("\n ******************Best*********************\n")
    print(f"BaseLine  {test_perf_record[1]['recall']}")
    print(f"Reranking {best_perf_record[1]['recall']}")








