


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
  def __init__(self,model,loss,experiment='deafault',lr = 0.01,epochs = 100,lr_step=100,val_report=1,tain_report_terminal=1,device='cuda',**args):
    super(ReRankingTrainer,self).__init__()
    
    self.device = device
    self.val_report = val_report
    self.tain_report_terminal= tain_report_terminal
    self.epochs     = epochs
    self.lr_step = lr_step
    self.model = model.to(device)
    self.loss  = loss
    self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr)
    self.top_cand = list(range(1,26))
    self.experiment = experiment


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
        
        rerank_perfm = retrieve_eval(rerank_loops,targets,top=1)
       
        delta = rerank_perfm['recall'] - test_perf_record[1]['recall']
        print(f"\nReRank Recall: {round(rerank_perfm['recall'],5)} | delta: {round(delta,5)} ")

        if rerank_perfm['recall']>old_perfm:
          old_perfm = rerank_perfm['recall']
          perf_record = {}

          for i in self.top_cand:
            perf_record[i] = retrieve_eval(rerank_loops,targets,top=i)

          self.save_checkpoint(epoch,perf_record,self.experiment)
          self.save_results_csv(self.experiment,perf_record,test_perf_record)
          print("\nBest performance\n")
          #save_log = [ed_sim,ed_loop,rerank_loops,scores,target_ord]
    
    print("\n ******************Best*********************\n")
    print(f"BaseLine  {test_perf_record[1]['recall']}")
    print(f"Reranking {old_perfm}")


  def save_checkpoint(self, epoch, best_log, filename):
    state = {
        'arch':str(self.model),
        'epoch': epoch,
        'state_dict': self.model.state_dict(),
        'monitor_best': best_log,
    }
    best = round(best_log[1]['recall'],2)
    checkpoint_dir = ''
    filename = os.path.join(checkpoint_dir, f'{filename}-{best}.pth')
    torch.save(state, filename)
    print("Saving current best: best_model.pth")



  def save_results_csv(self,file,results,base_results):
    import pandas as pd
    
    # Check if the results were generated
    #assert hasattr(self, 'results'), 'Results were not generated!'
    if file == None:
        raise NameError    #file = self.results_file # Internal File name 
    top_cand = np.array(list(results.keys())).reshape(-1,1)

    base_values   = np.array(list(base_results.values()))
    rerank_values = np.array(list(results.values()))

    array = []
    for b,r in zip(base_values,rerank_values):
      array.append([b['recall'],r['recall']])

    best = round(array[0][1],2)
    colum = ['top','base','reranked']
    rows = np.concatenate((top_cand,array),axis=1)
    df = pd.DataFrame(rows,columns = colum)
    #file_results = file + '_' + 'best_model.csv'
    checkpoint_dir = ''
    filename = os.path.join(checkpoint_dir,f'{file}-{str(best)}.csv')
    df.to_csv(filename)


