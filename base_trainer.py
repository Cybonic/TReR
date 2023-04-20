


import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import numpy as np

from utils import retrieve_eval

from torch.utils.data import DataLoader
from dataloaders.rankingdata import RankingDataset

import RERANKING



# =====================================================

class ReRankingTrainer(nn.Module):
  def __init__(self,model,loss,experiment='deafault',lr = 0.01,epochs = 100,lr_step=100,val_report=1,tain_report_terminal=1,device='cuda',max_top_cand=25,**args):
    super(ReRankingTrainer,self).__init__()
    
    self.device = device
    self.val_report = val_report
    self.tain_report_terminal= tain_report_terminal
    self.epochs     = epochs
    self.lr_step = lr_step
    self.model = model.to(device)
    self.loss  = loss
    self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
    if max_top_cand < 25:
      max_top_cand = 25
    self.top_cand = list(range(1,max_top_cand+1))
    self.experiment = experiment


  def Train(self,trainloader,testloader):
    old_perfm  = 0
    loss_log   = []
    top_mnt = 5
    
    try:
      btain_base_loop,base_sim =  trainloader.dataset.dataset.get_base_loops()
      train_targets = np.array(trainloader.dataset.dataset.get_targets())
      train_idx =  np.array(trainloader.dataset.indices)
      self.train_targets = train_targets[train_idx]
      train_base_loop = btain_base_loop[train_idx]

      # Test
      test_indices = np.array(testloader.dataset.indices)
      test_base_loop,test_base_sim =  testloader.dataset.dataset.get_base_loops()
      test_base_loop = test_base_loop[test_indices]
      test_targets = np.array(testloader.dataset.dataset.get_targets())
      test_targets = test_targets[test_indices]

      test_relevance = testloader.dataset.dataset.get_gt_relevance()
      self.test_relevance = test_relevance[test_indices]
      
    except:
      base_loop,base_sim =  trainloader.dataset.get_base_loops()
      targets =trainloader.dataset.get_targets()
      test_base_loop,test_base_sim =  testloader.dataset.get_base_loops()
      test_targets = testloader.dataset.get_targets() 

    # TRAIN PERFORMANCE
    base_perfm = retrieve_eval(train_base_loop,self.train_targets,top=25)

    print("\n========\n")
    print(base_perfm)
    print("\n========\n")
    # TEST PERFORMANCE
    test_perf_record = {}
    for i in self.top_cand:
      rerank_perfm = retrieve_eval(test_base_loop,test_targets,top=i)
      test_perf_record[i]=rerank_perfm

    # test_base_perfm = retrieve_eval(test_base_loop,test_targets,top=2)

    best_perf_record = []
    for epoch in range(self.epochs):
      
      loss,re_rank_idx  = self.train_epoch(epoch,trainloader)
      rerank_loops = np.array([loops[l] for loops,l in zip(train_base_loop,re_rank_idx)])
      train_rerank_perfm  = retrieve_eval(rerank_loops,self.train_targets ,top=top_mnt)

      value = np.round(np.mean(loss),3)
      loss_log.append(value)

      if epoch%self.lr_step == 0 and epoch > 0:
        self.optimizer.param_groups[0]['lr'] = self.optimizer.param_groups[0]['lr']/10
  
      if epoch%self.tain_report_terminal == 0:
        print('T ({}) | Loss {:.10f} Recall {:.3}'.format(epoch,np.mean(loss_log),train_rerank_perfm['recall']))

      # Val 
      if epoch % self.val_report == 0:
        
        re_rank_idx,standard_metrics = self.predict(testloader)
        rerank_loops = np.array([loops[l] for loops,l in zip(test_base_loop,re_rank_idx)])
        rerank_perfm = retrieve_eval(rerank_loops,test_targets,top=top_mnt)
       
        #delta = standard_metrics['recall_rr'][25][top_mnt-1] - test_perf_record[top_mnt]['recall']
        rerank_perfm = standard_metrics['recall_rr'][25][top_mnt-1]
        delta = standard_metrics['recall_rr'][25][top_mnt-1] - standard_metrics['recall'][25][top_mnt-1]
        print(f"\nReRank Recall: {round(rerank_perfm,5)} | delta: {round(delta,5)} ")

        if rerank_perfm>old_perfm:
          old_perfm = rerank_perfm
          perf_record = {}

          for i in self.top_cand:
            perf_record[i] = retrieve_eval(rerank_loops,test_targets,top=i)

          best_perf_record = standard_metrics
          
          print("\nBest performance\n")
          #self.save_checkpoint(epoch,best_perf_record,self.experiment,top_mnt)
          #save_log = [ed_sim,ed_loop,rerank_loops,scores,target_ord]
    
    
    self.save_results_csv2(self.experiment,best_perf_record,top_mnt)
    print("\n ******************Best*********************\n")
    recall = np.round(best_perf_record['recall'][25][top_mnt-1],2)
    recall_rr = np.round(best_perf_record['recall_rr'][25][top_mnt-1],2)
    print(f"BaseLine  {recall}")
    print(f"Reranking {recall_rr}")


  def save_checkpoint(self, epoch, best_log, filename,top_mnt):
    state = {
        'arch':str(self.model),
        'epoch': epoch,
        'state_dict': self.model.state_dict(),
        'monitor_best': best_log,
    }
    best = round(best_log[top_mnt]['recall'],2)
    checkpoint_dir = ''
    filename = os.path.join(checkpoint_dir, f'{filename}-{best}.pth')
    torch.save(state, filename)
    print("Saving current best: best_model.pth")


  def save_results_csv2(self,file,results,top,**argv):
    import pandas as pd
    if file == None:
      raise NameError    #file = self.results_file # Internal File name 
    
    metrics = list(results.keys())[5:]
    recall= []
    for tag in metrics[:2]:
      v = results[tag][25]
      recall.append(v)
    
    recall = np.transpose(np.array(recall))
    recall = np.round(recall,2)

    scores = np.zeros((recall.shape[0],3))
    scores[0,0]= np.round(results['MRR'][25],2)
    scores[0,1]= np.round(results['MRR_rr'][25],2)
    scores[0,2]= results['mean_t_RR']
    scores = np.hstack((recall,scores,))
    # Check if the results were generated
 

    colum = metrics
    #rows = np.concatenate((top_cand,array),axis=1)
    df = pd.DataFrame(scores,columns = metrics)
    #file_results = file + '_' + 'best_model.csv'
    best = np.round(results['recall_rr'][25][top-1],2)
    checkpoint_dir = ''
    filename = os.path.join(checkpoint_dir,f'{file}-{str(best)}.csv')
    df.to_csv(filename)


  def save_results_csv(self,file,results,base_results,top):
    import pandas as pd
    metrics = list(results.keys())
    values  = list(results.values())

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

    array = np.array(array).round(decimals=2)
    best = array[top-1,1]
    #colum = ['top','base-recall','base-precision','reranked-recall','reranked-precision']
    colum = ['top','base','reranked']
    rows = np.concatenate((top_cand,array),axis=1)
    df = pd.DataFrame(rows,columns = colum)
    #file_results = file + '_' + 'best_model.csv'
    checkpoint_dir = ''
    filename = os.path.join(checkpoint_dir,f'{file}-{str(best)}.csv')
    df.to_csv(filename)


