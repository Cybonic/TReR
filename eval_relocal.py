#!/usr/bin/env python3
# This file is covered by the LICENSE file in the root of this project.


# Getting latend space using Hooks :
#  https://towardsdatascience.com/the-one-pytorch-trick-which-you-should-know-2d5e9c1da2ca

# Binary Classification
# https://jbencook.com/cross-entropy-loss-in-pytorch/
from torch.utils.data import DataLoader
from dataloaders.rankingdata import RankingMSE,RankingNewRetreivalDataset
import numpy as np
from tqdm import tqdm

def load_cross_data(root,model_name,seq_train,seq_test):
  train_data = RankingNewRetreivalDataset(root,model_name,seq_train)
  #train_data = RankingMSE(root,model_name,seq_train)
  trainloader = DataLoader(train_data,batch_size = len(train_data),shuffle=True)
  # LOAD TEST DATA
  test_data = RankingNewRetreivalDataset(root,model_name,seq_test)
  #test_data = RankingMSE(root,model_name,seq_test)
  testloader = DataLoader(test_data,batch_size = len(test_data)) #
  return trainloader,testloader,test_data.get_max_top_cand()


def relocal_metric(relevant_hat,true_relevant):
    '''
    Difference between relocal metric and retrieval metric is that 
    retrieval proseposes that only evalautes positive queries
    ...

    input: 
    - relevant_hat (p^): indices of 
    '''
    n_samples = len(relevant_hat)
    recall,precision = 0,0
    
    for p,g in zip(relevant_hat,true_relevant):
        p = np.array(p).tolist()
        n_true_pos = len(g)
        n_pos_hat = len(p)
        tp = 0 
        fp = 0
        if n_true_pos > 0: # postive 
            # Loops exist, we want to know if it can retireve the correct frame
            num_tp = np.sum([1 for c in p if c in g])
            
            if num_tp>0: # found at least one loop 
                tp=1
            else:
                fn=1 
            
        else: # Negative
            # Loop does not exist: we want to know if it can retrieve a frame 
            # with a similarity > thresh
            if n_pos_hat == 0:
                tp=1

        recall += tp/1 
        precision += tp/n_pos_hat if n_pos_hat > 0 else 0
    
    recall/=n_samples
    precision/=n_samples
    return {'recall':recall, 'precision':precision}

def relocalize(queries,database,descriptors,top_cand,window=500,warmup = 600,sim_tresh=None):
    '''
    Retrieval function 
    
    '''
    #metric = 'euclidean' if metric == 'L2'
    from sklearn.neighbors import KDTree
    
    database_dptrs   = np.array([descriptors[i] for i in database])
    tree = KDTree(database_dptrs.squeeze(), leaf_size=2)

    scores,winner = [],[]
    for query in tqdm(queries,"Retrieval"):
        sim,loops = [],[]
        if query > warmup:
          query_dptrs = database_dptrs[query].reshape(1,-1)
          dist = np.linalg.norm(query_dptrs - database_dptrs,axis=-1)
 
          # Remove query index
          map = np.arange(query-window)
          sim = dist[map]
          cand_sort = np.argsort(sim)
          sim_sort = sim[cand_sort]
          
          if sim_tresh != None:
            sim =sim_sort[sim_sort<sim_tresh]
            loops = cand_sort[sim_sort<sim_tresh]
          else:
            sim =sim_sort
            loops = cand_sort

        scores.append(sim[:top_cand])
        winner.append(loops[:top_cand])

        #retrieved_loops,scores = retrieval_knn(query_dptrs, database_dptrs, top_cand =top, metric = eval_metric)
    return(np.array(winner),np.array(scores))



root = '/home/tiago/Dropbox/RAS-publication/predictions/paper/kitti'

train_size = 0.2
device = 'cuda:0'
#device = 'cpu'

Models = ['VLAD_pointnet', 'ORCHNet_pointnet' ,'SPoC_pointnet', 'GeM_pointnet']
#Models = ['VLAD_pointnet']
sequences = ['00','02','05','06','08']
import os
import pandas as pd
import torch
for model_name in Models:
    for seq in sequences:
      train,testloader,max_top_cand = load_cross_data(root,model_name,seq,seq)

      descriptors = testloader.dataset.get_descriptors()
      dataset_len = len(descriptors)

      queries = np.arange(1,dataset_len-1)
      database = np.arange(1,dataset_len)
      loops, sim = relocalize(queries,database,descriptors,top_cand=max_top_cand,window=500,warmup=600,sim_tresh=None)

      poses = testloader.dataset.get_pose()

      loops_gt, sim_gt = relocalize(queries,database,poses,top_cand=max_top_cand,window=500,warmup=600,sim_tresh=25)

      results = relocal_metric(loops,loops_gt)


    ###################################################################### 

      root_save = os.path.join('relocalization',seq,model_name)
      if not os.path.isdir(root_save):
        os.makedirs(root_save)

      file = os.path.join(root_save,'prediction.txt')

      f = open(file,"a")
      for l in loops:
        string = ' '.join([str(i) for i in l]) 
        f.write(string+'\n')
      
      file = os.path.join(root_save,'targets.txt')

      f = open(file,"a")
      for l in loops_gt:
        string = ' '.join([str(i) for i in l]) 
        f.write(string + '\n')
      
      file = os.path.join(root_save,'similarity.txt')

      f = open(file,"a")
      for l in sim:
        string = ' '.join([str(round(i,3)) for i in l]) 
        f.write(string + '\n')
      
      ground_truth = {"loops":loops_gt, "sim":sim_gt }
      prediction = {"loops":loops, "sim":sim }

      
      file = os.path.join(root_save,"data.torch")
      torch.save({"gt":ground_truth,"pred":prediction},file)

    
    



  
  