


import torch
import torch.nn as nn
import os, sys
import numpy as np
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir))
from utils import comp_loops
sequence_num = {'00':'02_05_06_08','02':'00_05_06_08','05':'00_02_06_08','06':'00_02_05_08','08':'00_02_05_06'}
max_loop_cand = {'00':37,'02':44,'05':24,'06':24,'08':38}

def compt_y_table(y):
  n = y.shape[1]
  batch_size = y.shape[0]
  table = np.ones((batch_size,n,n))*-1
  for z,(b) in enumerate(y):
    for i in range(n):
      for j in range(n):
        if b[i]<b[j]: #
          table[z,i,j] = 1
  return table

class AlphaQEData():
  def __init__(self,root,model,sequence):

    self.max_top_cand = max_loop_cand[sequence]
    self.file2load = os.path.join(root,sequence_num[sequence],model,'best_model.torch') 
    self.ground_truth_file = f'data/gt_kitti-{sequence}.torch'
    self.pose_dist_file = f'data/pose_distance-{sequence}-{model}.torch'
    self.feat_dist_file = f'data/feat_distance-{sequence}-{model}.torch'
    data = self.load_data()

    self.targets = data['t']
    target_relevance = data['p']['old']['ed']
    self.descriptors = data['d']
    queries = data['q']

    self.base_loops,self.base_relevance = comp_loops(data['f']['ed'], data['q'],window=500,max_top_cand=self.max_top_cand )
    
    self.base_descriptors = np.array([self.descriptors[l] for l in self.base_loops])
    self.base_query_descriptors = self.descriptors[queries]
    self.base_target_relevance = np.array([d[l] for d,l in zip(target_relevance,self.base_loops)])
    self.base_map_pose =  np.array([self.poses[l] for l in self.base_loops])
    self.table = compt_y_table(self.base_target_relevance)

  def __len__(self):
    return len(self.base_query_descriptors) 
  
  def get_gt_relevance(self):
    return self.base_target_relevance
  
  def get_base_loops(self):
    return self.base_loops,self.base_relevance
  
  def get_targets(self):
    return self.targets
  
  def get_descriptors(self):
    return self.descriptors
  def get_pose(self):
    return self.poses
  
  def __str__(self):
    return "AlphaQEData"

  def load_data(self):
    trining_data = torch.load(self.file2load)
    descriptors = trining_data['descriptors']
    data = np.array(list(descriptors.values()))
    # LOAD GROUND-TRUTH DATA
    ground_truth = torch.load(self.ground_truth_file)
    
    queries = ground_truth['anchors']
    map = ground_truth['targets']
    self.poses = ground_truth['poses']

    self.query_pos = self.poses[queries]
    self.map_pos = [self.poses[p] for p in map]

    pose_dist = torch.load(self.pose_dist_file)
    feat_dist = torch.load(self.feat_dist_file)

    return{'d':data,'f':feat_dist,'p':pose_dist,'t':map,'q':queries}

  def __getitem__(self,idx):
    query_pos = self.query_pos[idx]
    map_pos = self.base_map_pose[idx]
    target = self.table[idx]

    scores = self.base_target_relevance[idx]
    scores = torch.from_numpy(scores).float()
    
    query_emb = self.base_query_descriptors[idx]
    query_emb = torch.from_numpy(query_emb).unsqueeze(dim=0).float()

    map_emb = self.base_descriptors[idx]
    map_emb = torch.from_numpy(map_emb).float()

    emb = {'q':query_emb,'map':map_emb}
    pos = {'q':query_pos,'map':map_pos,'table':target}
    #target = self.norm(target)
    return emb,scores,pos
  
  def get_max_top_cand(self):
    return self.max_top_cand 