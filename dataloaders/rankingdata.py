


import torch
import torch.nn as nn
import os, sys
import numpy as np
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir))
from utils import comp_loops
sequence_num = {'00':'02_05_06_08','02':'00_05_06_08','05':'00_02_06_08','06':'00_02_05_08','08':'00_02_05_06'}
max_loop_cand = {'00':37,'02':44,'05':24,'06':24,'08':38}

class RankingDataset():
  def __init__(self,root,model,sequence):
    self.max_top_cand = max_loop_cand[sequence]
    self.norm = nn.LayerNorm(25)
    self.file2load = os.path.join(root,sequence_num[sequence],model,'best_model.torch') 
    self.ground_truth_file = f'data/gt_kitti-{sequence}.torch'
    self.pose_dist_file = f'data/pose_distance-{sequence}-{model}.torch'
    self.feat_dist_file = f'data/feat_distance-{sequence}-{model}.torch'
    data = self.load_data()

    self.targets = data['t']
    self.base_loops = data['pl']
    self.base_relevance = data['ps']
    target_relevance = data['ts']
    descriptors = data['d']
    queries = data['q']

    #self.base_loops,self.base_relevance= comp_loops(data['f']['ed'], data['q'],window=500,max_top_cand=self.max_top_cand)
    
    self.base_descriptors = np.array([descriptors[l] for l in self.base_loops])
    self.base_query_descriptors = descriptors[queries]
    self.base_target_relevance = np.array([d[l] for d,l in zip(target_relevance,self.base_loops)])

    #target_ord = np.array([l[np.argsort(d[l])] for d,l in zip(pose,self.targets)])

  def __len__(self):
    return len(self.targets)

  def get_max_top_cand(self):
    return self.max_top_cand 
  
  def get_base_loops(self):
    return self.base_loops,self.base_relevance
  
  def get_targets(self):
    return self.targets

  def load_data(self):
    trining_data = torch.load(self.file2load)
    descriptors = trining_data['descriptors']
    queries = trining_data['queries']
    targets = trining_data['targets']
    pred_loops = trining_data['prediction_loops']
    pred_scores = trining_data['prediction_scores']
    data = np.array(list(descriptors.values()))
    # LOAD GROUND-TRUTH DATA
    #ground_truth = torch.load(self.ground_truth_file)
   
    #poses = ground_truth['poses']

    pose_dist = torch.load(self.pose_dist_file)['ed']
    #feat_dist = torch.load(self.feat_dist_file)

    return{'d':data,'pl':pred_loops,'ps':pred_scores,'t':targets,'q':queries,'ts':pose_dist}

  def __getitem__(self,idx):
    target = np.argsort(self.base_target_relevance[idx])
    target_mat = np.zeros((25,25))
    for i,j in enumerate(target):
      target_mat[i,j]=1

    queries = self.base_query_descriptors[idx]
    queries = torch.from_numpy(queries).unsqueeze(dim=0).float()

    keys = self.base_descriptors[idx]
    #pred = {'q':self.base_query_descriptors[idx],'k': self.base_descriptors[idx]}
    #pred = self.base_descriptors[idx]
    keys = torch.from_numpy(keys).float()
    target = torch.from_numpy(target_mat).float()
    
    #target = self.norm(target)
    return keys,target


class RankingMSE(RankingDataset):
  def __init__(self,root,model_name,sequence):
    super().__init__(root,model_name,sequence)
    pass

  def __getitem__(self,idx):
    target = np.argsort(self.base_target_relevance[idx])
    max_cand =  self.max_top_cand
    target_ranked = np.zeros(max_cand)

    rank = np.arange(1,max_cand+1,1)[::-1]
    target_ranked[target] = rank
    keys = self.base_descriptors[idx]
    

 
    keys = torch.from_numpy(keys).float()
    target = torch.from_numpy(target_ranked).float()
    
    #target = self.norm(target)
    return keys,target






class RankingNewRetreivalDataset():
  def __init__(self,root,model,sequence):

    self.max_top_cand = max_loop_cand[sequence]
    self.file2load = os.path.join(root,sequence_num[sequence],model,'best_model.torch') 
    self.ground_truth_file = f'data/gt_kitti-{sequence}.torch'
    self.pose_dist_file = f'data/pose_distance-{sequence}-{model}.torch'
    self.feat_dist_file = f'data/feat_distance-{sequence}-{model}.torch'
    data = self.load_data()

    self.targets = data['t']
    target_relevance = data['p']['old']['ed']
    descriptors = data['d']
    queries = data['q']

    self.base_loops,self.base_relevance= comp_loops(data['f']['ed'], data['q'],window=500,max_top_cand=self.max_top_cand )
    
    self.base_descriptors = np.array([descriptors[l] for l in self.base_loops])
    self.base_query_descriptors = descriptors[queries]
    self.base_target_relevance = np.array([d[l] for d,l in zip(target_relevance,self.base_loops)])

    #target_ord = np.array([l[np.argsort(d[l])] for d,l in zip(pose,self.targets)])

  def __len__(self):
    return len(self.targets) 
  
  def get_base_loops(self):
    return self.base_loops,self.base_relevance
  
  def get_targets(self):
    return self.targets

  def load_data(self):
    trining_data = torch.load(self.file2load)
    descriptors = trining_data['descriptors']
    data = np.array(list(descriptors.values()))
    # LOAD GROUND-TRUTH DATA
    ground_truth = torch.load(self.ground_truth_file)
    queries = ground_truth['anchors']
    targets = ground_truth['targets']
    poses = ground_truth['poses']

    pose_dist = torch.load(self.pose_dist_file)
    feat_dist = torch.load(self.feat_dist_file)

    return{'d':data,'f':feat_dist,'p':pose_dist,'t':targets,'q':queries}

  def __getitem__(self,idx):
    target = np.argsort(self.base_target_relevance[idx])

    queries = self.base_query_descriptors[idx]
    queries = torch.from_numpy(queries).unsqueeze(dim=0).float()

    keys = self.base_descriptors[idx]
    keys = torch.from_numpy(keys).float()
    target = torch.from_numpy(target).float()
    
    #target = self.norm(target)
    return keys,target
  
  def get_max_top_cand(self):
    return self.max_top_cand 