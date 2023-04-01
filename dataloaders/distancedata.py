


import torch
import torch.nn as nn
import os, sys
import numpy as np
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir))
from utils import comp_loops

class RankingDataset():
  def __init__(self,root,model,sequence):
    sequence_num = {'00':'02_05_06_08','02':'00_05_06_08','05':'00_02_06_08','06':'00_02_05_08','08':'00_02_05_06'}

    self.norm = nn.LayerNorm(25)
    self.file2load = os.path.join(root,sequence_num[sequence],model,'best_model.torch') 
    self.ground_truth_file = f'data/gt_kitti-{sequence}.torch'
    self.pose_dist_file = f'data/pose_distance-{sequence}-{model}.torch'
    self.feat_dist_file = f'data/feat_distance-{sequence}-{model}.torch'
    data = self.load_data()

    self.targets = data['t']
    pose = data['p']['ed']

    self.baseloop,self.basesim= comp_loops(data['f']['ed'], data['q'],window=500)

    self.target_sim = np.array([d[l] for d,l in zip(pose,self.baseloop)])

    target_ord = np.array([l[np.argsort(d[l])] for d,l in zip(pose,self.targets)])

  def __len__(self):
    return len(self.targets) 
  
  def get_base_loops(self):
    return self.baseloop,self.basesim
  
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

    query_data = data[queries,:]
    query_pose = poses[queries,:]

    pose_dist = torch.load(self.pose_dist_file)
    feat_dist = torch.load(self.feat_dist_file)

    return{'f':feat_dist,'p':pose_dist,'t':targets,'q':queries}

  def __getitem__(self,idx):
    target = self.target_sim[idx]
    pred = self.basesim[idx]
    #pred = pred/pred.max()
    pred = torch.from_numpy(pred).float()
    #pred = self.norm(pred)
    #target = target/target.max()
    #target_mask = np.zeros((25,1))
    
    #arg_sort = np.argsort(target)
    #target_mask[arg_sort==0]=1
    #for i,idx in enumerate(arg_sort):
    #    target_mask[idx,i]=1
    target = torch.from_numpy(target).float()
    
    #target = self.norm(target)
    return pred,target