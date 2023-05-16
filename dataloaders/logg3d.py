


import torch
import torch.nn as nn
import os, sys
import numpy as np
from torch.utils.data import DataLoader

import pickle
from tqdm import tqdm

sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir))
from utils import comp_loops
sequence_num = {'00':'02_05_06_08','02':'00_05_06_08','05':'00_02_06_08','06':'00_02_05_08','08':'00_02_05_06'}
max_loop_cand = {'00':37,'02':44,'05':24,'06':24,'08':38}

def compute_distance(query_data,data):
  import time
  n = data.shape[0]
  m = len(query_data)
  md_matrix = np.empty((m,n))
  ed_matrix = np.empty((m,n))

  for i,x in tqdm(enumerate(query_data),total=m):
    
    eu_value = np.linalg.norm(x.reshape(1,-1) - data,axis=1)
    ed_matrix[i,:]=eu_value

  return(ed_matrix)


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

def load_torch_block(root,files):
  array = []
  for f in files:
      descriptor_file = os.path.join(root,f)
      #descriptor_file = os.path.join(root,sequence,'logg3d_descriptor.torch')
      descriptors =torch.load(descriptor_file)
      array.extend(descriptors)
  
  return np.array(array)

def load_data(root,sequence):

    root_seq = os.path.join(root,sequence)
    
    dir_list = os.listdir(root_seq)


    desctpr_files = np.sort([f for f in dir_list if f.startswith('logg3d_descriptor')])
    descriptors = load_torch_block(root_seq,desctpr_files)

    features_files = np.sort([f for f in dir_list if f.startswith('logg3d_feats')])
    features = load_torch_block(root_seq,features_files)

    keypts_files = np.sort([f for f in dir_list if f.startswith('logg3d_keypts')])
    keypts = load_torch_block(root_seq,keypts_files)

    
    ground_truth_file = f'data/gt_kitti-{sequence}.torch'

    # LOAD GROUND-TRUTH DATA
    ground_truth = torch.load(ground_truth_file)

    queries_idx = ground_truth['anchors']
    map_idx = ground_truth['targets']
    poses = ground_truth['poses']
  
    return{'d':descriptors,'f':features,'k':keypts,'p':poses,'m':map_idx,'q':queries_idx}

class LOGG3DData():
  def __init__(self,root,model,sequence):

    self.max_top_cand = 25 #max_loop_cand[sequence]

    data = load_data(root,sequence)

    self.targets     = data['m'] # ground truth Loop idx 
    self.poses = data['p'] # Ground truth metric distance
    self.descriptors = data['d'].squeeze()
    self.queries_idx = data['q'] # Query idx

    # Get pose 
    self.query_pose = self.poses[self.queries_idx]
    pose_dist = compute_distance(self.query_pose,self.poses)
    # Get Descriptor distance
    self.query_destps = self.descriptors[self.queries_idx]
    destprs_dist = compute_distance(self.query_destps,self.descriptors)

    self.base_loops,self.base_relevance = comp_loops(destprs_dist, self.queries_idx,window=500,max_top_cand=self.max_top_cand )
    
    self.base_descriptors = np.array([self.descriptors[l] for l in self.base_loops])
    self.base_query_descriptors = self.descriptors[self.queries_idx]

    self.base_target_relevance = np.array([d[l] for d,l in zip(pose_dist,self.base_loops)])
    self.base_map_pose =  np.array([self.poses[l] for l in self.base_loops])
    self.table = compt_y_table(self.base_target_relevance)

  def __len__(self):
    return len(self.base_query_descriptors) 
  

  def get_queries(self):
    return self.queries_idx
  
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
    return "LoGGNet3D"

  def load_data2(self):
    
    with open(self.descriptor_file, 'rb') as handle:
      descriptors = np.array(pickle.load(handle)).squeeze()

    # LOAD GROUND-TRUTH DATA
    ground_truth = torch.load(self.ground_truth_file)

    queries_idx = ground_truth['anchors']
    map_idx = ground_truth['targets']
    poses = ground_truth['poses']
  
    return{'d':descriptors,'p':poses,'m':map_idx,'q':queries_idx}

  def __getitem__(self,idx):
    query_pos = self.query_pose[idx]
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
  

class cros_seq_dataset():
  def __init__(self,root,model,sequences):
    
    query_pos_vec = []
    map_pos_vec = []
    target_buf = []
    scores_vec = []
    query_emb_vec = []
    map_emb_vec = []
    base_loops_buf = []
    base_relevance_buf = []
    table_buf = []
    queries_buf = []
    descriptors_buf = []
    poses_buf = []
    sum_n_frames = 0
    for i,(seq) in enumerate(sequences):
      data = LOGG3DData(root,model,seq)

      queries = np.array(data.get_queries())
      queries_buf.extend(sum_n_frames+ queries)

      sum_n_frames = sum_n_frames + len(data.get_queries())
      
      base_loops_buf.extend(data.base_loops)
      base_relevance_buf.extend(data.base_relevance)

      descriptors_buf.extend(data.get_descriptors())
      query_pos = data.query_pose
      query_pos_vec.extend(query_pos)

      map_pos = data.base_map_pose
      map_pos_vec.extend(map_pos)
      
      table = data.table
      table_buf.extend(table)
      
      scores = data.base_target_relevance
      scores_vec.extend(scores)
      
      query_emb = data.base_query_descriptors
      query_emb_vec.extend(query_emb)
      
      map_emb = data.base_descriptors
      map_emb_vec.extend(map_emb)
      target_buf.extend(data.targets)

      poses_buf.extend(data.get_pose())
   

    self.descriptors = np.array(descriptors_buf)
    self.queries_buf=np.array(queries_buf)
    # 
    self.base_loops_buf = np.array(base_loops_buf)
    self.base_relevance_buf = np.array(base_relevance_buf)

    # Poses
    self.query_pos_vec = np.array(query_pos_vec)
    self.map_pos_vec = np.array(map_pos_vec)
    # Embeddings
    self.map_emb_vec = np.array(map_emb_vec)
    self.query_emb_vec = np.array(query_emb_vec)
    # 
    self.target_buf = target_buf
    #self.target_buf = np.array(target_buf)
    self.scores_vec = np.array(scores_vec)
    self.table_buf = np.array(table_buf)

    self.poses_buf = np.array(poses_buf)
    
    

  def __len__(self):
    return len(self.target_buf)

  def __getitem__(self,idx):
    query_pos = self.query_pos_vec[idx]
    map_pos = self.map_pos_vec[idx]
    map_emb = self.map_emb_vec[idx]
    map_emb = torch.from_numpy(map_emb).float()
    table = self.table_buf[idx]
    query_emb = self.query_emb_vec[idx]
    query_emb = torch.from_numpy(query_emb).unsqueeze(dim=0).float()
    scores = self.scores_vec[idx]
    scores = torch.from_numpy(scores).float()

    emb = {'q':query_emb,'map':map_emb,'idx':idx}
    pos = {'q':query_pos,'map':map_pos,'table':table}
    return emb,scores,pos
  
  def get_base_loops(self):
    return self.base_loops_buf,self.base_relevance_buf
  
  def get_targets(self):
    return self.target_buf
  
  def get_descriptors(self):
    return self.descriptors
  
  def get_poses(self):
    return self.poses_buf
  


class LOGGNet3D_CROSS():
  def __init__(self,root,model,sequence):

    cross_test_seq = {'00':['02','05','06','08'],
                      '02':['00','05','06','08'],
                      '05':['00','02','06','08'],
                      '06':['00','02','05','08'],
                      '08':['00','02','05','06']}
    
    self.train_data = cros_seq_dataset(root,model,cross_test_seq[sequence])
    #self.train_data = cros_seq_dataset(root,model,[sequence])
    self.test_data = cros_seq_dataset(root,model,[sequence])
  
  def get_test_loader(self,batch_size=None):
    if batch_size == None:
      batch_size = len(self.test_data)
    return DataLoader(self.test_data,batch_size = batch_size) #
  
  def get_train_loader(self,batch_size=None):
    if batch_size == None:
      batch_size = len(self.train_data)
    return DataLoader(self.train_data,batch_size = batch_size,shuffle=True) #
  
  def get_train_base_loops(self):
    return self.train_data.get_base_loops()
  
  def get_test_base_loops(self):
    return self.test_data.get_base_loops()
  

  def __str__(self):
    return str(self.test_data)
    # Test data 
    
