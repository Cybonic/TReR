


import torch
import torch.nn as nn
import os
import numpy as np
from scipy.spatial import distance
from tqdm import tqdm
from data_helper import *



def compute_distance(query_data,data):
  import time
  cov = np.cov(data.T)
  inv_covmat = np.linalg.inv(cov)
  n = data.shape[0]
  m = len(query_data)
  md_matrix = np.empty((m,n))
  ed_matrix = np.empty((m,n))

  for i,x in tqdm(enumerate(query_data),total=m):
    #time.sleep(0.3)
    #for j,y in enumerate(data):
      #value2 = distance.mahalanobis(x,y,inv_covmat)
    x.reshape(1,-1) 
    value = calculateMahalanobis(x,data,inv_covmat)
    md_matrix[i,:]=value
    
    eu_value = np.linalg.norm(x - data,axis=1)
    ed_matrix[i,:]=eu_value

  return({'md':md_matrix,'ed':ed_matrix})


  

def compute_distance(query_data,data):
  import time
  cov = np.cov(data.T)
  inv_covmat = np.linalg.inv(cov)
  n = data.shape[0]
  m = len(query_data)
  md_matrix = np.empty((m,n))
  ed_matrix = np.empty((m,n))

  for i,x in tqdm(enumerate(query_data),total=m):
    #time.sleep(0.3)
    #for j,y in enumerate(data):
      #value2 = distance.mahalanobis(x,y,inv_covmat)
    x.reshape(1,-1) 
    value = calculateMahalanobis(x,data,inv_covmat)
    md_matrix[i,:]=value
    
    eu_value = np.linalg.norm(x - data,axis=1)
    ed_matrix[i,:]=eu_value

  return({'md':md_matrix,'ed':ed_matrix})


def comp_loops(sim_map,queries,window=500):
  loop_cand = []
  loop_sim = []
  for i,q in enumerate(queries):
    sim = sim_map[i]
    idx = q-window 
    elegible = sim[:idx]
    cand = np.argsort(elegible)[:25]
    sim = elegible[cand]
    loop_sim.append(sim)
    loop_cand.append(cand)
  return np.array(loop_cand), np.array(loop_sim)


# LOAD PREDICTION DATA
root = '/home/tiago/Dropbox/RAS-publication/predictions/paper/kitti'
sequence = '02_05_06_08'
model = 'ORCHNet_pointnet'


sequence = ['00','02','05','06','08']
sequence_num = ['02_05_06_08','00_05_06_08','00_02_06_08','00_02_05_08','00_02_05_06']

for i in range(0,5):

  
  file2load = os.path.join(root,sequence_num[i],model,'best_model.torch') 
  trining_data = torch.load(file2load)

  descriptors = trining_data['descriptors']
  
  targets = trining_data['targets']
  


  # LOAD GROUND-TRUTH DATA
  ground_truth = torch.load(f'data/gt_kitti-{sequence[i]}.torch')
  
  
  poses = ground_truth['poses']
  # Old approach is based on queries and positives that were saved previously, is a smaller set, performance is higher
  old_queries = ground_truth['anchors']
  query_pose = poses[old_queries,:]
  old_pose_dist = compute_distance(query_pose,poses)
  
  dist_file = f'feat_distance-{sequence[i]}-{model}.torch'
  data = np.array(list(descriptors.values()))
  query_data = data[old_queries,:]
  dist = compute_distance(query_data,data)
  torch.save(dist,dist_file)


  new_queries = trining_data['queries']
  query_pose = poses[new_queries,:]
  new_pose_dist = compute_distance(query_pose,poses)
  pose_dist_file = f'pose_distance-{sequence[i]}-{model}.torch'
  
  torch.save({'new':new_pose_dist,'old':old_pose_dist},pose_dist_file)
  #pose_dist = torch.load(pose_dist_file)
  #
  
# eat_dist = torch.load(dist_file)

