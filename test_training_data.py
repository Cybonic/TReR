


import torch
import torch.nn as nn
import os
import numpy as np

from utils import retrieve_eval

from torch.utils.data import DataLoader
from dataloaders.distancedata import RankingDataset

import RERANKING 







# LOAD TTRAINING DATA

root = '/home/tiago/Dropbox/RAS-publication/predictions/paper/kitti'
model_name = 'ORCHNet_pointnet'
sequence = '00'

train_data = RankingDataset(root,model_name,sequence)
trainloader = DataLoader(train_data,batch_size = 1,shuffle=True)



for m,g in trainloader:
    
    print(np.round(ou,3))
    print(g)

# LOAD TEST DATA
sequence = '00'
test_data = RankingDataset(root,model_name,sequence)
testloader = DataLoader(test_data,batch_size = 1)





