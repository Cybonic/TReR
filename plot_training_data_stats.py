


import torch
import torch.nn as nn
import os
import numpy as np



from torch.utils.data import DataLoader
from dataloaders.alphaqedata import AlphaQEData







# LOAD TTRAINING DATA


def load_data(root,model_name,seq,train_percentage,dataset='new',batch_size=50):
  



  return 


if __name__=='__main__':
  root ='/home/tiago/Dropbox/RAS-publication/predictions/paper/kitti/place_recognition'
  model_name = 'VLAD_pointnet'
  sequence = ['00','02','05','06','08']

  train_percentage = 0.2
  for seq in sequence:
    train_data = AlphaQEData(root,model_name,seq)
    train_size = int(len(train_data)*train_percentage)
    test_size = len(train_data) - train_size

    train, test =torch.utils.data.random_split(train_data,[train_size,test_size])

    print(seq)
    print(len(train))
    print(len(test))






