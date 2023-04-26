


import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import numpy as np


from torch.utils.data import DataLoader
from dataloaders.rankingdata import RankingDataset,RankingMSE,RankingNewRetreivalDataset
from dataloaders.alphaqedata import AlphaQEData

from base_trainer import ReRankingTrainer
from time import time

from losses.logistic_loss import logistic_loss
from losses.margin_ranking_loss import margin_ranking_loss

#rom models.TranformerEncoder import TranformerEncoder,TranformerEncoderWout
#from models.AttentionRanking import AttentionRanking,AttentionRankingWout
from models import pipeline
# =====================================================

      
class AttentionTrainer(ReRankingTrainer):
  def __init__(self,**args):
    super(AttentionTrainer,self).__init__(**args)
    self.max_top_cand = args['max_top_cand']

  def train_epoch(self,epoch,loader):
    #tain_report_terminal= 10
    self.model.train()
    loss_buffer = []
    re_rank_idx = []
    for emb,scores,pos in loader:
      em_map = emb['map'].to(self.device)
      self.optimizer.zero_grad()
      #batch = batch.to(self.device)
      gt = pos['table'].to(self.device)
      #std = 0.0001*epoch
      #noise = torch.randn(batch.size()) * std + 0
      #batch = batch + noise.to(self.device)
      out = self.model(em_map)

      loss_value = self.loss(out,gt)
      loss_value.backward()
      self.optimizer.step()

      values = torch.argsort(out,dim=-1,descending=True)
        
      values = values.detach().cpu().numpy().astype(np.uint8)
      re_rank_idx.extend(values)
      
      loss_buffer.append(loss_value.cpu().detach().numpy().item())
    return(loss_buffer,re_rank_idx) 


  def predict(self,testloader,radius=[25]):
      
      self.model.eval()
      k = self.max_top_cand
      global_metrics = {'tp': {r: [0] * k for r in radius}}
      global_metrics['tp_rr'] = {r: [0] * k for r in radius}
      global_metrics['RR'] = {r: [] for r in radius}
      global_metrics['RR_rr'] = {r: [] for r in radius}
      global_metrics['t_RR'] = []

      local_metrics_vec = []
      n_samples = len(testloader.dataset)
      re_rank_idx = []
      for emb,scores,pos in testloader:
        #x = 1-x # .cuda()
        query_pos = pos['q'].detach().numpy()
        map_positions = pos['map'].detach().numpy()

        x = emb['map'].to(self.device)
        tick = time()
        values = self.model(x)
        rr_nn_ndx = torch.argsort(values,dim=-1,descending=True)
        t_rerank = time() - tick

        rr_nn_ndx_vec = rr_nn_ndx.detach().cpu().numpy().astype(np.uint8)
        re_rank_idx.append(rr_nn_ndx_vec)

        
        # topk_gd_dists = scores.squeeze().detach().numpy()
        for i,(rr_nn_ndx,topk_gd_dists) in enumerate(zip(rr_nn_ndx_vec,scores.detach().numpy())):
          delta_rerank = query_pos[i] - map_positions[i,rr_nn_ndx,:]
          euclid_dist_rr = np.linalg.norm(delta_rerank, axis=-1)
          # Count true positives for different radius and NN number
          global_metrics['t_RR'].append(t_rerank)
          global_metrics['tp'] = {r: [global_metrics['tp'][r][nn] + (1 if (topk_gd_dists[:nn + 1] <= r).any() else 0) for nn in range(k)] for r in radius}
          global_metrics['tp_rr'] = {r: [global_metrics['tp_rr'][r][nn] + (1 if (euclid_dist_rr[:nn + 1] <= r).any() else 0) for nn in range(k)] for r in radius}
          global_metrics['RR'] = {r: global_metrics['RR'][r]+[next((1.0/(i+1) for i, x in enumerate(topk_gd_dists <= r) if x), 0)] for r in radius}
          global_metrics['RR_rr'] = {r: global_metrics['RR_rr'][r]+[next((1.0/(i+1) for i, x in enumerate(euclid_dist_rr <= r) if x), 0)] for r in radius}

      # Calculate mean metrics
      global_metrics["recall"] = {r: [global_metrics['tp'][r][nn] / n_samples for nn in range(k)] for r in radius}
      global_metrics["recall_rr"] = {r: [global_metrics['tp_rr'][r][nn] / n_samples for nn in range(k)] for r in radius}
      global_metrics['MRR'] = {r: np.mean(np.asarray(global_metrics['RR'][r])) for r in radius}
      global_metrics['MRR_rr'] = {r: np.mean(np.asarray(global_metrics['RR_rr'][r])) for r in radius}
      global_metrics['mean_t_RR'] = np.mean(np.asarray(global_metrics['t_RR']))
      
      #rerank_loops = test_base_loop
      return(re_rank_idx,global_metrics)


def load_data(root,model_name,seq,train_percentage,dataset='new',batch_size=50):
  if dataset == 'new':
    train_data = RankingNewRetreivalDataset(root,model_name,seq)
  elif dataset == 'AlphaQE':
     train_data = AlphaQEData(root,model_name,seq)
  else:
    train_data = RankingMSE(root,model_name,seq)

  train_size = int(len(train_data)*train_percentage)
  test_size = len(train_data) - train_size

  train, test =torch.utils.data.random_split(train_data,[train_size,test_size])
  #trainloader = DataLoader(train,batch_size = int(len(train)),shuffle=True)
  trainloader = DataLoader(train,batch_size = len(train),shuffle=True)
  testloader = DataLoader(test,batch_size =  len(test)) #

  return trainloader,testloader,train_data.get_max_top_cand(),str(train_data)
# LOAD TTRAINING DATA

if __name__=='__main__':


  root = '/home/tiago/Dropbox/RAS-publication/predictions/paper/kitti/place_recognition'

  train_size = 0.2
  #device = 'cuda:0'
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

  Models = ['VLAD_pointnet', 'ORCHNet_pointnet' ,'SPoC_pointnet', 'GeM_pointnet']
  #Models = ['VLAD_pointnet']
  sequences = ['00','02','05','06','08']
  #sequences = ['00']
  #model_list = ['tranformerencoder_max_fc_drop','tranformerencoder_wout_fc_drop','tranformerencoder_cnn_fc_drop',
  #              'attention_max_fc_drop','attention_wout_fc_drop','attention_cnn_fc_drop']
  model_list = ['tranformerencoder_max','tranformerencoder_wout','tranformerencoder_cnn',
                'attention_max','attention_wout','attention_cnn']
  
  model_list = ['attention_max','attention_wout','attention_cnn','attention_max_fc_drop','attention_wout_fc_drop','attention_cnn_fc_drop']
  model_list = ['max_fc']
  loss_list  = [logistic_loss]#,margin_ranking_loss]

  for model_obj in model_list:
    for loss_obj in loss_list:
      for model_name in Models:
        for seq in sequences:
          torch.manual_seed(0)
          np.random.seed(0)
          #seq = '02'
          trainloader,testloader,max_top_cand,datasetname = load_data(root,model_name,seq,train_size,'AlphaQE',batch_size=100)
          #===== RE-RANKING ========

          model_fun = pipeline.__dict__[model_obj](cand=max_top_cand,feat_size=256)
          # model_fun = model_obj(max_top_cand,256)
          loss_fun = loss_obj(max_top_cand)

          root_save = os.path.join('results',"paperv2",'noAttention',str(loss_fun),model_name,datasetname,seq,str(train_size))
          if not os.path.isdir(root_save):
            os.makedirs(root_save)

          experiment = os.path.join(root_save,f'{str(model_fun)}')
          rerank = AttentionTrainer(experiment=experiment,loss = loss_fun, model = model_fun,lr= 0.001,epochs = 500,lr_step=250,val_report=1,tain_report_terminal=1,device=device,max_top_cand = max_top_cand)

          rerank.Train(trainloader,testloader)




