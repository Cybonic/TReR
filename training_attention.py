


import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import numpy as np

from utils import retrieve_eval,comp_pair_permutations

from torch.utils.data import DataLoader
from dataloaders.rankingdata import RankingDataset,RankingMSE,RankingNewRetreivalDataset
from dataloaders.alphaqedata import AlphaQEData

from base_trainer import ReRankingTrainer
from time import time

class CNN(nn.Module):
    def __init__(self, d_model, hidden_dim, p):
        super().__init__()
        self.k1convL1 = nn.Linear(d_model,    hidden_dim)
        self.k1convL2 = nn.Linear(hidden_dim, d_model)
        self.activation = nn.ReLU()

    def forward(self, x):
        x = self.k1convL1(x)
        x = self.activation(x)
        x = self.k1convL2(x)
        return x

class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, conv_hidden_dim, p=0.1):
        super().__init__()

        self.mha = torch.nn.MultiheadAttention(d_model, num_heads,p)
        self.cnn = CNN(d_model, conv_hidden_dim, p)

        self.layernorm1 = nn.LayerNorm(normalized_shape=d_model, eps=1e-6)
        self.layernorm2 = nn.LayerNorm(normalized_shape=d_model, eps=1e-6)
    
    def forward(self, x):
        
        # Multi-head attention 
        attn_output, _ = self.mha(x, x, x)  # (batch_size, input_seq_len, d_model)
        
        # Layer norm after adding the residual connection 
        out1 = self.layernorm1(x + attn_output)  # (batch_size, input_seq_len, d_model)
        
        # Feed forward 
        cnn_output = self.cnn(out1)  # (batch_size, input_seq_len, d_model)
        
        #Second layer norm after adding residual connection 
        out2 = self.layernorm2(out1 + cnn_output)  # (batch_size, input_seq_len, d_model)

        return out2

class AttentionRanking(torch.nn.Module):
    def __init__(self,cand=35,feat_size = 256):
        super().__init__()
        
        self.Win =  nn.Parameter(torch.zeros(256,cand))
        self.Wout =  nn.Parameter(torch.zeros(256,1))
        
        nn.init.normal_(self.Win.data, mean=0, std=0.1)
        nn.init.normal_(self.Wout.data, mean=0, std=0.1)

        self.att = torch.nn.MultiheadAttention(cand,1)
        self.classifier = torch.nn.Conv1d(256, 1, 1)
        self.fc = nn.Linear(cand,cand)
        
        fc_drop = [nn.LazyLinear(cand),
             nn.ReLU(),
             nn.Dropout(0.1)
             ]
      
        self.fc_drop = nn.Sequential(*fc_drop)
        layer = [EncoderLayer(256,1,cand)]
        self.enc_n = 1
        for i in range(self.enc_n):
          layer += layer
        self.layer = nn.Sequential(*layer)

    def __str__(self):
      #return f"AttentionRanking_{self.enc_n}xEncoder_cnn"
      return f"AttentionRanking_CanAtt_cnn"
    
    def forward(self,k):
        k = torch.transpose(k,dim0=2,dim1=1)
        out, attn_output_weights = self.att(k,k,k)
        out = k + out
        #out = torch.transpose(out,dim0=2,dim1=1)
        #
        #
        #out = k
        #
       
        # out = self.layer(k)
        # out = self.fc_drop(out)
        # out = torch.transpose(out,dim0=2,dim1=1)
        out = self.classifier(out).squeeze()
        #  
        #out = torch.matmul(out,self.Wout).squeeze()
        #out,idx  = torch.max(out,dim=-1)
        #if self.training:
        #   return out.float()
        return out.float()

       

class MHAERanking(torch.nn.Module):
    def __init__(self,cand=35,feat_size = 256):
      super().__init__()
        

      h = cand
      self.atta = torch.nn.MultiheadAttention(cand,1)
      
      self.ln1 =  nn.LayerNorm(256)
      self.ln2 =  nn.LayerNorm(256)
      
      fc = [nn.LazyLinear(h),
             nn.ReLU(),
             nn.LazyLinear(cand)
             ]
      
      fc_drop = [nn.LazyLinear(h),
             nn.ReLU(),
             nn.Dropout(0.1)
             ]
      
      
      
      
      
      self.mlp = nn.Sequential(*fc_drop)
      self.drop1 = nn.Dropout(0.1)
      self.drop2 = nn.Dropout(0.1)

    def __str__(self):
      return "MaskRanking"
    
    def forward(self,x):
      #a,(b,c) = self.att(x,x,x)
      x, attn_output_weights = self.atta(x,x,x)
      #z = self.ln1(x + self.drop1(x))
      #z =  self.ln2(z + self.drop2(self.mlp(z)))
      return z.float()
    

class MaskRanking(torch.nn.Module):
    def __init__(self,cand=35,feat_size = 256):
      super().__init__()
        
      self.Win =  nn.Parameter(torch.zeros(cand,cand))
      self.Wout =  nn.Parameter(torch.zeros(cand,1))
        
      nn.init.normal_(self.Win.data, mean=0, std=0.5)
      nn.init.normal_(self.Wout.data, mean=0, std=0.5)

      self.att = torch.nn.MultiheadAttention(256,1)
      self.classifier = torch.nn.Conv1d(256, 1, 1)
      self.att2 = torch.nn.MultiheadAttention(cand,1)

      self.classifier = torch.nn.Conv1d(256, 1, 1)
      layer = [MHAERanking(cand)]
      for i in range(1):
        layer += layer

      self.fc = nn.Linear(cand,cand)
      self.layer = nn.Sequential(*layer)
      self.drop = nn.Dropout(0.2)
      self.tr = nn.Transformer(256,1,1,1,37)

    def __str__(self):
      return "MaskRanking"
    
    def forward(self,k):

     
      #out = []
      #for w in  self.Win:
      #  out.append(torch.matmul(k,w))
      #out = k
      #ou_stack = torch.stack(out,dim=1)
      #out,std_out = torch.max(ou_stack,dim=1)
      k = k.tranpose(dim0=2,dim1=1)
      ak,map= self.att(k,k,k)
      k = ak  + k
      out = self.drop(k)
      out = torch.transpose(out,dim0=2,dim1=1)
      k = self.classifier(out).squeeze()
      
      #k,idx  = torch.max(k,dim=-1)
      
      out = self.drop(k)
      out = self.fc(out)
      #
      #out = torch.matmul(out,self.Wout).squeeze()
       
      
      #out = self.fc(out)
      #out,_= self.att2(out,out,out)
      #out = out + out 

      if self.training:
          return out.float()
      return out.float()
    

# =====================================================
class logistic_loss:
  def __init__(self,cand=37):
    self.x1_perm,self.x2_perm = comp_pair_permutations(cand)
  
  def __call__(self,y_pred,y_true):
    x1 = y_pred[:,self.x1_perm]
    x2 = y_pred[:,self.x2_perm]
    y = y_true[:,self.x1_perm,self.x2_perm]
    value = torch.sum((y*torch.log2(1+torch.exp(-(x1-x2)))).clip(min=0),dim=-1)
    return torch.mean(value)

  def __str__(self):
     return 'logistic_loss'

class margin_ranking_loss:
  def __init__(self,cand=37):
        self.loss_fn = torch.nn.MarginRankingLoss(0.1)
        self.x1_perm,self.x2_perm = comp_pair_permutations(cand)

  def __call__(self,y_pred,y_true):
    x1 = y_pred[:,self.x1_perm]
    x2 = y_pred[:,self.x2_perm]
    y  = y_true[:,self.x1_perm,self.x2_perm]
    value = torch.sum((y*torch.log2(1+torch.exp(-(x1-x2)))).clip(min=0),dim=-1)
    return torch.mean(value)

  def __str__(self):
     return 'margin_ranking_loss'

      
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
      n_samples = len(testloader)
      re_rank_idx = []
      for emb,scores,pos in testloader:
        #x = 1-x # .cuda()
        query_pos = pos['q'].detach().numpy()
        map_positions = pos['map'].squeeze().detach().numpy()

        x = emb['map'].to(self.device)
        tick = time()
        values = self.model(x)

        rr_nn_ndx = torch.argsort(values,dim=-1,descending=True)
        t_rerank = time() - tick

        global_metrics['t_RR'].append(t_rerank)
        rr_nn_ndx = rr_nn_ndx.detach().cpu().numpy().astype(np.uint8)
        re_rank_idx.append(rr_nn_ndx)

        delta_rerank = query_pos - map_positions[rr_nn_ndx,:]
        euclid_dist_rr = np.linalg.norm(delta_rerank, axis=1)
        
        topk_gd_dists = scores.squeeze().detach().numpy()
        #euclid_dist_rr = values

        # Count true positives for different radius and NN number
        global_metrics['tp'] = {r: [global_metrics['tp'][r][nn] + (1 if (topk_gd_dists[:nn + 1] <= r).any() else 0) for nn in range(k)] for r in radius}
        global_metrics['tp_rr'] = {r: [global_metrics['tp_rr'][r][nn] + (1 if (euclid_dist_rr[:nn + 1] <= r).any() else 0) for nn in range(k)] for r in radius}
        global_metrics['RR'] = {r: global_metrics['RR'][r]+[next((1.0/(i+1) for i, x in enumerate(topk_gd_dists <= r) if x), 0)] for r in radius}
        global_metrics['RR_rr'] = {r: global_metrics['RR_rr'][r]+[next((1.0/(i+1) for i, x in enumerate(euclid_dist_rr <= r) if x), 0)] for r in radius}
      #re_rank_idx = np.argsort(self.test_relevance)
      
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
  testloader = DataLoader(test,batch_size = 1) #

  return trainloader,testloader,train_data.get_max_top_cand(),str(train_data)
# LOAD TTRAINING DATA

root = '/home/tiago/Dropbox/RAS-publication/predictions/paper/kitti/place_recognition'

train_size = 0.2
device = 'cuda:0'
#device = 'cpu'

Models = ['VLAD_pointnet', 'ORCHNet_pointnet' ,'SPoC_pointnet', 'GeM_pointnet']
#Models = ['VLAD_pointnet']
sequences = ['00','02','05','06','08']
#sequences = ['00']
dataset_type = 'AlphaQE'

loss_list =[logistic_loss,margin_ranking_loss]
for loss_obj in loss_list:
  for model_name in Models:
    for seq in sequences:
      torch.manual_seed(0)
      np.random.seed(0)
      #seq = '02'
      trainloader,testloader,max_top_cand,datasetname = load_data(root,model_name,seq,train_size,dataset_type,batch_size=100)
      #trainloader,testloader,max_top_cand  = load_cross_data(root,model_name,seq,seq)

      #===== RE-RANKING ========
      model = AttentionRanking(max_top_cand,256)

      loss_fun = loss_obj(max_top_cand)
      #loss_fun = margin_ranking_loss(max_top_cand)

      root_save = os.path.join('results',str(loss_fun),"ablation",model_name,datasetname,seq,str(train_size))
      if not os.path.isdir(root_save):
        os.makedirs(root_save)

      experiment = os.path.join(root_save,f'{str(model)}')
      rerank = AttentionTrainer(experiment=experiment,loss = loss_fun, model = model,lr= 0.001,epochs = 300,lr_step=150,val_report=1,tain_report_terminal=1,device=device,max_top_cand = max_top_cand)

      rerank.Train(trainloader,testloader)




