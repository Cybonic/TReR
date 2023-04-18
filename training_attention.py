


import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import numpy as np

from utils import retrieve_eval

from torch.utils.data import DataLoader
from dataloaders.rankingdata import RankingDataset,RankingMSE,RankingNewRetreivalDataset

import RERANKING
import loss 

from base_trainer import ReRankingTrainer

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
      return f"AttentionRanking_{self.enc_n}xEncoder_cnn"
    
    def forward(self,k):
        #k = torch.transpose(k,dim0=2,dim1=1)
        #out, attn_output_weights = self.att(k,k,k)
        #out = k + out
        #out = torch.transpose(out,dim0=2,dim1=1)
        #
        #
        #out = k
        #
       
        out = self.layer(k)
        #out = self.fc_drop(out)
        out = torch.transpose(out,dim0=2,dim1=1)
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
class rankloss():
  def __init__(self,cand=37):
        self.loss_fn = torch.nn.MarginRankingLoss(0.1)
        combo_idx = np.arange(cand)
        self.permute = torch.from_numpy(np.array([np.array([a, b]) for idx, a in enumerate(combo_idx) for b in combo_idx[idx + 1:]]))

  def __call__(self,pred,table):
        
      #pred = torch.softmax(pred,dim=-1) 
      b = torch.tensor(table.shape[0])
      n = table.shape[1]
      loss_vec = 0
      for p,batch in zip(pred,table):
        loss_value = 0  
        x1 = p[self.permute[:,0]]
        x2 = p[self.permute[:,1]]
        y = batch[self.permute[:,0],self.permute[:,1]]
        #value = self.loss_fn(x1,x2,y)
        

        value = torch.sum((y*torch.log2(1+torch.exp(-(x1-x2)))).clip(min=0))
        
        loss_vec +=value
        
      loss_ = loss_vec/b.float()
      return loss_

class MSERanking():
    def __init__(self):
        self.bce = nn.MSELoss()
    def __call__(self,x,y):
        # x = torch.round(x,decimals=0)
        #x = F.softmax(x)
        #y = F.softmax(y)

        #error = -torch.matmul(y,x.transpose())
        error = self.bce(x,y)
        return error.float()
      
  
class AttentionTrainer(ReRankingTrainer):
  def __init__(self,**args):
    super(AttentionTrainer,self).__init__(**args)
    

  def train_epoch(self,epoch,loader):
    #tain_report_terminal= 10
    self.model.train()
    loss_buffer = []
    re_rank_idx = []
    for batch,gt,t in loader:
      #batch = batch.to(self.device)
      self.optimizer.zero_grad()
      batch = batch.to(self.device)
      gt = gt.to(self.device)
      std = 0.0001*epoch
      noise = torch.randn(batch.size()) * std + 0
      #batch = batch + noise.to(self.device)
      out = self.model(batch)
      #print("\n\n")
      #a = torch.argsort(out[0],dim=-1,descending=True)
      #print(a.detach().cpu().numpy())

      #b = t[0]
      #print(b.detach().cpu().numpy())

      #print("\n\n")
      loss_value = self.loss(out,gt)
      loss_value.backward()
      self.optimizer.step()

      values = torch.argsort(out,dim=-1,descending=True)
        
      values = values.detach().cpu().numpy().astype(np.uint8)
      re_rank_idx.extend(values)
      
      loss_buffer.append(loss_value.cpu().detach().numpy().item())
    return(loss_buffer,re_rank_idx) 


  def predict(self,testloader):
      self.model.eval()
      re_rank_idx = []
      for x,_,t in testloader:
        #x = 1-x # .cuda()
        x = x.to(self.device)
        values = self.model(x)

        values = torch.argsort(values,dim=-1,descending=True)
        
        values = values.detach().cpu().numpy().astype(np.uint8)
        re_rank_idx.extend(values)

      #re_rank_idx = np.argsort(self.test_relevance)
      
      #rerank_loops = test_base_loop
      return(re_rank_idx)


def load_cross_data(root,model_name,seq_train,seq_test):
  train_data = RankingNewRetreivalDataset(root,model_name,seq_train)
  #train_data = RankingMSE(root,model_name,seq_train)
  trainloader = DataLoader(train_data,batch_size = len(train_data),shuffle=True)
  # LOAD TEST DATA
  test_data = RankingNewRetreivalDataset(root,model_name,seq_test)
  #test_data = RankingMSE(root,model_name,seq_test)
  testloader = DataLoader(test_data,batch_size = len(test_data)) #
  return trainloader,testloader,test_data.get_max_top_cand()


def load_data(root,model_name,seq,train_percentage,dataset='new',batch_size=50):
  if dataset == 'new':
    train_data = RankingNewRetreivalDataset(root,model_name,seq)
  else:
    train_data = RankingMSE(root,model_name,seq)

  train_size = int(len(train_data)*train_percentage)
  test_size = len(train_data) - train_size

  train, test =torch.utils.data.random_split(train_data,[train_size,test_size])
  #trainloader = DataLoader(train,batch_size = int(len(train)),shuffle=True)
  trainloader = DataLoader(train,batch_size = len(test),shuffle=True)
  testloader = DataLoader(test,batch_size = len(test)) #

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
dataset_type = 'new'

for j in range(1,10):
  # 'SPoC_pointnet', 'GeM_pointnet' ,
  #train_size = float(j)/10
  for model_name in Models:
    for seq in sequences:
      #torch.manual_seed(0)
      #np.random.seed(0)
      #seq = '02'
      trainloader,testloader,max_top_cand,datasetname = load_data(root,model_name,seq,train_size,dataset_type,batch_size=100)
      #trainloader,testloader,max_top_cand  = load_cross_data(root,model_name,seq,seq)

      #===== RE-RANKING ========
      model = AttentionRanking(max_top_cand,256)
      #
      #model = MaskRanking(max_top_cand,256)
      loss_fun = rankloss(max_top_cand)
      #loss_fun = MSERanking()

      #root_save = os.path.join('tests','loss_softmax',str(train_size),model_name,seq)
      # prob_rank_loss, margin_rank_loss
      root_save = os.path.join('results',"prob_rank_loss","ablation",model_name,datasetname,seq,str(train_size))
      if not os.path.isdir(root_save):
        os.makedirs(root_save)

      # experiment = os.path.join(root_save,f'{str(model)}-{str(train_size)}')
      experiment = os.path.join(root_save,f'{str(model)}')
      rerank = AttentionTrainer(experiment=experiment,loss = loss_fun, model = model,lr= 0.001,epochs = 300,lr_step=150,val_report=1,tain_report_terminal=1,device=device,max_top_cand = max_top_cand)

      rerank.Train(trainloader,testloader)




