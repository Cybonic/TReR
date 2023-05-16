


import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import numpy as np


from torch.utils.data import DataLoader
from dataloaders.alphaqedata import AlphaQEData,CROSS
from dataloaders.logg3d import LOGGNet3D_CROSS

from base_trainer import ReRankingTrainer
from time import time

from losses.logistic_loss import logistic_loss



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
      gt = pos['table'].to(self.device)
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
    
      global_metrics = {'t_RR':[]}

      local_metrics_vec = []
      n_samples = len(testloader.dataset)
      re_rank_idx = []
      for i,(emb,scores,pos) in enumerate(testloader):

        x = emb['map'].to(self.device)
        tick = time()
        values = self.model(x)
        rr_nn_ndx = torch.argsort(values,dim=-1,descending=True)
        t_rerank = time() - tick

        rr_nn_ndx_vec = rr_nn_ndx.detach().cpu().numpy().astype(np.uint8)
        re_rank_idx.append(rr_nn_ndx_vec)

        global_metrics['t_RR'].append(t_rerank)
      
      return(re_rank_idx,global_metrics)

def load_cross_data(root,model_name,seq_train,batch_size=None,**argv):
  if model_name == 'LoGGNet3D':
    loader = LOGGNet3D_CROSS(root,model_name,seq_train)
  else:
    loader = CROSS(root,model_name,seq_train)

  testloader  = loader.get_test_loader()
  trainloader = loader.get_train_loader(batch_size)
  max_cand = 25
  return trainloader,testloader,max_cand,str(loader)


def load_data(root,model_name,seq,train_percentage,dataset='new',batch_size=50):
  # Dataset
  train_data = AlphaQEData(root,model_name,seq)
  # Compute split sizes
  train_size = int(len(train_data)*train_percentage)
  test_size = len(train_data) - train_size
  # Split of the train and test sets
  train, test =torch.utils.data.random_split(train_data,[train_size,test_size])
  # Dataloaders
  trainloader = DataLoader(train,batch_size = len(train),shuffle=True)
  testloader = DataLoader(test,batch_size =  len(test)) #
  return trainloader,testloader,train_data.get_max_top_cand(),str(train_data)
# LOAD TTRAINING DATA


def main(root,model_name,seq,model_fun,loss_fun,**argv):
    torch.manual_seed(0)
    np.random.seed(0)
    
    head = argv['head']
    mha_i = argv['mha_i']
    mlp_h = argv['mlp_h']
    batch_size = argv['batch_size']
  # root_data = os.path.join(root, dataset)
    trainloader,testloader,max_top_cand,datasetname = load_cross_data(root,model_name,seq,batch_size=batch_size)
    #===== RE-RANKING ========

    model = pipeline.__dict__[model_fun](cand=25,feat_size=256,enc_n=head,mha=mha_i,mlp_features = mlp_h)
    # model_fun = model_obj(max_top_cand,256)
    loss = loss_fun(25)  # loss_fun = loss_obj(max_top_cand) 
    root_save = os.path.join('results',"cross_seq10k_mlp",str(mlp_h),str(batch_size),model_name,dataset,seq)
    #root_save = os.path.join('results',"paperv2",'final',str(loss_fun),model_name,datasetname,seq,str(train_size))
    if not os.path.isdir(root_save):
      os.makedirs(root_save)

    experiment = os.path.join(root_save,f'{str(model)}')
    rerank = AttentionTrainer(experiment=experiment,loss = loss, model = model,
                              lr= 0.001,epochs = 300,lr_step=250,val_report=1,
                              tain_report_terminal=1,device=device,max_top_cand = 25)


    ## 
    count = 0
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    print(f'\n# Model parameters: {params}\n')
    print(params)
      #count= count + len(parameter)
    
      
    rerank.Train(trainloader,testloader)

if __name__=='__main__':
  
  #root = "/home/deep/Dropbox/SHARE/deepisrpc/reranking/LoGG3D-Net/evaluation/10000pts/"
  root = '/home/tiago/Dropbox/SHARE/deepisrpc/reranking/LoGG3D-Net/evaluation/10000pts/'

  dataset = 'kitti'

  train_size = 0.2
  #device = 'cuda:0'
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

  Models = ['LoGGNet3D']#,'ORCHNet_pointnet','VLAD_pointnet','SPoC_pointnet', 'GeM_pointnet']
  #Models = ['VLAD_pointnet']
  sequences = ['00','02','05','06','08']
  #sequences = ['00']
  model_list = ['tranformerencoder_max_fc_drop','tranformerencoder_wout_fc_drop','tranformerencoder_cnn_fc_drop']
  #              'attention_max_fc_drop','attention_wout_fc_drop','attention_cnn_fc_drop']
  model_list = ['tranformerencoder_max_fc_drop']#'tranformerencoder_max_fc_drop']#,'attention_max_fc_drop'] 'tranformerencoder_mlpd']#
  
  #model_list = ['attention_max_fc_drop']
  #model_list = ['max_fc']
  loss_list  = [logistic_loss]#,margin_ranking_loss]
  #for train_size in range(1,9):
  batch_sizes = [1000,500,100,50,20,10]
  batch_sizes = [100]
  for model_obj in model_list:
    for loss_obj in loss_list:
      for model_name in Models:
        for batch_size in batch_sizes:
          for seq in sequences:
            for mha_i in [1]:#,4,8]:
              for head in [1]:
                
                
                mlp_h=512
                #root_data = root
                main(root,model_name,seq,
                      model_obj,loss_obj,
                      batch_size=batch_size, 
                      head=head,mlp_h=mlp_h,mha_i=mha_i)




