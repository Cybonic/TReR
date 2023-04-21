
import torch
import torch.nn as nn



class AttentionRanking(torch.nn.Module):
    def __init__(self,cand=35,feat_size = 256):
        super().__init__()
        
        self.Win =  nn.Parameter(torch.zeros(256,cand))
        self.Wout =  nn.Parameter(torch.zeros(256,1))
        
        nn.init.normal_(self.Win.data, mean=0, std=0.1)
        nn.init.normal_(self.Wout.data, mean=0, std=0.1)

        self.att = torch.nn.MultiheadAttention(feat_size,1)
        self.classifier = torch.nn.Conv1d(256, 1, 1)
        self.fc = nn.Linear(cand,cand)
        
        fc_drop = [nn.LazyLinear(cand),
             nn.ReLU(),
             nn.Dropout(0.1)
             ]
      
        self.fc_drop = nn.Sequential(*fc_drop)

    def __str__(self):
      return f"AttentionRanking_FeatAtt-Max-FC-Drop"
    
    def forward(self,k):
        #k = torch.transpose(k,dim0=2,dim1=1)
        out, attn_output_weights = self.att(k,k,k)
        out = k + out
        #out = torch.transpose(out,dim0=2,dim1=1)
        #
        #
        #out = k
        #
        out,idx  = torch.max(out,dim=-1)
        #out = self.layer(k)
        out = self.fc_drop(out)
        #out = torch.transpose(out,dim0=2,dim1=1)
        #out = self.classifier(out).squeeze()
        #  
        #out = torch.matmul(out,self.Wout).squeeze()
        #out,idx  = torch.max(out,dim=-1)
        #if self.training:
        #   return out.float()
        return out.float()