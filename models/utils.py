
import torch
import torch.nn as nn

class Wout(torch.nn.Module):
  def __init__(self,model=None,**argv):
    super().__init__()
    if model == None:
      self.model = ''
    else:
      self.model = model

    self.Wout =  nn.Parameter(torch.zeros(256,1))
    nn.init.normal_(self.Wout.data, mean=0, std=0.1)
    #self.model = model
    
  def __str__(self):
    return f"{str(self.model)}_wout"
  
  def forward(self,k):
    if self.model == '':
        out = k
    else:
      out = self.model(k) # outputs B,C,F
    out = torch.matmul(out,self.Wout).squeeze() # -> B,C
    return out.float()
  

class Max(torch.nn.Module):
  def __init__(self,model,**argv):
    super().__init__()
    self.model = model
    
  def __str__(self):
      return f"{str(self.model)}_max"
  
  def forward(self,k):
        out = self.model(k) # outputs B,C,F
        out,idx = torch.max(out,dim=-1) # -> B,C
        return out.float()

class Cnn(torch.nn.Module):
  def __init__(self,model=None,**argv):
    super().__init__()
    if model == None:
      self.model = ''
    else:
      self.model = model

    self.classifier = torch.nn.Conv1d(256, 1, 1)

  def __str__(self):
      return f"{str(self.model)}_cnn"
  
  def forward(self,k):
        if self.model == '':
           out = k
        else:
          out = self.model(k) # outputs B,C,F
        out = torch.transpose(out,dim0=2,dim1=1) # -> B,F,C
        out = self.classifier(out).squeeze() # -> B,C
        return out.float()
  

class FC_Drop(torch.nn.Module):
  def __init__(self,model,cand,**argv):
    super().__init__() 

    self.model = model
    fc_drop = [nn.LazyLinear(cand),
                    nn.ReLU(),
                    nn.Dropout(0.1)
                    ]
      
    self.fc_drop = nn.Sequential(*fc_drop)
  
  def __str__(self):
    return f"{str(self.model)}_fc_drop"
  
  def forward(self,k):
    out = self.model(k) # outputs B,C
    return self.fc_drop(out) # B,C -> B,C