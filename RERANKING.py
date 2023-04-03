import torch 
from torch import nn
from torch.nn import functional as F



class FC(nn.Module):
  def __init__(self,outdim):
    super(FC,self).__init__()
    outdim_list = [outdim,int(outdim/2),outdim]
    fc = [   nn.LazyLinear(outdim),
             nn.BatchNorm1d(outdim, momentum=0.01),
             #nn.LazyLinear(outdim),
             nn.ReLU()]
    
    self.fc = nn.Sequential(*[nn.LazyLinear(outdim_list[i]).float() for i in range(3)])
    #self.fc = nn.Sequential(*fc)
    
  def forward(self,x):
    x =  F.softmin(x,dim=1)
    return self.fc(x)
  

class Attention(nn.Module):
  def __init__(self,indim,n_head=1):
    super(Attention,self).__init__()
    self.net = nn.MultiheadAttention(indim, n_head)

  def forward(self,x):
    x = x.unsqueeze(1)
    x, attn_output_weights = self.net(x,x,x)
    return x



class MLPNet(torch.nn.Module):
    """ Multi-layer perception.
        [B, Cin, N] -> [B, Cout, N] or
        [B, Cin] -> [B, Cout]
    """
    def __init__(self, nch_input, nch_layers=[25], b_shared=True, bn_momentum=0.1, dropout=0.0):
        super().__init__()
        
        list_layers = mlp_layers(nch_input, nch_layers, b_shared, bn_momentum, dropout)
        self.layers = torch.nn.Sequential(*list_layers)
        self.classifier = nn.Softmax(dim=1)
        self.classifier  = torch.nn.Conv1d(5, 25, 1)

    def forward(self, inp):

        inp = inp.unsqueeze(1)
        #inp = 1-inp/inp.max()
        inp[inp <0.001] = 0.00001
        inp = torch.bmm(torch.transpose(inp,1,2),inp)
        
        out = self.layers(inp)
        #out = self.classifier(out)
        #out = out.detach().numpy()
        #out = torch.argmax(out,dim=1).float()
        return out.float()
    
def mlp_layers(nch_input, nch_layers, b_shared=True, bn_momentum=0.001, dropout=0.0):
    """ [B, Cin, N] -> [B, Cout, N] or
        [B, Cin] -> [B, Cout]
    """
    layers = []
    last = nch_input
    for i, outp in enumerate(nch_layers):
        if b_shared:
            weights = torch.nn.Conv1d(last, outp, 1)
        else:
            weights = torch.nn.Linear(last, outp)
        layers.append(weights)
        layers.append(torch.nn.BatchNorm1d(outp, momentum=bn_momentum))
        layers.append(torch.nn.ReLU())
        if b_shared == False and dropout > 0.0:
            layers.append(torch.nn.Dropout(dropout))
        last = outp
    return layers


class ML(torch.nn.Module):
    """ Multi-layer perception.
        [B, Cin, N] -> [B, Cout, N] or
        [B, Cin] -> [B, Cout]
    """
    def __init__(self,cand=35,feat_size = 256):
        super().__init__()
        
        self.Win =  nn.Parameter(torch.zeros(256,25))
        self.Wout =  nn.Parameter(torch.zeros(25,1))
        
        nn.init.normal_(self.Win.data, mean=0, std=0.1)
        nn.init.normal_(self.Wout.data, mean=0, std=0.1)

        #self.layers = torch.nn.Sequential(*list_layers)
        #self.classifier = nn.Softmax(dim=-1)
        layers  = []
        layers.append(torch.nn.Conv1d(25, 25, 1))
        #layers.append(weights)
        #layers.append(torch.nn.BatchNorm1d(25, momentum=0.1))
        layers.append(torch.nn.ReLU())
        self.classifier = torch.nn.Sequential(*layers)
        self.att = torch.nn.MultiheadAttention(256,1)


    def forward(self, q,k):
        #x1 =  F.softmin(x,dim=1)
        #print(' '.join([str(a) for a in x[0].detach().cpu().numpy()]))
        #print(x.detach().cpu().numpy())
        #inp = inp.unsqueeze(1)
        #inp = inp/inp.max()
        #inp[inp <0.001] = 0.00001
        #if self.training():
        #q = x['q']
        #k = x['k']
        #v = x['k']

        #out1 = torch.matmul(,self.Win)

        out = self.att(q,k,k)
        #out=self.classifier(out1)
        #out = out + out1

        if self.training:
           return out.float()
        
        outmax = torch.argmax(out,dim=-1)
        
        print(out[0][0].detach().cpu().numpy())
        print(outmax[0][0].detach().cpu().numpy())
        #out = torch.matmul(inp,self.Wout)
        
        #out = self.layers(inp)
        #out = self.classifier(out)
        #out = out.detach().numpy()
        #out = torch.argmax(out,dim=1).float()
        return outmax.float()
