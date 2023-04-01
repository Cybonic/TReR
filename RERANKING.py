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
    
    #self.fc = nn.Sequential(*[nn.LazyLinear(outdim_list[i]).float() for i in range(3)])
    self.fc = nn.Sequential(*fc)
    
  def forward(self,x):
    #return F.softmax(self.fc(x),dim=1)
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
    def __init__(self, nch_input, nch_layers=[25], b_shared=True, bn_momentum=0.1, dropout=0.0):
        super().__init__()
        
        self.param =  nn.Parameter(torch.zeros(25,25))
        nn.init.normal_(self.param.data, mean=0, std=0.1)
        #self.layers = torch.nn.Sequential(*list_layers)
        #self.classifier = nn.Softmax(dim=1)
        #self.classifier  = torch.nn.Conv1d(25, 25, 1)

    def forward(self, inp):

        inp = inp.unsqueeze(1)
        inp = inp/inp.max()
        inp[inp <0.001] = 0.00001
        inp = torch.matmul(inp,self.param)
        
        #out = self.layers(inp)
        #out = self.classifier(out)
        #out = out.detach().numpy()
        #out = torch.argmax(out,dim=1).float()
        return inp.float()
