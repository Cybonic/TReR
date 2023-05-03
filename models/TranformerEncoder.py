import torch
import torch.nn as nn
from .MHA import MultiHeadAttention
# Part of this code was taken from https://github.com/Atcold/pytorch-Deep-Learning/blob/master/15-transformer.ipynb

nn_Softargmax = nn.Softmax  # fix wrong name

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
        #self.mha = MultiHeadAttention(d_model, num_heads, p=0.1)
        self.mha = torch.nn.MultiheadAttention(d_model, num_heads,p,batch_first=True)
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


class TranformerEncoder(torch.nn.Module):
  def __init__(self,cand=35,feat_size = 256):
    super().__init__()
    self.heads = 1
    self.classifier = torch.nn.Conv1d(256, 1, 1)
    layer = [EncoderLayer(feat_size,self.heads,cand)]
    self.enc_n = 1
    for i in range(self.enc_n):
      layer += layer

    self.layer = nn.Sequential(*layer)

  def __str__(self):
      return f"TranformerEncoder_BFT_x{self.enc_n}_headx{self.heads}" # -> Batch first = True
  
  def forward(self,k):
        out = self.layer(k)
        return out.float()
