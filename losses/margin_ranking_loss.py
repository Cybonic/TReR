
import torch
from .loss_utils import pairwise_permutations

class margin_ranking_loss:
  def __init__(self,cand=37):
        self.loss_fn = torch.nn.MarginRankingLoss(0.1)
        self.x1_perm,self.x2_perm = pairwise_permutations(cand)

  def __call__(self,y_pred,y_true):
    x1 = y_pred[:,self.x1_perm]
    x2 = y_pred[:,self.x2_perm]
    y  = y_true[:,self.x1_perm,self.x2_perm]
    value = torch.sum((y*torch.log2(1+torch.exp(-(x1-x2)))).clip(min=0),dim=-1)
    return torch.mean(value)

  def __str__(self):
     return 'margin_ranking_loss'