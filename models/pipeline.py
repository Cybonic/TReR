
import torch
import torch.nn as nn
from .utils import *
from .TranformerEncoder import *
from .AttentionRanking import *

def tranformerencoder_max(**argv):
    #if argv['use_fc']:
    #   return  FC_Drop(Max(TranformerEncoder(**argv))**argv)
    return Max(TranformerEncoder(**argv))

def tranformerencoder_wout(**argv):
    return Wout(TranformerEncoder(**argv))

def tranformerencoder_cnn(**argv):
    return Cnn(TranformerEncoder(**argv))


def attention_max(**argv):
    return Max(AttentionRanking(**argv))

def attention_wout(**argv):
    return Wout(AttentionRanking(**argv))

def attention_cnn(**argv):
    return Cnn(AttentionRanking(**argv))