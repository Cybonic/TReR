
import torch
import torch.nn as nn
from .utils import *
from .TranformerEncoder import *
from .AttentionRanking import *

def tranformerencoder_max_fc_drop(**argv):
    return FC_Drop(tranformerencoder_max(**argv),**argv)

def tranformerencoder_wout_fc_drop(**argv):
    return FC_Drop(tranformerencoder_wout(**argv),**argv)

def tranformerencoder_cnn_fc_drop(**argv):
    return FC_Drop(tranformerencoder_cnn(**argv),**argv)

def attention_max_fc_drop(**argv):
    return FC_Drop(attention_max(**argv),**argv)

def attention_wout_fc_drop(**argv):
    return FC_Drop(attention_wout(**argv),**argv)

def attention_cnn_fc_drop(**argv):
    return FC_Drop(attention_cnn(**argv),**argv)

# =====================================================

def tranformerencoder_max(**argv):
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

def cnn(**argv):
    return Cnn(**argv)

def wout(**argv):
    return Wout(**argv)

def cnn_fc(**argv):
    return FC_Drop(Cnn(**argv),**argv)

def wout_fc(**argv):
    return FC_Drop(Wout(**argv),**argv)