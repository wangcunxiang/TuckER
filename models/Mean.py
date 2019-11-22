import torch
from torch import nn
from torch.nn.init import xavier_normal_
from torch.nn import LSTM
import numpy as np
from .Tucker_pn import TuckER

class MeanTuckER(nn.Module):
    """Text Encoding Model Mean"""

    def __init__(self, d, es_idx, ent_vec_dim, rel_vec_dim, cfg, Evocab=40990, Rvocab=13, **kwargs):
        super(MeanTuckER, self).__init__()
        self.Eembed = nn.Embedding(Evocab, ent_vec_dim, padding_idx=0)
        self.Rembed = nn.Embedding(Rvocab, rel_vec_dim, padding_idx=0)
        self.es_idx = es_idx
        self.tucker = TuckER(d, ent_vec_dim, rel_vec_dim, cfg)
        self.loss = torch.nn.BCELoss()


    def evaluate(self, e1, r):
        e1 = self.Eembed(e1)
        r = self.Rembed(r)
        es = self.Eembed(self.es_idx)

        e1_encoded = self.mean_(e1)
        r_encoded = self.mean_(r)
        es_encoded = self.mean_(es)

        return self.tucker.evaluate(e1_encoded, r_encoded, es_encoded)


    def mean_(self, tensor):
        # lens = torch.sum((tensor[:, :, 0] != 0).float(), dim=1)
        # lens = lens.unsqueeze(1)
        # tensor_  = torch.sum(tensor, dim=1)
        # tensor__ = torch.div(tensor_, lens)
        # return tensor__
        return torch.mean(tensor, dim=1)

    def forward(self, e1, r, e2p, e2n):

        e1 = self.Eembed(e1)
        r = self.Rembed(r)
        e2p = self.Eembed(e2p)
        e2n = self.Eembed(e2n)

        e1_encoded = self.mean_(e1)
        r_encoded = self.mean_(r)
        e2p_encoded = self.mean_(e2p)
        e2n_encoded = self.mean_(e2n)

        #print('e_encoded.szie:' + str(e1_encoded.size()))

        return self.tucker(e1_encoded, r_encoded, e2p_encoded, e2n_encoded)
