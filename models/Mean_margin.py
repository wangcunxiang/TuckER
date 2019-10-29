import torch
from torch import nn
from torch.nn.init import xavier_normal_
import numpy as np


class TuckER(torch.nn.Module):
    def __init__(self, d, d1, d2, **kwargs):
        super(TuckER, self).__init__()

        self.E = torch.nn.Embedding(len(d.entities), d1)  # , padding_idx=0)
        # self.R = torch.nn.Embedding(len(d.relations), d2, padding_idx=0)
        self.W = torch.nn.Parameter(torch.tensor(np.random.uniform(-1, 1, (d2, d1, d1)),
                                                 dtype=torch.float, device="cuda", requires_grad=True))

        self.input_dropout = torch.nn.Dropout(kwargs["input_dropout"])
        self.hidden_dropout1 = torch.nn.Dropout(kwargs["hidden_dropout1"])
        self.hidden_dropout2 = torch.nn.Dropout(kwargs["hidden_dropout2"])
        self.loss = torch.nn.MarginRankingLoss()

        self.bn0 = torch.nn.BatchNorm1d(d1)
        self.bn1 = torch.nn.BatchNorm1d(d1)
        # math:`C` from an expected input of size math:`(N, C, L)` or
        # math:`L` from input of size :math:`(N, L)`

    def init(self):
        xavier_normal_(self.E.weight.data)
        xavier_normal_(self.R.weight.data)

    def forward(self, e1, r, e2p, e2n):
        x = self.bn0(e1)
        x = self.input_dropout(x)
        x = x.view(-1, 1, e1.size(1))

        W_mat = torch.mm(r, self.W.view(r.size(1), -1))
        W_mat = W_mat.view(-1, e1.size(1), e1.size(1))
        W_mat = self.hidden_dropout1(W_mat)

        x = torch.bmm(x, W_mat)
        x = x.view(-1, e1.size(1))
        x = self.bn1(x)
        x = self.hidden_dropout2(x)
        x_p = torch.mm(x, e2p)
        x_n = torch.mm(x, e2n)
        pred_p = torch.sigmoid(x_p)
        pred_n = torch.sigmoid(x_n)
        return pred_p, pred_n


class MeanTuckER(nn.Module):
    """Text Encoding Model Mean"""

    def __init__(self, d, es_idx, ent_vec_dim, rel_vec_dim, Evocab=40990, Rvocab=13, **kwargs):
        super(MeanTuckER, self).__init__()
        self.Eembed = nn.Embedding(Evocab, ent_vec_dim, padding_idx=0)
        self.Rembed = nn.Embedding(Rvocab, rel_vec_dim, padding_idx=0)
        self.es_idx = es_idx
        self.tucker = TuckER(d, ent_vec_dim, rel_vec_dim, **kwargs)
        self.loss = torch.nn.BCELoss()

    def cal_es(self):

        es = self.Eembed(self.es_idx)
        es_encoded = self.mean_(es)

        return es_encoded


    def mean_(self, tensor):
        lens = torch.sum((tensor[:, :, 0] != 0).float(), dim=1)
        lens = lens.unsqueeze(1)
        tensor_  = torch.sum(tensor, dim=1)
        tensor__ = torch.div(tensor_, lens)
        return tensor__

    def forward(self, e1, r, e2p, e2n):

        e1 = self.Eembed(e1)
        r = self.Rembed(r)
        e2p = self.Eembed(e2p)
        e2n = self.Eembed(e2n)

        e1_encoded = self.mean_(e1)
        r_encoded = self.mean_(r)
        e2p_encoded = self.mean_(e2p)
        e2n_encoded = self.mean_(e2n)

        print('e_encoded.szie:' + str(e1_encoded.size()))

        return self.tucker(e1_encoded, r_encoded, e2p_encoded, e2n_encoded)



