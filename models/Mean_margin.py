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

        self.bn0 = torch.nn.BatchNorm1d(d1)
        self.bn1 = torch.nn.BatchNorm1d(d1)
        # math:`C` from an expected input of size math:`(N, C, L)` or
        # math:`L` from input of size :math:`(N, L)`

    def evaluate(self, e1, r, es):
        x = self.bn0(e1)
        x = self.input_dropout(x)
        x = x.view(-1, 1, e1.size(1))

        W_mat = torch.mm(r, self.W.view(r.size(1), -1))  # torch.mm() 矩阵相乘
        W_mat = W_mat.view(-1, e1.size(1), e1.size(1))
        W_mat = self.hidden_dropout1(W_mat)

        x = torch.bmm(x, W_mat)  # torch.mm() batch矩阵相乘
        x = x.view(-1, e1.size(1))
        x = self.bn1(x)
        x = self.hidden_dropout2(x)
        x = torch.mm(x, es.transpose(1, 0))
        pred = torch.sigmoid(x)

        return pred

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
        x_p = (x*e2p).sum(dim=1)
        x_n = (x*e2n).sum(dim=1)
        pred_p = torch.sigmoid(x_p)
        pred_n = torch.sigmoid(x_n)
        return pred_p, pred_n


class MeanTuckER(nn.Module):
    """Text Encoding Model Mean"""

    def __init__(self, d, es_idx, ent_vec_dim, rel_vec_dim, margin, Evocab=40990, Rvocab=13, **kwargs):
        super(MeanTuckER, self).__init__()
        self.Eembed = nn.Embedding(Evocab, ent_vec_dim, padding_idx=0)
        self.Rembed = nn.Embedding(Rvocab, rel_vec_dim, padding_idx=0)
        self.es_idx = es_idx
        self.tucker = TuckER(d, ent_vec_dim, rel_vec_dim, **kwargs)
        self.loss = torch.nn.MarginRankingLoss(margin=margin)


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
