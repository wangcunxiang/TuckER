import torch
from torch import nn
from torch.nn.init import xavier_normal_
from torch.nn import Conv1d, ReLU, MaxPool1d, Sequential
import numpy as np


class TuckER(torch.nn.Module):
    def __init__(self, d, d1, d2, **kwargs):
        super(TuckER, self).__init__()

        # self.E = torch.nn.Embedding(len(d.entities), d1)  # , padding_idx=0)
        # self.R = torch.nn.Embedding(len(d.relations), d2, padding_idx=0)
        self.W = torch.nn.Parameter(torch.tensor(np.random.uniform(-1, 1, (d2, d1, d1)),
                                                 dtype=torch.float, requires_grad=True)).cuda()

        self.input_dropout = torch.nn.Dropout(kwargs["input_dropout"])
        self.hidden_dropout1 = torch.nn.Dropout(kwargs["hidden_dropout1"])
        self.hidden_dropout2 = torch.nn.Dropout(kwargs["hidden_dropout2"])
        #self.loss = torch.nn.BCELoss()

        self.bn0 = torch.nn.BatchNorm1d(d1)
        self.bn1 = torch.nn.BatchNorm1d(d1)
        # math:`C` from an expected input of size math:`(N, C, L)` or
        # math:`L` from input of size :math:`(N, L)`

    def init(self):
        xavier_normal_(self.E.weight.data)
        xavier_normal_(self.R.weight.data)

    def update_es(self, es):
        self.E.weight.data.copy_(es)
        self.E.weight.requires_grad = False

    def forward(self, e1, r, es):
        #print("e1 size:"+str(e1.size()))
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
        # x = torch.mm(x, self.E.weight.transpose(1, 0))
        # print('self.E.weight='+str(self.E.weight.size()))
        x = torch.mm(x, es.transpose(1, 0))
        pred = torch.sigmoid(x)
        return pred


class CNNTuckER(nn.Module):
    """Text Encoding Model CNN"""

    def __init__(self, d, es_idx, ent_vec_dim, rel_vec_dim, cfg, max_length, window_size, Evocab=40990, Rvocab=13, **kwargs):
        super(CNNTuckER, self).__init__()
        self.Eembed = nn.Embedding(Evocab, cfg.hSize, padding_idx=0)
        self.Rembed = nn.Embedding(Rvocab, cfg.hSize, padding_idx=0)
        self.es_idx = es_idx
        self.Evocab = Evocab
        self.ecnn = Sequential(Conv1d(in_channels=cfg.hSize, out_channels=ent_vec_dim, kernel_size=window_size),
                             MaxPool1d(kernel_size=max_length-window_size+1))
        self.rcnn = Sequential(Conv1d(in_channels=cfg.hSize, out_channels=ent_vec_dim, kernel_size=window_size),
                             MaxPool1d(kernel_size=max_length-window_size+1))

        self.tucker = TuckER(d, ent_vec_dim, rel_vec_dim, **kwargs)
        self.loss = torch.nn.BCELoss()

    def cal_es(self):

        es = self.Eembed(self.es_idx)
        es = es.permute(0, 2, 1)
        es_encoded = self.ecnn(es)
        es_encoded = es_encoded.reshape(-1, es_encoded.size(1))

        return es_encoded



    def forward(self, e, r):
        e = self.Eembed(e)
        r = self.Rembed(r)

        e = e.permute(0, 2, 1)
        r = r.permute(0, 2, 1)
        #print('e size = ' + str(e.size()))
        e_encoded = self.ecnn(e)
        r_encoded = self.rcnn(r)

        #print('e_encoded size = ' + str(e_encoded.size()))
        e_encoded = e_encoded.reshape(-1, e_encoded.size(1))
        #print('e_encoded size = '+str(e_encoded.size()))
        r_encoded = r_encoded.reshape(-1, r_encoded.size(1))


        return self.tucker(e_encoded, r_encoded, self.cal_es())



