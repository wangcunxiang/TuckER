import torch
from torch import nn
from torch.nn.init import xavier_normal_
from torch.nn import LSTM
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


class MeanTuckER(nn.Module):
    """Text Encoding Model Mean"""

    def __init__(self, d, es_idx, ent_lens, rel_lens, ent_vec_dim, rel_vec_dim, Evocab=40990, Rvocab=13, **kwargs):
        super(MeanTuckER, self).__init__()
        self.Eembed = nn.Embedding(Evocab, ent_vec_dim, padding_idx=0)
        self.Rembed = nn.Embedding(Rvocab, rel_vec_dim, padding_idx=0)
        self.es_idx = es_idx
        self.Evocab = Evocab
        self.ent_lens = ent_lens
        self.rel_lens = rel_lens
        self.tucker = TuckER(d, ent_vec_dim, rel_vec_dim, **kwargs)
        self.loss = torch.nn.BCELoss()

    def cal_es(self):

        es = self.Eembed(self.es_idx)
        es_encoded = self.mean_(es)

        return es_encoded


    def mean_(self, tensor):
        # #print('tensor[:, :, 0] != 0: '+str(tensor[:, :, 0] != 0))
        # lens = torch.sum((tensor[:, :, 0] != 0).float(), dim=1)
        # assert torch.nonzero(lens).size(0) == lens.size(0)
        # lens = lens.unsqueeze(1)
        # tensor_  = torch.sum(tensor, dim=1)
        # #print('tensor_ =' + str(tensor_))
        # tensor__ = torch.div(tensor_, lens)
        return torch.mean(tensor, dim=1)

    def forward(self, e, r):
        # #print('e.szie:' + str(e.size()))
        #         # e = e.view(-1, e.size(-1))
        #         # #print('e.szie:' + str(e.size()))
        #         # e = self.Eembed(e)
        #         # #print('e.szie:'+str(e.size()))
        #         # e_encoded, tmp = self.elstm(e)
        #         # e_encoded = e_encoded[:, -1,:]  # use last word's output
        #         # #print('e_encoded.szie:' + str(e_encoded.size()))
        #         # #print('e_encoded:'+str(e_encoded))
        #         #
        #         # r = r.view(-1, r.size(-1))
        #         # r = self.Rembed(r)
        #         # r_encoded, tmp = self.rlstm(r)
        #         # r_encoded = r_encoded[:,-1,:]#use last word's output
        #print('e_lens = '+str(e_lens))
        e = self.Eembed(e)
        r = self.Rembed(r)

        e_encoded = self.mean_(e)
        r_encoded = self.mean_(r)

        #print('e_encoded.szie:' + str(e_encoded.size()))

        return self.tucker(e_encoded, r_encoded, self.cal_es())



