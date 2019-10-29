import torch
from torch import nn
from torch.nn.init import xavier_normal_
from torch.nn import LSTM
import numpy as np



class TuckER(torch.nn.Module):
    def __init__(self, d, d1, d2, **kwargs):
        super(TuckER, self).__init__()

        self.E = torch.nn.Embedding(len(d.entities), d1)#, padding_idx=0)
        self.R = torch.nn.Embedding(len(d.relations), d2, padding_idx=0)
        self.W = torch.nn.Parameter(torch.tensor(np.random.uniform(-1, 1, (d2, d1, d1)),
                                                 dtype=torch.float, device="cuda", requires_grad=True))

        self.input_dropout = torch.nn.Dropout(kwargs["input_dropout"])
        self.hidden_dropout1 = torch.nn.Dropout(kwargs["hidden_dropout1"])
        self.hidden_dropout2 = torch.nn.Dropout(kwargs["hidden_dropout2"])
        self.loss = torch.nn.BCELoss()

        #print("d1="+str(d1))
        self.bn0 = torch.nn.BatchNorm1d(d1)
        self.bn1 = torch.nn.BatchNorm1d(d1)
        #math:`C` from an expected input of size math:`(N, C, L)` or
        #math:`L` from input of size :math:`(N, L)`

    def init(self):
         xavier_normal_(self.E.weight.data)
         xavier_normal_(self.R.weight.data)

    def evaluate(self, e1, r, es):
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

        x = torch.mm(x, es.transpose(1, 0))
        pred = torch.sigmoid(x)
        return pred

    def forward(self, e1, r, e2p, e2n):
        # e1 = self.E(e1)
        # r = self.R(r)
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
        x_p = (x * e2p).sum(dim=1)
        x_n = (x * e2n).sum(dim=1)
        pred_p = torch.sigmoid(x_p)
        pred_n = torch.sigmoid(x_n)
        return pred_p, pred_n

class LSTMTuckER(nn.Module):
    """Text Encoding Model LSTM"""

    def __init__(self, d, ent_vec_dim, rel_vec_dim, cfg, Evocab=40990, Rvocab=13,  **kwargs):
        super(LSTMTuckER, self).__init__()
        self.Eembed = nn.Embedding(Evocab, cfg.hSize, padding_idx=0)
        self.Rembed = nn.Embedding(Rvocab, cfg.hSize, padding_idx=0)
        self.tucker = TuckER(d, ent_vec_dim, rel_vec_dim, **kwargs)
        self.elstm = LSTM(cfg.hSize, int(ent_vec_dim/2), num_layers=1, batch_first=True, dropout=0., bidirectional=True)
        #batch_first: If ``True``, then the input and output tensors are provided as (batch, seq, feature). Default: ``False``
        self.rlstm = LSTM(cfg.hSize, int(rel_vec_dim/2), num_layers=1, batch_first=True, dropout=0., bidirectional=True)
        self.loss = torch.nn.BCELoss()

    def evaluate(self, e, r, es):
        e = self.Eembed(e)
        e_encoded, tmp = self.elstm(e)
        e_encoded = e_encoded[:, -1, :]  # use last word's output

        r = self.Rembed(r)
        r_encoded, tmp = self.rlstm(r)
        r_encoded = r_encoded[:, -1, :]  # use last word's output

        es = self.Eembed(es)
        es_encoded, tmp = self.elstm(es)
        es_encoded = es_encoded[:, -1, :]  # use last word's output
        # print('e_encoded size:'+str(e_encoded.size()))

        return self.tucker.evaluate(e_encoded, r_encoded, es_encoded)

    def forward(self, e, r, e2p, e2n):

        e = self.Eembed(e)
        e_encoded, tmp = self.elstm(e)
        e_encoded = e_encoded[:, -1,:]  # use last word's output

        e2p = self.Eembed(e2p)
        e2p_encoded, tmp = self.elstm(e2p)
        e2p_encoded = e2p_encoded[:, -1, :]  # use last word's output

        e2n = self.Eembed(e2n)
        e2n_encoded, tmp = self.elstm(e2n)
        e2n_encoded = e2n_encoded[:, -1, :]  # use last word's output


        r = self.Rembed(r)
        r_encoded, tmp = self.rlstm(r)
        r_encoded = r_encoded[:,-1,:]#use last word's output

        #print('e_encoded size:'+str(e_encoded.size()))

        return self.tucker(e_encoded, r_encoded, e2p_encoded, e2n_encoded)
        #return self.tucker(e, r)





