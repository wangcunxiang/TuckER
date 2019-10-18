import torch
from torch import nn
from torch.nn.init import xavier_normal_
from torch.nn import LSTM
import numpy as np



class TuckER(torch.nn.Module):
    def __init__(self, d, d1, d2, **kwargs):
        super(TuckER, self).__init__()

        self.E = torch.nn.Embedding(len(d.entities), d1)#, padding_idx=0)
        #self.R = torch.nn.Embedding(len(d.relations), d2, padding_idx=0)
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

    def update_es(self, es):
        self.E.weight.data.copy_(es)
        self.E.weight.requires_grad = False

    def forward(self, e1, r):
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
        x = torch.mm(x, self.E.weight.transpose(1, 0))
        pred = torch.sigmoid(x)
        return pred

class LSTMTuckER(nn.Module):
    """Text Encoding Model LSTM"""

    def __init__(self, d, ent_vec_dim, rel_vec_dim, cfg, vocab=40990, **kwargs):
        super(LSTMTuckER, self).__init__()
        self.embed = nn.Embedding(vocab, cfg.hSize, padding_idx=0)
        self.tucker = TuckER(d, ent_vec_dim, rel_vec_dim, **kwargs)
        self.elstm = LSTM(cfg.hSize, int(ent_vec_dim/2), num_layers=2, batch_first=True, dropout=0.2, bidirectional=True)
        #batch_first: If ``True``, then the input and output tensors are provided as (batch, seq, feature). Default: ``False``
        self.rlstm = LSTM(cfg.hSize, int(rel_vec_dim/2), num_layers=2, batch_first=True, dropout=0.2, bidirectional=True)
        self.loss = torch.nn.BCELoss()

    def cal_es(self, es):
        es = self.embed(es)
        #print("es size:"+str(es.size()))
        #print("es[0] size:" + str(es[0].size()))
        #print("torch.unsqueeze(es[0], 0):"+str(torch.unsqueeze(es[0], 0).size()))
        es_encoded, tmp = self.elstm(torch.unsqueeze(es[0], 0))
        es_encoded = es_encoded[:, -1, :]
        for i in range(1, es.size(0)):
            i_tmp, tmp = self.elstm(torch.unsqueeze(es[i],0))
            i_tmp = i_tmp[:, -1, :]
            es_encoded = torch.cat((es_encoded, i_tmp),0)
        #print("es_encoded size:"+str(es_encoded.size()))

        self.tucker.update_es(es_encoded)


    def forward(self, e, r):
        #print('e.szie:' + str(e.size()))
        e = e.view(-1, e.size(-1))
        #print('e.szie:' + str(e.size()))
        e = self.embed(e)
        #print('e.szie:'+str(e.size()))
        e_encoded, tmp = self.elstm(e)
        e_encoded = e_encoded[:, -1,:]  # use last word's output
        #print('e_encoded.szie:' + str(e_encoded.size()))
        #print('e_encoded:'+str(e_encoded))

        r = r.view(-1, r.size(-1))
        r = self.embed(r)
        r_encoded, tmp = self.rlstm(r)
        r_encoded = r_encoded[:,-1,:]#use last word's output

        return self.tucker(e_encoded, r_encoded)



