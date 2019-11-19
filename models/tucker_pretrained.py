import numpy as np
import torch
from torch.nn import Linear
from torch.nn.init import xavier_normal_



class TuckER(torch.nn.Module):
    def __init__(self, d, d1, d2, **kwargs):
        super(TuckER, self).__init__()

        self.E = torch.nn.Embedding(len(d.entities), 768, padding_idx=0)
        self.E2E = Linear(768, d1)
        self.R = torch.nn.Embedding(len(d.relations), 768, padding_idx=0)
        self.R2R = Linear(768, d2)
        self.W = torch.nn.Parameter(torch.tensor(np.random.uniform(-1, 1, (d2, d1, d1)),
                                                 dtype=torch.float, device="cuda", requires_grad=True))

        self.input_dropout = torch.nn.Dropout(kwargs["input_dropout"])
        self.hidden_dropout1 = torch.nn.Dropout(kwargs["hidden_dropout1"])
        self.hidden_dropout2 = torch.nn.Dropout(kwargs["hidden_dropout2"])
        self.loss = torch.nn.BCELoss()

        self.bn0 = torch.nn.BatchNorm1d(d1)
        self.bn1 = torch.nn.BatchNorm1d(d1)

    def init(self):
        xavier_normal_(self.E.weight.data)
        xavier_normal_(self.R.weight.data)

    def evaluate(self, e1_idx, r_idx, es_idx):
        e1 = self.E(e1_idx)
        es = self.E(es_idx)
        x = self.bn0(e1)
        x = self.input_dropout(x)
        x = x.view(-1, 1, e1.size(1))

        r = self.R(r_idx)
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

    def forward(self, e1_idx, r_idx):
        e1 = self.E(e1_idx)
        e1 = self.E2E(e1)
        x = self.bn0(e1)
        x = self.input_dropout(x)
        x = x.view(-1, 1, e1.size(1))

        r = self.R(r_idx)
        r = self.R2R(r)
        W_mat = torch.mm(r, self.W.view(r.size(1), -1))  # torch.mm() 矩阵相乘
        W_mat = W_mat.view(-1, e1.size(1), e1.size(1))
        W_mat = self.hidden_dropout1(W_mat)

        x = torch.bmm(x, W_mat)  # torch.mm() batch矩阵相乘
        x = x.view(-1, e1.size(1))
        x = self.bn1(x)
        x = self.hidden_dropout2(x)
        x = torch.mm(x, self.E2E(self.E.weight).transpose(1, 0))
        pred = torch.sigmoid(x)
        return pred


    def forward(self, e1_idx, r_idx, e2p_idx, e2n_idx):
        e1 = self.E(e1_idx)
        e2p = self.E(e2p_idx)
        e2n = self.E(e2n_idx)
        x = self.bn0(e1)
        x = self.input_dropout(x)
        x = x.view(-1, 1, e1.size(1))

        r = self.R(r_idx)
        W_mat = torch.mm(r, self.W.view(r.size(1), -1))  # torch.mm() 矩阵相乘
        W_mat = W_mat.view(-1, e1.size(1), e1.size(1))
        W_mat = self.hidden_dropout1(W_mat)

        x = torch.bmm(x, W_mat)  # torch.mm() batch矩阵相乘
        x = x.view(-1, e1.size(1))
        x = self.bn1(x)
        x = self.hidden_dropout2(x)
        x_p = (x*e2p).sum(dim=1)
        x_n = (x*e2n).sum(dim=1)
        pred_p = torch.sigmoid(x_p)
        pred_n = torch.sigmoid(x_n)
        return pred_p, pred_n
