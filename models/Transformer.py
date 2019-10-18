import copy
import json
import math
import re

import numpy as np
import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
from torch.nn.init import xavier_normal_


class TuckER(torch.nn.Module):
    def __init__(self, d, d1, d2, **kwargs):
        super(TuckER, self).__init__()

        # self.E = torch.nn.Embedding(len(d.entities), d1, padding_idx=0)
        # self.R = torch.nn.Embedding(len(d.relations), d2, padding_idx=0)
        self.W = torch.nn.Parameter(torch.tensor(np.random.uniform(-1, 1, (d2, d1, d1)),
                                                 dtype=torch.float, device="cuda", requires_grad=True))

        self.input_dropout = torch.nn.Dropout(kwargs["input_dropout"])
        self.hidden_dropout1 = torch.nn.Dropout(kwargs["hidden_dropout1"])
        self.hidden_dropout2 = torch.nn.Dropout(kwargs["hidden_dropout2"])
        self.loss = torch.nn.BCELoss()

        #print("d1="+str(d1))
        self.bn0 = torch.nn.BatchNorm1d(d1)
        self.bn1 = torch.nn.BatchNorm1d(d1)

    def init(self):
        xavier_normal_(self.E.weight.data)
        xavier_normal_(self.R.weight.data)

    def forward(self, e1, r):
        print("e1 size:"+str(e1.size()))
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


'''
Much of this code is taken from HuggingFace's OpenAI LM Implementation here:

https://github.com/huggingface/pytorch-openai-transformer-lm
'''


def gelu(x):
    return (0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) *
            (x + 0.044715 * torch.pow(x, 3)))))


def swish(x):
    return x * torch.sigmoid(x)


ACT_FNS = {
    'relu': nn.ReLU,
    'swish': swish,
    'gelu': gelu
}


class LayerNorm(nn.Module):
    "Construct a layernorm module in the OpenAI style \
    (epsilon inside the square root)."

    def __init__(self, n_state, e=1e-5):
        super(LayerNorm, self).__init__()
        self.g = nn.Parameter(torch.ones(n_state))
        self.b = nn.Parameter(torch.zeros(n_state))
        self.e = e

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.e)
        return self.g * x + self.b


class Conv1D(nn.Module):# return w * x + b x:nx b:nf
    def __init__(self, nf, rf, nx):
        super(Conv1D, self).__init__()
        self.rf = rf
        self.nf = nf
        if rf == 1:  # faster 1x1 conv
            w = torch.empty(nx, nf)
            nn.init.normal_(w, std=0.02)
            self.w = Parameter(w)
            self.b = Parameter(torch.zeros(nf))
        else:  # was used to train LM
            raise NotImplementedError

    def forward(self, x):
        if self.rf == 1:
            size_out = x.size()[:-1] + (self.nf,)
            x = torch.addmm(self.b, x.view(-1, x.size(-1)), self.w)
            x = x.view(*size_out)
        else:
            raise NotImplementedError
        return x


class Attention(nn.Module):
    def __init__(self, nx, n_ctx, cfg, scale=False):
        super(Attention, self).__init__()
        n_state = nx  # in Attention: n_state=768 (nx=n_embd)

        assert n_state % cfg.nH == 0
        self.register_buffer('b', torch.tril(torch.ones( #tril()返回下三角阵
            n_ctx, n_ctx)).view(1, 1, n_ctx, n_ctx))
        self.n_head = cfg.nH
        self.split_size = n_state
        self.scale = scale
        self.c_attn = Conv1D(n_state * 3, 1, nx)
        self.c_proj = Conv1D(n_state, 1, nx)
        self.attn_dropout = nn.Dropout(cfg.adpt)
        self.resid_dropout = nn.Dropout(cfg.rdpt)

    # dimensions of w: (batch_size x num_heads x seq_length x seq_length)
    def _attn(self, q, k, v, sequence_mask):
        w = torch.matmul(q, k)
        if self.scale:
            w = w / math.sqrt(v.size(-1))

        b_subset = self.b[:, :, :w.size(-2), :w.size(-1)]

        if sequence_mask is not None:
            b_subset = b_subset * sequence_mask.view(
                sequence_mask.size(0), 1, -1)
            b_subset = b_subset.permute(1, 0, 2, 3)

        w = w * b_subset + -1e9 * (1 - b_subset)
        w = nn.Softmax(dim=-1)(w)
        w = self.attn_dropout(w)
        return torch.matmul(w, v)

    def merge_heads(self, x):
        x = x.permute(0, 2, 1, 3).contiguous()
        new_x_shape = x.size()[:-2] + (x.size(-2) * x.size(-1),)
        return x.view(*new_x_shape)  # in Tensorflow implem: fct merge_states

    def split_heads(self, x, k=False):
        new_x_shape = x.size()[:-1] + (self.n_head, x.size(-1) // self.n_head)
        x = x.view(*new_x_shape)  # in Tensorflow implem: fct split_states
        if k:
            return x.permute(0, 2, 3, 1)
        else:
            return x.permute(0, 2, 1, 3)

    def forward(self, x, sequence_mask):
        x = self.c_attn(x)
        query, key, value = x.split(self.split_size, dim=2)
        query = self.split_heads(query)
        key = self.split_heads(key, k=True)
        value = self.split_heads(value)
        a = self._attn(query, key, value, sequence_mask)
        a = self.merge_heads(a)
        a = self.c_proj(a)
        a = self.resid_dropout(a)
        return a


class MLP(nn.Module):
    def __init__(self, n_state, cfg):  # in MLP: n_state=3072 (4 * n_embd)
        super(MLP, self).__init__()
        nx = cfg.hSize
        self.c_fc = Conv1D(n_state, 1, nx)
        self.c_proj = Conv1D(nx, 1, n_state)
        self.act = ACT_FNS[cfg.afn]
        self.dropout = nn.Dropout(cfg.rdpt)

    def forward(self, x):
        h = self.act(self.c_fc(x))
        h2 = self.c_proj(h)
        return self.dropout(h2)


class Block(nn.Module):
    def __init__(self, n_ctx, cfg, scale=False):
        super(Block, self).__init__()
        nx = cfg.hSize
        self.attn = Attention(nx, n_ctx, cfg, scale)
        self.ln_1 = LayerNorm(nx)
        self.mlp = MLP(4 * nx, cfg)
        self.ln_2 = LayerNorm(nx)

    def forward(self, x, sequence_mask):
        a = self.attn(x, sequence_mask)
        n = self.ln_1(x + a)
        m = self.mlp(n)
        h = self.ln_2(n + m)
        return h


class TransformerModel(nn.Module):
    """ Transformer model """

    def __init__(self, cfg, vocab=40990, n_ctx=512):
        super(TransformerModel, self).__init__()
        self.vocab = vocab
        self.embed = nn.Embedding(vocab, cfg.hSize, padding_idx=0)
        self.drop = nn.Dropout(cfg.edpt)
        block = Block(n_ctx, cfg, scale=True)
        self.h = nn.ModuleList([copy.deepcopy(block)
                                for _ in range(cfg.nL)])

        nn.init.normal_(self.embed.weight, std=0.02)


    def forward(self, x, sequence_mask):
        x = x.view(-1, x.size(-2), x.size(-1))
        #print("x size:"+str(x.size()))
        e = self.embed(x)
        #print("e size:" + str(e.size()))
        # Add the position information to the input embeddings
        h = e.sum(dim=2)
        #print("h size before block:" + str(h.size()))
        for block in self.h:
            h = block(h, sequence_mask)
        #print("h size after block:" + str(h.size()))
        return h



class TransformerTucker(nn.Module):
    """Text Encoding Model"""

    def __init__(self, d, ent_vec_dim, rel_vec_dim, cfg, vocab=40990, n_ctx=512, **kwargs):
        super(TransformerTucker, self).__init__()
        self.n_ctx = n_ctx
        self.tucker = TuckER(d, ent_vec_dim, rel_vec_dim, **kwargs)
        self.transformer = TransformerModel(cfg, vocab=vocab, n_ctx=n_ctx)
        self.translationE = Conv1D(ent_vec_dim, 1, cfg.hSize)#last layer into one vector
        self.translationR = Conv1D(rel_vec_dim, 1, cfg.hSize)
        self.E = torch.nn.Embedding(len(d.entities), ent_vec_dim, padding_idx=0)
        self.R = torch.nn.Embedding(len(d.relations), rel_vec_dim, padding_idx=0)
        self.loss = torch.nn.BCELoss()


    # def init(self):
    #     xavier_normal_(self.E.weight.data)
    #     xavier_normal_(self.R.weight.data)

    def forward(self, e, r, n_special=0, sequence_mask = None):

        #print('e size:'+str(e.size()))
        he = self.transformer(e, sequence_mask)
        he = he[:,0,:]#get the first word's hidden layer
        #he = he.reshape(he.size(0), he.size(-2) * he.size(-1))
        #print('he size:' + str(he.size()))
        te = self.translationE(he)
        #print('te size:' + str(te.size()))


        hr = self.transformer(r, sequence_mask)
        hr = hr[:,0,:]#get the first word's hidden layer
        #hr = hr.reshape(hr.size(0), hr.size(-2) * hr.size(-1))
        tr = self.translationR(hr)

        return self.tucker(te, tr)




def load_openai_pretrained_model(model, n_ctx=-1, n_special=-1, n_transfer=12,
                                 n_embd=768, path='./model/', path_names='./model/'):
    # Load weights from TF model
    print("Loading weights...")
    names = json.load(open(path_names + 'parameters_names.json'))
    shapes = json.load(open(path + 'params_shapes.json'))
    offsets = np.cumsum([np.prod(shape) for shape in shapes])
    init_params = [np.load(path + 'params_{}.npy'.format(n)) for n in range(10)]
    init_params = np.split(np.concatenate(init_params, 0), offsets)[:-1]
    init_params = [param.reshape(shape) for param, shape in zip(init_params, shapes)]
    if n_ctx > 0:
        init_params[0] = init_params[0][:n_ctx]
    if n_special > 0:
        init_params[0] = np.concatenate(
            [init_params[1],
             (np.random.randn(n_special, n_embd) * 0.02).astype(np.float32),
             init_params[0]
             ], 0)
    else:
        init_params[0] = np.concatenate(
            [init_params[1],
             init_params[0]
             ], 0)
    del init_params[1]
    if n_transfer == -1:
        n_transfer = 0
    else:
        n_transfer = 1 + n_transfer * 12
    init_params = [arr.squeeze() for arr in init_params]

    try:
        assert model.embed.weight.shape == init_params[0].shape
    except AssertionError as e:
        e.args += (model.embed.weight.shape, init_params[0].shape)
        raise

    model.embed.weight.data = torch.from_numpy(init_params[0])

    for name, ip in zip(names[1:n_transfer], init_params[1:n_transfer]):
        name = name[6:]  # skip "model/"
        assert name[-2:] == ":0"
        name = name[:-2]
        name = name.split('/')
        pointer = model
        for m_name in name:
            if re.fullmatch(r'[A-Za-z]+\d+', m_name):
                l = re.split(r'(\d+)', m_name)
            else:
                l = [m_name]
            pointer = getattr(pointer, l[0])
            if len(l) >= 2:
                num = int(l[1])
                pointer = pointer[num]
        try:
            assert pointer.shape == ip.shape
        except AssertionError as e:
            e.args += (pointer.shape, ip.shape)
            raise
        pointer.data = torch.from_numpy(ip)


class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


DEFAULT_CONFIG = dotdict({
    'n_embd': 768,
    'n_head': 12,
    'n_layer': 12,
    'embd_pdrop': 0.1,
    'attn_pdrop': 0.1,
    'resid_pdrop': 0.1,
    'afn': 'gelu',
    'clf_pdrop': 0.1})

def prepare_position_embeddings(encoder_vocab, sequences):
    vocab_size = len(encoder_vocab)
    #print('sequences size:')
    #print(sequences.size())
    num_positions = sequences.size(-2)
    position_embeddings = torch.LongTensor(
        range(vocab_size, vocab_size + num_positions)).to(sequences.device)
    sequences = sequences.repeat(1, 1, 2)
    sequences[:, :, 1] = position_embeddings
    #print(sequences.size())
    return sequences
