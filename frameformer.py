import torch
from block import Dense
from torch.autograd import Variable
import math, copy
from torch import nn as nn
import numpy as np
from torchvision import transforms
from torch.utils.data import Dataset
from lstm import get_temporal_abaw5_dataset, get_temporal_lsd_dataset
from transformer import get_projection, PositionwiseFeedForward, attention

class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        "Take in model size and number of heads."
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)
        
    def forward(self, query, key, value, mask=None):
        "Implements Figure 2"
        if mask is not None:
            # Same mask applied to all h heads.
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)
        
        # 1) Do all the linear projections in batch from d_model => h x d_k 
        query, key, value = \
            [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
             for l, x in zip(self.linears, (query, key, value))]
        
        # 2) Apply attention on all the projected vectors in batch. 
        x, self.attn = attention(query, key, value, mask=mask, 
                                 dropout=self.dropout)
        
        # 3) "Concat" using a view and apply a final linear. 
        x = x.transpose(1, 2).contiguous() \
             .view(nbatches, -1, self.h * self.d_k)
        return self.linears[-1](x)


def clones(module, N):
    "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

class LayerNorm(nn.Module):
    "Construct a layernorm module (See citation for details)."
    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2



class SublayerConnection(nn.Module):
    """
    A residual connection followed by a layer norm.
    Note for code simplicity the norm is first as opposed to last.
    """
    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        "Apply residual connection to any sublayer with the same size."
        return x + self.dropout(sublayer(self.norm(x)))

class EncoderLayer(nn.Module):
    "Encoder is made up of self-attn and feed forward (defined below)"
    def __init__(self, size, self_attn, feed_forward, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 2)
        self.size = size

    def forward(self, x, mask):
        "Follow Figure 1 (left) for connections."
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
        return self.sublayer[1](x, self.feed_forward)

class OriginEncoder(nn.Module):
    "Core encoder is a stack of N layers"
    def __init__(self, layer, N):
        super(OriginEncoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)
        
    def forward(self, x, mask):
        "Pass the input (and mask) through each layer in turn."
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)

class DecoderLayer(nn.Module):
    "Decoder is made of self-attn, src-attn, and feed forward (defined below)"
    def __init__(self, size, self_attn, src_attn, feed_forward, dropout):
        super(DecoderLayer, self).__init__()
        self.size = size
        self.self_attn = self_attn
        self.src_attn = src_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 3)
 
    def forward(self, x, memory, src_mask, tgt_mask):
        "Follow Figure 1 (right) for connections."
        m = memory
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, tgt_mask))
        x = self.sublayer[1](x, lambda x: self.src_attn(x, m, m, src_mask))
        return self.sublayer[2](x, self.feed_forward)

class OriginDecoder(nn.Module):
    "Generic N layer decoder with masking."
    def __init__(self, layer, N):
        super(OriginDecoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)
        
    def forward(self, x, memory, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, memory, src_mask, tgt_mask)
        return self.norm(x)

def subsequent_mask(size):
    "Mask out subsequent positions."
    attn_shape = (1, size, size)
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    return torch.from_numpy(subsequent_mask) == 0

class Frameformer(nn.Module):
    def __init__(self, input, output, size, h, feed_forward, dropout, N):
        super(Frameformer, self).__init__()
        c = copy.deepcopy
        self.src_project = get_projection(input, 512, 'minimal')
        self.tgt_project = get_projection(output, 512, 'minimal')
        ff = PositionwiseFeedForward(512, feed_forward, dropout)
        attn = MultiHeadedAttention(h, 512)
        self.encoder = OriginEncoder(EncoderLayer(512, c(attn), c(ff), dropout), N)
        self.decoder = OriginDecoder(DecoderLayer(512, c(attn), c(attn), 
                             c(ff), dropout), N)
        self.head = Dense(512, output)

    def forward(self, src, tgt, src_mask, tgt_mask):
        memory = self.encode(src, src_mask)
        output = self.decode(memory, src_mask, tgt, tgt_mask)
        output = self.head(output)
        return output
    
    def encode(self, src, src_mask):
        x = self.src_project(src)
        y = self.encoder(x, src_mask)
        return y
    
    def decode(self, memory, src_mask, tgt, tgt_mask):
        x = self.tgt_project(tgt)
        y = self.decoder(x, memory, src_mask, tgt_mask)
        return y

def greedy_decode(model, src, src_mask, max_len):
    memory = model.encode(src, src_mask)
    ys = torch.zeros(1, 1, 8).fill_(0).type_as(src.data)
    for i in range(max_len-1):
        tgt = Variable(ys)
        tgt_mask = Variable(subsequent_mask(ys.size(1)).type_as(src.data))
        out = model.decode(memory, src_mask, tgt, tgt_mask)
        out = out[:, -1]
        prob = model.head(out)
        _, next_word = torch.max(prob, dim = 1)
        next_word = next_word.data[0]
        pred = torch.zeros(1, 1, 8).type_as(src.data)
        pred[0,0,next_word] = 1.0
        ys = torch.cat([ys, pred], dim=1)
    return ys


class FrameABAW5(Dataset):
    def __init__(self, annoation_paths, img_path, feature_dicts, max_length):
        super(FrameABAW5, self).__init__()
        path_abaw5, path_lsd_t, path_lsd_v = annoation_paths
        dict_abaw5, dict_lsd_t, dict_lsd_v = feature_dicts
        X_abaw5, Y_abaw5 = get_temporal_abaw5_dataset(path_abaw5, img_path, dict_abaw5, max_length, True)
        X_lsd_t, Y_lsd_t = get_temporal_lsd_dataset(path_lsd_t, dict_lsd_t, max_length)
        X_lsd_v, Y_lsd_v = get_temporal_lsd_dataset(path_lsd_v, dict_lsd_v, max_length)

        self.src = []
        X = X_abaw5 + X_lsd_t + X_lsd_v
        self.src_mask = [np.ones((x.shape[0]+1)) for x in X]

        for x in X:
            self.src.append(np.concatenate([np.zeros((1, 1288)), x], axis=0))
 
        self.tgt_y = []
        Y = Y_abaw5 + Y_lsd_t + Y_lsd_v
        self.weigth = self.ex_weight(Y)

        for y in Y:
            one_hot = np.zeros((y.shape[0], 8))
            one_hot[np.arange(y.shape[0]), y] = 1
            self.tgt_y.append(one_hot)

        self.tgt = []
        for y in self.tgt_y:
            self.tgt.append(np.concatenate([np.zeros((1, 8)), y[:-1,:]], axis=0))
        
        self.tgt_mask = []
        for tgt in self.tgt:
            mask = subsequent_mask(tgt.shape[0]).squeeze(0)
            self.tgt_mask.append(mask)

    def __getitem__(self, i):
        return self.src[i] , self.src_mask[i], self.tgt[i], self.tgt_mask[i], self.tgt_y[i]

    def __len__(self):
        return len(self.src)

    def ex_weight(self, y):
        y = np.concatenate(y, axis=0).flatten()
        unique, counts = np.unique(y.astype(int), return_counts=True)
        emo_cw = 1 / counts
        emo_cw/= emo_cw.min()
        return emo_cw

class VFrameABAW5(Dataset):
    def __init__(self, annoation_paths, img_path, feature_dict, max_length):
        super(VFrameABAW5, self).__init__()
        X_abaw5, Y_abaw5 = get_temporal_abaw5_dataset(annoation_paths, img_path, feature_dict, max_length, False)

        self.src = []
        X = X_abaw5
        self.src_mask = [np.ones((x.shape[0]+1)) for x in X]

        for x in X:
            self.src.append(np.concatenate([np.zeros((1, 1288)), x], axis=0))
 
        self.tgt_y = []
        Y = Y_abaw5
        self.weigth = self.ex_weight(Y)

        for y in Y:
            one_hot = np.zeros((y.shape[0], 8))
            one_hot[np.arange(y.shape[0]), y] = 1
            self.tgt_y.append(one_hot)

        self.tgt = []
        for y in self.tgt_y:
            self.tgt.append(np.concatenate([np.zeros((1, 8)), y[:-1,:]], axis=0))
        
        self.tgt_mask = []
        for tgt in self.tgt:
            mask = subsequent_mask(tgt.shape[0]).squeeze(0)
            self.tgt_mask.append(mask)

    def __getitem__(self, i):
        return self.src[i] , self.src_mask[i], self.tgt[i], self.tgt_mask[i], self.tgt_y[i]

    def __len__(self):
        return len(self.src)

    def ex_weight(self, y):
        y = np.concatenate(y, axis=0).flatten()
        unique, counts = np.unique(y.astype(int), return_counts=True)
        emo_cw = 1 / counts
        emo_cw/= emo_cw.min()
        return emo_cw

