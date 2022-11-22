"""
Code from: https://towardsdatascience.com/how-to-code-the-transformer-in-pytorch-24db27c8f9ec
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import math
import dgl
from utils.dgl_gnn import UnsupervisedGAT, UnsupervisedGIN
from graph_norm import GraphNorm


class PositionalEncoder(nn.Module):
    def __init__(self, d_model, device, max_seq_len=150):
        super().__init__()
        self.d_model = d_model
        self.device = device

        # create constant 'pe' matrix with values dependant on
        # pos and i
        pe = torch.zeros(max_seq_len, d_model)
        for pos in range(max_seq_len):
            for i in range(0, d_model, 2):
                pe[pos, i] = \
                    math.sin(pos / (10000 ** ((2 * i) / d_model)))
                pe[pos, i + 1] = \
                    math.cos(pos / (10000 ** ((2 * (i + 1)) / d_model)))

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # make embeddings relatively larger
        x = x * math.sqrt(self.d_model)
        # add constant to embedding
        seq_len = x.size(1)
        x = x + Variable(self.pe[:, :seq_len], requires_grad=False).to(self.device)
        return x


class GatedFusion(nn.Module):
    def __init__(self, d_model):
        super(GatedFusion, self).__init__()
        self.hid_dim = d_model
        self.HS_fc = nn.Linear(self.hid_dim, self.hid_dim)
        self.HT_fc = nn.Linear(self.hid_dim, self.hid_dim)

    def forward(self, HS, HT):
        '''
        gated fusion
        HS:     (1, batch_size, hid_dim)
        HT:     (1, batch_size, hid_dim)
        return: (1, batch_size, hid_dim)
        '''
        XS = F.leaky_relu(self.HS_fc(HS))
        XT = F.leaky_relu(self.HT_fc(HT))
        z = torch.sigmoid(torch.add(XS, XT))
        H = torch.add(torch.multiply(z, HS), torch.multiply(1 - z, HT))
        return H


class MultiHeadAttention(nn.Module):
    def __init__(self, heads, d_model, dropout=0.1):
        super().__init__()

        self.d_model = d_model
        self.d_k = d_model // heads
        self.h = heads

        self.q_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(d_model, d_model)

    def forward(self, q, k, v, mask=None):
        bs = q.size(0)

        # perform linear operation and split into h heads

        k = self.k_linear(k).view(bs, -1, self.h, self.d_k)
        q = self.q_linear(q).view(bs, -1, self.h, self.d_k)
        v = self.v_linear(v).view(bs, -1, self.h, self.d_k)

        # transpose to get dimensions bs * h * sl * d_model

        k = k.transpose(1, 2)
        q = q.transpose(1, 2)
        v = v.transpose(1, 2)
        # calculate attention using function we will define next
        scores = self.attention(q, k, v, self.d_k, mask, self.dropout)

        # concatenate heads and put through final linear layer
        concat = scores.transpose(1, 2).contiguous().view(bs, -1, self.d_model)

        output = self.out(concat)

        return output

    def attention(self, q, k, v, d_k, mask=None, dropout=None):
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)

        if mask is not None:
            mask = mask.unsqueeze(1)
            scores = scores.masked_fill(mask == 0, -1e9)
        scores = F.softmax(scores, dim=-1)

        if dropout is not None:
            scores = self.dropout(scores)

        output = torch.matmul(scores, v)
        return output


class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff=512, dropout=0.1):
        super().__init__()
        # We set d_ff as a default to 2048
        self.linear_1 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        x = self.dropout(F.relu(self.linear_1(x)))
        x = self.linear_2(x)
        return x


class FeedForwardGNN(nn.Module):
    def __init__(self, gnn_type, input_dim, output_dim, num_layers=1, num_heads=8, dropout=0.1):
        super().__init__()
        self.gnn_type = gnn_type
        self.node_input_dim = input_dim
        self.node_hidden_dim = output_dim
        self.num_layers = num_layers
        self.dropout = nn.Dropout(dropout)
        if self.gnn_type == 'gat':
            self.gnn_1 = UnsupervisedGAT(self.node_input_dim, self.node_hidden_dim, edge_input_dim=0,
                                       num_layers=self.num_layers, num_heads=num_heads)
            # self.gnn_2 = UnsupervisedGAT(self.node_input_dim, self.node_hidden_dim, edge_input_dim=0,
            #                            num_layers=self.num_layers, num_heads=num_heads)
        else:
            self.gnn_1 = UnsupervisedGIN(self.node_input_dim, self.node_hidden_dim, edge_input_dim=0,
                                       num_layers=self.num_layers)
            # self.gnn_2 = UnsupervisedGIN(self.node_input_dim, self.node_hidden_dim, edge_input_dim=0,
            #                            num_layers=self.num_layers)

    def forward(self, g, x):
        '''
        :param x: road emb id with size [node size, id dim]
        :return: road hidden emb with size [graph size, hidden dim] if readout
                 else [node size, hidden dim]
        '''
        if 'w' in g.edata:
            # x = self.dropout(F.relu(self.gnn_1(g, x, g.edata['w'])))
            # x = self.gnn_2(g, x, g.edata['w'])
            x = self.gnn_1(g, x, g.edata['w'])
            return x
        else:
            # x = self.dropout(F.relu(self.gnn_1(g, x)))
            # x = self.gnn_2(g, x)
            x = self.gnn_1(g, x)
            return x


class Norm(nn.Module):
    def __init__(self, d_model, eps=1e-6):
        super().__init__()

        self.size = d_model
        # create two learnable parameters to calibrate normalisation
        self.alpha = nn.Parameter(torch.ones(self.size))
        self.bias = nn.Parameter(torch.zeros(self.size))
        self.eps = eps

    def forward(self, x):
        norm = self.alpha * (x - x.mean(dim=-1, keepdim=True)) \
               / (x.std(dim=-1, keepdim=True) + self.eps) + self.bias
        return norm


class GraphRefinementLayer(nn.Module):
    def __init__(self, gnn_type, d_model, num_heads, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.norm_1 = GraphNorm(d_model)
        self.norm_2 = GraphNorm(d_model)
        self.attn = GatedFusion(d_model)
        self.ff = FeedForwardGNN(gnn_type, d_model, d_model, num_heads=num_heads)
        self.dropout_1 = nn.Dropout(dropout)
        self.dropout_2 = nn.Dropout(dropout)

    def forward(self, y, g, mask2d=None):
        """
        :param hidden: [bs, src len, hid dim]
        :param g: batched DGLGraph (with bs first)
        :return: refined graph and refined hidden state
        """
        bs = y.size(0)
        max_src_len = y.size(1)

        y = y.reshape(-1, self.d_model)
        y = dgl.broadcast_nodes(g, y)  # [v, hid dim]

        x = g.ndata['x']  # [v, hid dim]

        x2 = self.norm_1(x, g, mask2d)
        x = x + self.dropout_1(self.attn(x2, y))
        x2 = self.norm_2(x, g, mask2d)
        x = x + self.dropout_2(self.ff(g, x2))  # [v, hid dim]

        g.ndata['x'] = x

        # x3 = dgl.sum_nodes(g, 'x').reshape(bs, max_src_len, -1)
        x3 = dgl.mean_nodes(g, 'x').reshape(bs, max_src_len, -1)
        return x3, g


class EncoderLayer(nn.Module):
    def __init__(self, d_model, heads=8, dropout=0.1):
        super().__init__()
        self.norm_1 = Norm(d_model)
        self.norm_2 = Norm(d_model)
        self.attn = MultiHeadAttention(heads, d_model)
        self.ff = FeedForward(d_model)
        self.dropout_1 = nn.Dropout(dropout)
        self.dropout_2 = nn.Dropout(dropout)
        self.norm_3 = Norm(d_model)

    def forward(self, x, mask, norm=False):
        x2 = self.norm_1(x)
        x = x + self.dropout_1(self.attn(x2, x2, x2, mask))
        x2 = self.norm_2(x)
        x = x + self.dropout_2(self.ff(x2))
        return x if not norm else self.norm_3(x)


class Encoder(nn.Module):
    def __init__(self, gnn_type, d_model, N, device, heads=8):
        super().__init__()
        self.N = N
        self.pe = PositionalEncoder(d_model, device)
        self.layers = nn.ModuleList([
            EncoderLayer(d_model, heads) for _ in range(N)
        ])
        self.refines = nn.ModuleList([
            GraphRefinementLayer(gnn_type, d_model, heads) for _ in range(N)
        ])
        self.norm = Norm(d_model)

    def forward(self, src, g, mask3d=None, mask2d=None):
        """
        :param src: [bs, src len, hid dim]
        :param g: batched DGLGraph (with bs first)
        :param mask: [bs, src len, src len]
        :return: encoder hidden, refined graph
        """
        x = self.pe(src)
        for i in range(self.N):
            x = self.layers[i](x, mask3d)
            x, g = self.refines[i](x, g, mask2d)
        return self.norm(x), g