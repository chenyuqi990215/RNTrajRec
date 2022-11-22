import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn.pytorch import GATConv, GINConv, EGATConv, GraphConv


class UnsupervisedGAT(nn.Module):
    def __init__(
        self, node_input_dim, node_hidden_dim, edge_input_dim, num_layers, num_heads=8,
    last_activate=False):
        super(UnsupervisedGAT, self).__init__()
        self.hid_dim = node_hidden_dim
        assert node_hidden_dim % num_heads == 0
        self.layers = nn.ModuleList(
            [
                GATConv(
                    in_feats=node_input_dim if i == 0 else node_hidden_dim,
                    out_feats=node_hidden_dim // num_heads,
                    num_heads=num_heads,
                    feat_drop=0.0,
                    attn_drop=0.0,
                    residual=False,
                    activation=F.leaky_relu if i + 1 < num_layers or last_activate else None,
                )
                for i in range(num_layers)
            ]
        )

    def forward(self, g, n_feat, e_feat=None):
        for i, layer in enumerate(self.layers):
            n_feat = layer(g, n_feat)
            n_feat = n_feat.reshape(-1, self.hid_dim)
        return n_feat


class UnsupervisedGIN(nn.Module):
    def __init__(
        self, node_input_dim, node_hidden_dim, edge_input_dim, num_layers, last_activate=False
    ):
        super(UnsupervisedGIN, self).__init__()
        self.layers = nn.ModuleList(
            [
                GINConv(
                    apply_func=nn.Linear(node_input_dim, node_hidden_dim) \
                        if i == 0 else nn.Linear(node_hidden_dim, node_hidden_dim),
                    activation=F.leaky_relu if i + 1 < num_layers or last_activate else None,
                    learn_eps=True,
                    aggregator_type='max'
                )
                for i in range(num_layers)
            ]
        )

    def forward(self, g, n_feat, e_feat=None):
        for i, layer in enumerate(self.layers):
            n_feat = layer(g, n_feat, e_feat)
        return n_feat


class UnsupervisedGCN(nn.Module):
    def __init__(
        self, node_input_dim, node_hidden_dim, edge_input_dim, num_layers, num_heads=8,
    last_activate=False):
        super(UnsupervisedGCN, self).__init__()
        self.hid_dim = node_hidden_dim
        assert node_hidden_dim % num_heads == 0
        self.layers = nn.ModuleList(
            [
                GraphConv(
                    in_feats=node_input_dim if i == 0 else node_hidden_dim,
                    out_feats=node_hidden_dim,
                    activation=F.leaky_relu if i + 1 < num_layers or last_activate else None,
                )
                for i in range(num_layers)
            ]
        )

    def forward(self, g, n_feat, e_feat=None):
        for i, layer in enumerate(self.layers):
            n_feat = layer(g, n_feat)
            n_feat = n_feat.reshape(-1, self.hid_dim)
        return n_feat


class UnsupervisedEGAT(nn.Module):
    def __init__(
        self, node_input_dim, node_hidden_dim, edge_input_dim, num_layers, num_heads=8,
    ):
        super(UnsupervisedEGAT, self).__init__()
        self.hid_dim = node_hidden_dim
        self.node_in_dim = node_input_dim
        self.edge_in_dim = edge_input_dim
        assert node_hidden_dim % num_heads == 0
        self.layers = nn.ModuleList(
            [
                EGATConv(
                    in_node_feats=node_input_dim if i == 0 else node_hidden_dim,
                    out_node_feats=node_hidden_dim // num_heads,
                    in_edge_feats=edge_input_dim,
                    out_edge_feats=8,
                    num_heads=num_heads
                )
                for i in range(num_layers)
            ]
        )

    def forward(self, g, n_feat, e_feat=None):
        n_feat = n_feat.reshape(-1, self.node_in_dim)
        e_feat = e_feat.reshape(-1, self.edge_in_dim)
        for i, layer in enumerate(self.layers):
            n_feat, _ = layer(g, n_feat, e_feat)
            n_feat = n_feat.reshape(-1, self.hid_dim)
        return n_feat