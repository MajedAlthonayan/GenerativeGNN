import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import matplotlib.pyplot as plt
from initializations import *
from preprocessing import normalize_adj_torch


class GSRLayer(nn.Module):

    def __init__(self, hr_dim):
        super(GSRLayer, self).__init__()

        self.weights = torch.from_numpy(
            weight_variable_glorot(hr_dim)).type(torch.FloatTensor)
        self.weights = torch.nn.Parameter(
            data=self.weights, requires_grad=True)

    def forward(self, A, X):
        with torch.autograd.set_detect_anomaly(True):

            lr = A
            lr_dim = lr.shape[0]
            f = X
            eig_val_lr, U_lr = torch.linalg.eigh(lr, UPLO='U')

            # U_lr = torch.abs(U_lr)
            eye_mat = torch.eye(lr_dim).type(torch.FloatTensor)
            s_d = torch.cat((eye_mat, eye_mat), 0)

            a = torch.matmul(self.weights, s_d)
            b = torch.matmul(a, torch.t(U_lr))
            f_d = torch.matmul(b, f)
            f_d = torch.abs(f_d)
            f_d = f_d.fill_diagonal_(1)
            adj = f_d

            X = torch.mm(adj, adj.t())
            X = (X + X.t())/2
            X = X.fill_diagonal_(1)
        return adj, torch.abs(X)


class GraphConvolution(nn.Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, dropout, act=F.relu):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.dropout = dropout
        self.act = act
        self.weight = torch.nn.Parameter(
            torch.FloatTensor(in_features, out_features))
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.weight)

    def forward(self, input, adj):
        input = F.dropout(input, self.dropout, self.training)
        support = torch.mm(input, self.weight)
        output = torch.mm(adj, support)
        output = self.act(output)
        return output


class GATLayer(nn.Module):
    def __init__(self, in_features, out_features, dropout_rate=0.4, alpha=0.2):
        super(GATLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.dropout_rate = dropout_rate
        self.alpha = alpha

        self.W = nn.Parameter(torch.empty(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.a = nn.Parameter(torch.empty(size=(2*out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        self.dropout = nn.Dropout(dropout_rate)
        self.leakyrelu = nn.LeakyReLU(alpha)

    def forward(self, x, adj):
        h = torch.mm(x, self.W)
        N = h.size()[0]

        a_input = torch.cat([h.repeat(1, N).view(N * N, -1), h.repeat(N, 1)], dim=1).view(N, N, 2 * self.out_features)
        e = self.leakyrelu(torch.matmul(a_input, self.a).squeeze(2))

        zero_vec = -9e15*torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)
        attention = F.softmax(attention, dim=1)
        attention = self.dropout(attention)
        h_prime = torch.matmul(attention, h)

        return h_prime


class MultiHeadGATLayer(nn.Module):
    def __init__(self, in_features, out_features, num_heads=3, dropout_rate=0.4, alpha=0.2):
        super(MultiHeadGATLayer, self).__init__()
        self.num_heads = num_heads
        self.head_dim = out_features // num_heads
        assert self.head_dim * num_heads == out_features, "out_features must be divisible by num_heads"

        self.attention_heads = nn.ModuleList([
            GATLayer(in_features, self.head_dim, dropout_rate, alpha)
            for _ in range(num_heads)
        ])
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x, adj):
        head_outputs = [attn_head(x, adj) for attn_head in self.attention_heads]
        # Concatenate the outputs of different heads along the feature dimension
        output = torch.cat(head_outputs, dim=1) 
        return self.dropout(output)


class GINLayer(nn.Module):
    """Graph Isomorphism Network layer with an MLP."""
    def __init__(self, in_features, out_features, mlp_layers=2, dropout=0.5):
        super(GINLayer, self).__init__()
        self.mlp = self.create_mlp(in_features, out_features, mlp_layers, dropout)

    def create_mlp(self, in_features, out_features, mlp_layers, dropout):
        layers = [nn.Linear(in_features, out_features), nn.ReLU(), nn.Dropout(dropout)]
        for _ in range(mlp_layers - 1):
            layers.extend([nn.Linear(out_features, out_features), nn.ReLU(), nn.Dropout(dropout)])
        return nn.Sequential(*layers)

    def forward(self, input, adj):
        neighbor_sum = torch.mm(adj, input)  # Aggregate neighbor features
        self_feats = input  # Self-features
        total = neighbor_sum + self_feats  # Combine self and neighbor features
        return self.mlp(total)
