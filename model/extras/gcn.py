#!/usr/bin/env python3
import torch
import torch.nn as nn


class GCN(nn.Module):
    '''
    GCN: Graph Convolutional Network, ICLR 2017
    https://arxiv.org/pdf/1609.02907.pdf
    '''
    def __init__(self, nfeat, nhid, nclass, n_layers = 1, dropout=0.5):
        super().__init__()
        self.n_layers = n_layers    
        self.gcn1 = GraphConv(nfeat, nhid)
        self.gcn2 = GraphConv(nhid, nclass)
        self.acvt = nn.Sequential(nn.ReLU(), nn.Dropout(dropout))
        self.linear = nn.Linear(nclass, nclass)
        
    def forward(self, x, adj):
    
    # TO-DO : Try using and not using the relu in sencond leayer and also the number of layers in GCN

        
        x = self.gcn1(x, adj)
        x = self.acvt(x)
        if self.n_layers == 1:
            return x

        x = self.gcn2(x, adj)
        x = self.acvt(x)

        ## For one feature per graph (Graph Classification Tasks)
        # x = torch.sum(x, 0)
        # x = self.linear(x)
        
        return x


class GraphConv(nn.Module):
    def __init__(self, in_features, out_features, bias=False):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias)

    def forward(self, x, adj):
        return adj @ self.linear(x)