import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATv2Conv, global_mean_pool
from torch_geometric.data import Batch

class SceneGraphEmbedding(nn.Module):
    def __init__(self, categorical_dim, hidden_categorical_dim, continuous_dim, hidden_dim, out_dim, \
                 conv_type='gat', heads=4, dropout=0.6, pool=True):
        super(SceneGraphEmbedding, self).__init__()
        self.pool = pool
        self.categorical_dim = categorical_dim
        self.hidden_categorical_dim = hidden_categorical_dim
        in_dim = continuous_dim + hidden_categorical_dim

        self.categorical_init = nn.Linear(categorical_dim, hidden_categorical_dim)
        self.lin_init = nn.Linear(in_dim, hidden_dim)

        if conv_type == 'gcn':
            self.conv1 = GCNConv(hidden_dim, hidden_dim)
            self.conv2 = GCNConv(hidden_dim, hidden_dim)
        elif conv_type == 'gat':
            self.conv1 = GATv2Conv(hidden_dim, hidden_dim, heads=heads, dropout=dropout)
            self.conv2 = GATv2Conv(hidden_dim * heads, hidden_dim, heads=1)
        self.lin_out = nn.Linear(hidden_dim, out_dim)
        #TODO: try with GAT

    def forward(self, x, edge_index, batch):

        #encoding is set up such that the very first dim is the one-hot encoding for 'ego'
        ego_index = torch.argmax(x.T[0]).item()

        #embed categorical variables first, then reconcat with x
        s = x[:, :self.categorical_dim]
        x = x[:, self.categorical_dim:] 
        s = F.relu(self.categorical_init(s))
        x = torch.cat((x,s), dim=1) #x now has dim in_dim

        # Initial node embeddings
        x = F.relu(self.lin_init(x))

        # Apply GNN layers
        x = F.elu(self.conv1(x, edge_index))
        x = F.elu(self.conv2(x, edge_index))
        
        if self.pool:
            x = global_mean_pool(x, batch)
            x = self.lin_out(x)
        else:
            x = self.lin_out(x)
            x = x[ego_index]
            
        return x