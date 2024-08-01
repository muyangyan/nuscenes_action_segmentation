import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATv2Conv, global_mean_pool
from torch_geometric.data import Batch
from dataset_utils import edge_labels

class SceneGraphEmbedding(nn.Module):
    def __init__(self, categorical_dim, hidden_categorical_dim, hidden_edge_dim, continuous_dim, hidden_dim, out_dim, \
                 conv_type='gat', heads=4, dropout=0.6, pool=True):
        super(SceneGraphEmbedding, self).__init__()
        self.conv_type = conv_type
        self.pool = pool
        self.categorical_dim = categorical_dim
        self.hidden_categorical_dim = hidden_categorical_dim
        in_dim = continuous_dim + hidden_categorical_dim

        self.categorical_init = nn.Linear(categorical_dim, hidden_categorical_dim)
        self.edge_embed = nn.Linear(len(edge_labels), hidden_edge_dim)
        self.lin_init = nn.Linear(in_dim, hidden_dim)

        flow = 'source_to_target'
        if not pool:
            #if we are just taking the final ego embedding, we need message passing to converge to ego
            flow = 'target_to_source'

        if conv_type == 'gcn':
            self.conv1 = GCNConv(hidden_dim, hidden_dim, flow=flow)
            self.conv2 = GCNConv(hidden_dim, hidden_dim, flow=flow)
        elif conv_type == 'gat':
            self.conv1 = GATv2Conv(hidden_dim, hidden_dim, edge_dim=hidden_edge_dim, heads=heads, dropout=dropout, flow=flow)
            self.conv2 = GATv2Conv(hidden_dim * heads, hidden_dim, edge_dim=hidden_edge_dim, heads=1, flow=flow)
        self.lin_out = nn.Linear(hidden_dim, out_dim)

    def forward(self, x, edge_index, edge_attr, batch):

        #encoding is set up such that the very first dim is the one-hot encoding for 'ego'
        ego_mask = [i[0]==1 for i in x]
        ego_mask = [e.item() for e in ego_mask]

        #embed categorical variables first, then reconcat with x
        s = x[:, :self.categorical_dim]
        x = x[:, self.categorical_dim:] 
        s = F.relu(self.categorical_init(s))
        x = torch.cat((x,s), dim=1) #x now has dim in_dim

        # Initial node embeddings
        x = F.relu(self.lin_init(x))

        # Apply GNN layers
        if self.conv_type == 'gat':

            edge_attr = self.edge_embed(edge_attr)
            
            x = F.elu(self.conv1(x, edge_index, edge_attr))
            x = F.elu(self.conv2(x, edge_index, edge_attr))
        else:
            x = F.elu(self.conv1(x, edge_index))
            x = F.elu(self.conv2(x, edge_index))
        
        if self.pool:
            x = global_mean_pool(x, batch)
            x = self.lin_out(x)
        else:
            #mask = [b==ego_index for b in batch]
            x = x[ego_mask]
            remainder = max(batch) - len(ego_mask) 
            if remainder > 0:
                x = torch.cat((x,torch.zeros(remainder, x.size()[1])), 0)

            x = self.lin_out(x)
            
        return x