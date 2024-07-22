import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.data import Batch

class SceneGraphEmbedding(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(SceneGraphEmbedding, self).__init__()
        self.lin_init = nn.Linear(in_channels, hidden_channels)
        self.conv1 = GCNConv(hidden_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.lin_out = nn.Linear(hidden_channels, out_channels)
        #TODO: try with GAT

    def forward(self, x, edge_index, batch):

        # Initial node embeddings
        x = F.relu(self.lin_init(x))

        # Apply GNN layers
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        
        # Global pooling
        x = global_mean_pool(x, batch)
        
        # Optional: Apply final linear layer
        x = self.lin_out(x)
        
        return x