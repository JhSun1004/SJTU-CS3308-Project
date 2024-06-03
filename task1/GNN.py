import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch.nn import Linear

class GCN(torch.nn.Module):
    def __init__(self):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(2, 64)
        self.conv2 = GCNConv(64, 64)
        self.lin = Linear(64, 1)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index).relu()
        x = torch.mean(x, dim=0)  # Global pooling.
        x = self.lin(x)
        return x