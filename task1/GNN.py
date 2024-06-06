import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.datasets import TUDataset
from torch_geometric.data import DataLoader
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
class GCN(torch.nn.Module):
    def __init__(self, num_features):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(num_features, 32)
        self.conv2 = GCNConv(32, 64)
        self.fc = torch.nn.Linear(64, 8)
        self.fc2 = torch.nn.Linear(8, 1)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = self.conv1(x, edge_index)
        x = torch.relu(x)
        # x = torch.dropout(x, p=0.5, train=self.training)

        x = self.conv2(x, edge_index)
        x = torch.relu(x)
        # x = torch.dropout(x, p=0.5, train=self.training)

        x = global_mean_pool(x, batch) # Add global mean pooling

        x = self.fc(x)
        x = torch.relu(x)

        x = self.fc2(x)
        x = x.reshape(-1)
        return x
