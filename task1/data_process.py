import torch
from torch_geometric.data import Data
from sklearn.model_selection import train_test_split
dataset = torch.load('data_set.pt')
processed_data = []
for i, data in enumerate(dataset):
    data_batch = {}
    x = torch.cat([data['node_type'], data['num_inverted_predecessors']], dim=1)
    x = x.reshape(2, -1).T
    x = torch.tensor(x, dtype=torch.float32)
    edge_index = data['edge_index']
    batch = torch.zeros((x.size(0),), dtype=torch.long)
    new_data = Data(x = x, edge_index = edge_index, y = data['y'], batch = batch)
    processed_data.append(new_data)
train_data, test_data = train_test_split(processed_data, test_size=0.2, shuffle=True, random_state=42)
torch.save(train_data, 'split_train_data.pt')
torch.save(test_data, 'split_test_data.pt')
