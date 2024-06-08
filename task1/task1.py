from GCN import GCN
import torch
import numpy as np
from tqdm import *
from torch_geometric.loader import DataLoader
device = torch.device('cuda:3' if torch.cuda.is_available() else 'cpu')
MODEL_PATH = './model/'
class AIGTrain():
    def __init__(self):
        self.model = GCN().to(device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.01)
        self.loss = torch.nn.MSELoss()
        self.batch_size = 16
        self.trainset = torch.load('split_train_data.pt')
        self.testset = torch.load('split_test_data.pt')
        
    def train(self):
        trainloader = DataLoader(self.trainset, batch_size=self.batch_size, shuffle=True)
        print(self.model)
        print('Start training')
        # self.model.train()
        for epoch in tqdm(range(100)):
            train_loss = []
            for data in trainloader:
                data = data.to(device)
                self.optimizer.zero_grad()
                out = self.model(data).squeeze()
                loss = self.loss(out, data.y)
                loss.backward()
                train_loss.append(loss.item())
                self.optimizer.step()    
            avg_train_loss = np.mean(train_loss)
            print('Epoch: ', epoch, 'Loss: ', avg_train_loss)     
        torch.save(self.model.state_dict(), MODEL_PATH + 'final.pth') 

if __name__ == '__main__':
    aig = AIGTrain()
    aig.train()
    
