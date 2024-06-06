from GCN import GCN
import torch
import numpy as np
from matplotlib import pyplot as plt
from tqdm import *
from torch_geometric.loader import DataLoader
from dataset import load_data
device = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')
MODEL_PATH = './model/'
class AIGTrain():
    def __init__(self):
        self.model = GCN(2).to(device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.1)
        self.lr_scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=20, gamma=0.5)
        self.loss = torch.nn.MSELoss()
        self.batch_size = 16
        self.trainset = torch.load('split_train_data.pt')
        self.testset = torch.load('split_test_data.pt')
    
    def train(self):
        trainloader = DataLoader(self.trainset, batch_size=self.batch_size, shuffle=True)
        min_loss = 100
        print(self.model)
        print('Start training')
        for i in range(1000):
            self.model.train()
            for data in trainloader:
                data = data.to(device)
                self.optimizer.zero_grad()
                out = self.model(data)
                loss = self.loss(out, data.y)
                loss.backward()
                self.optimizer.step()
                self.lr_scheduler.step()
            new_loss = self.test()
            print('\nEpoch: {}, Loss: {}'.format(i, new_loss))
            if new_loss < min_loss:
                min_loss = new_loss.clone()
                torch.save(self.model.state_dict(), MODEL_PATH + str(i) + '.pth')
                print('Model saved')
        # self.final_test()    

    def test(self):
        self.model.eval()
        testloader = DataLoader(self.testset, batch_size=self.batch_size, shuffle=True)
        # testloader = torch.utils.data.DataLoader(self.testset, batch_size=self.batch_size, shuffle=True)    
        # plt.scatter(np.array(y), np.array(out))
        # plt.plot(np.array([0, 1]), np.array([0, 1]))
        # plt.xlabel('y')
        # plt.ylabel('out')
        # plt.title('Prediction of AIG evaluation')
        # plt.savefig('./graph/prediction.png')
        loss = 0
        for data in testloader:
            data = data.to(device)
            out = self.model(data)
            loss += self.loss(out, data.y)
        return loss
    
    def final_test(self):
        self.model.eval()
        y = []
        out = []
        testloader = DataLoader(self.testset, batch_size=self.batch_size, shuffle=True)   
        for data in testloader:
            data = data.to(device)
            new_out = self.model(data, data.batch)
            out.append(new_out)
            y.append(data.y)
        plt.scatter(np.array(y), np.array(out))
        plt.plot(np.array([0, 1]), np.array([0, 1]))
        plt.xlabel('y')
        plt.ylabel('out')
        plt.title('Prediction of AIG evaluation')
        plt.savefig('./prediction.png')

if __name__ == '__main__':
    aig = AIGTrain()
    aig.train()
    
