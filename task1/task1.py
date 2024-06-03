from GNN import GCN
import torch
from dataset import *
from matplotlib import pyplot as plt
import tqdm
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
MODEL_PATH = './model/'
class AIGTrain():
    def __init__(self):
        self.model = GCN().to(device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.01, weight_decay=5e-4)
        self.loss = torch.nn.MSELoss()
        self.batch_size = 32
        self.trainset, self.testset = get_dataset()
    
    def train(self):
        trainloader = torch.utils.data.DataLoader(self.trainset, batch_size=self.batch_size, shuffle=True)
        self.model.train()
        loss = int('inf')
        for i in tqdm(range(100)):
            for data in trainloader:
                data = data.to(device)
                self.optimizer.zero_grad()
                out = self.model(data)
                loss = self.loss(out, data.y)
                loss.backward()
                self.optimizer.step()
            new_loss = self.test()
            if new_loss < loss:
                loss = new_loss
                torch.save(self.model.state_dict(), MODEL_PATH + str(i) + '.pth')

    def test(self):
        self.model.eval()
        testloader = torch.utils.data.DataLoader(self.testset, batch_size=self.batch_size, shuffle=True)    
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

if __name__ == '__main__':
    aig = AIGTrain()
    aig.train()
