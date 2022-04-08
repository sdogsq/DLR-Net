from optparse import OptParseError
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from time import time
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader
from src.Noise import Noise
from src.Rule import Rule
from src.SPDEs import SPDE
from src.Model import Model
from src.Graph import Graph
from model.RSlayer import ParabolicIntegrate


def parabolic_graph(data):
    R = Rule(kernel_deg = 2, noise_deg = -1.5, free_num = 3) # create rule with additive width 3
    R.add_component(1, {'xi':1}) # add multiplicative width = 1

    I = SPDE(BC = 'P', T = data['T'], X = data['X']).Integrate_Parabolic_trees # initialize integration map I

    G = Graph(integration = I, rule = R, height = 2, deg = 5) # initialize graph

    extra_deg = 2
    key = "I_c[u_0]"

    graph = G.create_model_graph(data['W'][0], extra_planted = {key: data['W'][0]}, extra_deg = {key : extra_deg})
    return graph

def xi_graph(graph): # select subgraph without u_0
    return {key : graph[key] for key in graph.keys() if 'u_0' not in key}


class rsnet(nn.Module):
    def __init__(self, graph, T, X):
        super().__init__()
        self.graph = graph
        self.T = T
        self.X = X
        self.RSLayer0 = ParabolicIntegrate(graph, T = T, X = X).to(device)
        self.fc1 = nn.Linear(len(self.graph), 1)
        self.RSLayer1 = ParabolicIntegrate(graph, T = T, X = X).to(device)
        self.fc2 = nn.Linear(len(self.graph), 1)
        
    def forward(self, U0, W, Feature_Xi = None):
        '''
        U0: [B, N] initial condition
        W: [B, T, N] realizations of white noise
        Feature_Xi: [B, T, N, F] pre-computed features only containing Xi
        '''
        F1 = self.RSLayer0(W, U0, Feature_Xi) # [B, T, N, F]
        F2 = self.fc1(F1[:,-1,:,:]).squeeze() # [B, N]
        F3 = self.RSLayer1(W, F2, Feature_Xi)
        F4 = self.fc2(F3[:,-1,:,:]).squeeze() # [B, N]
        return F4 # [B, N]


def train(model, device, train_loader, optimizer, criterion, epoch):
    model.train()
    train_loss = 0
    for batch_idx, (W, U0, F_Xi, Y) in enumerate(train_loader):
        # data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(U0, W, F_Xi)
        loss = criterion(output, Y[:,-1,:])
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    print('Train Epoch: {} \tLoss: {:.6f}'.format(epoch, train_loss / len(train_loader.dataset)))

def test(model, device, test_loader, criterion):
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for batch_idx, (W, U0, F_Xi, Y) in enumerate(test_loader):
            # data, target = data.to(device), target.to(device)
            output = model(U0, W, F_Xi)
            loss = criterion(output, Y[:,-1,:])
            test_loss += loss.item()
    print('Test Loss: {:.6f}'.format(test_loss / len(test_loader.dataset)))

if __name__ == '__main__':
    data = np.load("./data/parabolic_additive.npz")
    train_W, test_W, train_U0, test_U0, train_Y, test_Y = train_test_split(data['W'], data['U0'], data['Soln_add'], train_size=1000, shuffle=False)
    print(f"train_W: {train_W.shape}, train_U0: {train_U0.shape}, train_Y: {train_Y.shape}")
    print(f"test_W: {test_W.shape}, test_U0: {test_U0.shape}, test_Y: {test_Y.shape}")

    graph = parabolic_graph(data)
    print(graph)
    device = torch.device('cuda:0')
    train_W = torch.Tensor(train_W).to(device)
    train_U0 = torch.Tensor(train_U0).to(device)
    train_Y = torch.Tensor(train_Y).to(device)

    test_W = torch.Tensor(test_W).to(device)
    test_U0 = torch.Tensor(test_U0).to(device)
    test_Y = torch.Tensor(test_Y).to(device)

    InteLayer = ParabolicIntegrate(graph, T = data['T'], X = data['X']).to(device)
    Feature_Xi = InteLayer(train_W)
    trainset = TensorDataset(train_W, train_U0, Feature_Xi, train_Y)
    train_loader = DataLoader(trainset, batch_size=32, shuffle=True)

    test_F_Xi = InteLayer(test_W)
    testset = TensorDataset(test_W, test_U0, test_F_Xi, test_Y)
    test_loader = DataLoader(testset, batch_size=100, shuffle=True)
    torch.autograd.set_detect_anomaly(True)

    model = rsnet(graph, data['T'], data['X']).to(device)

    lossfn = nn.MSELoss(reduction = 'sum')
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(1, 50):
        train(model, device, train_loader, optimizer, lossfn, epoch)
        test(model, device, test_loader, lossfn)
        # scheduler.step()

