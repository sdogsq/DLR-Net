from itertools import groupby
from optparse import OptParseError
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from time import time
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader
from src.Noise import Noise
from src.Rule import Rule
from src.SPDEs import SPDE
from src.Model import Model
from src.Graph import Graph
from model.RSlayer import ParabolicIntegrate, FNO_layer
from utils import LpLoss

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
        self.RSLayer0 = ParabolicIntegrate(graph, T = T, X = X)
        self.down1 = nn.Sequential(
            nn.Linear(len(self.graph), 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
        self.L = 4
        self.padding = 6 
        modes1, modes2, width = 32, 24, 32
        self.net = [FNO_layer(modes1, modes2, width) for i in range(self.L-1)]
        self.net += [FNO_layer(modes1, modes2, width, last=True)]
        self.net = nn.Sequential(*self.net)
        self.fc0 = nn.Linear(len(self.graph) + 2, width)
        self.decoder = nn.Sequential(
            nn.Linear(width, 128),
            nn.GELU(),
            nn.Linear(128, 1)
        )
        
    def get_grid(self, shape, device):
        batchsize, size_x, size_t = shape[0], shape[1], shape[2]
        gridx = torch.tensor(np.linspace(0, 1, size_x), dtype=torch.float)
        gridx = gridx.reshape(1, size_x, 1, 1).expand([batchsize, size_x, size_t, 1])
        gridt = torch.tensor(np.linspace(0, 1, size_t), dtype=torch.float)
        gridt = gridt.reshape(1, 1, size_t, 1).expand([batchsize, size_x, size_t, 1])
        return torch.cat((gridx, gridt), dim=-1).to(device)

    def forward(self, U0, W, Feature_Xi = None):
        '''
        U0: [B, N] initial condition
        W: [B, T, N] realizations of white noise
        Feature_Xi: [B, T, N, F] pre-computed features only containing Xi
        '''
        R1 = self.RSLayer0(W = W, U0 = U0, Xi_feature = Feature_Xi) # [B, T, N, F]
        # U0 = U0 + self.down1(R1[:,-1,:,:]).squeeze() # [B, N]
        # R1 = self.RSLayer0(W = W, U0 = U0, Xi_feature = Feature_Xi)
        grid = self.get_grid(R1.shape, R1.device)
        R1 = torch.cat((R1, grid), dim=-1)
        R1 = R1.to(torch.float32)
        R1 = self.fc0(R1)
        R1 = R1.permute(0, 3, 2, 1) # [B, Hidden, N, T]
        R1 = F.pad(R1, [0,self.padding]) 
        R1 = self.net(R1)
        R1 = R1[..., :-self.padding]
        R1 = R1.permute(0, 3, 2, 1) # [B, T, N, Hidden]
        R1 = self.decoder(R1) # [B, T, N, 1]
        return R1[:,-1,:,:].squeeze() # [B, N]

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

# use torch.profiler.profile to get the time of each layer
def profile(model, device, train_loader, optimizer, criterion, epoch):
    model.train()
    for batch_idx, (W, U0, F_Xi, Y) in enumerate(train_loader):
        with torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
            record_shapes = False,
            with_modules = True,
            with_stack = True,
            profile_memory=False,
            with_flops = False) \
        as prof:
            output = model(U0, W, F_Xi)
        print(prof.key_averages().table(sort_by="self_cpu_time_total"))
        break

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
    data = np.load("./data/parabolic_additive_randomU0.npz")
    train_W, test_W, train_U0, test_U0, train_Y, test_Y = train_test_split(data['W'], data['U0'], data['Soln_add'], train_size=1000, shuffle=False)
    print(f"train_W: {train_W.shape}, train_U0: {train_U0.shape}, train_Y: {train_Y.shape}")
    print(f"test_W: {test_W.shape}, test_U0: {test_U0.shape}, test_Y: {test_Y.shape}")

    graph = parabolic_graph(data)
    print(graph)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    train_W = torch.Tensor(train_W).to(device)
    train_U0 = torch.Tensor(train_U0).to(device)
    train_Y = torch.Tensor(train_Y).to(device)

    test_W = torch.Tensor(test_W).to(device)
    test_U0 = torch.Tensor(test_U0).to(device)
    test_Y = torch.Tensor(test_Y).to(device)

    # BEGIN
    InteLayer = ParabolicIntegrate(graph, T = data['T'], X = data['X']).to(device)
    Feature_Xi = InteLayer(train_W) # cache the features of Xi

    trainset = TensorDataset(train_W, train_U0, Feature_Xi, train_Y)
    train_loader = DataLoader(trainset, batch_size=20, shuffle=True)

    test_F_Xi = InteLayer(test_W)
    testset = TensorDataset(test_W, test_U0, test_F_Xi, test_Y)
    test_loader = DataLoader(testset, batch_size=100, shuffle=False)

    model = rsnet(graph, data['T'], data['X']).to(device)

    lossfn = LpLoss(size_average=False)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.5)

    for epoch in range(1, 500):
        train(model, device, train_loader, optimizer, lossfn, epoch)
        test(model, device, test_loader, lossfn)
        scheduler.step()
        # if (epoch == 5):
        #     profile(model, device, train_loader, optimizer, lossfn, epoch)
        #     break

