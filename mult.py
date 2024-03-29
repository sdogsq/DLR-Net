import time
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader
from src.Rule import Rule
from src.SPDEs import SPDE
from src.Graph import Graph
from model.RSlayer import ParabolicIntegrate, FNO_layer
from utils import LpLoss, cacheXiFeature

parser = argparse.ArgumentParser()
parser.add_argument('-N', '--N', type=int, default=1000, metavar='N',
                    help = 'number of training realizations')
parser.add_argument('-k', '--k', type=float, default=0.1, metavar='N',
                    help = 'parameter k in U0')
parser.add_argument('--batch_size', type=int, default=32, metavar='N',
                    help='input batch size for training (default: 32)')
parser.add_argument('--epochs', type=int, default=500, metavar='N',
                    help='number of epochs to train (default: 500)')
parser.add_argument('--lr', type=float, default=1e-3, metavar='N',
                    help='learning rate')
parser.add_argument('--weight_decay', type=float, default=1e-5, metavar='N',
                    help='weight decay')
parser.add_argument('-H', '--height', type=int, default=2, metavar='N',
                    help = 'feature height')
parser.add_argument('--nlog', type=int, default=5, metavar='N',
                    help='frequency of log printing (default: 5)')
args = parser.parse_args()

def parabolic_graph(data):
    # create rule with additive width 3
    R = Rule(kernel_deg = 2, noise_deg = -1.5, free_num = 3) 
    # add multiplicative width = 1
    R.add_component(1, {'xi':1}) 
    # initialize integration map I
    I = SPDE(BC = 'P', T = data['T'], X = data['X']).Integrate_Parabolic_trees 

    G = Graph(integration = I, rule = R, height = args.height, deg = 5) # initialize graph

    extra_deg = 2
    key = "I_c[u_0]"

    graph = G.create_model_graph(data['W'][0],
                                 extra_planted = {key: data['W'][0]},
                                 extra_deg = {key : extra_deg})
    return graph

class rsnet(nn.Module):
    def __init__(self, graph, T, X):
        super().__init__()
        self.graph = graph
        self.F = len(graph) - 1
        self.FU0 = len([key for key in graph.keys() if 'u_0' in key])
        self.T = len(T)
        self.X = len(X)
        self.RSLayer0 = ParabolicIntegrate(graph, T = T, X = X)
        self.down0 = nn.Sequential(
            nn.Linear(self.F, 32),
            nn.GELU(),
            nn.Linear(32, 1)
        )
        self.L = 4
        self.padding = 6 
        modes1, modes2, width = 16, 16, 8
        self.net = [FNO_layer(modes1, modes2, width) for i in range(self.L-1)]
        self.net += [FNO_layer(modes1, modes2, width, last=True)]
        self.net = nn.Sequential(*self.net)
        self.fc0 = nn.Linear(self.F + self.FU0 + 2, width)
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
        U0 = self.RSLayer0.I_c(U0) # [B, T, N]

        R1 = self.RSLayer0(W = W, Latent = U0, XiFeature = Feature_Xi) # [B, T, N, F + 1]

        O1 = R1[..., 1:] # [B, T, N, F],  drop Xi
        U0 = self.down0(O1).squeeze() # [B, T, N]
        R1 = self.RSLayer0(W = W, Latent = U0, XiFeature = Feature_Xi, returnU0Feature = True)

        R1 = torch.cat((O1, R1), dim = 3) # [B,T,N, F + FU0]
        grid = self.get_grid(R1.shape, R1.device)
        R1 = torch.cat((R1, grid), dim=-1)
        R1 = self.fc0(R1)
        R1 = R1.permute(0, 3, 2, 1) # [B, Hidden, N, T]
        R1 = F.pad(R1, [0,self.padding]) 
        R1 = self.net(R1)
        R1 = R1[..., :-self.padding]
        R1 = R1.permute(0, 3, 2, 1) # [B, T, N, Hidden]
        R1 = self.decoder(R1) # [B, T, N, 1]
        return R1.squeeze() # [B, T, N]


def train(model, device, train_loader, optimizer, criterion, epoch):
    model.train()
    train_loss = 0
    for batch_idx, (W, U0, F_Xi, Y) in enumerate(train_loader):
        W, U0, F_Xi, Y = W.to(device), U0.to(device), F_Xi.to(device), Y.to(device)
        optimizer.zero_grad()
        output = model(U0, W, F_Xi)
        loss = criterion(output[:,1:,:], Y[:,1:,:])
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    return train_loss / len(train_loader.dataset)

def test(model, device, test_loader, criterion):
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for batch_idx, (W, U0, F_Xi, Y) in enumerate(test_loader):
            W, U0, F_Xi, Y = W.to(device), U0.to(device), F_Xi.to(device), Y.to(device)
            output = model(U0, W, F_Xi)
            loss = criterion(output[:,1:,:], Y[:,1:,:])
            test_loss += loss.item()
    return test_loss / len(test_loader.dataset)

if __name__ == '__main__':
    data = np.load(f"./data/parabolic_multiplicative_{args.N}_{args.k}.npz")
    train_W, test_W, train_U0, test_U0, train_Y, test_Y = train_test_split(data['W'],
                                                                           data['U0'],
                                                                           data['Solution'],
                                                                           train_size=args.N, 
                                                                           shuffle=False)
    val_W, test_W, val_U0, test_U0, val_Y, test_Y = train_test_split(test_W,
                                                                     test_U0,
                                                                     test_Y,
                                                                     train_size=0.5,
                                                                     shuffle=False)

    print(f"train_W: {train_W.shape}, train_U0: {train_U0.shape}, train_Y: {train_Y.shape}")
    print(f"val_W: {val_W.shape}, val_U0: {val_U0.shape}, val_Y: {val_Y.shape}")
    print(f"test_W: {test_W.shape}, test_U0: {test_U0.shape}, test_Y: {test_Y.shape}")

    graph = parabolic_graph(data)
    print(graph)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    train_W, train_U0, train_Y = torch.Tensor(train_W), torch.Tensor(train_U0), torch.Tensor(train_Y)
    val_W, val_U0, val_Y = torch.Tensor(val_W), torch.Tensor(val_U0), torch.Tensor(val_Y)
    test_W, test_U0, test_Y = torch.Tensor(test_W), torch.Tensor(test_U0), torch.Tensor(test_Y)

    # cache Xi fatures
    Feature_Xi = cacheXiFeature(graph, T = data['T'], X = data['X'], W = train_W, device = device)

    trainset = TensorDataset(train_W, train_U0, Feature_Xi, train_Y)
    train_loader = DataLoader(trainset,
                              batch_size=args.batch_size,
                              shuffle=True,
                              pin_memory=True,
                              persistent_workers=True,
                              drop_last=True,
                              num_workers=4)

    val_F_Xi = cacheXiFeature(graph, T = data['T'], X = data['X'], W = val_W, device = device)

    valset = TensorDataset(val_W, val_U0, val_F_Xi, val_Y)
    val_loader = DataLoader(valset,
                             batch_size=100,
                             shuffle=True,
                             pin_memory=True,
                             persistent_workers=True,
                             drop_last=False,
                             num_workers=4)

    # ------ begin training ------
    model = rsnet(graph, data['T'], data['X']).to(device)
    def count_params(model):
        c = 0
        from functools import reduce
        import operator
        for p in list(model.parameters()):
            c += reduce(operator.mul, list(p.size()))
        return c
    print("Params", count_params(model))
    lossfn = LpLoss(size_average=False)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs, verbose = False)
    #torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.5)

    trainTime = 0
    for epoch in range(1, args.epochs + 1):
        tik = time.time()
        trainLoss = train(model, device, train_loader, optimizer, lossfn, epoch)
        tok = time.time()
        trainTime += tok - tik
        scheduler.step()

        if (epoch-1) % args.nlog == 0:
            testLoss = test(model, device, val_loader, lossfn)

            print('Epoch: {:04d} \tTrain Loss: {:.6f} \tVal Loss: {:.6f} \t\
                   Training Time per Epoch: {:.3f} \t'\
                   .format(epoch, trainLoss, testLoss, trainTime / epoch))

    ## ----------- test ------------

    test_F_Xi = cacheXiFeature(graph, T = data['T'], X = data['X'], W = test_W, device = device)

    testset = TensorDataset(test_W, test_U0, test_F_Xi, test_Y)
    test_loader = DataLoader(testset,
                             batch_size=100,
                             shuffle=True,
                             pin_memory=True,
                             persistent_workers=True,
                             drop_last=False,
                             num_workers=4)

    testLoss = test(model, device, test_loader, lossfn)
    print(f'Final Test Loss: {testLoss:.6f}')