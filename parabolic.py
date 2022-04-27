import time
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader
import wandb
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
args = parser.parse_args()

def parabolic_graph(data):
    # create rule with additive width 3
    R = Rule(kernel_deg = 2, noise_deg = -1.5, free_num = 3) 
    # add multiplicative width = 1
    R.add_component(1, {'xi':1}) 
    # initialize integration map I
    I = SPDE(BC = 'P', T = data['T'], X = data['X']).Integrate_Parabolic_trees 

    G = Graph(integration = I, rule = R, height = 2, deg = 5) # initialize graph

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
        self.T = T
        self.X = X
        self.RSLayer0 = ParabolicIntegrate(graph, T = T, X = X)
        self.down0 = nn.Sequential(
            nn.Linear(1+self.F, 32),
            nn.GELU(),
            nn.Linear(32, 1)
        )
        self.L = 4
        self.padding = 6 
        modes1, modes2, width = 32, 24, 32
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
        R1 = self.RSLayer0(W = W, U0 = U0, XiFeature = Feature_Xi) # [B, T, N, F + 1]

        O1 = R1[..., 1:] # [B, T, N, F],  drop Xi
        U0 = self.down0(torch.cat((U0.unsqueeze(2), O1[:,-1,:,:]), dim = 2)).squeeze() # [B, N]
        R1 = self.RSLayer0(W = W, U0 = U0, XiFeature = Feature_Xi, returnU0Feature = True)

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
        return R1[:,-1,:,:].squeeze() # [B, N]


def train(model, device, train_loader, optimizer, criterion, epoch):
    model.train()
    train_loss = 0
    for batch_idx, (W, U0, F_Xi, Y) in enumerate(train_loader):
        W, U0, F_Xi, Y = W.to(device), U0.to(device), F_Xi.to(device), Y.to(device)
        optimizer.zero_grad()
        output = model(U0, W, F_Xi)
        loss = criterion(output, Y[:,-1,:])
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
            loss = criterion(output, Y[:,-1,:])
            test_loss += loss.item()
    return test_loss / len(test_loader.dataset)

if __name__ == '__main__':
    data = np.load(f"./data/parabolic_additive_{args.N}_{args.k}.npz")
    train_W, test_W, train_U0, test_U0, train_Y, test_Y = train_test_split(data['W'],
                                                                           data['U0'],
                                                                           data['Soln_add'],
                                                                           train_size=args.N, 
                                                                           shuffle=False)

    print(f"train_W: {train_W.shape}, train_U0: {train_U0.shape}, train_Y: {train_Y.shape}")
    print(f"test_W: {test_W.shape}, test_U0: {test_U0.shape}, test_Y: {test_Y.shape}")

    graph = parabolic_graph(data)
    print(graph)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    train_W, train_U0, train_Y = torch.Tensor(train_W), torch.Tensor(train_U0), torch.Tensor(train_Y)
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

    test_F_Xi = cacheXiFeature(graph, T = data['T'], X = data['X'], W = test_W, device = device)

    testset = TensorDataset(test_W, test_U0, test_F_Xi, test_Y)
    test_loader = DataLoader(testset,
                             batch_size=100,
                             shuffle=True,
                             pin_memory=True,
                             persistent_workers=True,
                             drop_last=False,
                             num_workers=4)

    # ------ begin training ------
    model = rsnet(graph, data['T'], data['X']).to(device)

    lossfn = LpLoss(size_average=False)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs, verbose = False)
    #torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.5)

    wandb.init(project="DeepRS", entity="sdogsq", config=args)

    trainTime = 0
    for epoch in range(1, args.epochs + 1):
        tik = time.time()
        trainLoss = train(model, device, train_loader, optimizer, lossfn, epoch)
        tok = time.time()
        testLoss = test(model, device, test_loader, lossfn)
        scheduler.step()

        trainTime += tok - tik
        wandb.log({"Train Loss": trainLoss, "Test Loss": testLoss})
        print('Epoch: {:04d} \tTrain Loss: {:.6f} \tTest Loss: {:.6f} \tTime per Epoch: {:.3f}'\
              .format(epoch, trainLoss, testLoss, trainTime / epoch))
        # if (epoch == 5):
        #     profile(model, device, train_loader, optimizer, lossfn, epoch)
        #     break