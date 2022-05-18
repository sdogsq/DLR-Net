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
from model.RSlayer_2d import ParabolicIntegrate_2d, FNO_layer
from utils import LpLoss, cacheXiFeature_2d

parser = argparse.ArgumentParser()
parser.add_argument('-N', '--N', type=int, default=1000, metavar='N',
                    help = 'number of training realizations')
parser.add_argument('-k', '--k', type=float, default=0.1, metavar='N',
                    help = 'parameter k in U0')
parser.add_argument('-nu', '--nu', type=float, default=1e-4, metavar='N',
                    help = 'viscosity nu')
parser.add_argument('--batch_size', type=int, default=32, metavar='N',
                    help='input batch size for training (default: 32)')
parser.add_argument('--epochs', type=int, default=500, metavar='N',
                    help='number of epochs to train (default: 500)')
parser.add_argument('--lr', type=float, default=1e-3, metavar='N',
                    help='learning rate')
parser.add_argument('--weight_decay', type=float, default=1e-5, metavar='N',
                    help='weight decay')
args = parser.parse_args()

def NS_graph(data):
    # create rule with additive width 2
    R = Rule(kernel_deg = 2, noise_deg = -2, free_num = 2) 

    # initialize integration map I
    I = SPDE(BC = 'P', T = data['T'], X = data['X']).Integrate_Parabolic_trees_2d

    G = Graph(integration = I, rule = R, height = 2, deg = 7.5, derivative = False)  # initialize graph

    extra_deg = 2
    key = "I_c[u_0]"

    graph = G.create_model_graph_2d(data['W'][0], data['X'],
                                 extra_planted = {key: data['W'][0]},
                                 extra_deg = {key : extra_deg})
    # delete unused derivative features
    used = set().union(*[{IZ for IZ in graph[key].keys()} for key in graph.keys() if key[:2] == 'I['])
    graph = {IZ: graph[IZ] for IZ in graph if IZ[:2] =='I[' or IZ in used}

    return graph


def train(model, device, train_loader, optimizer, criterion, epoch):
    model.train()
    train_loss = 0
    for batch_idx, (W, U0, F_Xi, Y) in enumerate(train_loader):
        W, U0, F_Xi, Y = W.to(device), U0.to(device), F_Xi.to(device), Y.to(device)
        optimizer.zero_grad()
        output = model(U0, W, F_Xi)
        loss = criterion(output[:,1:,...], Y[:,1:,...])
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
            loss = criterion(output[:,1:,...], Y[:,1:,...])
            test_loss += loss.item()
    return test_loss / len(test_loader.dataset)

def Inference(model, device, test_loader, criterion):
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for batch_idx, (W, U0, F_Xi, Y) in enumerate(test_loader):
            W, U0, Y = W.to(device), U0.to(device), Y.to(device)
            output = model(U0, W)
            loss = criterion(output[:,1:,...], Y[:,1:,...])
            test_loss += loss.item()
    return test_loss / len(test_loader.dataset)

class rsnet_2d(nn.Module):
    def __init__(self, graph, T, X, Y):
        super().__init__()
        self.graph = graph
        self.F = len(graph) - 1
        self.FU0 = len([key for key in graph.keys() if 'u_0' in key])
        self.T = len(T)
        self.X = len(X)
        self.Y = len(Y)
        self.RSLayer0 = ParabolicIntegrate_2d(graph, T = T, X = X, Y = Y, eps = args.nu)
        self.down0 = nn.Sequential(
            nn.Linear(self.F, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 1)
        )
        self.down1 = nn.Sequential(
            nn.Conv1d(self.T * self.F, 32 * self.T, kernel_size=1, groups = self.T),
            nn.ReLU(inplace=True),
            nn.Conv1d(32 * self.T, self.T, kernel_size = 1, groups = self.T)
        )
        self.L = 4
        self.padding = 6
        # modes1, modes2, width = 12, 12, 32
        modes1, modes2, modes3, width = 8, 8, 8, 20
        self.net = [FNO_layer(modes1, modes2, modes3, width) for i in range(self.L-1)]
        self.net += [FNO_layer(modes1, modes2, modes3, width, last=True)]
        self.net = nn.Sequential(*self.net)
        self.fc0 = nn.Linear(1 + self.F + self.FU0 + 3, width)
        self.decoder = nn.Sequential(
            nn.Linear(width, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 1)
        )

    def get_grid(self, shape, device):
        batchsize, size_x, size_y, size_z = shape[0], shape[1], shape[2], shape[3]
        gridx = torch.tensor(np.linspace(0, 1, size_x), dtype=torch.float)
        gridx = gridx.reshape(1, size_x, 1, 1, 1).repeat([batchsize, 1, size_y, size_z, 1])
        gridy = torch.tensor(np.linspace(0, 1, size_y), dtype=torch.float)
        gridy = gridy.reshape(1, 1, size_y, 1, 1).repeat([batchsize, size_x, 1, size_z, 1])
        gridz = torch.tensor(np.linspace(0, 1, size_z), dtype=torch.float)
        gridz = gridz.reshape(1, 1, 1, size_z, 1).repeat([batchsize, size_x, size_y, 1, 1])
        return torch.cat((gridx, gridy, gridz), dim=-1).to(device)
        
    # def get_grid(self, shape, device):
    #     batchsize, size_x, size_y = shape[0], shape[1], shape[2]
    #     gridx = torch.tensor(np.linspace(0, 1, size_x), dtype=torch.float)
    #     gridx = gridx.reshape(1, size_x, 1, 1).repeat([batchsize, 1, size_y, 1])
    #     gridy = torch.tensor(np.linspace(0, 1, size_y), dtype=torch.float)
    #     gridy = gridy.reshape(1, 1, size_y, 1).repeat([batchsize, size_x, 1, 1])
    #     return torch.cat((gridx, gridy), dim=-1).to(device)

    def forward(self, U0, W, Feature_Xi = None):
        '''
        U0: [B, X, Y] initial condition
        W: [B, T, X, Y] realizations of white noise
        Feature_Xi: [B, T, X, Y, F] pre-computed features only containing Xi
        '''
        U0 = self.RSLayer0.I_c(U0) # [B, T, X, Y]
        R1 = self.RSLayer0(W = W, Latent = U0, XiFeature = Feature_Xi, returnFeature = 'normal')
        O1 = R1 # [B, T, X, Y, F + 1] with xi
        U0 = self.down0(O1).squeeze() # [B, T, X, Y]
        R1 = self.RSLayer0(W = W, Latent = U0, XiFeature = Feature_Xi, returnFeature = 'U0')
        grid = self.get_grid(R1.shape, R1.device)
        R1 = torch.cat((R1, grid), dim=-1) # [B, T, X, Y, F + FU0 + 3]
        R1 = self.fc0(R1)
        R1 = R1.permute(0, 4, 2, 3, 1) # [B, Hidden, X, Y, T]
        R1 = F.pad(R1, [0,self.padding]) 
        # print("C")
        R1 = self.net(R1)
        R1 = R1[..., :-self.padding]
        R1 = R1.permute(0, 4, 2, 3, 1) # [B, T, X, Y, Hidden]
        R1 = self.decoder(R1) # [B, T, X, Y, 1]
        # print("D")
        return R1.squeeze() # [B, T, X, Y]


if __name__ == '__main__':
    data = np.load(f"./data/NS.npz")
    # Solution = Soln, W = forcing, T = time.numpy(), X = X, Y = Y, U0 = IC
    train_W, test_W, train_U0, test_U0, train_Y, test_Y = train_test_split(data['W'],
                                                                           data['U0'],
                                                                           data['Solution'],
                                                                           train_size=args.N, 
                                                                           shuffle=False)

    print(f"train_W: {train_W.shape}, train_U0: {train_U0.shape}, train_Y: {train_Y.shape}")
    print(f"test_W: {test_W.shape}, test_U0: {test_U0.shape}, test_Y: {test_Y.shape}")
    print(f"data['T']: {data['T'].shape}, data['X']: {data['X'].shape}, data['Y']: {data['Y'].shape}")

    graph = NS_graph(data)
    for key, item in graph.items():
        print(key, item)
    print(len(graph))
    G2 = {IZ: graph[IZ] for IZ in graph if IZ[1] != "1" and IZ[1] != "2"}
    print( graph.keys() - G2.keys())
    print(len(G2))

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    train_W, train_U0, train_Y = torch.Tensor(train_W), torch.Tensor(train_U0), torch.Tensor(train_Y)
    test_W, test_U0, test_Y = torch.Tensor(test_W), torch.Tensor(test_U0), torch.Tensor(test_Y)


    # cache Xi fatures
    Feature_Xi = cacheXiFeature_2d(graph, T = data['T'], X = data['X'][:,0], Y = data['Y'][0],\
                                   W = train_W, eps = args.nu, device = device)
    print(Feature_Xi.shape)
    trainset = TensorDataset(train_W, train_U0, Feature_Xi, train_Y)
    train_loader = DataLoader(trainset,
                              batch_size=args.batch_size,
                              shuffle=True,
                              pin_memory=True,
                              persistent_workers=True,
                              drop_last=True,
                              num_workers=4)

    test_F_Xi = cacheXiFeature_2d(graph, T = data['T'], X = data['X'][:,0], Y = data['Y'][0],\
                                  W = test_W, eps = args.nu, device = device)

    testset = TensorDataset(test_W, test_U0, test_F_Xi, test_Y)
    test_loader = DataLoader(testset,
                             batch_size=16,
                             shuffle=True,
                             pin_memory=True,
                             persistent_workers=True,
                             drop_last=False,
                             num_workers=4)


    # ------ begin training ------
    model = rsnet_2d(graph, data['T'], X = data['X'][:,0], Y = data['Y'][0]).to(device)

    lossfn = LpLoss(size_average=False)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs, verbose = False)
    #torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.5)

    wandb.init(project="DeepRS", entity="sdogsq", config=args)

    trainTime = 0
    inferenceTime = 0
    for epoch in range(1, args.epochs + 1):
        tik = time.time()
        trainLoss = train(model, device, train_loader, optimizer, lossfn, epoch)
        tok = time.time()
        testLoss = test(model, device, test_loader, lossfn)
        scheduler.step()

        trainTime += tok - tik

        tik = time.time()
        inferenceLoss = Inference(model, device, test_loader, lossfn)
        tok = time.time()
        inferenceTime += tok - tik

        wandb.log({"Train Loss": trainLoss, "Test Loss": testLoss})
        print('Epoch: {:04d} \tTrain Loss: {:.6f} \tTest Loss: {:.6f} \tTime per Epoch: {:.3f} \tInference Time per Epoch: {:.3f}'\
              .format(epoch, trainLoss, testLoss, trainTime / epoch, inferenceTime / epoch))