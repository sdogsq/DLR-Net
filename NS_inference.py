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
from utils import LpLoss, cacheXiFeature_2d, mkdir, MatReader

parser = argparse.ArgumentParser()
parser.add_argument('-N', '--N', type=int, default=1000, metavar='N',
                    help = 'number of training realizations')
parser.add_argument('-T', '--T', type=float, default=1, metavar='N',
                    help = 'parameter k in U0')
parser.add_argument('-nu', '--nu', type=float, default=1e-4, metavar='N',
                    help = 'viscosity nu')
parser.add_argument('-H', '--height', type=int, default=2, metavar='N',
                    help = 'feature height')
parser.add_argument('--sub_x', type=int, default=4, metavar='N',
                    help = 'space discretilization')
parser.add_argument('--sub_t', type=int, default=1, metavar='N',
                    help = 'time discretilization')
parser.add_argument('--fixU0', action='store_true',
                    help = 'fixU0 and generate xi->u data')
parser.add_argument('-bs', '--batch_size', type=int, default=64, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--epochs', type=int, default=500, metavar='N',
                    help='number of epochs to train (default: 500)')
parser.add_argument('--nlog', type=int, default=5, metavar='N',
                    help='frequency of log printing (default: 5)')
parser.add_argument('--lr', type=float, default=1e-2, metavar='N',
                    help='learning rate')
parser.add_argument('--weight_decay', type=float, default=1e-5, metavar='N',
                    help='weight decay')
args = parser.parse_args()

def NS_graph(data):
    # create rule with additive width 2
    R = Rule(kernel_deg = 2, noise_deg = -2, free_num = 2) 

    # initialize integration map I
    I = SPDE(BC = 'P', T = data['T'], X = data['X']).Integrate_Parabolic_trees_2d

    G = Graph(integration = I, rule = R, height = args.height, deg = 7.5, derivative = True)  # initialize graph

    extra_deg = 2
    key = "I_c[u_0]"
    SZ = data['X'].shape
    graph = G.create_model_graph_2d(np.zeros((len(data['T']),*SZ)), data['X'],
                                 extra_planted = {key: np.zeros((len(data['T']),*SZ))},
                                 extra_deg = {key : extra_deg})
    # delete unused derivative features
    used = set().union(*[{IZ for IZ in graph[key].keys()} for key in graph.keys() if key[:2] == 'I['])
    graph = {IZ: graph[IZ] for IZ in graph if IZ[:2] =='I[' or IZ in used}
    if (key not in graph.keys()):
        graph = list(graph.items())
        graph.insert(1,(key,dict()))
        graph = dict(graph)
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

def saveplot(pred, u, epoch):
    import matplotlib.pyplot as plt
    T = pred.shape[1]
    fig, ax = plt.subplots(2,T,figsize=(2*T,4))
    for i in range(T):
        if torch.is_tensor(u):
            ax[0][i].contourf(u[0,i,...].detach().cpu().numpy())
            ax[1][i].contourf(pred[0,i,...].detach().cpu().numpy())
        else:
            ax[0][i].contourf(u[0,i,...])
            ax[1][i].contourf(pred[0,i,...])
        ax[0][i].set_title(f't = {i}')
    plt.savefig(f"./fig/{epoch}.pdf", bbox_inches='tight')
    plt.clf()

def test(model, device, test_loader, criterion, epoch):
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for batch_idx, (W, U0, F_Xi, Y) in enumerate(test_loader):
            W, U0, F_Xi, Y = W.to(device), U0.to(device), F_Xi.to(device), Y.to(device)
            output = model(U0, W, F_Xi)
            loss = criterion(output[:,1:,...], Y[:,1:,...])
            test_loss += loss.item()
        # saveplot(output, Y, epoch)
    return test_loss / len(test_loader.dataset)

def Inference(model, xiLayer, device, test_loader):
    from tqdm import tqdm
    for batch_idx, (W, U0, Y) in enumerate(test_loader):
        dummy_U0 = torch.rand_like(U0, dtype=torch.float, device = device)
        dummy_W = torch.rand_like(W, dtype=torch.float, device = device)
        print(U0.shape, W.shape)
        print(f"Test U0: {dummy_U0.shape}, Test W: {dummy_W.shape}")
        break

    starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    repetitions = 300
    timings=np.zeros((repetitions,1))
    # MEASURE PERFORMANCE
    model.eval()
    with torch.no_grad():
        #GPU-WARM-UP：dummy example
        for _ in tqdm(range(30)):
            xiCache = xiLayer(W = dummy_W)
            _ = model(dummy_U0, dummy_W, xiCache)
        for rep in tqdm(range(repetitions)):
            starter.record()
            xiCache = xiLayer(W = dummy_W)
            _ = model(dummy_U0, dummy_W, xiCache)
            ender.record()
            # WAIT FOR GPU SYNC
            torch.cuda.synchronize()
            curr_time = starter.elapsed_time(ender)
            timings[rep] = curr_time
    mean_syn = np.sum(timings) / repetitions
    std_syn = np.std(timings)
    print(f"Mean: {mean_syn} (ms), Std:{std_syn}")


class rsnet_2d(nn.Module):
    def __init__(self, graph, T, X, Y):
        super().__init__()
        self.graph = graph
        self.vkeys = [key for key in graph.keys() if key[-1] is ']']
        self.F = len(self.vkeys)
        self.FU0 = len([key for key in self.vkeys if 'u_0' in key])
        self.T = len(T)
        self.X = len(X)
        self.Y = len(Y)
        self.RSLayer0 = ParabolicIntegrate_2d(graph, T = T, X = X, Y = Y, eps = args.nu)
        self.down0 = nn.Sequential(
            nn.Linear(1 + self.F, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 1)
        )

        self.L = 4
        self.padding = 6
        modes1, modes2, modes3, width = 8, 8, 8, 8 #16, 16, 10, 8 #8, 8, 8, 20
        self.net = [FNO_layer(modes1, modes2, modes3, width) for i in range(self.L-1)]
        self.net += [FNO_layer(modes1, modes2, modes3, width, last=True)]
        self.net = nn.Sequential(*self.net)
        self.fc0 = nn.Linear(1 + self.F + self.FU0 + 3, width)
        self.decoder = nn.Sequential(
            nn.Linear(width, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 1)
        )
        grid = self.get_grid(self.T, self.X, self.Y)
        self.register_buffer("grid", grid)

    def get_grid(self, T, X, Y):
        batchsize, size_x, size_y, size_z = 1, T, X, Y
        gridx = torch.tensor(np.linspace(0, 1, size_x), dtype=torch.float)
        gridx = gridx.reshape(1, size_x, 1, 1, 1).repeat([batchsize, 1, size_y, size_z, 1])
        gridy = torch.tensor(np.linspace(0, 1, size_y), dtype=torch.float)
        gridy = gridy.reshape(1, 1, size_y, 1, 1).repeat([batchsize, size_x, 1, size_z, 1])
        gridz = torch.tensor(np.linspace(0, 1, size_z), dtype=torch.float)
        gridz = gridz.reshape(1, 1, 1, size_z, 1).repeat([batchsize, size_x, size_y, 1, 1])
        return torch.cat((gridx, gridy, gridz), dim=-1)
        
    def forward(self, U0, W, Feature_Xi = None):
        '''
        U0: [B, X, Y] initial condition
        W: [B, T, X, Y] realizations of white noise
        Feature_Xi: [B, T, X, Y, F] pre-computed features only containing Xi
        '''
        U0 = self.RSLayer0.I_c(U0) # [B, T, X, Y]
        R1 = self.RSLayer0(W = W, Latent = U0, XiFeature = Feature_Xi, returnFeature = 'normal')
        O1 = R1 # [B, T, X, Y, F + 1] with xi
        U0 = self.down0(O1).squeeze(-1) # [B, T, X, Y]
        R1 = self.RSLayer0(W = W, Latent = U0, XiFeature = Feature_Xi, returnFeature = 'U0')
        R1 = torch.cat((R1, O1, self.grid.expand(R1.shape[0],-1,-1,-1,-1)), dim=-1) # [B, T, X, Y, 1 + F + FU0 + 3]
        R1 = self.fc0(R1)
        R1 = R1.permute(0, 4, 2, 3, 1) # [B, Hidden, X, Y, T]
        R1 = F.pad(R1, [0,self.padding]) 
        R1 = self.net(R1)
        R1 = R1[..., :-self.padding]
        R1 = R1.permute(0, 4, 2, 3, 1) # [B, T, X, Y, Hidden]
        R1 = self.decoder(R1) # [B, T, X, Y, 1]
        return R1.squeeze() # [B, T, X, Y]

def dataloader_2d(u, xi=None, ntrain=1000, ntest=200, T=51, sub_t=1, sub_x=4):

    if xi is None:
        print('There is no known forcing')

    u0_train = u[:ntrain, ::sub_x, ::sub_x, 0]#.unsqueeze(1)
    u_train = u[:ntrain, ::sub_x, ::sub_x, :T:sub_t]

    if xi is not None:
        xi_train = xi[:ntrain, ::sub_x, ::sub_x, 0:T:sub_t]#.unsqueeze(1)
    else:
        xi_train = torch.zeros_like(u_train)

    u0_test = u[-ntest:, ::sub_x, ::sub_x, 0]#.unsqueeze(1)
    u_test = u[-ntest:, ::sub_x, ::sub_x, 0:T:sub_t]

    if xi is not None:
        xi_test = xi[-ntest:, ::sub_x, ::sub_x, 0:T:sub_t]#.unsqueeze(1)
    else:
        xi_test = torch.zeros_like(u_test)

    return (xi_train.transpose(0,3,1,2), xi_test.transpose(0,3,1,2),
            u0_train, u0_test,
            u_train.transpose(0,3,1,2), u_test.transpose(0,3,1,2))

def mat2data(reader):
    data = {}
    data['T'] = reader.read_field('t').squeeze()[:1000:10*args.sub_t].squeeze()
    data['Solution'] = reader.read_field('sol')
    data['W'] = reader.read_field('forcing')
    spoints = np.linspace(0, 1, data['W'].shape[1]//args.sub_x)
    data['Y'], data['X'] = np.meshgrid(spoints, spoints)
    return data

if __name__ == '__main__':
    print(f"args: {args}")
    reader = MatReader(f"./data/NS_{'xi' if args.fixU0 else 'u0_xi'}.mat", to_torch = False)
    print(f"Use ./data/NS_{'xi' if args.fixU0 else 'u0_xi'}.mat")
    data = mat2data(reader)
    train_W, test_W, train_U0, test_U0, train_Y, test_Y = dataloader_2d(
                                                            u=data['Solution'], xi=data['W'], ntrain=1000,
                                                            ntest = 200, T = 100,
                                                            sub_t = args.sub_t, sub_x = args.sub_x)

    print(f"train_W: {train_W.shape}, train_U0: {train_U0.shape}, train_Y: {train_Y.shape}")
    print(f"test_W: {test_W.shape}, test_U0: {test_U0.shape}, test_Y: {test_Y.shape}")
    print(f"data['T']: {data['T'].shape}, data['X']: {data['X'].shape}, data['Y']: {data['Y'].shape}")

    graph = NS_graph(data)
    for key, item in graph.items():
        print(key, item)
    print("Total Feature Number:", len(graph))

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    train_W, train_U0, train_Y = torch.Tensor(train_W), torch.Tensor(train_U0), torch.Tensor(train_Y)
    test_W, test_U0, test_Y = torch.Tensor(test_W), torch.Tensor(test_U0), torch.Tensor(test_Y)

    model = rsnet_2d(graph, data['T'], X = data['X'][:,0], Y = data['Y'][0]).to(device)
    print("Trainable parameter number: ", sum(p.numel() for p in model.parameters() if p.requires_grad))

    testset = TensorDataset(test_W, test_U0, test_Y)
    test_loader = DataLoader(testset,
                             batch_size=100,
                             shuffle=False,
                             pin_memory=True,
                             persistent_workers=True,
                             drop_last=False,
                             num_workers=4)

    xiLayer = ParabolicIntegrate_2d(graph, T = data['T'], X = data['X'][:,0], Y = data['Y'][0], eps = args.nu).to(device)
    Inference(model, xiLayer, device, test_loader)
