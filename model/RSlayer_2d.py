import torch
from torch import nn
import torch.nn.functional as F
from operator import itemgetter

class ParabolicIntegrate_2d(nn.Module):
    def __init__(self, graph, T, X, Y, BC = 'P', eps = 1, device = None, dtype = None):
            self.factory_kwargs = {'device': device, 'dtype': dtype}
            super().__init__()
            keys = list(graph)
            self.graph = [{keys.index(it): graph[key][it] for it in graph[key]} for key in keys] # model graph
            self.isDerivative = [(int(key[1]) if key[1].isdigit() else False) for key in graph.keys()]
            self.only_xi = [(True if 'u_0' not in key else False) for key in graph.keys()] # if feature is only determined by xi
            self.U0FeatureIndex = [i for i, key in enumerate(self.only_xi) if not key] # index of features containing U0
            self.BC = BC #Boundary condition 'D' - Dirichlet, 'N' - Neuman, 'P' - periodic
            self.eps = eps # viscosity
            self.X_points = X # discretization of space (O_X space)
            self.Y_points = Y # discretization of space (O_Y space)
            self.T_points = T # discretization of time (O_T space)
            self.X = len(self.X_points) # number of space X points
            self.Y = len(self.Y_points) # number of space Y points
            self.T = len(self.T_points) # number of time points

            self.dt, self.dx = self.T_points[1] - self.T_points[0], self.X_points[1] - self.X_points[0]  # for equaly spaced points

    def Laplace_2d(self, arr):
        if not isinstance(arr, torch.Tensor):
            raise TypeError('arr must be a torch tensor')
        out = torch.zeros(arr.shape, dtype = arr.dtype, device = arr.device)
        out[:,1:-1,:] += torch.diff(torch.diff(arr, axis=1), axis = 1)
        out[:,:,1:-1] += torch.diff(torch.diff(arr, axis=2), axis = 2)
        if torch.isinf(out).any():
            print(arr)
            raise ValueError('tmp is nan')
        if self.BC =='P':
            out[:,0,1:-1] = arr[:,1,1:-1] + arr[:,-1,1:-1] + arr[:,0,0:-2] + arr[:,0,2:] - 4*arr[:,0,1:-1]
            out[:,-1,1:-1] = arr[:,0,1:-1] + arr[:,-2,1:-1] + arr[:,-1,0:-2] + arr[:,-1,2:] - 4*arr[:,-1,1:-1]
            out[:,1:-1,0] = arr[:,1:-1,1] + arr[:,1:-1,-1] + arr[:,0:-2,0] + arr[:,2:,0] - 4*arr[:,1:-1,0]
            out[:,1:-1,-1] = arr[:,1:-1,0] + arr[:,1:-1,-2] + arr[:,0:-2,-1] + arr[:,2:,-1] - 4*arr[:,1:-1,-1]
        if torch.isinf(out).any():
            raise ValueError('tmp is nan')
        return out*self.dt/self.dx**2

    def I_c(self, U0):
        '''
            U0: [B, X, Y]
            return: [B, T, X, Y]
        '''
        factory_kwargs = {'device': U0.device, 'dtype': U0.dtype}
        Solution = torch.zeros(len(U0), self.T, self.X, self.Y, **factory_kwargs)
        # Initialize
        Solution[:,0,:,:] = U0
        
        # Finite difference method.
        # u_{n+1} = u_n + mu(u_n)*dt + sigma(u_n)*dW_{n} + (dx)^{-2} A*u_{n}*dt 
        # mu = sigma = 0 for solving I_c[u_0]
        for i in range(1, self.T):
            Solution[:,i,:,:] = Solution[:,i-1,:,:] + self.eps * self.Laplace_2d(Solution[:,i-1,:,:])

        return Solution

    def forward(self, W = None, Latent = None, XiFeature = None, returnU0Feature = False, diff = True):
        '''
            W: [B, T, X, Y]
            Latent: [B, T, X, Y]
            XiFeature: [B, T, X, Y, F]
            diff: bool

            Return: [B, T, X, Y, F]
        '''
        factory_kwargs = {'device': W.device, 'dtype': W.dtype}
        # differentiate noise/forcing to create dW 
        integrated = []

        # add xi features as integrated[0]
        if XiFeature is not None:
            integrated.append(XiFeature[...,0])
        elif W is not None:
            # differentiate noise/forcing to create dW 
            if diff:
                dW = torch.zeros(W.shape, **factory_kwargs)
                dW[:,1:,:] = torch.diff(W, dim = 1)/self.dt
            else:
                dW = W*self.dt
            integrated.append(dW)
            if torch.isinf(integrated[-1]).any():
                raise ValueError('dW is nan')
        else:
            raise "empty itorchut"

        firiter = 1

        # if itorchut is given, substitude I_c[u_0] by itorchut, recorded in integrated[1]
        if Latent is not None:
            integrated.append(Latent)
            firiter = 2

        for k, dic in enumerate(self.graph[firiter:],firiter):
            if (self.only_xi[k] and XiFeature is not None): # have cached XiFeature
                integrated.append(XiFeature[...,k])
                continue
            
            if (self.isDerivative[k]): # derivative
                integrated.append(self.discrete_diff_2d(integrated[list(dic.keys())[0]], self.X, axis = self.isDerivative[k], higher = False)/self.dx)
                continue
            
            # compute the integral with u_0
            B = len(W) if W is not None else len(Latent)
            tmp = torch.ones(B, self.T, self.X, self.Y,  **factory_kwargs) # [B, T, X, Y]
            for it, p in dic.items():
                if (p == 1):
                    tmp = tmp * integrated[it] #.clone()
                else:
                    tmp = tmp * torch.pow(integrated[it], p)

            if torch.isinf(tmp).any():
                raise ValueError('tmp is nan')
            res = [torch.zeros(B, self.X, self.Y, **factory_kwargs)] # T * [B, X, Y]
            for i in range(1,self.T):
                res.append(res[-1] + self.eps * self.Laplace_2d(res[-1]) + tmp[:,i-1,:,:] * self.dt)
                if torch.isinf(res[-1]).any():
                    raise ValueError('res is nan')
            # for i in range(1,len(self.T)): 
            #     integrated[:,i,:,:] = (integrated[:,i-1,:,:] + self.eps * self.Laplace_2d(integrated[:,i-1,:,:], self.dx, dt) + taus[:,i-1,:,:] * dt).reshape((len(trees), self.X.shape[0], self.X.shape[1]))
        
            # tmp = torch.matmul(tmp.reshape(B, -1), self.M_PowSq).reshape(B, self.T, self.N) * self.dt

            integrated.append(torch.stack(res, dim = 1))

        if returnU0Feature:
            U0Feature = torch.stack(itemgetter(*self.U0FeatureIndex)(integrated), dim = -1)
            return U0Feature
        else:
            integrated = torch.stack(integrated, dim = -1)
            return integrated

        #extract the trees from dictionary which are not purely polyniomials and were not already integrated
        trees = [tree for tree in tau.keys() if 'I[{}]'.format(tree) not in done and 'I[{}]'.format(tree) not in exceptions] 

        dt, dx = self.T[1]-self.T[0], self.X[1,0]-self.X[0,0]
        taus = torch.array([tau[t] for t in trees])

        integrated = torch.zeros(shape = (len(trees), len(self.T), self.X.shape[0], self.X.shape[1]))
        
        # Finite difference method.
        # Compute M*(integrated[i-1]+taus[i]). M^T (transpose) and reshaping for the same reason as in Parabolic_many
        for i in range(1,len(self.T)): 
            integrated[:,i,:,:] = (integrated[:,i-1,:,:] + self.eps * self.Laplace_2d(integrated[:,i-1,:,:], dx, dt) + taus[:,i-1,:,:] * dt).reshape((len(trees), self.X.shape[0], self.X.shape[1]))
        
        Jtau = {}
        for i, t in enumerate(trees): #update dictionary and return integrated taus.
            Jtau['I[{}]'.format(t)] = integrated[i]
            if derivative and "I1[{}]".format(t) not in exceptions and "I2[{}]".format(t) not in exceptions and t[1] == "[":
                # If derivative is true include also functions of the form \partial_x I[f] that are denoted as I'[f]
                if derivative == 1 or derivative is True:
                    Jtau["I1[{}]".format(t)] = self.discrete_diff_2d(integrated[i], self.X.shape[0], axis = 1, flatten = False, higher = False)/dx
                    Jtau["I2[{}]".format(t)] = self.discrete_diff_2d(integrated[i], self.X.shape[0], axis = 2, flatten = False, higher = False)/dx
                else:
                    # centralised differentiation
                    Jtau["I1[{}]".format(t)] = self.discrete_diff_2d(integrated[i], self.X.shape[0], axis = 1, flatten = False, higher = True)/dx
                    Jtau["I2[{}]".format(t)] = self.discrete_diff_2d(integrated[i], self.X.shape[0], axis = 2, flatten = False, higher = True)/dx
        return Jtau

    def discrete_diff_2d(self, vec, N, axis, higher = True):
        a = torch.zeros_like(vec)
        if len(a.shape) == 1:
            a = a.reshape(len(vec)//N, N)
        if axis == 1:
            if higher: # central approximation of a dervative
                a[:,:-1,:] = (torch.roll(vec[:,:-1,:], -1, dims = 1) - torch.roll(vec[:,:-1,:], 1, dims = 1))/2
            else:
                a[:,:-1,:] = a[:,:-1,:] - torch.roll(vec[:,:-1,:], 1, dims = 1)
            a[:,-1,:] = a[:,0,:] # enforce periodic boundary condions
        if axis == 2:
            if higher: # central approximation of a dervative
                a[:,:,:-1] = (torch.roll(vec[:,:,:-1], -1, dims = 2) - torch.roll(vec[:,:,:-1], 1, dims = 2))/2
            else:
                a[:,:,:-1] = a[:,:,:-1] - torch.roll(vec[:,:,:-1], 1, dims = 2)
            a[:,:,-1] = a[:,:,0] # enforce periodic boundary condions

        return a

class SpectralConv3d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2, modes3):
        super(SpectralConv3d, self).__init__()

        """
        3D Fourier layer. It does FFT, linear transform, and Inverse FFT.    
        """

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1 #Number of Fourier modes to multiply, at most floor(N/2) + 1
        self.modes2 = modes2
        self.modes3 = modes3

        self.scale = (1 / (in_channels * out_channels))
        self.weights1 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, self.modes3, dtype=torch.cfloat))
        self.weights2 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, self.modes3, dtype=torch.cfloat))
        self.weights3 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, self.modes3, dtype=torch.cfloat))
        self.weights4 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, self.modes3, dtype=torch.cfloat))

    # Complex multiplication
    def compl_mul3d(self, input, weights):
        # (batch, in_channel, x,y,t ), (in_channel, out_channel, x,y,t) -> (batch, out_channel, x,y,t)
        return torch.einsum("bixyz,ioxyz->boxyz", input, weights)

    def forward(self, x):
        batchsize = x.shape[0]
        #Compute Fourier coeffcients up to factor of e^(- something constant)
        x_ft = torch.fft.rfftn(x, dim=[-3,-2,-1])

        # Multiply relevant Fourier modes
        out_ft = torch.zeros(batchsize, self.out_channels, x.size(-3), x.size(-2), x.size(-1)//2 + 1, dtype=torch.cfloat, device=x.device)
        out_ft[:, :, :self.modes1, :self.modes2, :self.modes3] = \
            self.compl_mul3d(x_ft[:, :, :self.modes1, :self.modes2, :self.modes3], self.weights1)
        out_ft[:, :, -self.modes1:, :self.modes2, :self.modes3] = \
            self.compl_mul3d(x_ft[:, :, -self.modes1:, :self.modes2, :self.modes3], self.weights2)
        out_ft[:, :, :self.modes1, -self.modes2:, :self.modes3] = \
            self.compl_mul3d(x_ft[:, :, :self.modes1, -self.modes2:, :self.modes3], self.weights3)
        out_ft[:, :, -self.modes1:, -self.modes2:, :self.modes3] = \
            self.compl_mul3d(x_ft[:, :, -self.modes1:, -self.modes2:, :self.modes3], self.weights4)

        #Return to physical space
        x = torch.fft.irfftn(out_ft, s=(x.size(-3), x.size(-2), x.size(-1)))
        return x


class FNO_layer(nn.Module):
    def __init__(self, modes1, modes2, modes3, width, last=False):
        super(FNO_layer, self).__init__()
        """ ...
        """
        self.last = last

        self.conv = SpectralConv3d(width, width, modes1, modes2, modes3)
        self.w = nn.Conv3d(width, width, 1)
        # self.bn = torch.nn.BatchNorm2d(width)


    def forward(self, x):
        """ x: (batch, hidden_channels, dim_x, dim_y, dim_t)"""

        x1 = self.conv(x)
        x2 = self.w(x)
        x = x1 + x2
        if not self.last:
            x = F.gelu(x)
            
        return x

# ################################################################
# #  2d fourier layer
# ################################################################
# class SpectralConv2d(nn.Module):
#     def __init__(self, in_channels, out_channels, modes1, modes2):
#         super(SpectralConv2d, self).__init__()

#         """
#         2D Fourier layer. It does FFT, linear transform, and Inverse FFT.    
#         """

#         self.in_channels = in_channels
#         self.out_channels = out_channels
#         self.modes1 = modes1 #Number of Fourier modes to multiply, at most floor(N/2) + 1
#         self.modes2 = modes2

#         self.scale = (1 / (in_channels * out_channels))
#         self.weights1 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat))
#         self.weights2 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat))

#     # Complex multiplication
#     def compl_mul2d(self, input, weights):
#         # (batch, in_channel, x,y ), (in_channel, out_channel, x,y) -> (batch, out_channel, x,y)
#         return torch.einsum("bixy,ioxy->boxy", input, weights)

#     def forward(self, x):
#         batchsize = x.shape[0]
#         #Compute Fourier coeffcients up to factor of e^(- something constant)
#         x_ft = torch.fft.rfft2(x)

#         # Multiply relevant Fourier modes
#         out_ft = torch.zeros(batchsize, self.out_channels,  x.size(-2), x.size(-1)//2 + 1, dtype=torch.cfloat, device=x.device)
#         out_ft[:, :, :self.modes1, :self.modes2] = \
#             self.compl_mul2d(x_ft[:, :, :self.modes1, :self.modes2], self.weights1)
#         out_ft[:, :, -self.modes1:, :self.modes2] = \
#             self.compl_mul2d(x_ft[:, :, -self.modes1:, :self.modes2], self.weights2)

#         #Return to physical space
#         x = torch.fft.irfft2(out_ft, s=(x.size(-2), x.size(-1)))
#         return x

# class FNO_layer(nn.Module):
#     def __init__(self, modes1, modes2, width, last=False):
#         super(FNO_layer, self).__init__()
#         """ ...
#         """
#         self.last = last

#         self.conv = SpectralConv2d(width, width, modes1, modes2)
#         self.w = nn.Conv2d(width, width, 1)
#         # self.bn = torch.nn.BatchNorm2d(width)


#     def forward(self, x):
#         """ x: (batch, hidden_channels, dim_x, dim_t)"""

#         x1 = self.conv(x)
#         x2 = self.w(x)
#         x = x1 + x2
#         if not self.last:
#             x = F.gelu(x)
            
#         return x
