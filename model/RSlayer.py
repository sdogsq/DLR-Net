import torch
from torch import nn
import torch.nn.functional as F
from operator import itemgetter

class ParabolicIntegrate(nn.Module):
    def __init__(self, graph, BC = 'P', eps = 1, T = None, X = None, device = None, dtype = None):
        self.factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        keys = list(graph)
        self.graph = [{keys.index(it): graph[key][it] for it in graph[key]} for key in keys] # model graph
        self.only_xi = [(True if 'u_0' not in key else False) for key in graph.keys()] # if feature is only determined by xi
        self.U0FeatureIndex = [i for i, key in enumerate(self.only_xi) if not key] # index of features containing U0
        self.BC = BC #Boundary condition 'D' - Dirichlet, 'N' - Neuman, 'P' - periodic
        self.eps = eps # viscosity
        self.X_points = X # discretization of space (O_X space)
        self.T_points = T # discretization of time (O_T space)
        self.N = len(self.X_points) # number of space points
        self.T = len(self.T_points) # number of time points

        self.dt, self.dx = self.T_points[1] - self.T_points[0], self.X_points[1] - self.X_points[0]  # for equaly spaced points
        
        M = self.Parabolic_Matrix(self.N-1, self.dt, self.dx, BC).T     # approximate inverse of (I - Laplacian)
        M_c = self.Parabolic_Matrix(self.N-1, self.dt, self.dx, 'D').T  # M = (I-\Delta*dt)^{-1}
        self.register_buffer("M", M)

        O = torch.zeros(self.N, self.N, **self.factory_kwargs) # [N, N]
        M_PowMat = [M]
        for i in range(1, self.T):
            M_PowMat.append(torch.mm(M_PowMat[-1], M))
        M_PowMat = torch.cat(M_PowMat, axis = 1) # [M^1, M^2, ..., M^(T-1), M^T] with the shape of [N, T*N]

        self.register_buffer("M_PowMat", M_PowMat)

        M_PowSq = [M_PowMat]
        for i in range(1, self.T):
            M_PowSq.append(torch.cat((O, M_PowSq[-1][:, :-self.N]), axis = 1))
        M_PowSq[0] = O.repeat(1, self.T)
        M_PowSq = torch.cat(M_PowSq, axis = 0) # [0  , 0  , ..., 0      , 0      ] 
                                               # [0  , M^1, ..., N^(T-2), M^(T-1)]
                                               # [..., ..., ..., ...    , ...    ]
                                               # [0  , 0  , ..., M^1    , M^2    ]
                                               # [0  , 0  , ..., 0      , M^1    ] with the shape of [T*N, T*N]
        # M_PowSq = M_PowSq.reshape(self.T,self.N,self.T,self.N).permute(2,0,1,3)
        self.register_buffer("M_PowSq", M_PowSq)

        self.register_buffer("M_c", M_c)
        M_c_PowMat = [torch.eye(self.N, **self.factory_kwargs)] # [M_c^0, M_c^1, ..., M_c^(T-1)] with the shape of [T, N, N]
        for i in range(1, self.T):
            M_c_PowMat.append(torch.mm(M_c_PowMat[-1], M_c))
        M_c_PowMat = torch.stack(M_c_PowMat)
        self.register_buffer("M_c_PowMat", M_c_PowMat)

    def Parabolic_Matrix(self, N, dt, dx, BC, inverse = True):
        '''
        (N+1)x(N+1) Matrix approximating (Id - eps * \Delta*dt)^{-1}
        'D' corresponds to Dirichlet, 'N' to Neuman, 'P' to periodic BC
        Approximate sceletot of the Laplacian
        '''
        A = torch.diag(-2 * torch.ones(N + 1)) + torch.diag(torch.ones(N), diagonal=1) + torch.diag(torch.ones(N), diagonal=-1)
        A = A.to(**self.factory_kwargs)
        if BC == 'D': # if Dirichlet BC adjust # u(X[0]) = u(X[N]) = 0
            A[0,0], A[0,1], A[1,0], A[-1,-1], A[-1,-2], A[-2,-1] = 0, 0, 0, 0, 0, 0
        if BC == 'N': # if Neuman BC adjust
            A[0,1], A[-1,-2] = 2, 2
        if BC == 'P':
            A[-1, 1], A[0, -2] = 1, 1
        
        if inverse:
            return torch.linalg.inv(torch.eye(N + 1, **self.factory_kwargs) - self.eps*dt * A / (dx ** 2))
        
        # Matrix approximation of eps * \Delta*dt
        return self.eps*dt * A / (dx ** 2)

    def I_c(self, U0):
        '''
            U0: [B, N]
            return: [B, T, N]
        '''

        Ret = torch.matmul(U0, self.M_c_PowMat).transpose(0,1)

        return Ret # [B, T, N]
    
    def forward(self, W = None, U0 = None, XiFeature = None, returnU0Feature = False, diff = True):
        '''
            W : [B, T, N]
            U0 : [B, N]
            XiFeature : [B, T, N, F]
            diff : bool

            Return : [B, T, N, F]
        '''
        factory_kwargs = {'device': W.device, 'dtype': W.dtype}
        #----- use matmul -----
        integrated = []
        if XiFeature is not None:
            integrated.append(XiFeature[:,:,:,0])
        elif W is not None:
            # differentiate noise/forcing to create dW 
            if diff:
                dW = torch.zeros(W.shape, **factory_kwargs)
                dW[:,1:,:] = torch.diff(W, dim = 1)/self.dt
            else:
                dW = W*self.dt
            integrated.append(dW)
        else:
            raise "empty input"

        firiter = 1
        if U0 is not None:
            integrated.append(self.I_c(U0))
            firiter = 2

        for k, dic in enumerate(self.graph[firiter:],firiter):
            if (self.only_xi[k] and XiFeature is not None): # have cached XiFeature
                integrated.append(XiFeature[:,:,:,k])
                continue

            # compute the integral with u_0
            B = len(W) if W is not None else len(U0)
            tmp = torch.ones(B, self.T, self.N,  **factory_kwargs) # [B, T, N]
            for it, p in dic.items():
                if (p == 1):
                    tmp = tmp * integrated[it] #.clone()
                else:
                    tmp = tmp * torch.pow(integrated[it], p)

            # tmpA = torch.einsum("bik,tikn->btn", tmp, self.M_PowSq) * self.dt #### might be right

            # tmp = tmp.transpose(0, 1) # [T, B, N]
            # tmpB = torch.zeros_like(tmp)
            # print(tmpB.shape, tmp.shape, self.M_PowSq.shape)
            # for i in range(self.T):
            #     for j in range(self.T):
            #         tmpB[i] = tmpB[i] + torch.matmul(tmp[j], self.M_PowSq[i,j]) * self.dt
            #         if k == 1 and i == 1 and j <=2:
            #             print("KIJ",k,i,j, tmpB[i].sum(), tmp[j,0], self.M_PowSq[i,j])
            #     if k == 1 and i == 1:
            #         print("TMB", tmpB[i].sum())
            # tmpB = tmpB.transpose(0, 1) # [B, T, N]
            # tmp = tmp.transpose(0, 1) # [B, T, N]


            # print("AB", k, torch.norm(tmpA - tmpB))
            # tmp = tmpB

            # print(torch.norm(tmpA - tmp))
            # tmp = torch.einsum("bik,tikn->btn", tmp, self.M_PowSq)
            # print(tmp.shape)
            
            tmp = torch.matmul(tmp.reshape(B, -1), self.M_PowSq).reshape(B, self.T, self.N) * self.dt

            integrated.append(tmp)

        if returnU0Feature:
            U0Feature = torch.stack(itemgetter(*self.U0FeatureIndex)(integrated), dim = 3)
            return U0Feature
        else:
            integrated = torch.stack(integrated, dim = 3)
            return integrated
        #----- use append ----- 
        integrated = []
        if XiFeature is not None:
            integrated.append(XiFeature[:,:,:,0])
        elif W is not None:
            # differentiate noise/forcing to create dW 
            if diff:
                dW = torch.zeros(W.shape, **factory_kwargs)
                dW[:,1:,:] = torch.diff(W, dim = 1)/self.dt
            else:
                dW = W*self.dt
            integrated.append(dW)
        else:
            raise "empty input"
        firiter = 1
        if U0 is not None:
            integrated.append(self.I_c(U0))
            firiter = 2

        for k, dic in enumerate(self.graph[firiter:],firiter):
            if (self.only_xi[k] and XiFeature is not None): # have cached XiFeature
                integrated.append(XiFeature[:,:,:,k])
                continue

            # compute the integral with u_0
            B = len(W) if W is not None else len(U0)
            tmp = torch.ones(B, self.T, self.N, **factory_kwargs) # [B, T, N]
            for it, p in dic.items():
                if (p == 1):
                    tmp = tmp * integrated[it]
                else:
                    tmp = tmp * torch.pow(integrated[it], p)
            res = [torch.zeros(B, self.N, **factory_kwargs)]
            for i in range(1,self.T): 
                res.append(torch.matmul((res[-1] + tmp[:,i,:] * self.dt), self.M))

            integrated.append(torch.stack(res, dim = 1)) 
        integrated = torch.stack(integrated, dim = 3)
        return integrated
        # ------------------ check feature correctness --------------------
        if diff:
            dW = torch.zeros(W.shape)
            dW[:,1:,:] = torch.diff(W, dim = 1)/self.dt
        else:
            dW = W*self.dt
        int_true = torch.zeros(len(W), len(self.graph), self.T, self.N).to(**factory_kwargs) #[B, F, T, N]
        
        int_true[:,0,:,:] = dW.cuda()
        firiter = 1
        if U0 is not None:
            int_true[:,1,:,:] = self.I_c(U0)
            firiter = 2
        
        for k, dic in enumerate(self.graph[firiter:],firiter):
            tmp2 = torch.ones(len(W), self.T, self.N).to(**factory_kwargs)
            for it, p in dic.items():
                if (p == 1):
                    tmp2 *= int_true[:,it,:,:]
                else:
                    tmp2 *= torch.pow(int_true[:,it,:,:], p)
            for i in range(1,self.T): 
                int_true[:,k,i,:] = torch.matmul((int_true[:,k,i-1,:] + tmp2[:,i,:] * self.dt), self.M)
        # print(int_true.permute(0,2,3,1).shape, integrated.shape)
        # print("NORM", torch.max(int_true.permute(0,2,3,1) - integrated))
        return int_true.permute(0, 2, 3, 1) # [B, T, N, F]
        #------------------------------------------------------
        # differentiate noise/forcing to create dW 
        # if diff:
        #     dW = torch.zeros(W.shape)
        #     dW[:,1:,:] = torch.diff(W, dim = 1)/self.dt
        # else:
        #     dW = W*self.dt
        # # W_0 = np.zeros((1, len(T), len(X)))
        # # I_c = SPDE(BC = 'D', T = T, X = X, IC = IC, mu = lambda x: 0, sigma = lambda x: 0).Parabolic(W_0, T, X)
        # # self.register_buffer('integrated', torch.zeros(len(W), len(self.T), len(self.X), len(self.graph))
        # integrated = torch.zeros(len(W), self.T, self.N, len(self.graph)).to(**factory_kwargs) #[B, T, N, F]
        # # print("integrated.device: ", integrated.device)
        # integrated[:,:,:,0] = dW
        # firiter = 1
        # if U0 is not None:
        #     # print("W.device: ", W.device, "U0.device: ", U0.device)
        #     # print(integrated.shape, dW.shape, U0.shape)
        #     # integrated[:,:,:,1] = U0
        #     integrated[:,:,:,1] = self.I_c(U0)
        #     firiter = 2
        
        # for k, dic in enumerate(self.graph[firiter:],firiter):
        #     if (self.only_xi[k] and XiFeature is not None): # have cached XiFeature
        #         integrated[:,:,:,k] = XiFeature[:,:,:,k]
        #         continue

        #     # compute the integral with u_0
        #     tmp = torch.zeros(len(W), self.T, self.N).to(**factory_kwargs)
        #     nn.init.ones_(tmp)
        #     for it, p in dic.items():
        #         if (p == 1):
        #             tmp = tmp * integrated[:,:,:,it].clone()
        #         else:
        #             tmp = tmp * torch.pow(integrated[:,:,:,it].clone(), p)
        #     for i in range(1,self.T): 
        #         integrated[:,i,:,k] = torch.matmul((integrated[:,i-1,:,k] + tmp[:,i,:] * self.dt), self.M)
        
        # return integrated
#         # Finite difference method.
#         # Compute M*(integrated[i-1]+taus[i]). M^T (transpose) and reshaping for the same reason as in Parabolic_many
#         for i in range(1,len(self.T)): 
#             integrated[:,:,i,:] = torch.matmul((integrated[:,:,i-1,:] + taus[:,:,i,:] * dt).reshape(len(W), (len(self.graph), len(self.X))), self.M)
        
#         Jtau = {}
#         for i, t in enumerate(trees): #update dictionary and return integrated taus.
#             Jtau['I[{}]'.format(t)] = integrated[i]
#             if derivative and "I'[{}]".format(t) not in exceptions:
#                 # If derivative is true include also functions of the form \partial_x I[f] that are denoted as I'[f]
#                 if derivative == 1 or derivative is True:
#                     Jtau["I'[{}]".format(t)] = self.discrete_diff(integrated[i], len(self.X), flatten = False, higher = False)/dx
#                 else:
#                     # centralised differentiation
#                     Jtau["I'[{}]".format(t)] = self.discrete_diff(integrated[i], len(self.X), flatten = False, higher = True)/dx
#         return Jtau


#===========================================================================
# 2d fourier layers
#===========================================================================
class SpectralConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2):
        super(SpectralConv2d, self).__init__()

        """
        2D Fourier layer. It does FFT, linear transform, and Inverse FFT.    
        """

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1 #Number of Fourier modes to multiply, at most floor(N/2) + 1
        self.modes2 = modes2

        self.scale = (1 / (in_channels * out_channels))
        self.weights1 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat))
        self.weights2 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat))

    # Complex multiplication
    def compl_mul2d(self, input, weights):
        # (batch, in_channel, x,t), (in_channel, out_channel, x,t) -> (batch, out_channel, x,t)
        return torch.einsum("bixt,ioxt->boxt", input, weights)

    def forward(self, x):
        batchsize = x.shape[0]
        #Compute Fourier coeffcients up to factor of e^(- something constant)
        x_ft = torch.fft.rfft2(x)

        # Multiply relevant Fourier modes
        out_ft = torch.zeros(batchsize, self.out_channels,  x.size(-2), x.size(-1)//2 + 1, dtype=torch.cfloat, device=x.device)
        out_ft[:, :, :self.modes1, :self.modes2] = \
            self.compl_mul2d(x_ft[:, :, :self.modes1, :self.modes2], self.weights1)
        out_ft[:, :, -self.modes1:, :self.modes2] = \
            self.compl_mul2d(x_ft[:, :, -self.modes1:, :self.modes2], self.weights2)

        #Return to physical space
        x = torch.fft.irfft2(out_ft, s=(x.size(-2), x.size(-1)))
        return x


class FNO_layer(nn.Module):
    def __init__(self, modes1, modes2, width, last=False):
        super(FNO_layer, self).__init__()
        """ ...
        """
        self.last = last

        self.conv = SpectralConv2d(width, width, modes1, modes2)
        self.w = nn.Conv2d(width, width, 1)
        # self.bn = torch.nn.BatchNorm2d(width)


    def forward(self, x):
        """ x: (batch, hidden_channels, dim_x, dim_t)"""

        x1 = self.conv(x)
        x2 = self.w(x)
        x = x1 + x2
        if not self.last:
            x = F.gelu(x)
            
        return x