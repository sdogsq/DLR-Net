import torch
from torch import nn

class ParabolicIntegrate(nn.Module):
    def __init__(self, graph, BC = 'P', eps = 1, T = None, X = None):
        super().__init__()
        keys = list(graph)
        self.graph = [{keys.index(it): graph[key][it] for it in graph[key]} for key in keys] # model graph
        self.only_xi = [(True if 'u_0' not in key else False) for key in graph.keys()] # if feature is only determined by xi

        self.BC = BC #Boundary condition 'D' - Dirichlet, 'N' - Neuman, 'P' - periodic
        self.eps = eps # viscosity
        self.X = X # discretization of space (O_X space)
        self.T = T # discretization of time (O_T space)
        
        self.dt, self.dx = self.T[1]-self.T[0], self.X[1]-self.X[0]  # for equaly spaced points
        
        # approximate inverse of (I - Laplacian)
        self.M = self.Parabolic_Matrix(len(self.X)-1, self.dt, self.dx, BC).T.cuda()
        self.M_c = self.Parabolic_Matrix(len(self.X)-1, self.dt, self.dx, 'D').T.cuda() #M = (I-\Delta*dt)^{-1}
        
    def Parabolic_Matrix(self, N, dt, dx, BC, inverse = True): #(N+1)x(N+1) Matrix approximating (Id - eps * \Delta*dt)^{-1}
        # 'D' corresponds to Dirichlet, 'N' to Neuman, 'P' to periodic BC
        # Approximate sceletot of the Laplacian
        A = torch.diag(-2 * torch.ones(N + 1)) + torch.diag(torch.ones(N), diagonal=1) + torch.diag(torch.ones(N), diagonal=-1) 
        if BC == 'D': # if Dirichlet BC adjust # u(X[0]) = u(X[N]) = 0
            A[0,0], A[0,1], A[1,0], A[-1,-1], A[-1,-2], A[-2,-1] = 0, 0, 0, 0, 0, 0
        if BC == 'N': # if Neuman BC adjust
            A[0,1], A[-1,-2] = 2, 2
        if BC == 'P':
            A[-1, 1], A[0, -2] = 1, 1
        
        if inverse:
            return torch.linalg.inv(torch.eye(N + 1) - self.eps*dt * A / (dx ** 2))
        
        # Matrix approximation of eps * \Delta*dt
        return self.eps*dt * A / (dx ** 2)

    def I_c(self, U0):
        
        # W_0 = np.zeros((1, len(T), len(X)))
        # I_c = SPDE(BC = 'D', T = T, X = X, IC = IC, mu = lambda x: 0, sigma = lambda x: 0).Parabolic(W_0, T, X)

        # M = self.Parabolic_Matrix(len(X)-1, dt, dx).T #M = (I-\Delta*dt)^{-1}

        Solution = torch.zeros(len(U0), len(self.T), len(self.X)).cuda()

        # Initialize
        Solution[:,0,:] = U0
        
        # Finite difference method.
        # u_{n+1} = u_n + (dx)^{-2} A*u_{n+1}*dt + mu(u_n)*dt + sigma(u_n)*dW_{n+1}
        # Hence u_{n+1} = (I - dt/(dx)^2 A)^{-1} (u_n + mu(u_n)*dt + sigma(u_n)*dW_{n+1})
        # Solve equations in paralel for every noise/IC simultaneosly
        for i in range(1,len(self.T)):
            Solution[:,i,:] = torch.matmul(Solution[:,i-1,:], self.M_c)
            
        # Because Solution.iloc[i-1] and thus current is a vector of length len(noises)*len(X)
        # need to reshape it to matrix of the shape (W.shape[0], len(X)) and multiply on the right by the M^T (transpose).
        # M*(current.reshape(...)) does not give the correct value.
        
        
        return Solution
    
#     def Integrate_Parabolic(self, W)
    def forward(self, W, U0 = None, Xi_feature = None, diff = True): # W <- [B, T, N]
        # differentiate noise/forcing to create dW 
        if diff:
            dW = torch.zeros(W.shape)
            dW[:,1:,:] = torch.diff(W, dim = 1)/self.dt
        else:
            dW = W*self.dt
        
        # W_0 = np.zeros((1, len(T), len(X)))
        # I_c = SPDE(BC = 'D', T = T, X = X, IC = IC, mu = lambda x: 0, sigma = lambda x: 0).Parabolic(W_0, T, X)
        # self.register_buffer('integrated', torch.zeros(len(W), len(self.T), len(self.X), len(self.graph))
        integrated = torch.zeros(len(W), len(self.T), len(self.X), len(self.graph)).cuda() #[B, T, N, F]
        # print("integrated.device: ", integrated.device)
        integrated[:,:,:,0] = dW
        firiter = 1
        if U0 is not None:
            # print("W.device: ", W.device, "U0.device: ", U0.device)
            # print(integrated.shape, dW.shape, U0.shape)
            # integrated[:,:,:,1] = U0
            integrated[:,:,:,1] = self.I_c(U0)
            firiter = 2
        
        for k, dic in enumerate(self.graph[firiter:],firiter):
            if (self.only_xi[k] and Xi_feature is not None): # have cached Xi_feature
                integrated[:,:,:,k] = Xi_feature[:,:,:,k]
                continue

            # compute the integral with u_0
            tmp = torch.zeros(len(W), len(self.T), len(self.X)).cuda()
            nn.init.ones_(tmp)
            for it, p in dic.items():
                if (p == 1):
                    tmp = tmp * integrated[:,:,:,it].clone()
                else:
                    tmp = tmp * torch.pow(integrated[:,:,:,it].clone(), p)
            for i in range(1,len(self.T)): 
                integrated[:,i,:,k] = torch.matmul((integrated[:,i-1,:,k] + tmp[:,i,:] * self.dt), self.M)
        
        return integrated
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