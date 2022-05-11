from Classes.SPDEs import *
from Classes.Rule import *
from Classes.Model import *
from Classes.Noise import *

from Data.generator_sns import navier_stokes_2d
from Data.random_forcing import GaussianRF

import numpy as np
import torch
from timeit import default_timer
import math
import os

torch.manual_seed(0)
np.random.seed(0)

def mkdir(path):
    folder = os.path.exists(path)
    if not folder:
        os.makedirs(path)

################################################################
#  configurations
################################################################
ntrain = 1000
ntest = 200
# N = 5
N = 1200

sub_x = 4
sub_t = 1

t_tradition = 0
t_RS = 0

################################################################
# generate data
################################################################

device = torch.device('cuda')

# Viscosity parameter
nu = 1e-4

# Spatial Resolution
s = 64

# domain where we are solving
a = [1,1]

# Temporal Resolution   
T = 5e-2
delta_t = 1e-3

# Set up 2d GRF with covariance parameters
GRF = GaussianRF(2, s, alpha=3, tau=3, device=device)

# Forcing function: 0.1*(sin(2pi(x+y)) + cos(2pi(x+y)))
t = torch.linspace(0, 1, s+1, device=device)
t = t[0:-1]

X,Y = torch.meshgrid(t, t)
dx = X[1,0] - X[0,0]
f = 0.1*(torch.sin(2*math.pi*(X + Y)) + torch.cos(2*math.pi*(X + Y)))

# Stochastic forcing function: sigma*dW/dt 
stochastic_forcing = {'alpha':0.005, 'kappa':10, 'sigma':0.05}

# Number of snapshots from solution
record_steps = int(T/(delta_t))

# Solve equations in batches (order of magnitude speed-up)

# Batch size
bsize = 100

c = 0

#Sample random fields
# w0 = GRF.sample(1).repeat(bsize,1,1)

for j in range(N//bsize):
    
    t0 = default_timer()

    w0 = GRF.sample(bsize) # (u0,xi)->u
    # w0 = torch.zeros((bsize, X.shape[0], X.shape[1]), device = device) # xi->u

    sol, sol_t, force = navier_stokes_2d(a, w0, f, nu, T, delta_t, record_steps, stochastic_forcing)  

    # add time 0
    time = torch.zeros(record_steps+1)
    time[1:] = sol_t.cpu()
    sol = torch.cat([w0[...,None],sol],dim=-1)
    force = torch.cat([torch.zeros_like(w0)[...,None],force],dim=-1)

    if j == 0:
        Soln = sol.cpu()
        forcing = force.cpu()
        IC = w0.cpu()
        Soln_t = sol_t.cpu()
    else:
        Soln = torch.cat([Soln, sol.cpu()], dim=0)
        forcing = torch.cat([forcing, force.cpu()], dim=0)
        IC = torch.cat([IC, w0.cpu()],dim=0)

    c += bsize
    t1 = default_timer()
    print(j, c, (t1-t0)/bsize)
    t_tradition = t_tradition + t1 - t0

# Soln: [sample, x, y, step]
# Soln_t: [t=step*delta_t]

Soln = Soln.cpu()
Soln_t = Soln_t.cpu()
forcing = forcing.cpu()
X, Y = X.cpu(), Y.cpu()
IC = IC.cpu()

Soln = Soln.transpose(2,3).transpose(1,2).numpy()
Soln_t = Soln_t.numpy()
forcing = forcing.transpose(2,3).transpose(1,2).numpy()
X, Y = X.numpy(), Y.numpy()
IC = IC.numpy()

print(f"Soln shape: {Soln.shape}, time shape: {time.shape}, forcing shape: {forcing.shape}, X shape: {X.shape}, Y shape: {Y.shape}, IC shape: {IC.shape}")
print(f"t {time}")
mkdir("./data/")
np.savez(f"./data/NS.npz", Solution = Soln, W = forcing, T = time.numpy(), X = X, Y = Y, U0 = IC)


Soln = Soln[:,::sub_t,::sub_x,::sub_x]
Soln_t = Soln_t[::sub_t]
forcing = forcing[:,::sub_t,::sub_x,::sub_x]
X, Y = X[::sub_x,::sub_x], Y[::sub_x,::sub_x]
IC = IC[:,::sub_x,::sub_x]


# scipy.io.savemat(filename+'{}.mat'.format(j), mdict={'t':time.numpy(), 'sol': sol.cpu().numpy(), 'forcing': forcing.cpu().numpy(), 'param':stochastic_forcing})
    # scipy.io.savemat(filename+'small_{}.mat'.format(j), mdict={'t': time[::4].numpy(), 'sol': sol[:,::4,::4,::4].cpu().numpy(), 'forcing': forcing[:,::4,::4,::4].cpu().numpy(), 'param':stochastic_forcing})