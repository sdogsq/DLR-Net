from Data.generator_sns import navier_stokes_2d
from Data.random_forcing import GaussianRF

import numpy as np
import torch
from timeit import default_timer
import math
import os
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('-N', '--N', type=int, default=1200, metavar='N',
                    help = 'number of training realizations')
parser.add_argument('-S', '--S', type=int, default=64, metavar='N',
                    help = 'Spatial Resolution')
parser.add_argument('--sub_x', type=int, default=1, metavar='N',
                    help = 'number of training realizations')
parser.add_argument('--sub_t', type=int, default=1, metavar='N',
                    help = 'number of training realizations')
parser.add_argument('--nu', type=float, default=1e-4, metavar='N',
                    help = 'Viscosity parameter')
parser.add_argument('-T', '--T', type=float, default=1, metavar='N',
                    help = 'Simulation time')
parser.add_argument('--dt', type=float, default=1e-3, metavar='N',
                    help = 'Temporal Resolution')
parser.add_argument('--fixU0', action='store_true',
                    help = 'fixU0 and generate xi->u data')
parser.add_argument('-B', '--bs', type=float, default=100, metavar='N',
                    help = 'Simulation batchsize')
args = parser.parse_args()

torch.manual_seed(0)
np.random.seed(0)

def mkdir(path):
    folder = os.path.exists(path)
    if not folder:
        os.makedirs(path)

################################################################
#  configurations
################################################################
N = args.N

sub_x = args.sub_x
sub_t = args.sub_t

t_tradition = 0

################################################################
# generate data
################################################################

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Viscosity parameter
nu = args.nu

# Spatial Resolution
s = args.S

# domain where we are solving
a = [1,1]

# Temporal Resolution   
T = args.T
delta_t = args.dt

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
bsize = min(N,args.bs)

c = 0

#Sample random fields
# w0 = GRF.sample(1).repeat(bsize,1,1)

for j in range(N//bsize):
    
    t0 = default_timer()

    if args.fixU0:
        w0 = torch.zeros((bsize, X.shape[0], X.shape[1]), device = device) # xi->u
    else:
        w0 = GRF.sample(bsize) # (u0,xi)->u

    sol, sol_t, force = navier_stokes_2d(a, w0, f, nu, T, delta_t, record_steps, stochastic_forcing)  

    # add time 0
    time = torch.zeros(record_steps+1)
    time[1:] = sol_t.cpu()
    time = time[::sub_t]
    sol = torch.cat([w0[...,None],sol],dim=-1)
    force = torch.cat([torch.zeros_like(w0)[...,None],force],dim=-1)

    if j == 0:
        Soln = sol.cpu()[:,::sub_x,::sub_x,::sub_t]
        forcing = force.cpu()[:,::sub_x,::sub_x,::sub_t]
        print(forcing.shape)
        IC = w0.cpu()[:,::sub_x,::sub_x]
        Soln_t = sol_t.cpu()[::sub_t]
    else:
        Soln = torch.cat([Soln, sol.cpu()[:,::sub_x,::sub_x,::sub_t]], dim=0)
        forcing = torch.cat([forcing, force.cpu()[:,::sub_x,::sub_x,::sub_t]], dim=0)
        IC = torch.cat([IC, w0.cpu()[:,::sub_x,::sub_x]],dim=0)

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
np.savez(f"./data/NS_{'xi' if args.fixU0 else 'u0_xi'}_{N}_{T:g}.npz", Solution = Soln, W = forcing, T = time.numpy(), X = X, Y = Y, U0 = IC)