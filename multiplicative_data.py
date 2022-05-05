r'''
(\partial_t - \Delta) u &= 3u - u^3 + u\cdot\xi\,\quad\text{for $(t,x) \in [0,1]\times [0,1]$,}
u(t,0) &= u(t,1) 
u(0,x) &= x(1-x) + k \eta(x)
'''
import numpy as np
import argparse
from tqdm import tqdm
import os
from src.Noise import Noise
from src.SPDEs import SPDE
import time
parser = argparse.ArgumentParser()
parser.add_argument('-N', '--N', type=int, default=1000)
parser.add_argument('-k', '--k', type=float, default=0.1)
args = parser.parse_args()

def mkdir(path):
    folder = os.path.exists(path)
    if not folder:
        os.makedirs(path)

dx, dt = 1/127, 0.001 #space-time increments 
N = int(args.N * 1.2) # Number of realizations
a, b, s, t = 0, 1, 0, 0.05 # space-time boundaries

X, T = Noise().partition(a,b,dx), Noise().partition(s,t,dt) # space grid O_X and time grid O_T

W = 0.1 * Noise().WN_space_time_many(s, t, dt, a, b, dx, N) # Create realizations of space time white noise

ic = np.vectorize(lambda x: x*(1-x)) # initial condition
IC_1 = args.k * Noise().initial(N, X, scaling = 2) # 2 cycle
IC_2 = np.tile(ic(X), (N, 1))
IC = IC_1 + IC_2
# (u0,xi)->u: IC = IC_1+IC_2 / xi->u: IC = IC_2
# IC = np.array([[ic(dx*i) for i in range(len(X))] for n in range(k)]) # initial condition

mu = lambda x: 3*x-x**3 # drift
sigma1 = lambda x: x # multiplicative diffusive term
sigma2 = lambda x: 1 # additive diffusive term


# solutions to the multiplicative equation 
Solution = SPDE(BC = 'P', IC = IC, mu = mu, sigma = sigma1).Parabolic(W, T, X)

mkdir("./data/")
np.savez(f"./data/parabolic_multiplicative_{args.N}_{args.k}.npz", Solution = Solution, W = W, T = T, X = X, U0 = IC)
print(Solution.shape)