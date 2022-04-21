import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import math
from tqdm import tqdm
from time import time
import os
from src.Noise import Noise
from src.Rule import Rule
from src.SPDEs import SPDE
from src.Model import Model

def mkdir(path):
    folder = os.path.exists(path)
    if not folder:
        os.makedirs(path)

dx, dt = 1/127, 0.001 #space-time increments 
k = 1200 # Number of realizations
a, b, s, t = 0, 1, 0, 0.05 # space-time boundaries

X, T = Noise().partition(a,b,dx), Noise().partition(s,t,dt) # space grid O_X and time grid O_T

W = Noise().WN_space_time_many(s, t, dt, a, b, dx, k) # Create realizations of space time white noise

ic = lambda x: x*(1-x) # initial condition
IC_1 = 0.1 * Noise().initial(k, X, scaling = 2) # 2 cycle
IC_2 = np.array([[ic(dx*i) for i in range(len(X))] for n in range(k)])
IC = IC_1 + IC_2
# (u0,xi)->u: IC = IC_1+IC_2 / xi->u: IC = IC_2
# IC = np.array([[ic(dx*i) for i in range(len(X))] for n in range(k)]) # initial condition

mu = lambda x: 3*x-x**3 # drift
sigma1 = lambda x: x # multiplicative diffusive term
sigma2 = lambda x: 1 # additive diffusive term



# solutions to the additive equation 
Soln_add = SPDE(BC = 'P', IC = IC, mu = mu, sigma = sigma2).Parabolic(W, T, X)

mkdir("./data/")
np.savez("./data/parabolic_additive_randomU0.npz", Soln_add = Soln_add, W = W, T = T, X = X, U0 = IC)
print(Soln_add.shape)