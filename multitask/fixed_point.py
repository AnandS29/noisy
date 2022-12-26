import torch
import numpy as np
from scipy import optimize

prob_geq = {}
prob_less = {}
for i in range(4):
    for j in range(4):
        prob = None
        if (i == 0 and j == 0) or (i == 1 and j == 1) or (i == 1 and j == 0):
            prob = 1
        elif (i==0 and j==1):
            prob = 0
        elif ((i == 2) or (i==3)) and (j==0):
            prob = 1
        elif ((i == 2) or (i==3)) and (j==1):
            prob = 0.5
        elif ((j == 2) or (j==3)) and (i==0):
            prob = 0.5
        elif ((j == 2) or (j==3)) and (i==1):
            prob = 0.5
        elif (i==2 and j==2) or (i==3 and j==3) or (i==3 and j==2):
            prob = 0.75
        elif (i==2 and j==3):
            prob = 0.5

        prob_geq[(i,j)] = prob
        prob_less[(i,j)] = 1-prob

def obj(r):
    print(f"r: {r}")
    r0, r1, r2, r3 = r[0], r[1], r[2], r[3]
    total = 0
    rs = {0:r0, 1:r1, 2:r2, 3:r3}
    for i in range(4):
        for j in range(4):
            r_i, r_j = rs[i], rs[j]
            total += prob_geq[(i,j)]*r_i + prob_less[(i,j)]*r_j - np.log(np.exp(r_i) + np.exp(r_j))
    return total

x0 = np.array([0,1,2,3])
opt_args = optimize.minimize(obj, x0, method='Nelder-Mead')
opt_val = obj(opt_args)

print(f"Opt Val = {opt_val}, Opt Args = {opt_args}")