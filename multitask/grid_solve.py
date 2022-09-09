import torch
import numpy as np

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

def obj(r0, r1, r2, r3):
    total = 0
    rs = {0:r0, 1:r1, 2:r2, 3:r3}
    for i in range(4):
        for j in range(4):
            r_i, r_j = rs[i], rs[j]
            total += prob_geq[(i,j)]*r_i + prob_less[(i,j)]*r_j - np.log(np.exp(r_i) + np.exp(r_j))
    return total

def grid_search(disc, n):
    opt_val = None
    opt_args = None
    i = 0
    for r0 in disc:
        for r1 in disc:
            for r2 in disc:
                for r3 in disc:
                    curr = obj(r0, r1, r2, r3)
                    if opt_val is None or curr > opt_val:
                        opt_val = curr
                        opt_args = (r0, r1, r2, r3)

                    i += 1
                    if i % (n**4 // 20) == 0: print(f"Done {np.round(i/(n**4)*100)}%")
    return opt_val, opt_args

n = 1000
disc = np.linspace(-5, 5, n)

opt_val, opt_args = grid_search(disc, n)

print(f"Opt Val = {opt_val}, Opt Args = {opt_args}")