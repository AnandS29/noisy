import torch
import numpy as np

def partial_obj(r0, r1, r2, r3, i, j):
    if i > j:
        i, j = j, i

    if i == 0 and j == 0:
        return 2*r0 - np.log(2*np.exp(r0))

    if i == 1 and j == 1:
        return 2*r1 - np.log(2*np.exp(r1))

    if i == 2 and j == 2:
        return 0.75*r2 - np.log(2*np.exp(r2))

    if i == 3 and j == 3:
        return 0.75*r3 - np.log(2*np.exp(r3))

    if i == 0 and j == 1:
        return r1 - np.log(np.exp(r0) + np.exp(r1))

    if i == 0 and j == 2:
        return 0.5*r0 + r2 - np.log(np.exp(r0) + np.exp(r2))

    if i == 0 and j == 3:
        return 0.5*r0 + r3 - np.log(np.exp(r0) + np.exp(r3))

    if i == 1 and j == 2:
        return 0.5*r1 + 0.5*r2 - np.log(np.exp(r1) + np.exp(r2))

    if i == 1 and j == 3:
        return 0.5*r1 + 0.5*r3 - np.log(np.exp(r1) + np.exp(r3))

    if i == 2 and j == 3:
        return 0.5*r2 + 0.75*r3 - np.log(np.exp(r1) + np.exp(r3))

def obj(r0, r1, r2, r3):
    total = 0
    for i in range(4):
        for j in range(4):
            total += partial_obj(r0, r1, r2, r3, i, j)
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

n = 500
disc = np.linspace(-5, 5, n)

opt_val, opt_args = grid_search(disc, n)

print(f"Opt Val = {opt_val}, Opt Args = {opt_args}")