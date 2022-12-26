from scipy import integrate
import numpy as np

t = 0.8

def obj(r):
    def fn(x,y,ex,ey):
        if x < t:
            p_ex = 1 if ex == 0 else 0
        else:
            p_ex = 0.5 if ex in [-x,x] else 0
        if y < t:
            p_ey = 1 if ey == 0 else 0
        else:
            p_ey = 0.5 if ey in [-y,y] else 0

        prob = 1*1*p_ex*p_ey
        geq = 1 if x + ex >= y+ey else 0
        less = 1 if x + ex < y+ey else 0
        return prob*(geq*r(x) + less*r(y) - np.log(np.exp(r(x)) + np.exp(r(y))))
    return fn

def create_linear_fn(a, b):
    def fn(x):
        return a*x + b
    return fn

def create_piecewise_fn(a, b, c, d):
    def fn(x):
        if x < t:
            return a*x + b
        else:
            return c*x + d
    return fn

param_range = np.linspace(-10, 10, 100)
linear_params = [(a,b) for a in param_range for b in param_range]
piecewise_params = [(a,b,c,d) for a in param_range for b in param_range for c in param_range for d in param_range]

def get_optimal_params(params, obj, create_fn, verbose=False):
    opt_val = None
    opt_args = None
    val, args = [], []
    i = 0
    for p in params:
        if verbose:
            if (i*100/len(params)) % 5 == 0:
                print(f'{i*100/len(params)}%')
        fn = create_fn(*p)
        curr = integrate.nquad(obj(fn), [[0,1], [0,1], [-1,1], [-1,1]])
        if opt_val is None or curr > opt_val:
            opt_val = curr
            opt_args = p
        val.append(curr)
        args.append(p)
        i += 1
    return opt_val, opt_args, val, args

# Linear
opt_val_linear, opt_args_linear, val_linear, args_linear = get_optimal_params(linear_params, obj, create_linear_fn, verbose=True)
print(f"Opt Val Linear = {opt_val_linear}, Opt Args Linear = {opt_args_linear}")

# Piecewise
opt_val_piecewise, opt_args_piecewise, val_piecewise, args_piecewise = get_optimal_params(piecewise_params, obj, create_piecewise_fn, verbose=True)
print(f"Opt Val Piecewise = {opt_val_piecewise}, Opt Args Piecewise = {opt_args_piecewise}")

# Save
np.savez("linear.npz", val=val_linear, args=args_linear)
np.savez("piecewise.npz", val=val_piecewise, args=args_piecewise)