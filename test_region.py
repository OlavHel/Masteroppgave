import numpy as np
import matplotlib.pyplot as plt
import time
from scipy.stats import gamma
from scipy.optimize import fsolve, bisect, brentq, toms748
from loss_functions import *
import pickle

# code used for the method of regions


def sim_any_pivot_g(X, Y, n, m, g):
    # can simulate the CD corresponding to the APF phi(x,y)=g(x)-g(y) numerically
    def f_inv(y, x):
        return g(y) - g(x)

    def func_to_solve(x,S1,S2,a):
        if x >= 1:
            return np.infty
        elif x <= -1:
            return -np.infty
        return f_inv(S2/(2*(1-x)),S1/(2*(1+x)))-a

    U1 = np.random.chisquare(n, size=m)
    U2 = np.random.chisquare(n, size=m)

    S1 = np.sum((X+Y)**2)
    S2 = np.sum((X-Y)**2)

    samples = np.array([
        brentq(func_to_solve, -1, 1, args=(S1, S2, f_inv(U2[j], U1[j]))) for j in range(m)
    ])

    return samples

