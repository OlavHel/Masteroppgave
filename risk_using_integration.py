import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad
from loss_functions import *
from posteriors import Posterior
from scipy.special import gamma
from scipy.stats import gamma as dist_gamma
from time import time

def risk(rho,density, loss, n, S1,S2):
    T1 = 1/2*(S1+S2)
    T2 = 1/4*(S1-S2)
    c = quad(density, -1, 1, args=(n, T1, T2))[0]

    loss_density = lambda x: loss(rho,x)*density(x,n, T1, T2)/c
    return quad(loss_density, -1,1)

jeff = Posterior("jeffrey").distribution

rhos = [0.5]
losses = [fisher_information_metric]

risks = np.empty((len(rhos),len(losses)))

for rho in rhos:
    n = 3

    quantiles1 = dist_gamma.ppf([0.001,0.999],a=n/2, scale = 4*(1+rho))
    quantiles2 = dist_gamma.ppf([0.001,0.999],a=n/2, scale = 4*(1-rho))
    start = time()

    S1S2_func = lambda S1, S2: risk(rho, jeff, fisher_information_metric, n, S1, S2)[0]*S1**(n/2-1)*np.exp(-1/(4*(1+rho))*S1)
    S2_func = lambda S2: quad(S1S2_func,quantiles1[0],quantiles1[1], args=S2)[0]*S2**(n/2-1)*np.exp(-1/(4*(1-rho))*S2)
    result = quad(S2_func,quantiles2[0],quantiles2[1])
    print(result, result/(gamma(n/2)**2*4**n*(1-rho**2)**(n/2)))
    print(time()-start)


