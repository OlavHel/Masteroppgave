import numpy as np
import matplotlib.pyplot as plt
from simulate_CD import gen_data, one_simulate_1, one_simulate_2
from MCMC_test2 import one_simulation, posterior_distr
from posteriors import Posterior

def compare_widths(sim_funcs, alphas, n_samples, sample_size, n_MCMC, rho):
    ## Create multiple simulations with different data set from the CF of rho given
    ##      2 unknown variables (common variance + correlation).
    ## The output consists of
    ##      1. The number of samples below the known correlation for each simulation
    ##      2. A vector with the mean and variance of each simulation
    n_funcs = len(sim_funcs)
    n_alphas = len(alphas)
    results = np.empty((n_samples,n_funcs,n_alphas))

    for i in range(n_samples):
        print(i)
        X,Y = gen_data(sample_size, rho)

        for j in range(n_funcs):
            func = sim_funcs[j]
            samples = func(X, Y, sample_size, n_MCMC)

            upper_quantiles = np.quantile(samples,alphas)
            results[i,j,:] = upper_quantiles

    return results

def table_it(widths, func_names, n_funcs, alpha_index):
    tab_form = "{:^4d}"

    print("")
    print("func_numbers:",func_names)
    print(tab_form.format(0),end=" ")
    for i in range(n_funcs):
        print(tab_form.format(i),end=" ")
    print("")
    for i in range(n_funcs):
        print(tab_form.format(i),end=" ")
        for j in range(n_funcs):
            print(tab_form.format(np.sum(widths[:,i,alpha_index]>widths[:,j,alpha_index])),end = " ")
        print("")
    print("")


if __name__ == "__main__":
    if False:
        start = 0
        s2 = 0.01
        n = 3
        rho = 0.8

        m = 100000
        s2 = 0.01
        start = 0

        n_samples = 1000

        poster = Posterior("jeffrey", lam=10 ** (-4)).distribution
        distr = lambda x, n, X, Y: posterior_distr(x, n, X, Y, poster)
        sim_post = lambda X, Y, sample_size, n_MCMC: one_simulation(n_MCMC, sample_size, X, Y, distr, s2, start)

        sim_funcs = [sim_post, one_simulate_1, one_simulate_2]
        func_names = ["jeffrey", "CD1","CD2"]
        alphas = [0.99,0.975,0.95,0.9,0.75,0.5,0.05,0.025]
        n_funcs = len(sim_funcs)
        n_alphas = len(alphas)

        widths = compare_widths(sim_funcs,alphas,n_samples,n,m,rho)

        print(widths)

        cols = ["blue","red","green"]

        print("name","mean","var")
        for i in range(n_funcs):
            print(func_names[i],np.mean(widths[:,i,:],axis=0), np.var(widths[:,i,:],axis=0))

        for i in range(n_alphas):
            print("For alpha="+str(alphas[i]))
            table_it(widths, func_names, n_funcs, i)

        plt.figure()
        for i in range(n_funcs):
            plt.plot(widths[:,i,0],"o",color = cols[i], label=func_names[i])
        plt.legend()
        plt.show()
    elif True:
        start = 0
        s2 = 0.01
        n = 3
        rho = 0.5

        m = 100000
        s2 = 0.01
        start = 0

        n_samples = 1000

        poster = Posterior("jeffrey", lam=10 ** (-4)).distribution
        distr = lambda x, n, X, Y: posterior_distr(x, n, X, Y, poster)
        sim_post = lambda X, Y, sample_size, n_MCMC: one_simulation(n_MCMC, sample_size, X, Y, distr, s2, start)

        sim_funcs = [sim_post, one_simulate_1, one_simulate_2]
        func_names = ["jeffrey", "CD1", "CD2"]
        n_funcs = len(sim_funcs)

        X, Y = gen_data(n,rho)


#        samples = np.empty((m,n_funcs))
#        for i in range(n_funcs):
#            samples[:,i] = sim_funcs[i](X, Y, n, m)

        rhos = np.linspace(-0.999,0.999,1000)
        from MCMC_test2 import taraldsen_dens
        from scipy.special import gamma

        S1 = np.sum((X+Y)**2)
        S2 = np.sum((X-Y)**2)
        jef = distr(rhos,n, X, Y)/Posterior("jeffrey").normalization(n,1/2*(S1+S2),1/4*(S1-S2))
        cd1 = taraldsen_dens(rhos, n, X, Y)
        us = S2/S1*(1+rhos)/(1-rhos)
        cd2 = 2*gamma(n)/(gamma(n/2)**2)*((1+us)*(1+1/us))**(-n/2)/(1-rhos**2)

        plt.figure()
#        for i in range(n_funcs):
#            plt.hist(samples[:,i],bins = 100, density=True, label=func_names[i])
        plt.plot(rhos,jef,label="jeffrey")
        plt.plot(rhos,cd1,label="CD1")
        plt.plot(rhos,cd2,label="CD2")
        plt.legend()
        plt.show()





