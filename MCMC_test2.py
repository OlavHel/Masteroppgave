import numpy as np
import matplotlib.pyplot as plt
from posteriors import Posterior
from scipy.special import hyp2f1
from scipy.special import gamma
import pickle
from loss_functions import *
import time


def one_simulation(n_MCMC, sample_size, X, Y, dens,s2,start):
    # USE MCMC TO SIMULATE n_MCMC SAMPLES FROM THE DENSITY dens with n=sample_size and data X,Y.
    s = np.sqrt(s2)
    data = np.empty(n_MCMC) # the list of sampled values
    randoms = s*np.random.standard_normal(n_MCMC) # proposal distribution is z_i=N(z_{i-1},s^2).
    unifs = np.random.uniform(0,1,n_MCMC)

    post = lambda x: dens(x, sample_size, X, Y)

    sample = start # the last sample
    for i in range(n_MCMC):
        new_guess = sample+randoms[i]

        alpha = post(new_guess)/post(sample)
        if unifs[i] <= alpha:
            sample = new_guess

        data[i] = sample

    return data

def gen_data(n,rho):
    # generate n data points X,Y given correlation rho
    data = np.random.multivariate_normal(
        np.array([0, 0]),
        np.array([[1, rho], [rho, 1]]),
        size=n
    )
    X = data[:, 0]
    Y = data[:, 1]

    return X, Y



def mult_simulations(n_samples,sample_size,n_MCMC,s2,start, dens, rho):
    # Run n_samples MCMC simulations, calculate the CFD value at rho and the confidence loss for each simulation
    all_samples = np.empty(n_samples)

    properties = np.empty((n_samples, 12)) # list of all confidence losses and expected values for various parametrizations

    start_time = time.time() # to keep track of time
    prev_time = start_time
    for i in range(n_samples):
        print(i)
        print("time since start:",time.time()-start_time)
        print("last time:",time.time()-prev_time)
        prev_time = time.time()
        X, Y = gen_data(sample_size,rho) # generate data
        
        samples = one_simulation(n_MCMC, sample_size, X, Y, dens, s2, start) # run simulations

        properties[i,:] = np.array([ # calculate the loss and means
            np.mean(samples),
            np.var(samples),
            MAE(samples,rho),
            MSE(samples,rho),
            np.mean(fisher_information_metric(samples,rho)),
            np.mean(kullback_leibler(samples,rho)),
            z_transMean(samples),
            z_transMSE(samples,rho),
            w_transMean(samples),
            w_transMSE(samples, rho),
            fishMean(samples),
            fishMSE(samples,rho)
        ])

        # calculate the CDF value at rho
        num_below_rho = np.sum(samples < rho)
        all_samples[i] = num_below_rho / n_MCMC
    
    return all_samples, properties

def posterior_distr(x, n, X, Y, post):
    # a function used to generalize the structure of the functions above. Lets densities that uses statistics
    # T1 and T2 to be used in the one_simulation function which uses data on the form (X,Y)
    T1 = np.sum(X**2+Y**2)
    T2 = np.sum(X*Y)

    return post(x, n, T1, T2)



if __name__ == "__main__":
    # code for testing frequentistic coverage and calculating risks for a posterior or GFD under different correlations
    prior_name = "fiduc_2" # the name of the "prior" to be used. Is linked to the names given in the Posterior class
    poster = Posterior(prior_name,lam=10**(-4)).distribution
    distr = lambda x, n, X, Y: posterior_distr(x, n, X, Y, poster) # make sure that the density can use original data

    n = 3 # number of data points in a data set

    rhos = [0.9] # list of all correlations to simulate using
    dist_name = "fiduc2" # name of the prior to be used when saving the results

    m = 100000 # number of simulations in MCMC
    s2 = 0.01 # variance for the proposal distribution of the MCMC
    start = 0 # starting value for the MCMC

    n_samples = 1000 # number of simulations to create

    for rho in rhos:
        rho_to_print = round(10*rho)
        print("Simulation of "+dist_name+" for rho="+str(rho_to_print))
        samples, properties = mult_simulations(n_samples, n, m, s2, start, distr, rho)

        # store the results in a pickle file
        pickle.dump({
            "samples": samples,
            "properties": properties
        }, open("CD_samples_n_"+str(n)+"/"+dist_name+
                f"{rho_to_print:02}" +
                 "1000.p", "wb"))




