import numpy as np
import matplotlib.pyplot as plt
from posteriors import Posterior
from scipy.special import hyp2f1
from scipy.special import gamma
import pickle
from loss_functions import *
import time


def one_simulation(n_MCMC, sample_size, X, Y, dens,s2,start):
    s = np.sqrt(s2)
    data = np.empty(n_MCMC)
    randoms = s*np.random.standard_normal(n_MCMC)
    unifs = np.random.uniform(0,1,n_MCMC)

    post = lambda x: dens(x, sample_size, X, Y)

    sample = start
    for i in range(n_MCMC):
        new_guess = sample+randoms[i]

        alpha = post(new_guess)/post(sample)
        if unifs[i] <= alpha:
            sample = new_guess

        data[i] = sample

    return data

def gen_data(n,rho):
    data = np.random.multivariate_normal(
        np.array([0, 0]),
        np.array([[1, rho], [rho, 1]]),
        size=n
    )
    X = data[:, 0]
    Y = data[:, 1]

    return X, Y



def mult_simulations(n_samples,sample_size,n_MCMC,s2,start, dens, rho):
    all_samples = np.empty(n_samples)

    properties = np.empty((n_samples, 12))

    start_time = time.time()
    prev_time = start_time
    for i in range(n_samples):
        print(i)
        print("time since start:",time.time()-start_time)
        print("last time:",time.time()-prev_time)
        prev_time = time.time()
        X, Y = gen_data(sample_size,rho)
        
        samples = one_simulation(n_MCMC, sample_size, X, Y, dens, s2, start)

        properties[i,:] = np.array([
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

        num_below_rho = np.sum(samples < rho)
        
        all_samples[i] = num_below_rho / n_MCMC
    
    return all_samples, properties

def posterior_distr(x, n, X, Y, post):
    T1 = np.sum(X**2+Y**2)
    T2 = np.sum(X*Y)

    return post(x, n, T1, T2)

def taraldsen_dens(x, n, X, Y):
    if type(x) == type(np.array([0.1])) and len(x) == 1:
        x = x[0]
    if type(x) == type(np.array([])):
        x[x <= -1] = 0
        x[x >= 1] = 0
    elif (x <= (-1) or x >= (1)):
        return 0

    mu_X = np.sum(X)/n
    mu_Y = np.sum(Y)/n

    SSx = np.sum((X-mu_X)**2)
    SSy = np.sum((Y-mu_Y)**2)
    SSxy = np.sum((X-mu_X)*(Y-mu_Y))

    r = SSxy/np.sqrt(SSx*SSy)

    nu = n

    c = nu*(nu-1)*gamma(nu-1)/(np.sqrt(2*np.pi)*gamma(nu+1/2))*(1-r**2)**((nu-1)/2)

    return c*(1-x**2)**((nu-2)/2)*(1-x*r)**((1-2*nu)/2)*hyp2f1(3/2,-1/2,nu+1/2,(1+r*x)/2)


def loccond(theta, x, n, X, Y):
    if type(x) == type(np.array([0.1])) and len(x) == 1:
        x = x[0]
    if type(x) == type(np.array([])):
        x[x <= -1] = 0
        x[x >= 1] = 0
    elif (x <= (-1) or x >= (1)):
        return 0

    S1 = np.sum((X + Y) ** 2)
    S2 = np.sum((X - Y) ** 2)

    if theta >= 1:
        f_theta = (1 + theta) ** (1 / 2 + theta / 2)
        f_x = (1 - x) ** (1 / 2 + theta / 2)
    elif theta <= -1:
        f_theta = (1 - theta) ** (1 / 2 - theta / 2)
        f_x = (1 + x) ** (1 / 2 - theta / 2)
    else:
        f_theta = (1+theta)**(1/2+theta/2)*(1-theta)**(1/2-theta/2)
        f_x = (1+x)**(1/2-theta/2)*(1-x)**(1/2+theta/2)

    return f_x**n*1/(1-x**2)*(1-x**2)**(-n/2)*np.exp(-1/2*f_x/f_theta*(S1/(1+x)+S2/(1-x)))




if __name__ == "__main__":
    if True:
        poster = Posterior("arcsine",lam=10**(-4)).distribution
        distr = lambda x, n, X, Y: posterior_distr(x, n, X, Y, poster)

        n = 20
        rho = 0.0

        m = 100000
        s2 = 0.01
        start = 0

        n_samples = 1000

        samples, properties = mult_simulations(n_samples, n, m, s2, start, distr, rho)

        pickle.dump({
            "samples": samples,
            "properties": properties
            }, open("CD_samples_n_20/arcsine001000.p", "wb")
        )

        risks = np.mean(properties, axis=0)
        risk_names = ["Mean mean:", "Mean var:", "Mean MAE:", "Mean MSE:", "Mean FIM:", "Mean KLD:", "Mean z_mean:",
                      "Mean z_MSE:", "Mean w_mean:", "Mean w_MSE:", "Mean f_mean:", "Mean f_MSE:"]
        print("")

        for i in range(len(risks)):
            print(risk_names[i], risks[i])
        print("")


        print(samples)

        print("for alpha=0.95")
        print(np.sum(samples < 0.95)/n_samples)

        print("for alpha=0.9")
        print(np.sum(samples < 0.9)/n_samples)

        print("for alpha=0.75")
        print(np.sum(samples < 0.75)/n_samples)

        print("for alpha=0.5")
        print(np.sum(samples < 0.5)/n_samples)

        alphas = np.linspace(0,1,100)
        confs = np.array([np.sum((samples < 1-(1-alpha)/2) & (samples > (1-alpha)/2))/n_samples for alpha in alphas])

        plt.figure()
        plt.plot(alphas,confs, label="confs")
        plt.plot(alphas,alphas, label="linear")
        plt.legend()
        plt.show()

    elif True:
        poster = Posterior("fiduc_infty",lam=10**(-4)).distribution
        distr = lambda x, n, T1, T2: posterior_distr(x, n, T1, T2, poster)

        n = 3
        rho = 0.8

        m = 100000
        s2 = 0.01
        start = 0

        X, Y = gen_data(n,rho)

        samples = one_simulation(m, n, X, Y, distr,s2,start)

        def sym_test(samples, median = None):
            if median is None:
                median = np.median(samples)
            plt.figure()
            pos_samples = samples[samples >= median]
            neg_samples = samples[samples < median]

            plt.hist(2*median-pos_samples, bins=100,density=True, histtype="step")
            plt.hist(neg_samples, bins=100,density=True, histtype="step")

            plt.show()


        f_samples = samples#np.arctanh(samples)#fisher_information(samples)
        f_mean = np.mean(f_samples)
        f_var = np.var(f_samples)
        f_median = np.median(f_samples)

#        sym_test(f_samples, f_median)

        print(np.sum(f_samples - f_median))
#        plt.figure(1)
        a_list = np.linspace(0, 3 * np.sqrt(f_var), 100)
        pos_side = np.array([np.sum((f_samples - f_mean) > a) for a in a_list])
        neg_side = np.array([np.sum((f_samples - f_mean) < -a) for a in a_list])
        #        plt.plot(a_list, pos_side)
        #        plt.plot(a_list, neg_side)
#        plt.plot(a_list, (pos_side - neg_side) / m)

        plt.figure(2)
        #        plt.hist(1/2*np.log((1+samples)/(1-samples)), density=True, bins=100)
        plt.hist(f_samples, density=True, bins=100)
        plt.axvline(x=fisher_information(rho), color="green")
        plt.axvline(x=f_mean, color="red")
        plt.axvline(x=f_median, color="yellow")

        rhos = np.linspace(-0.99,0.99,100)
#        plt.plot(rhos, distr(rhos,n, X, T))
        plt.show()




