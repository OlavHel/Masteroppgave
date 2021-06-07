import numpy as np
import matplotlib.pyplot as plt
from scipy.special import gamma
from loss_functions import *

def gen_data(n, rho):
    # generate n data points given the correlation rho
    data = np.random.multivariate_normal(
        np.array([0, 0]),
        np.array([[1, rho], [rho, 1]]),
        size=n
    )
    X = data[:, 0]
    Y = data[:, 1]

    return X, Y


def one_simulate_1(X, Y, sample_size, n_MCMC):
    ## simulate from the CF of rho given 3 unknown variables (2 variances + correlation)
    u = np.random.chisquare(sample_size, n_MCMC)
    v = np.random.chisquare(sample_size - 1, n_MCMC)

    z = np.random.standard_normal(n_MCMC)

    SSx = np.sum(X**2)
    SSy = np.sum(Y**2)
    SSxy = np.sum(X*Y)

    r = SSxy / np.sqrt(SSx * SSy)

    gammas = z + np.sqrt(v) * r / np.sqrt(1 - r ** 2)

    signs = np.sign(gammas)

    samples = signs * np.sqrt(gammas**2 / (gammas**2 + u))

    return samples

def mult_simulate_1(n_samples, sample_size, n_MCMC, rho):
    ## Create multiple simulations with different data set from the CF of rho given
    ##      3 unknown variables (2 variances + correlation).
    ## The output consists of
    ##      1. The number of samples below the known correlation for each simulation
    ##      2. A vector with the mean and variance of each simulation

    print("Simulating from CD1")

    all_samples = np.empty(n_samples)

    properties = np.empty((n_samples, 12))

    for i in range(n_samples):
        if i%100==0:
            print(i)
        X,Y = gen_data(sample_size, rho)

        samples = one_simulate_1(X, Y, sample_size, n_MCMC)

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


def one_simulate_2(X, Y, sample_size, n_MCMC):
    ## simulate from the CF of rho given 2 unknown variables (common variance + correlation)
    S1 = np.sum((X+Y)**2)
    S2 = np.sum((X-Y)**2)

    u = np.random.chisquare(sample_size, n_MCMC) # fix so that drop when below S/4 and generate new samples
    v = np.random.chisquare(sample_size, n_MCMC)

    eta = u/v*S1/S2

    samples = (eta-1)/(eta+1)

    return samples

def mult_simulate_2(n_samples, sample_size, n_MCMC, rho):
    ## Create multiple simulations with different data set from the CF of rho given
    ##      2 unknown variables (common variance + correlation).
    ## The output consists of
    ##      1. The number of samples below the known correlation for each simulation
    ##      2. A vector with the mean and variance of each simulation

    print("Simulating from CD2")

    all_samples = np.empty(n_samples)

    properties = np.empty((n_samples, 12))

    for i in range(n_samples):
        if i%100==0:
            print(i)
        X,Y = gen_data(sample_size, rho)

        samples = one_simulate_2(X, Y, sample_size, n_MCMC)

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


def one_simulate_3(X, Y, sample_size, n_MCMC):
    ## simulate from the CF of rho given 3 unknown variables (2 variances + correlation)
    S1 = np.sum((X+Y)**2)
    S2 = np.sum((X-Y)**2)

    U1 = np.random.chisquare(sample_size, n_MCMC) # fix so that drop when below S/4 and generate new samples
    U2 = np.random.chisquare(sample_size, n_MCMC)

    a = U2 - U1
    b1 = (S1 + S2) / a
    b2 = (S2 - S1) / a

    samples = -b1 / 4 + np.sign(a) * 1 / 4 * np.sqrt(b1 ** 2 + 16 * (1 - b2 / 2))

    return samples

def mult_simulate_3(n_samples, sample_size, n_MCMC, rho):
    ## Create multiple simulations with different data set from the CF of rho given
    ##      3 unknown variables (2 variances + correlation).
    ## The output consists of
    ##      1. The number of samples below the known correlation for each simulation
    ##      2. A vector with the mean and variance of each simulation

    print("Simulating from CD3")

    all_samples = np.empty(n_samples)


    properties = np.empty((n_samples, 12))

    for i in range(n_samples):
        if i%100==0:
            print(i)
        X,Y = gen_data(sample_size, rho)

        samples = one_simulate_3(X, Y, sample_size, n_MCMC)

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


def one_simulate_4(X, Y, sample_size, n_MCMC):
    gamma = np.prod((X+Y)**2/(X-Y)**2)
    z1s = np.random.standard_normal((sample_size,n_MCMC))
    z2s = np.random.standard_normal((sample_size,n_MCMC))
    Z = np.prod(z1s**2/z2s**2,axis=0)

    eta = (Z/gamma)**(1/sample_size)

    samples = (1-eta)/(eta+1)

    return samples

def mult_simulate_4(n_samples, sample_size, n_MCMC, rho):
    ## Create multiple simulations with different data set from the CF of rho given
    ##      2 unknown variables (common variance + correlation).
    ## The output consists of
    ##      1. The number of samples below the known correlation for each simulation
    ##      2. A vector with the mean and variance of each simulation

    print("Simulating from CD4")

    all_samples = np.empty(n_samples)

    properties = np.empty((n_samples, 12))

    for i in range(n_samples):
        if i%100==0:
            print(i)
        X,Y = gen_data(sample_size, rho)

        samples = one_simulate_4(X, Y, sample_size, n_MCMC)

        plt.figure()
        plt.hist(samples, bins = 100)
        plt.show()
        plt.figure()
        plt.hist(np.arctanh(samples), bins=100)
        plt.show()

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


def one_simulate_5(X, Y, sample_size, n_MCMC):
    gamma = np.sum((X+Y)**2/(X-Y)**2)
    z1s = np.random.standard_normal((sample_size,n_MCMC))
    z2s = np.random.standard_normal((sample_size,n_MCMC))
    Z = np.sum(z1s**2/z2s**2,axis=0)

    eta = Z/gamma

    samples = (1-eta)/(eta+1)

    return samples

def mult_simulate_5(n_samples, sample_size, n_MCMC, rho):
    ## Create multiple simulations with different data set from the CF of rho given
    ##      2 unknown variables (common variance + correlation).
    ## The output consists of
    ##      1. The number of samples below the known correlation for each simulation
    ##      2. A vector with the mean and variance of each simulation

    print("Simulating from CD5")

    all_samples = np.empty(n_samples)

    properties = np.empty((n_samples, 12))

    for i in range(n_samples):
        if i%100==0:
            print(i)
        X,Y = gen_data(sample_size, rho)

        samples = one_simulate_5(X, Y, sample_size, n_MCMC)

#        plt.figure()
#        plt.hist(samples, bins = 100)
#        plt.show()
#        plt.figure()
#        plt.hist(np.arctanh(samples), bins=100)
#        plt.show()

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



if __name__ == "__main__":
    if True:
        import pickle

        rhos = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        n = 3
        CD_number = 5

        m = 100000
        s2 = 0.01
        start = 0

        n_samples = 1000

        for rho in rhos:
            print("rho",rho)
            CD_list = [mult_simulate_1, mult_simulate_2, mult_simulate_3, mult_simulate_4, mult_simulate_5]


            # HER KAN DU VELGE FORDELING VED Å ENDRE 1 til 2 eller motsatt
            samples, properties = CD_list[CD_number-1](n_samples, n, m, rho)

            rho_to_print = round(10*rho)

            pickle.dump({
                "samples": samples,
                "properties": properties
            }, open("CD_samples_n_"+str(n)+"/CD"+str(CD_number)+
                    f"{rho_to_print:02}" +
                     "1000.p", "wb"))

        risks = np.mean(properties, axis=0)
        risk_names = ["Mean mean:", "Mean var:", "Mean MAE:", "Mean MSE:", "Mean FIM:", "Mean KLD:", "Mean z_mean:",
                      "Mean z_MSE:", "Mean w_mean:", "Mean w_MSE:", "Mean f_mean:", "Mean f_MSE:"]

        if len(risks) != len(risk_names):
            print("Number of names and properties do not match!")
            1 / 0

        for i in range(len(risks)):
            print(risk_names[i], risks[i])

        print(samples)

        print("for alpha=0.95")
        print(np.sum(samples < 0.95) / n_samples)

        print("for alpha=0.9")
        print(np.sum(samples < 0.9) / n_samples)

        print("for alpha=0.75")
        print(np.sum(samples < 0.75) / n_samples)

        print("for alpha=0.5")
        print(np.sum(samples < 0.5) / n_samples)

        print("Mean mean:", np.mean(properties[:,0]))
        print("Mean var + var mean:", np.mean(properties[:,1]) + np.var(properties[:,0]))

        alphas = np.linspace(0, 1, 100)
        confs = np.array([np.sum(samples < alpha) / n_samples for alpha in alphas])

        plt.figure()
        plt.plot(alphas, confs, label="confs")
        plt.plot(alphas, alphas, label="linear")
        plt.legend()
        plt.show()

    elif True:
        n = 3
        rho = 0.5

        m = 1000000
        s2 = 0.01
        start = 0

        X, Y = gen_data(n, rho)

        S1 = np.sum((X+Y)**2)
        S2 = np.sum((X-Y)**2)

        print("S_1",S1,"S_2",S2)

        samples = one_simulate_2(X, Y, n, m)

        def sym_test(samples, median = None):
            if median is None:
                median = np.median(samples)
            plt.figure()
            pos_samples = samples[samples >= median]
            neg_samples = samples[samples < median]

            plt.hist(2*median-pos_samples, bins=100,density=True, histtype="step")
            plt.hist(neg_samples, bins=100,density=True, histtype="step")

            plt.show()

        f_samples = 2*np.arctanh(samples)#fisher_information(samples)
        f_mean = np.mean(f_samples)
        f_var = np.var(f_samples)
        f_median = np.median(f_samples)

        sym_test(f_samples, f_median)

        print(np.sum(f_samples-f_median))

        print((S1-S2)/(S2+S1),2*np.sum(X*Y)/np.sum(X**2+Y**2))

        plt.figure(2)
        #        plt.hist(1/2*np.log((1+samples)/(1-samples)), density=True, bins=100)
        plt.hist(samples, density = True, bins = 100)
        #        plt.hist(f_samples, density=True, bins=100)
        plt.axvline(x = fisher_information(rho), color = "green")
        plt.axvline(x = f_mean, color = "red")
        plt.axvline(x = f_median, color = "yellow")
        #        plt.axvline(x = np.log(S1/S2),color="green")
        #        plt.axvline(x = (S1-S2)/(S2+S1), color="green")
        #        plt.axvline(x = np.median(samples), color="red")

        #        rhos = np.linspace(-0.99, 0.99, 100)
        #        S1 = np.sum((X+Y)**2)
        #        S2 = np.sum((X-Y)**2)
        #        pdfs = 2*gamma(n)/(gamma(n/2)**2)*(S2/S1*(1+rhos)/(1-rhos))**(n/2-1)/(S2/S1*(1+rhos)/(1-rhos)+1)**(n)*1/(1-rhos)**2*S2/S1
        #        plt.plot(rhos,pdfs)
        plt.show()





