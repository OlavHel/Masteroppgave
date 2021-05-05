import numpy as np
import matplotlib.pyplot as plt
import time
from scipy.stats import gamma
from scipy.optimize import fsolve, bisect, brentq, toms748
from loss_functions import *
import pickle


def dist_func(S1, S2, U1, U2):
    a = U2 - U1
    b1 = (S1 + S2) / a
    b2 = (S2 - S1) / a

    return -b1 / 4 + np.sign(a) * 1 / 4 * np.sqrt(b1 ** 2 + 16 * (1 - b2 / 2))

def sim_pivot_diff(X, Y, n, m):
    U1 = np.random.chisquare(n, size=m)
    U2 = np.random.chisquare(n, size=m)

    S1 = np.sum((X+Y)**2)
    S2 = np.sum((X-Y)**2)

    return dist_func(S1, S2, U1, U2)

def sim_any_pivot_g(X, Y, n, m, g):
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



if not __name__ == "__main__":
    pass
elif False:
    rho = 0.5
    n = 3
    n_samples = 100000

    ##########################
    ## 0.95: 7.81, 0.90: 6.25, 0.999: 16.27
    ## 0.05: 0.35
    ##

    bound = 0.35#16.27#6.25#7.81

    Ss = np.empty((n_samples, 2))

    S1 = np.random.gamma(shape=n/2, scale=4*(1+rho), size=n_samples)
    S2 = np.random.gamma(shape=n/2, scale=4*(1-rho), size=n_samples)

    Ss[:,0] = S1
    Ss[:,1] = S2

    alphas = np.linspace(0, 1, 100)

    confs_lower = np.array(
        [np.sum((gamma.cdf(Ss[:, 1], a=n/2, scale=4*(1-rho)) < alpha)) / n_samples for alpha in alphas])

    plt.figure(2)
    plt.plot(alphas, np.array(
        [np.sum((gamma.cdf(Ss[:, 1], a=n/2, scale=8) > alpha)) / n_samples for alpha in alphas]))

    plt.figure(1)
    plt.plot(alphas,alphas)
    plt.plot(alphas,confs_lower)
    plt.show()

elif False:
    rho = 0.0
    n = 3
    n_samples = 100000

    S1 = np.random.gamma(shape=n / 2, scale=4 * (1 + rho), size=n_samples)
    S2 = np.random.gamma(shape=n / 2, scale=4 * (1 - rho), size=n_samples)

    U1 = np.random.chisquare(n, size=n_samples)
    U2 = np.random.chisquare(n, size=n_samples)

    ratios = U2/U1

    alphas = np.linspace(0, 1, 200)

    cums = np.quantile(ratios, alphas)

    confs_lower = np.array(
        [np.sum(rho <= (a*S1-S2)/(a*S1+S2)) / n_samples for a in cums])

    confs_upper = np.flip(np.array(
        [np.sum(rho > (a*S1-S2)/(a*S1+S2)) / n_samples for a in cums]))

    g = lambda x,a: 4*a*x-S1*a

    confs_lower = np.array(
        [np.sum(S2 <= g(S1/(2*(1+rho)),a)) / n_samples for a in cums])

    plt.figure()
    plt.plot(alphas, alphas)
    plt.plot(alphas, confs_lower)
    plt.plot(alphas, confs_upper)
    plt.show()

elif False:
    rhos = np.linspace(-0.99,0.99,10)
    plt.figure()
    for rho in rhos:
        n = 3
        n_samples = 100000

        S1 = np.random.gamma(shape=n / 2, scale=4 * (1 + rho), size=n_samples)
        S2 = np.random.gamma(shape=n / 2, scale=4 * (1 - rho), size=n_samples)

        U1 = np.random.chisquare(n, size=n_samples)
        U2 = np.random.chisquare(n, size=n_samples)

        ratios = U2/U1

        alphas = np.linspace(0, 1, 100)

        cums = np.quantile(ratios, alphas)

        confs_lower = np.array(
            [np.sum(rho <= (a*S1-S2)/(a*S1+S2)) / n_samples for a in cums])

        confs_both = np.array(
            [
                np.sum( (rho <= (np.quantile(ratios, 1/2+alpha/2)*S1-S2)/(np.quantile(ratios, 1/2+alpha/2)*S1+S2) ) &
                        (rho > (np.quantile(ratios, 1/2-alpha/2)*S1-S2)/(np.quantile(ratios, 1/2-alpha/2)*S1+S2) ) )/n_samples for alpha in alphas
            ]
        )

#        plt.plot(alphas, alphas)
        plt.plot(alphas, confs_lower,label=rho)
#        plt.plot(alphas, confs_both)
    plt.show()

elif False:
    rho = -0.99
    n = 3
    n_samples = 100000

    S1 = np.random.gamma(shape=n / 2, scale=4 * (1 + rho), size=n_samples)
    S2 = np.random.gamma(shape=n / 2, scale=4 * (1 - rho), size=n_samples)

    U1 = np.random.chisquare(n, size=n_samples)
    U2 = np.random.chisquare(n, size=n_samples)

    ratios = U2/U1**2

    alphas = np.linspace(0, 1, 100)

    cums = np.quantile(ratios, alphas)

    g = lambda x, a: (4-S1/x)*(a*x**2)

    confs_lower = np.array(
        [np.sum(rho <= (3-np.sqrt(1+16*S2/(a*S1)))/(1+np.sqrt(1+16*S2/(a*S1)))) / n_samples for a in cums])

    confs_lower = np.array(
        [np.sum( g(S1/(2*(1+rho)), a) >= S2) / n_samples for a in cums])

    plt.figure()
    plt.plot(alphas, alphas)
    plt.plot(alphas, confs_lower)
    plt.show()

elif False:
    rho = 0.0
    n = 3
    n_samples = 100000

    S1 = np.random.gamma(shape=n / 2, scale=4 * (1 + rho), size=n_samples)
    S2 = np.random.gamma(shape=n / 2, scale=4 * (1 - rho), size=n_samples)

    U1 = np.random.chisquare(n, size=n_samples)
    U2 = np.random.chisquare(n, size=n_samples)

    ratios = np.log(U2+1)/np.log(U1+1)

    alphas = np.linspace(0, 1, 100)

    cums = np.quantile(ratios, alphas)

    print(cums[:10])

    def g(x,a):
#        print(x,a)
        return (4-S1/x)*((x+1)**a-1)


    confs_lower = np.array(
        [np.sum( g(S1/(2*(1+rho)), a) >= S2) / n_samples for a in cums])

    confs_upper = np.flip(np.array(
        [np.sum( g(S1/(2*(1+rho)), a) < S2) / n_samples for a in cums]))

    plt.figure()
    plt.plot(alphas, alphas)
    plt.plot(alphas, confs_lower)
    plt.plot(alphas, confs_upper)
    plt.show()

elif False:
    rho = -0.3
    n = 3
    n_samples = 100000

    b = 10

    S1 = np.random.gamma(shape=n / 2, scale=4 * (1 + rho), size=n_samples)
    S2 = np.random.gamma(shape=n / 2, scale=4 * (1 - rho), size=n_samples)

    U1 = np.random.chisquare(n, size=n_samples)
    U2 = np.random.chisquare(n, size=n_samples)

    ratios = U2**b-U1**b

    alphas = np.linspace(0, 0.99, 100)

    cums = -np.quantile(ratios, alphas)

    def g(x,a):
#        print(x,a)
        return (4-S1/x)*(U1**b-a)**(1/b)


    confs_lower = np.array(
        [np.sum( g(S1/(2*(1+rho)), a) >= S2) / n_samples for a in cums])

#    confs_upper = np.flip(np.array(
#        [np.sum( g(S1/(2*(1+rho)), a) < S2) / n_samples for a in cums]))

    plt.figure()
    plt.plot(alphas, alphas)
    plt.plot(alphas, confs_lower)
#    plt.plot(alphas, confs_upper)
    plt.show()


elif False:
    rho = 0.0
    n = 3
    n_samples = 10000
    n_MCMC = 100000

    S1 = np.random.gamma(shape=n / 2, scale=4 * (1 + rho), size=n_samples)
    S2 = np.random.gamma(shape=n / 2, scale=4 * (1 - rho), size=n_samples)

    samples = np.empty(n_samples)
    for i in range(n_samples):
        if i%100 == 0:
            print(i)
        U1 = np.random.chisquare(n, size=n_MCMC)
        U2 = np.random.chisquare(n, size=n_MCMC)

        rhos = (S1[i]*U2-S2[i]*U1)/(S1[i]*U2+S2[i]*U1)
        samples[i] = np.sum(rhos < rho)/n_MCMC

#        plt.figure(2)
#        plt.hist(rhos, bins=100, density=True)
#        plt.show()

    alphas = np.linspace(0,1,100)


    confs_lower = np.array(
        [np.sum(samples < alpha) / n_samples for alpha in alphas])

    plt.figure()
    plt.plot(alphas, alphas)
    plt.plot(alphas, confs_lower)
    plt.show()


elif False:
    rho = 0.0
    n = 3
    n_samples = 10000
    n_MCMC = 100000

    S1 = np.random.gamma(shape=n / 2, scale=4 * (1 + rho), size=n_samples)
    S2 = np.random.gamma(shape=n / 2, scale=4 * (1 - rho), size=n_samples)

    samples = np.empty(n_samples)
    for i in range(n_samples):
        if i%100 == 0:
            print(i)
        U1 = np.random.chisquare(n, size=n_MCMC)
        U2 = np.random.chisquare(n, size=n_MCMC)

        beta = U2*S1[i]**2/(2*S2[i]*U1**2)

        rhos = -(2+beta)/2+np.sqrt((2+beta)**2/2**2-1+beta)
        samples[i] = np.sum(rhos < rho)/n_MCMC

        plt.figure(2)
        plt.hist(rhos, bins=100, density=True)
        plt.show()

    alphas = np.linspace(0,1,100)


    confs_lower = np.array(
        [np.sum(samples < alpha) / n_samples for alpha in alphas])

    plt.figure()
    plt.plot(alphas, alphas)
    plt.plot(alphas, confs_lower)
    plt.show()


elif True:
    from math import log, exp
    rho = 0.5
    n = 3
    n_samples = 1000
    n_MCMC = 100000

    S1 = np.random.gamma(shape=n / 2, scale=4 * (1 + rho), size=n_samples)
    S2 = np.random.gamma(shape=n / 2, scale=4 * (1 - rho), size=n_samples)

    def g(x):
        return log(x+exp(x))

    def f_inv(y,x):
        return g(y)-g(x)

    def dist_func(S1, S2, U1, U2):
        a = U2-U1
        b1 = (S1+S2)/a
        b2 = (S2-S1)/a

        return -b1/4+np.sign(a)*1/4*np.sqrt(b1**2+16*(1-b2/2))

    def func_to_solve(x,S1,S2,a):
        if x == 1:
            return np.infty
        elif x == -1:
            return -np.infty
        return f_inv(S2/(2*(1-x)),S1/(2*(1+x)))-a#f_inv(U2,U1)

    all_samples = np.empty(n_samples)
    properties = np.empty((n_samples, 12))

    start_time = time.time()
    last_time = start_time
    for i in range(n_samples):
        print(i)
        print("Time of last set:",time.time()-last_time,"Total time elapse:", time.time()-start_time)
        last_time = time.time()

        U1 = np.random.chisquare(n, size=n_MCMC)
        U2 = np.random.chisquare(n, size=n_MCMC)

        samples = np.array([
            brentq(func_to_solve, -1, 1, args=(S1[i],S2[i],f_inv(U2[j], U1[j]))) for j in range(n_MCMC)
        ])
        all_samples[i] = np.sum(samples < rho)/n_MCMC

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

        print("Mean",np.mean(np.arctanh(samples)), "var",np.var(np.arctanh(samples)), "MSE:",np.mean((np.arctanh(samples)-np.arctanh(rho))**2))
        print("Mean",np.mean(np.arctanh(dist_func(S1[i],S2[i],U1,U2))), "var",np.var(np.arctanh(dist_func(S1[i],S2[i],U1,U2))), "MSE:",np.mean((np.arctanh(dist_func(S1[i],S2[i],U1,U2))-np.arctanh(rho))**2))

        plt.figure(2)
        plt.title("S1 "+str(S1[i])+", S2 "+str(S2[i]))
        plt.axvline(x=np.arctanh(rho), color="green")
        plt.hist(np.arctanh(samples), bins=100, density=True,label="samples")
        plt.hist(np.arctanh(dist_func(S1[i],S2[i],U1,U2)), bins=100, density=True, histtype="step",label="g(x)=x")
        plt.hist(np.arctanh((S1[i]*U2-S2[i]*U1)/(S1[i]*U2+S2[i]*U1)), bins=100, density=True, histtype="step",label="g(x)=ln(x)")
        plt.legend()
        plt.show()

    alphas = np.linspace(0,1,100)

    pickle.dump({
        "samples": all_samples,
        "properties": properties
    }, open("CD_samples/regtestcomposite_deg2051000.p", "wb")
    )

    risks = np.mean(properties, axis=0)
    risk_names = ["Mean mean:", "Mean var:", "Mean MAE:", "Mean MSE:", "Mean FIM:", "Mean KLD:", "Mean z_mean:",
                  "Mean z_MSE:", "Mean w_mean:", "Mean w_MSE:", "Mean f_mean:", "Mean f_MSE:"]
    print("")

    for i in range(len(risks)):
        print(risk_names[i], risks[i])
    print("")

    confs_lower = np.array(
        [np.sum(all_samples < alpha) / n_samples for alpha in alphas])

    plt.figure()
    plt.plot(alphas, alphas)
    plt.plot(alphas, confs_lower)
    plt.show()

elif False:
    rho = 0.8
    n = 3
    n_samples = 1000
    n_MCMC = 100000

    def dist_func(S1, S2, U1, U2):
        a = U2-U1
        b1 = (S1+S2)/2
        b2 = (S2-S1)/2

        return -b1/(2*a)+1/(2*a)*np.sqrt(b1**2+4*a*(a-b2))

    S1 = np.random.gamma(shape=n / 2, scale=4 * (1 + rho), size=n_samples)
    S2 = np.random.gamma(shape=n / 2, scale=4 * (1 - rho), size=n_samples)

    all_samples = np.empty(n_samples)
    properties = np.empty((n_samples, 12))

    start_time = time.time()
    last_time = start_time
    for i in range(n_samples):
        if i%100==0:
            print(i)
            print("Time of last set:",time.time()-last_time,"Total time elapse:", time.time()-start_time)
            last_time = time.time()
        U1 = np.random.chisquare(n, size=n_MCMC)
        U2 = np.random.chisquare(n, size=n_MCMC)

        samples = dist_func(S1[i], S2[i], U1, U2)
        all_samples[i] = np.sum(samples < rho)/n_MCMC

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

        from posteriors import Posterior
        from MCMC_test2 import one_simulation
        jeff = Posterior("jeffrey", lam=10 ** (-4)).distribution
        T1 = 1/2*(S1[i]+S2[i])
        T2 = 1/4*(S1[i]-S2[i])
        post_samples = one_simulation(n_MCMC, n, T1, T2, jeff, 0.1, 0)


#        print("Mean",np.mean(samples), "var",np.var(samples), "MSE:",np.mean((samples-rho)**2))

        plt.figure(2)
        plt.title("S1 "+str(S1[i])+", S2 "+str(S2[i]))
        plt.hist(samples, bins=100, density=True)
        plt.hist(post_samples, bins=100, density=True, histtype="step")
        plt.hist((S1[i]*U2-S2[i]*U1)/(S1[i]*U2+S2[i]*U1), bins=100, density=True, histtype="step")
        plt.show()

    alphas = np.linspace(0,1,100)

    pickle.dump({
        "samples": all_samples,
        "properties": properties
    }, open("CD_samples/regtest1081000.p", "wb")
    )

    risks = np.mean(properties, axis=0)
    risk_names = ["Mean mean:", "Mean var:", "Mean MAE:", "Mean MSE:", "Mean FIM:", "Mean KLD:", "Mean z_mean:",
                  "Mean z_MSE:", "Mean w_mean:", "Mean w_MSE:", "Mean f_mean:", "Mean f_MSE:"]
    print("")

    for i in range(len(risks)):
        print(risk_names[i], risks[i])
    print("")

    confs_lower = np.array(
        [np.sum(all_samples < alpha) / n_samples for alpha in alphas])

    plt.figure()
    plt.plot(alphas, alphas)
    plt.plot(alphas, confs_lower)
    plt.show()

elif False:
    rho = 0.2
    n = 3
    n_samples = 1000
    n_MCMC = 10000

    def dist_func(S1, S2, U1, U2):
        a = U2-U1
        b1 = (S1+S2)/2
        b2 = (S2-S1)/2

        return -b1/(2*a)+1/(2*a)*np.sqrt(b1**2+4*a*(a-b2))

    S1 = np.random.gamma(shape=n / 2, scale=4 * (1 + rho), size=n_samples)
    S2 = np.random.gamma(shape=n / 2, scale=4 * (1 - rho), size=n_samples)

    all_samples = np.empty(n_samples)
    properties = np.empty((n_samples, 12))

    start_time = time.time()
    last_time = start_time
    for i in range(n_samples):
        if i%1==0:
            print(i)
            print("Time of last set:",time.time()-last_time,"Total time elapse:", time.time()-start_time)
            last_time = time.time()
        U1 = np.random.chisquare(n, size=n_MCMC)
        U2 = np.random.chisquare(n, size=n_MCMC)

        orig_samples = dist_func(S1[i], S2[i], U1, U2)
        mean = np.mean(orig_samples)

        samples = np.empty(n_MCMC)

        for j in range(n_MCMC):
            if j%1000==0:
                print(j)
            new_S1 = np.random.gamma(shape = n/2, scale=4*(1+mean), size=1)[0]
            new_S2 = np.random.gamma(shape = n/2, scale=4*(1-mean), size=1)[0]
            new_U1 = np.random.chisquare(n, size=n_MCMC)
            new_U2 = np.random.chisquare(n, size=n_MCMC)

            new_samples = dist_func(new_S1, new_S2, new_U1, new_U2)
            samples[j] = np.mean(new_samples)

        all_samples[i] = np.sum(samples < rho)/n_MCMC



#        print("Mean",np.mean(samples), "var",np.var(samples), "MSE:",np.mean((samples-rho)**2))

#        plt.figure(2)
#        plt.title("S1 "+str(S1[i])+", S2 "+str(S2[i]))
#        plt.hist(samples, bins=100, density=True)
#        plt.hist(orig_samples, bins=100, density=True, histtype="step")
#        plt.hist((S1[i]*U2-S2[i]*U1)/(S1[i]*U2+S2[i]*U1), bins=100, density=True, histtype="step")
#        plt.show()

    alphas = np.linspace(0,1,100)

    pickle.dump({
        "samples": all_samples,
        "properties": properties
    }, open("CD_samples/regtest1081000.p", "wb")
    )

    risks = np.mean(properties, axis=0)
    risk_names = ["Mean mean:", "Mean var:", "Mean MAE:", "Mean MSE:", "Mean FIM:", "Mean KLD:", "Mean z_mean:",
                  "Mean z_MSE:", "Mean w_mean:", "Mean w_MSE:", "Mean f_mean:", "Mean f_MSE:"]
    print("")

    for i in range(len(risks)):
        print(risk_names[i], risks[i])
    print("")

    confs_lower = np.array(
        [np.sum(all_samples < alpha) / n_samples for alpha in alphas])

    plt.figure()
    plt.plot(alphas, alphas)
    plt.plot(alphas, confs_lower)
    plt.show()


elif False:
    rho = 0.0
    n = 3
    n_samples = 1000
    n_MCMC = 100000


    def dist_func(S1, S2, U1, U2):
        a = U2 - U1
        b1 = (S1 + S2) / 2
        b2 = (S2 - S1) / 2

        return -b1 / (2 * a) + 1 / (2 * a) * np.sqrt(b1 ** 2 + 4 * a * (a - b2))


    S1 = np.random.gamma(shape=n / 2, scale=4 * (1 + rho), size=n_samples)
    S2 = np.random.gamma(shape=n / 2, scale=4 * (1 - rho), size=n_samples)

    all_samples = np.empty(n_samples)
    properties = np.empty((n_samples, 12))
    means = np.empty(n_samples)
    meds = np.empty(n_samples)
    modes = np.empty(n_samples)
    zmeans = np.empty(n_samples)

    start_time = time.time()
    last_time = start_time
    for i in range(n_samples):
        if i % 100 == 0:
            print(i)
            print("Time of last set:", time.time() - last_time, "Total time elapse:", time.time() - start_time)
            last_time = time.time()
        U1 = np.random.chisquare(n, size=n_MCMC)
        U2 = np.random.chisquare(n, size=n_MCMC)

        samples = dist_func(S1[i], S2[i], U1, U2)
        all_samples[i] = np.sum(samples < rho) / n_MCMC

        properties[i, :] = np.array([
            np.mean(samples),
            np.var(samples),
            MAE(samples, rho),
            MSE(samples, rho),
            np.mean(fisher_information_metric(samples, rho)),
            np.mean(kullback_leibler(samples, rho)),
            z_transMean(samples),
            z_transMSE(samples, rho),
            w_transMean(samples),
            w_transMSE(samples, rho),
            fishMean(samples),
            fishMSE(samples, rho)
        ])

        means[i] = np.mean(samples)
        meds[i] = np.median(samples)
        histogram = np.histogram(samples, bins=100)
        index = np.argmax(histogram[0])
        modes[i] = histogram[1][index]
        zmeans[i] = np.tanh( np.mean( np.arctanh(samples) ) )



#        print("Mean",np.mean(samples), "var",np.var(samples), "MSE:",np.mean((samples-rho)**2))

#        plt.figure(2)
#        plt.title("S1 " + str(S1[i]) + ", S2 " + str(S2[i]))
#        plt.axvline(x=means[i], color="yellow")
#        plt.axvline(x=meds[i], color="blue")
#        plt.axvline(x=modes[i], color="red")
#        plt.axvline(x=zmeans[i], color="green")
#        plt.hist(samples, bins=100, density=True)
#        plt.hist((S1[i] * U2 - S2[i] * U1) / (S1[i] * U2 + S2[i] * U1), bins=100, density=True, histtype="step")
#        plt.show()

    alphas = np.linspace(0, 1, 100)

#    pickle.dump({
#        "samples": all_samples,
#        "properties": properties
#    }, open("CD_samples/regtest1081000.p", "wb")
#    )

    print("FIM mean",np.mean(fisher_information_metric(rho, means)), "KL mean",np.mean(kullback_leibler(rho, means)),
          "MSE mean",np.mean((rho-means)**2), "z_MSE mean",np.mean((np.arctanh(rho)-np.arctanh(means))**2))
    print("FIM med",np.mean(fisher_information_metric(rho, meds)), "KL med",np.mean(kullback_leibler(rho, meds)),
          "MSE med",np.mean((rho-meds)**2), "z_MSE med",np.mean((np.arctanh(rho)-np.arctanh(meds))**2))
    print("FIM mode",np.mean(fisher_information_metric(rho, modes)), "KL mode",np.mean(kullback_leibler(rho, modes)),
          "MSE mode",np.mean((rho-modes)**2), "z_MSE mode",np.mean((np.arctanh(rho)-np.arctanh(modes))**2))
    print("FIM zmean",np.mean(fisher_information_metric(rho, zmeans)), "KL zmean",np.mean(kullback_leibler(rho, zmeans)),
          "MSE zmean",np.mean((rho-zmeans)**2), "z_MSE zmean",np.mean((np.arctanh(rho)-np.arctanh(zmeans))**2))

    risks = np.mean(properties, axis=0)
    risk_names = ["Mean mean:", "Mean var:", "Mean MAE:", "Mean MSE:", "Mean FIM:", "Mean KLD:", "Mean z_mean:",
                  "Mean z_MSE:", "Mean w_mean:", "Mean w_MSE:", "Mean f_mean:", "Mean f_MSE:"]
    print("")

    for i in range(len(risks)):
        print(risk_names[i], risks[i])
    print("")

    confs_lower = np.array(
        [np.sum(all_samples < alpha) / n_samples for alpha in alphas])

    plt.figure()
    plt.plot(alphas, alphas)
    plt.plot(alphas, confs_lower)
    plt.show()




elif False:
    rho = 0.5
    n = 3
    n_samples = 1000
    n_MCMC = 100000

    S1 = np.random.gamma(shape=n / 2, scale=4 * (1 + rho), size=n_samples)
    S2 = np.random.gamma(shape=n / 2, scale=4 * (1 - rho), size=n_samples)

    def dist_func(S1, S2, U1, U2):
        a = U2-U1
        b1 = (S1+S2)/2
        b2 = (S2-S1)/2

        return -b1/(2*a)+1/(2*a)*np.sqrt(b1**2+4*a*(a-b2))

    def spes_func(S1, S2, U1, U2):
        temp = (S1+S2)/4
        ret_vals = np.empty(len(U1))
        ret_vals[(U1 <= temp) & (U2 <= temp)] = (S1-S2)/(S1+S2)

        U2_big = (U2 > U1) & (U2 > temp)
        ret_vals[U2_big] = 1-1/2*S2/U2[U2_big]

        U1_big = (U1 > U2) & (U1 > temp)
        ret_vals[U1_big] = 1/2*S1/U1[U1_big]-1

        return ret_vals

    all_samples = np.empty(n_samples)
    properties = np.empty((n_samples, 12))

    start_time = time.time()
    last_time = start_time
    for i in range(n_samples):
        if i%100==0:
            print(i)
            print("Time of last set:",time.time()-last_time,"Total time elapse:", time.time()-start_time)
            last_time = time.time()
        U1 = np.random.chisquare(n, size=n_MCMC)
        U2 = np.random.chisquare(n, size=n_MCMC)

        samples = spes_func(S1[i], S2[i], U1, U2)#np.array([spes_func(S1[i], S2[i], U1[j], U2[j]) for j in range(n_MCMC)])
        all_samples[i] = np.sum(samples < rho)/n_MCMC

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

#        print("Mean",np.mean(samples), "var",np.var(samples), "MSE:",np.mean((samples-rho)**2))

        trans = lambda x: 2*np.arctanh(x)

        lim = gamma.cdf((S1[i]+S2[i])/4, n/2, scale=2)

        plt.figure(2)
        plt.title("S1 "+str(S1[i])+", S2 "+str(S2[i]))
        plt.ylim((0,2))
        plt.axvline(x=trans(rho), color="green")
        plt.hist(trans(samples), bins=100, density=True)
        plt.hist(trans(dist_func(S1[i],S2[i],U1,U2)), bins=100, density=True, histtype="step")
        plt.hist(trans((S1[i]*U2-S2[i]*U1)/(S1[i]*U2+S2[i]*U1)), bins=100, density=True, histtype="step")
        plt.axvline(x=trans((S1[i]-S2[i])/(S1[i]+S2[i])),ymin=0,ymax=lim**2,color="red")
        plt.hist(np.log(S1[i]/(4*U1[U1>(S1[i]+S2[i])/8]-1/2*S1[i])), bins=100, density=True, histtype="step",color="red")
        plt.hist(-np.log(S2[i]/(4*U2[U2>(S1[i]+S2[i])/8]-1/2*S2[i])), bins=100, density=True, histtype="step",color="red")
        plt.show()

    alphas = np.linspace(0,1,100)

    pickle.dump({
        "samples": all_samples,
        "properties": properties
    }, open("CD_samples/regtest12001000.p", "wb")
    )

    risks = np.mean(properties, axis=0)
    risk_names = ["Mean mean:", "Mean var:", "Mean MAE:", "Mean MSE:", "Mean FIM:", "Mean KLD:", "Mean z_mean:",
                  "Mean z_MSE:", "Mean w_mean:", "Mean w_MSE:", "Mean f_mean:", "Mean f_MSE:"]
    print("")

    for i in range(len(risks)):
        print(risk_names[i], risks[i])
    print("")

    confs_lower = np.array(
        [np.sum(all_samples < alpha) / n_samples for alpha in alphas])

    plt.figure()
    plt.plot(alphas, alphas)
    plt.plot(alphas, confs_lower)
    plt.show()

elif True:
    rho = 0.5
    n = 3
    n_samples = 1000
    n_MCMC = 100000

    all_samples = np.empty(n_samples)
    properties = np.empty((n_samples, 12))

    start_time = time.time()
    last_time = start_time
    for i in range(n_samples):
        if i%100==0:
            print(i)
            print("Time of last set:",time.time()-last_time,"Total time elapse:", time.time()-start_time)
            last_time = time.time()
        X = np.random.multivariate_normal(mean=np.array([0,0]), cov=np.array([[1,rho],[rho,1]]), size=n)

        Z = np.random.chisquare(n, size=(n, n_MCMC))

        XY_sum = np.sum(1/(X[:,0]-X[:,1]))
        print(XY_sum)

        samples = np.sqrt(1-XY_sum*np.sum(1/Z,axis=0))
        all_samples[i] = np.sum(samples < rho)/n_MCMC

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

#        print("Mean",np.mean(samples), "var",np.var(samples), "MSE:",np.mean((samples-rho)**2))

        trans = lambda x: 2*np.arctanh(x)

        plt.figure(2)
        plt.ylim((0,2))
        plt.axvline(x=trans(rho), color="green")
        plt.hist(trans(samples), bins=100, density=True)
        plt.show()

    alphas = np.linspace(0,1,100)

    pickle.dump({
        "samples": all_samples,
        "properties": properties
    }, open("CD_samples/regtest12001000.p", "wb")
    )

    risks = np.mean(properties, axis=0)
    risk_names = ["Mean mean:", "Mean var:", "Mean MAE:", "Mean MSE:", "Mean FIM:", "Mean KLD:", "Mean z_mean:",
                  "Mean z_MSE:", "Mean w_mean:", "Mean w_MSE:", "Mean f_mean:", "Mean f_MSE:"]
    print("")

    for i in range(len(risks)):
        print(risk_names[i], risks[i])
    print("")

    confs_lower = np.array(
        [np.sum(all_samples < alpha) / n_samples for alpha in alphas])

    plt.figure()
    plt.plot(alphas, alphas)
    plt.plot(alphas, confs_lower)
    plt.show()

