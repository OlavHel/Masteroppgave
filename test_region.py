import numpy as np
import matplotlib.pyplot as plt
import time
from scipy.stats import gamma


if False:
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

#    confs_both = np.array(
#        [
#            np.sum( (rho <= (np.quantile(ratios, 1/2+alpha/2)*S1-S2)/(np.quantile(ratios, 1/2+alpha/2)*S1+S2) ) &
#                    (rho > (np.quantile(ratios, 1/2-alpha/2)*S1-S2)/(np.quantile(ratios, 1/2-alpha/2)*S1+S2) ) )/n_samples for alpha in alphas
#        ]
#    )

#    confs_both = np.array(
#        [
#            np.sum((rho <= (np.quantile(ratios, alphas[np.where(confs_lower >= (1+alpha)/2)[0][0]]) * S1 - S2) / (
#                        np.quantile(ratios, alphas[np.where(confs_lower >= (1+alpha)/2)[0][0]]) * S1 + S2)) &
#                   (rho > (np.quantile(ratios, alphas[np.where(confs_lower >= (1-alpha)/2)[0][0]]) * S1 - S2) / (
#                               np.quantile(ratios, alphas[np.where(confs_lower >= (1-alpha)/2)[0][0]]) * S1 + S2))) / n_samples
#            for alpha in np.linspace(0,1,100)
#        ]
#    )

    plt.figure()
    plt.plot(alphas, alphas)
    plt.plot(alphas, confs_lower)
    plt.plot(alphas, confs_upper)
#    plt.plot(alphas, -0.1*np.sin(alphas*2*np.pi))
#    plt.plot(np.linspace(0,1,100), confs_both)
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

elif True:
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

elif True:
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



