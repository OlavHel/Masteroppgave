import numpy as np



def MAE(samples, theta = None):
    if theta is None:
        mean = np.mean(samples)
        return np.mean(np.abs(samples-mean))
    else:
        return np.mean(np.abs(samples-theta))


def MSE(samples, theta = None):
    N = len(samples)
    if theta is None:
        return np.var(samples)
    else:
        squared_errors = (samples-theta)**2
        return 1/N*np.sum(squared_errors)


def z_transMean(samples):
    z_samples = 1/2*np.log((1+samples)/(1-samples))
    return np.mean(z_samples)

def z_transMSE(samples, theta):
    z_samples = 1/2*np.log((1+samples)/(1-samples))
    z_theta = 1/2*np.log((1+theta)/(1-theta))
    return MSE(z_samples,z_theta)


def w_transMean(samples):
    w_samples = samples/np.sqrt(1-samples**2)
    return np.mean(w_samples)

def w_transMSE(samples, theta):
    w_samples = samples/np.sqrt(1-samples**2)
    w_theta = theta/np.sqrt(1-theta**2)
    return MSE(w_samples, w_theta)


def fishMean(samples):
    f_samples = fisher_information(samples)
    return np.mean(f_samples)

def fishMSE(samples, theta):
    f_samples = fisher_information(samples)
    f_theta = fisher_information(theta)
    return MSE(f_samples, f_theta)


def kullback_leibler(rho, rho_hat):
    return -1 / 2 * np.log((1 - rho ** 2) / (1 - rho_hat ** 2)) + (1 - rho * rho_hat) / (1 - rho_hat ** 2) - 1


def fisher_information(x):
    if type(x) == type(np.array([])):
        x[x <= -1] = 0
        x[x >= 1] = 0
    elif (x <= (-1) or x >= (1)):
        return 0
    return np.sqrt(2) * np.arctanh(np.sqrt(2) * x / np.sqrt(1 + x ** 2)) - np.arcsinh(x)

def fisher_information_metric(rho, rho_hat):
    return np.abs(fisher_information(rho) - fisher_information(rho_hat))


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    rhos = np.linspace(-0.99999,0.99999, 1000000)
    rho = 0.9

    loss_names = ["Squared error", "z-transformed squared error", "Fisher Information Metric", "Kullback-Leibler"]
    loss_funcs = [lambda x, rho: (x-rho)**2, lambda x, rho: (np.arctanh(x)-np.arctanh(rho))**2, fisher_information_metric, kullback_leibler]

    lims = [[0,(-1-0.9)**2], [0,8], [0,3.4], [0,9]]

    plt.figure()
    for i in range(len(loss_funcs)):
        plt.subplot(2,2,i+1)
        plt.title(loss_names[i],fontsize=14)
        plt.plot(rhos,loss_funcs[i](rho,rhos))
        plt.ylim(lims[i][0],lims[i][1])
        plt.xlabel(r"$\rho$", fontsize = 14)
        plt.ylabel("loss", fontsize = 14)
    plt.subplots_adjust(left=0.1,
                        bottom=0.1,
                        right=0.96,
                        top=0.9,
                        wspace=0.3,
                        hspace=0.5)
    plt.show()


