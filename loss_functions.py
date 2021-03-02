import numpy as np



def MAE(samples, theta = None):
    N = len(samples)
    if theta is None:
        mean = np.mean(samples)
        return np.mean(np.abs(samples-mean))
    else:
        squared_errors = (samples-theta)**2
        return 1/N*np.sum(squared_errors)


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


def kullback_leibler(rho1, rho2):
    return -1 / 2 * np.log((1 - rho1 ** 2) / (1 - rho2 ** 2)) + (1 - rho1 * rho2) / (1 - rho2 ** 2) - 1


def fisher_information(x):
    if type(x) == type(np.array([])):
        x[x <= -1] = 0
        x[x >= 1] = 0
    elif (x <= (-1) or x >= (1)):
        return 0
    return np.sqrt(2) * np.arctanh(np.sqrt(2) * x / np.sqrt(1 + x ** 2)) - np.arcsinh(x)

def fisher_information_metric(rho, rho_hat):
    return np.abs(fisher_information(rho) - fisher_information(rho_hat))






