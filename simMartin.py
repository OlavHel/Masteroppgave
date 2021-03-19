import numpy as np
import matplotlib.pyplot as plt
from MCMC_test2 import mult_simulations, loccond
import pickle

rho = 0.8

m = 100000
s2 = 0.01
start = 0

distr = lambda x, n, X, Y: loccond(rho, x, n, X, Y)

n = 3
n_samples = 10000

samples, properties = mult_simulations(n_samples, n, m, s2, start, distr, rho)

pickle.dump({
    "samples": samples,
    "properties": properties
    }, open("CD_samples/test0510000.p", "wb")
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










