import numpy as np
import matplotlib.pyplot as plt
from posteriors import Posterior
from MCMC_test2 import one_simulation, gen_data, posterior_distr
from simulate_CD import one_simulate_1, one_simulate_2

start = 0.0
s2 = 0.01
posterior_names = ["fiduc_2", "fiduc_infty"]
posts1 = lambda x, n, X, Y: posterior_distr(x, n, X, Y, Posterior(posterior_names[0], lam=10**(-4)).distribution)
posts2 = lambda x, n, X, Y: posterior_distr(x, n, X, Y, Posterior(posterior_names[1], lam=10**(-4)).distribution)
call_posts = [lambda X, Y, n, m: one_simulation(m, n, X, Y, posts1, s2, start),
              lambda X, Y, n, m: one_simulation(m, n, X, Y, posts2, s2, start)]

exact_CD_names = ["CD1","CD2"]
exact_CDs = [one_simulate_1, one_simulate_2]

all_names = np.concatenate((posterior_names, exact_CD_names))
all_CDs = np.concatenate((call_posts, exact_CDs))

print(all_names)
print(all_CDs)

n = 3
rho = 0.5

m = 100000
s2 = 0.01
start = 0

X, Y = gen_data(n,rho)

print("r=",np.sum(X*Y)/np.sqrt(np.sum(X**2)*np.sum(Y**2)))
print("S_1=",np.sum((X+Y)**2), "S_2=",np.sum((X-Y)**2))

from scipy.stats import mannwhitneyu, ks_2samp
samples = np.empty((len(all_names),m))
for i in range(len(all_names)):
        samples[i,:] = all_CDs[i](X, Y, n, m)

print("Mann-Witney U test mellom:", all_names[0],all_names[1],mannwhitneyu(samples[0,:], samples[1,:]))
print("Mann-Witney U test mellom:", all_names[2],all_names[3],mannwhitneyu(samples[2,:], samples[3,:]))
print("KS_2 test mellom:", all_names[0],all_names[1],ks_2samp(samples[0,:], samples[1,:]))
print("KS_2 test mellom:", all_names[2],all_names[3],ks_2samp(samples[2,:], samples[3,:]))

fid2 = Posterior("fiduc_2")
fidinf = Posterior("fiduc_infty")
T1 = np.sum(X**2+Y**2)
T2 = np.sum(X*Y)

rhos = np.linspace(-0.9999, 0.9999, 100)

plt.figure(1)
for i in range(2):#len(all_names)):
        plt.hist(samples[i, :], bins = 100, density = True, histtype = "step", label = all_names[i])
plt.plot(rhos, fid2.distribution(rhos, n, T1, T2)/fid2.normalization(n, T1, T2), label = "fid2")
plt.plot(rhos, fidinf.distribution(rhos, n, T1, T2)/fidinf.normalization(n, T1, T2), label = "fidinf")
plt.legend()
plt.show()






