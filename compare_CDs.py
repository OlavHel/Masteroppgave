import numpy as np
import matplotlib.pyplot as plt
from posteriors import Posterior
from MCMC_test2 import one_simulation, gen_data, posterior_distr
from simulate_CD import one_simulate_1, one_simulate_2, one_simulate_3
from test_region import sim_pivot_diff, sim_any_pivot_g
from loss_functions import *

posterior_names = []#["jeffrey", "uniform", "PC", "new", "arcsine", "test"]
posteriors  = []#[Posterior(name, lam=10**(-4)).norm_distribution for name in posterior_names]

all_names = ["CD1","CD2","Diff"]
all_CDs = [one_simulate_1, one_simulate_2, one_simulate_3]

print(all_names)
print(all_CDs)

n = 20
rho = 0.9

m = 10000000

X, Y = gen_data(n,rho)

print("r=",np.sum(X*Y)/np.sqrt(np.sum(X**2)*np.sum(Y**2)))
print("S_1=",np.sum((X+Y)**2), "S_2=",np.sum((X-Y)**2))

samples = np.empty((len(all_names),m))
for i in range(len(all_names)):
        samples[i,:] = all_CDs[i](X, Y, n, m)

T1 = np.sum(X**2+Y**2)
T2 = np.sum(X*Y)
S1 = np.sum((X+Y)**2)
S2 = np.sum((X-Y)**2)

rhos = np.linspace(-0.9999, 0.9999, 1000)

for i in range(len(all_names)):
    print(all_names[i], np.mean(z_transMSE(samples[i,:],rho)))

plt.figure(1)
plt.title(r"$S_1$="+str(S1)+r", $S_2$="+str(S2))
#plt.subplot(1,2,1)
for i in range(len(posterior_names)):
    plt.plot(rhos, posteriors[i](rhos, n, T1, T2), label=posterior_names[i])
for i in range(len(all_names)):
    plt.hist(samples[i, :], bins = 1000, density = True, histtype = "step", label = all_names[i])
plt.legend()
#trans = np.arctanh
#plt.subplot(1,2,2)
#for i in range(len(all_names)):
#    plt.hist(trans(samples[i, :]), bins = 100, density = True, histtype = "step", label = all_names[i])
#plt.legend()
plt.show()






