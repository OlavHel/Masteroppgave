import numpy as np
import matplotlib.pyplot as plt
from test_region import sim_pivot_diff, sim_any_pivot_g
from simulate_CD import gen_data
from posteriors import Posterior

n = 3
rho = 0.0

m = 100000

X, Y = gen_data(n,rho)

from math import log

samples = sim_any_pivot_g(X, Y, n, m,
                          lambda x: log(x)) #sim_pivot_diff(X, Y, n, m)

temp = np.histogram(samples, bins=100)

rhos = np.array([(temp[1][i+1]+temp[1][i])/2 for i in range(len(temp[1])-1)])

print(rhos)
T1 = np.sum(X**2+Y**2)
T2 = np.sum(X*Y)
unif = Posterior("uniform")
fs = unif.distribution(rhos, n, T1, T2)/unif.normalization(n, T1, T2)

cds = temp[0]/np.sum(temp[0])*1/np.diff(temp[1])

plt.figure()
plt.subplot(1,2,1)
plt.plot(rhos, fs)
plt.plot(rhos, cds)
plt.hist(samples, bins = 100, density=True)
plt.subplot(1,2,2)
plt.plot(rhos,np.minimum(cds/fs, 100))
plt.show()






