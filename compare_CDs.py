import numpy as np
import matplotlib.pyplot as plt
from posteriors import Posterior
from MCMC_test2 import one_simulation, gen_data, posterior_distr
from simulate_CD import one_simulate_1, one_simulate_2

start = 0.0
s2 = 0.01
posterior_names = ["fiduc_2", "fiduc_infty"]
posts = [lambda x, n, X, Y: posterior_distr(x, n, X, Y, Posterior(name, lam=10**(-4)).distribution) for name in posterior_names]
call_posts = []
for i in range(len(posts)):
        call_posts.append( lambda X, Y, n, m: one_simulation(m, n, X, Y, posts[i], s2, start) )

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

plt.figure(1)
for i in range(len(all_names)):
        plt.hist(all_CDs[i](X,Y,n,m), bins = 100, density = True, histtype = "step", label = all_names[i])
plt.legend()
plt.show()







