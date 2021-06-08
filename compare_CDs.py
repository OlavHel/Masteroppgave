import numpy as np
import matplotlib.pyplot as plt
from posteriors import Posterior
from MCMC_test2 import one_simulation, gen_data, posterior_distr
from simulate_CD import one_simulate_1, one_simulate_2, one_simulate_3, one_simulate_4, one_simulate_5
from test_region import sim_pivot_diff, sim_any_pivot_g
from loss_functions import *

posterior_names = []#["jeffrey", "uniform", "PC", "arctanh", "arcsine"]#["fiduc_orig_2", "fiduc_orig_infty","fiduc_2","fiduc_infty"]["jeffrey", "uniform", "PC", "new", "arcsine", "test"]
posteriors  = [Posterior(name, lam=10**(-4)).norm_distribution for name in posterior_names]
posterior_input = []#[True, True, True, True, True]#[False, False, True, True]

all_names = ["Tarldsen-CD","CVCD","Diff","CD4"]#,"CD5"]#["Diff"]#["CD1","CD2","Diff"]
all_CDs = [one_simulate_1, one_simulate_2, one_simulate_3, one_simulate_4]#, one_simulate_5]#[one_simulate_3]#[one_simulate_1, one_simulate_2, one_simulate_3]

print(all_names)
print(all_CDs)

n = 3
rho = 0.5

m = 1000000

X, Y = gen_data(n,rho)

print("r=",np.sum(X*Y)/np.sqrt(np.sum(X**2)*np.sum(Y**2)))
print("S_1=",np.sum((X+Y)**2), "S_2=",np.sum((X-Y)**2))

samples = np.empty((len(all_names),m))
for i in range(len(all_names)):
    print(i)
    samples[i,:] = all_CDs[i](X, Y, n, m)

print("hei")

T1 = np.sum(X**2+Y**2)
T2 = np.sum(X*Y)
S1 = np.sum((X+Y)**2)
S2 = np.sum((X-Y)**2)

rhos = np.linspace(-0.9999, 0.9999, 1000)

for i in range(len(all_names)):
    print(all_names[i], np.mean(z_transMSE(samples[i,:],rho)))

#cj = Posterior("jeffrey").normalization(n, T1, T2)
#cn = Posterior("arctanh").normalization(n, T1, T2)
#cu = Posterior("uniform").normalization(n, T1, T2)
#ca= Posterior("arcsine").normalization(n, T1, T2)
#cpc= Posterior("PC", lam=10**(-4)).normalization(n, T1, T2)

#print(cj, cn)
#test_rho = np.sqrt((cj/cn)**2-1)
#print(test_rho)
#print(cu, cn*(2-(cj/cn)**2))
#print(ca, cn*np.sqrt(2-(cj/cn)**2))
#print(cpc, cn*test_rho/np.sqrt(-np.log(1-test_rho**2)))

plt.figure(1)
plt.title(r"$S_1$="+str(S1)+r", $S_2$="+str(S2))
#plt.subplot(1,2,1)
for i in range(len(posterior_names)):
    if posterior_input[i]:
        plt.plot(rhos, posteriors[i](rhos, n, T1, T2), label=posterior_names[i])
    else:
        plt.plot(rhos, posteriors[i](rhos, n, X, Y), label=posterior_names[i])
for i in range(len(all_names)):
    plt.hist(np.arctanh(samples[i, :]), bins = 1000, density = True, histtype = "step", label = all_names[i])
#plt.axvline(x=np.sqrt((cj/cn)**2-1))
plt.legend()
#trans = np.arctanh
#plt.subplot(1,2,2)
#for i in range(len(all_names)):
#    plt.hist(trans(samples[i, :]), bins = 100, density = True, histtype = "step", label = all_names[i])
#plt.legend()


plt.figure(2)
plt.title(r"$S_1$="+str(S1)+r", $S_2$="+str(S2))
colors = ["red","blue","green","purple","orange"]
for i in range(len(all_names)):
    plt.hist(samples[i, :], bins = 1000, density = True, histtype = "step", label = all_names[i])
#    plt.axvline(np.median(samples[i,:]),label="median "+ all_names[i],color=colors[i])
plt.legend()
plt.show()





