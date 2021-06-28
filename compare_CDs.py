import numpy as np
import matplotlib.pyplot as plt
from posteriors import Posterior
from MCMC_test2 import one_simulation, gen_data, posterior_distr
from simulate_CD import one_simulate_1, one_simulate_2, one_simulate_3, one_simulate_4, one_simulate_5
from loss_functions import *

# code for comparing the density functions of different distribution esitmators

# the first set of lists are for the posteriors to plot.
# the posterior names correspond to the names in the Posterior class
posterior_names = ["jeffrey", "uniform", "PC", "arctanh", "arcsine"]
posteriors  = [Posterior(name, lam=10**(-4)).norm_distribution for name in posterior_names] # just let this stay
# list that determines if the posterior uses sufficient statistics or original data. True for sufficient statistics
posterior_input = [True, True, True, True, True] # only the "orig" fiducials will need False

# the second set of lists are for the exact confidence distributions
all_names = ["UVCD","CVCD","DiffCD","CD1"] # is used only for the legend
all_CDs = [one_simulate_1, one_simulate_2, one_simulate_3, one_simulate_4] # list of CDs to sample

print(all_names)
print(all_CDs)

n = 3
rho = 0.5

m = 10000000 # number of samples in the MCMC for the exact CDs

X, Y = gen_data(n,rho) # generate data

# the following set of data sets correspond to those defined in the appendix in the thesis

#X = np.array([-1.73,-0.85,2.28]) #DATA 1
#Y = np.array([-0.41,0.57,-0.01]) #DATA 1
X = np.array([1.01,-0.74,2.12]) #DATA 3
Y = np.array([1.19,-0.23,0.35]) #DATA 3
#X = np.array([0.98, -0.11, 0.87]) #DATA 4
#Y = np.array([0.96, 0.67, 0.82]) #DATA 4

print("r=",np.sum(X*Y)/np.sqrt(np.sum(X**2)*np.sum(Y**2)))
print("S_1=",np.sum((X+Y)**2), "S_2=",np.sum((X-Y)**2))

# sample from the exact CDs
samples = np.empty((len(all_names),m))
for i in range(len(all_names)):
    print(i)
    samples[i,:] = all_CDs[i](X, Y, n, m)

T1 = np.sum(X**2+Y**2)
T2 = np.sum(X*Y)
S1 = np.sum((X+Y)**2)
S2 = np.sum((X-Y)**2)

r = np.sum(X*Y)/np.sqrt(np.sum(X**2)*np.sum(Y**2))

rhos = np.linspace(-0.9999, 0.9999, 1000)

# visualization of the densities
plt.figure(1)
# first subplot is for densities of the correlation
plt.subplot(1,2,1)
plt.title(r"$S_1$="+str(np.round(S1,3))+r", $S_2$="+str(np.round(S2,3)), fontsize=14)
for i in range(len(posterior_names)):
    if posterior_input[i]:
        plt.plot(rhos, posteriors[i](rhos, n, T1, T2), label=posterior_names[i])
    else:
        plt.plot(rhos, posteriors[i](rhos, n, X, Y), label=posterior_names[i])
for i in range(len(all_names)):
    plt.hist(samples[i, :], bins = 1000, density = True, histtype = "step", label = all_names[i])
#plt.axvline(x=r,label="Empirical correlation") # if one wants the empirical correlation
plt.xlabel(r"$\rho$", fontsize=14)
plt.legend(fontsize=14)

# second subplot is for the parameter z(rho)=arctanh(rho)
plt.subplot(1,2,2)
plt.title(r"$n$="+str(n))
#plt.title(r"$S_1$="+str(S1)+r", $S_2$="+str(S2))
colors = ["red","blue","green","purple","orange"] # colors to be used for the densities

zs = np.linspace(-2.5,2.5,100)
# to limit the range on the x-axis. can be too large
plt.xlim(-4,4)

for i in range(len(posterior_names)):
    dens_func = lambda x, X, Y: posteriors[i](np.tanh(x), n, X, Y)/(np.cosh(x)**2)
    if posterior_input[i]:
        plt.plot(zs, dens_func(zs, T1, T2), label=posterior_names[i])
    else:
        plt.plot(zs, dens_func(zs, X, Y), label=posterior_names[i])
for i in range(len(all_names)):
    plt.hist(np.arctanh(samples[i, :]), bins = 1000, density = True, histtype = "step", label = all_names[i])
#plt.axvline(np.arctanh(r),label="Empirical correlation") # if one wants the empirical correlation
plt.xlabel(r"$z=\arctanh(\rho)$", fontsize=14)
plt.show()






