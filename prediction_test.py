import numpy as np
import matplotlib.pyplot as plt

rho = 0.5
n = 3
m = 100000
n_tests = 1000

S1 = np.random.gamma(shape=n/2, scale=4*(1+rho), size=n_tests)
S2 = np.random.gamma(shape=n/2, scale=4*(1-rho), size=n_tests)

normals = np.random.multivariate_normal(mean = [0,0], cov = [[1, rho], [rho, 1]], size = n_tests)

cums = np.empty(n_tests)
real_cums = np.empty(n_tests)
rho_cums = np.empty(n_tests)

for i in range(n_tests):
    if i % 100==0:
        print(i)
    U1 = np.random.gamma(shape=n / 2, scale=2, size=m)
    U2 = np.random.gamma(shape=n / 2, scale=2, size=m)

    Z2 = np.random.standard_normal(size=m)

#    rhos = (S1[i]*U1-S2[i]*U2)/(S1[i]*U1+S2[i]*U2)
    a = U2 - U1
    b1 = (S1[i] + S2[i]) / a
    b2 = (S2[i] - S1[i]) / a
    rhos = -b1 / 4 + np.sign(a) * 1 / 4 * np.sqrt(b1 ** 2 + 16 * (1 - b2 / 2))
#    eta = U1/U2*S1[i]/S2[i]
#    rhos = (eta-1)/(eta+1)

    samples = normals[i, 0]*rhos+Z2*np.sqrt(1-rhos**2)

#    plt.figure()
#    plt.hist(samples, bins=100, density=True)
#    plt.axvline(x=normals[i,1], color="red")
#    plt.show()

    cums[i] = np.sum(samples <= normals[i,1])/m
    real_cums[i] = np.sum( normals[i,0]*rho+np.sqrt(1-rho**2)*Z2 <= normals[i,1])/m
    rho_cums[i] = np.sum( (S1[i]*U1-S2[i]*U2)/(S1[i]*U1+S2[i]*U2) <= rho)/m


plt.figure()
plt.hist(cums, bins=100)
plt.hist(real_cums, bins = 100)
plt.show()

alphas = np.linspace(0,1,100)

confs_lower = np.array(
    [np.sum(cums < alpha) / n_tests for alpha in alphas])
real_confs_lower = np.array(
    [np.sum(real_cums < alpha) / n_tests for alpha in alphas])
rho_confs_lower = np.array(
    [np.sum(rho_cums < alpha) / n_tests for alpha in alphas])

plt.figure()
plt.plot(alphas, alphas)
plt.plot(alphas, confs_lower)
plt.plot(alphas, real_confs_lower)
plt.show()




