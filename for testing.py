import numpy as np
import matplotlib.pyplot as plt
from MCMC_test2 import one_simulation, mult_simulations, gen_data


def test_distr(theta, h, x,n, X, Y):
    S1 = np.sum((X+Y)**2)
    S2 = np.sum((X-Y)**2)

    return np.exp(n*theta*x/2-np.cosh(x/2)*np.exp((h+theta*x)/2))

def pos_distr(theta, x, n, X, Y):
    if type(x) == type(np.array([0.1])) and len(x) == 1:
        x = x[0]
    if type(x) == type(np.array([])):
        x[x <= -1] = 0
        x[x >= 1] = 0
    elif (x <= (-1) or x >= (1)):
        return 0

    S1 = np.sum((X + Y) ** 2)
    S2 = np.sum((X - Y) ** 2)

    if theta >= 1:
        f_theta = (1 + theta) ** (1 / 2 + theta / 2)
        f_x = (1 - x) ** (1 / 2 + theta / 2)
    elif theta <= -1:
        f_theta = (1 - theta) ** (1 / 2 - theta / 2)
        f_x = (1 + x) ** (1 / 2 - theta / 2)
    else:
        f_theta = (1+theta)**(1/2+theta/2)*(1-theta)**(1/2-theta/2)
        f_x = (1+x)**(1/2-theta/2)*(1-x)**(1/2+theta/2)

    return f_x**n*1/(1-x**2)*(1-x**2)**(-n/2)*np.exp(-1/2*f_x/f_theta*(S1/(1+x)+S2/(1-x)))

rho = -0.5

n = 5
data = np.random.multivariate_normal(mean=[0,0], cov=[[1,rho],[rho,1]],size=n)

est = np.sum(data[:,0]*data[:,1])/np.sqrt(np.sum(data[:,0]**2)*np.sum(data[:,1]**2))

plt.figure(1)
plt.scatter(data[:,0], data[:,1])
plt.show()

est = 0.5

print(est)

m = 10000000

new_data = np.random.multivariate_normal(mean=[0,0], cov=[[1,est],[est,1]],size=m)

X1, Y1 = np.meshgrid(new_data[:,0], new_data[:,1])



true_data = np.random.multivariate_normal(mean=[0,0], cov=[[1,rho],[rho,1]],size=m)



plt.figure(2)
plt.hist2d(true_data[:,0], true_data[:,1], bins=(100,100), density=True, cmap=plt.cm.Reds)
plt.hist2d(new_data[:,0], new_data[:,1], bins=(100,100), density=True, cmap=plt.cm.Blues, )
plt.colorbar()
plt.scatter(data[:,0],data[:,1])
props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
plt.text(-2, 2, r"$\rho=$"+str(rho), fontsize=14,
        verticalalignment='top', bbox=props)
plt.xlabel("X")
plt.ylabel("Y")
plt.xlim(-3,3)
plt.ylim(-3,3)
plt.show()


1/0

theta_0 = 1
rho = 0.5
n = 10
m = 100000

X, Y = gen_data(n, rho)
S1 = np.sum((X+Y)**2)
S2 = np.sum((X-Y)**2)

if theta_0 <= -1:
    h_0 = (1 - theta_0) * np.log(S1 / (1 - theta_0))
    theta_0 = -1
elif theta_0 >= 1:
    h_0 = (1 + theta_0) * np.log(S2 / (1 + theta_0))
    theta_0 = 1
else:
    h_0 = (1-theta_0)*np.log(S1/(1-theta_0))+(1+theta_0)*np.log(S2/(1+theta_0))
T = np.log(S1/S2)

distr = lambda x, n, X, Y: test_distr(theta_0, h_0, x, n, X, Y)

v_samples = one_simulation(m, n, X, Y, distr, 0.1, 0)

samples = np.tanh((T-v_samples)/2)

distr2 = lambda x, n, X, Y: pos_distr(theta_0, x, n, X, Y)
samples2 = one_simulation(m, n, X, Y, distr2, 0.1, 0)

plt.figure(1)
plt.hist(samples, bins=100, density=True)
plt.hist(samples2, bins=100, density=True, histtype="step")
plt.show()



