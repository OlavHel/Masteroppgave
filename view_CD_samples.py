import numpy as np
import matplotlib.pyplot as plt
import pickle

rho = 0.5
dict = pickle.load(open("CD_samples_n_3/jeffrey081000.p","rb"))

samples = dict["samples"]
properties = dict["properties"]

risks = np.mean(properties,axis=0)
risk_names = ["Mean mean:", "Mean var:", "Mean MAE:", "Mean MSE:", "Mean FIM:", "Mean KLD:", "Mean z_mean:",
              "Mean z_MSE:", "Mean w_mean:", "Mean w_MSE:", "Mean f_mean:", "Mean f_MSE:"]

if len(risks) != len(risk_names):
    print("Number of names and properties do not match!")
    1/0

for i in range(len(risks)):
    print(risk_names[i],risks[i])

print("")

print("Mean mean:", np.mean(properties[:, 0]))
print("Mean var + var mean:", np.mean(properties[:, 1]) + np.var(properties[:, 0]))
print("Mean var:", np.mean(properties[:,1]))
print("Mean MSE", np.mean(properties[:,1]+properties[:,0]**2+0.0**2-2*0.0*properties[:,0]))

print("Mean z_var:", np.mean(properties[:,7]+2*rho*properties[:,6]-rho**2-properties[:,6]**2))
print("Mean w_var:", np.mean(properties[:,9]+2*rho*properties[:,8]-rho**2-properties[:,8]**2))
print("Mean f_var:", np.mean(properties[:,11]+2*rho*properties[:,10]-rho**2-properties[:,10]**2))

n = len(samples)

alphas = np.linspace(0, 1, 1000)
confs_lower = np.array(
    [np.sum((samples < alpha)) / n for alpha in alphas])

confs_both = np.array(
    [np.sum((samples < (1 + alpha) / 2) & (samples > (1 - alpha) / 2)) / n for alpha in alphas])

confs_upper = np.array(
    [np.sum((samples > 1-alpha)) / n for alpha in alphas])

plt.figure()
plt.plot(alphas, confs_lower, label="lower confs")
#plt.plot(alphas, confs_both, label="two-sided confs")
#plt.plot(alphas, confs_upper, label="upper confs")
plt.plot(alphas, alphas, label="linear")
#plt.plot(np.linspace(0,1,50), confs_lower[50:100]-np.flip((confs_lower)[0:50]))
plt.legend()
plt.show()

if True:
    prior_names = ["jeffrey", "PC10-4", "arcsine", "arctanh", "uniform"]

    rho = 0.8

    plt.figure(2)
    for name in prior_names:
        dict = pickle.load(open("CD_samples_n_3/"+name+f"{round(10*rho):02}" +
                                "1000.p", "rb"))

        samples = dict["samples"]
        n = len(samples)

        alphas = np.linspace(0, 1, 1000)
        confs_lower = np.array(
            [np.sum((samples < alpha)) / n for alpha in alphas])

        plt.plot(alphas, confs_lower-alphas, label=name)
    plt.plot(alphas, alphas-alphas,label="linear")
    plt.xlabel(r"$\alpha$", fontsize=14)
    plt.ylabel("Error in coverage rate", fontsize=14)
    plt.legend(fontsize=16)
    plt.show()


