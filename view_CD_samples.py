import numpy as np
import matplotlib.pyplot as plt
import pickle

# code used to visualize the frequentistic coverage
# the file names can be seen in the folders "CD_samples_n_<number of data points>"


if True:
    # this code will visualize the coverage of only one distribution for one correlation and a given number of data points
    rho = 0.0
    dict = pickle.load(open("CD_samples_n_3/org_fiduc2051000.p","rb"))

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

    #plt.figure()
    #plt.subplot(1,2,1)
    #plt.plot(alphas, confs_lower-alphas)
    #plt.subplot(1,2,2)
    #plt.plot(alphas, (confs_lower-alphas)/alphas)
    #plt.show()

    confs_both = np.array(
        [np.sum((samples < (1 + alpha) / 2) & (samples > (1 - alpha) / 2)) / n for alpha in alphas])

    confs_upper = np.array(
        [np.sum((samples > 1-alpha)) / n for alpha in alphas])

    plt.figure()
    plt.plot(alphas, confs_lower, label="lower confs")
    plt.plot(alphas, confs_both, label="two-sided confs")
    #plt.plot(alphas, confs_upper, label="upper confs")
    plt.plot(alphas, alphas, label="linear")
    #plt.plot(np.linspace(0,1,50), confs_lower[50:100]-np.flip((confs_lower)[0:50]))
    plt.legend()
    plt.show()

elif False:
    # this code will visualize the coverage for multiple distributions for one correlation and a given number of data points

    prior_names = ["jeffrey","uniform","arctanh", "PC10-4", "arcsine"]

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
        confs_both = np.array(
            [np.sum((samples < (1 + alpha) / 2) & (samples > (1 - alpha) / 2)) / n for alpha in alphas])

        plt.plot(alphas, confs_both-alphas, label=name)
    plt.plot(alphas, alphas-alphas,label="linear")
    plt.xlabel(r"$\alpha$", fontsize=14)
    plt.ylabel("Error in coverage rate", fontsize=14)
    plt.legend(fontsize=16)
    plt.show()

elif False:
    # this code will visualize the coverage a distributions for one correlation and multipel choices for numbers of data points
    prior_name = "uniform"
    ns = ["3","10","20"]
    rho = 0.0

    plt.figure(2)
    for m in ns:
        dict = pickle.load(open("CD_samples_n_"+m+"/"+prior_name+f"{round(10*rho):02}" +
                                "1000.p", "rb"))

        samples = dict["samples"]
        n = len(samples)

        alphas = np.linspace(0, 1, 1000)
        confs_lower = np.array(
            [np.sum((samples < alpha)) / n for alpha in alphas])
        confs_both = np.array(
            [np.sum((samples < (1 + alpha) / 2) & (samples > (1 - alpha) / 2)) / n for alpha in alphas])

        plt.plot(alphas, confs_both, label=m)
    plt.plot(alphas, alphas,label="linear")
    plt.xlabel(r"$\alpha$", fontsize=14)
    plt.ylabel("Error in coverage rate", fontsize=14)
    plt.legend(fontsize=16)
    plt.show()

elif True:
    # this code will visualize the coverage for a distributions for multiple correlations and a given number of data points
    prior_name = "fiducinf"
    m = "20"
    rhos = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

    plt.figure(2)
    for rho in rhos:
        dict = pickle.load(open("CD_samples_n_"+m+"/"+prior_name+f"{round(10*rho):02}" +
                                "1000.p", "rb"))

        samples = dict["samples"]
        n = len(samples)

        alphas = np.linspace(0, 1, 1000)
        confs_lower = np.array(
            [np.sum((samples < alpha)) / n for alpha in alphas])
        confs_both = np.array(
            [np.sum((samples < (1 + alpha) / 2) & (samples > (1 - alpha) / 2)) / n for alpha in alphas])

        plt.plot(alphas, confs_both-alphas, label=r"$\rho$="+str(rho))
    plt.plot(alphas, alphas-alphas,label="linear")
    plt.xlabel(r"$\alpha$", fontsize=14)
    plt.ylabel("Error in coverage rate", fontsize=14)
    plt.legend(fontsize=16)
    plt.show()




