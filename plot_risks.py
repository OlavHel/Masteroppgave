import numpy as np
import matplotlib.pyplot as plt
import pickle

# code for plotting the risks in the stored data

folder = "CD_samples_n_20"

rhos = [0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
CD_numbers = [1,2,3,4]

n = len(rhos)
m = len(CD_numbers)

names = ["UVCD","CVCD", "DiffCD", "CD1"]
files = [["/CD"+str(CD_numbers[j])+
                    f"{round(10*rhos[i]):02}" +
                     "1000.p" for i in range(n)] for j in range(m)]


risk_names = ["Mean mean:", "Mean var:", "Mean MAE:", "Mean MSE:", "Mean FIM:", "Mean KLD:", "Mean z_mean:",
              "Mean z_MSE:", "Mean w_mean:", "Mean w_MSE:", "Mean f_mean:", "Mean f_MSE:"]

# the list of indexes for the risks to be visualized, see list risk_names
risks_of_interest = [3,4,5,7]
data = np.empty((m,n,len(risks_of_interest)))

for i in range(m):
    for j in range(n):
        dict = pickle.load(open(folder+"/"+files[i][j],"rb"))

        samples = dict["samples"]
        properties = dict["properties"]

        risks = np.mean(properties,axis=0)

        data[i,j,:] = risks[risks_of_interest]

plt.figure()
for j in range(len(risks_of_interest)):
    plt.subplot(2,2,j+1)
    plt.title(risk_names[risks_of_interest[j]], fontsize=20)
    plt.xlabel(r"$\rho$", fontsize=22)
    plt.ylabel("risk", fontsize=22)
    for i in range(m):
        plt.plot(rhos, data[i,:,j], label = names[i])
if True:
    # if the risk for the fiduc_2 should be added
    fid_rhos = rhos
    fid_data = np.empty((len(fid_rhos),len(risks_of_interest)))
    for i in range(len(fid_rhos)):
        dict = pickle.load(open(folder + "/" + "fiduc2"+f"{round(10*fid_rhos[i]):02}" +
                     "1000.p", "rb"))
        properties = dict["properties"]
        risks = np.mean(properties, axis=0)
        print(risks)
        fid_data[i, :] = risks[risks_of_interest]
    for i in range(len(risks_of_interest)):
        plt.subplot(2,2,i+1)
        plt.plot(fid_rhos, fid_data[:,i],label="fiduc2")
if True:
    # if the risk of fiduc_infinity should be added
    fid_rhos = rhos
    fid_data = np.empty((len(fid_rhos),len(risks_of_interest)))
    for i in range(len(fid_rhos)):
        dict = pickle.load(open(folder + "/" + "fiducinf"+f"{round(10*fid_rhos[i]):02}" +
                     "1000.p", "rb"))
        properties = dict["properties"]
        risks = np.mean(properties, axis=0)
        print(risks)
        fid_data[i, :] = risks[risks_of_interest]
    for i in range(len(risks_of_interest)):
        plt.subplot(2,2,i+1)
        plt.plot(fid_rhos, fid_data[:,i],label="fiducinf")
plt.subplots_adjust(left=0.1,
                    bottom=0.1,
                    right=0.9,
                    top=0.9,
                    wspace=0.2,
                    hspace=0.3)
plt.legend(fontsize=16)
plt.show()









