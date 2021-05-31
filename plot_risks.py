import numpy as np
import matplotlib.pyplot as plt
import pickle

folder = "CD_samples_n_20"

rhos = [0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
CD_numbers = [1,2,3]

n = len(rhos)
m = len(CD_numbers)

files = [["/CD"+str(CD_numbers[j])+
                    f"{round(10*rhos[i]):02}" +
                     "1000.p" for i in range(n)] for j in range(m)]


risk_names = ["Mean mean:", "Mean var:", "Mean MAE:", "Mean MSE:", "Mean FIM:", "Mean KLD:", "Mean z_mean:",
              "Mean z_MSE:", "Mean w_mean:", "Mean w_MSE:", "Mean f_mean:", "Mean f_MSE:"]

risks_of_interest = [2,3,4,5,7,9,11]
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
    plt.subplot(3,3,j+1)
    plt.title(risk_names[risks_of_interest[j]])
    plt.xlabel(r"$\rho$", fontsize=14)
    plt.ylabel("risk", fontsize=14)
    for i in range(m):
        plt.plot(rhos, data[i,:,j], label = "CD"+str(i+1))
plt.subplot_tool()
#plt.subplots_adjust(left=0.1,
#                    bottom=0.1,
#                    right=0.9,
#                    top=0.9,
#                    wspace=0.4,
#                    hspace=0.4)
plt.legend(loc="center left", bbox_to_anchor=(1,0.5), fontsize=16)
plt.show()









