import numpy as np
from scipy.integrate import quad


class Posterior:
    # class for all the posteriors
    def __init__(self,posterior=None,lam=10**(-1)):
        self.posterior = posterior
        self.lam = lam
        self.distr = {
            "jeffrey":self.jeffreys,
            "PC":lambda x,n,T1,T2: self.PC(x,n,T1,T2,self.lam),
            "uniform":self.uniformprior,
            "arcsine":self.arcsine,
            "arctanh":self.arctanh,
            "fiduc_2": self.fiducial_2, # fiduc_2 for sufficient statistics
            "fiduc_infty": self.fiducial_infinity, # fiduc_infinity for sufficient statistics
            "fiduc_orig_2": self.fiducial_orig_2, # fiduc_2 for original data but not "symmetric"
            "fiduc_orig_infty": self.fiducial_orig_infinity, # fiduc_infinity for original data but not "symmetric"
            "fiduc_orig_2_new": self.fiducial_orig_2_new, ## fiduc_2 for original data and "symmetric"
            "fiduc_orig_infty_new": self.fiducial_orig_infinity_new # fiduc_infinity for original data and "symmetric"
        }

    def get_list_of_posteriors(self):
        # can be used to find the numbering to choose to set the posterior class in set_posterior
        return [
            [0,"jeffrey"],
            [1,"PC"],
            [2,"uniform"],
            [3,"arcsine"],
            [4,"arctanh"],
            [5,"fiducial 2"],
            [6,"fiducial infinity"],
            [7,"fiducial original 2"],
            [8, "fiducial original infinity"],
            [9, "fiducial original 2 symmetric"],
            [10, "fiducial original infinity symmetric"]
        ]

    def set_posterior(self,i):
        # can be used to set the posterior for the Posterior class
        # for instance, i=1 makes the Posterior object work as the PC posterior
        self.posterior = {
            0:"jeffrey",
            1:"PC",
            2:"uniform",
            3:"arcsine",
            4:"arctanh",
            5:"fiduc_2",
            6:"fiduc_infty",
            7:"fiduc_orig_2",
            8:"fiduc_orig_infinity",
            9: "fiduc_orig_2_new",
            10: "fiduc_orig_infinity_new"
        }[i]

    def distribution(self,rho,n,T1,T2):
        return self.distr[self.posterior](rho,n,T1,T2)

    def norm_distribution(self,rho,n,T1,T2):
        return self.distribution(rho,n,T1,T2)/self.normalization(n,T1,T2)

    def normalization(self,n,T1,T2):
        distr = self.distr[self.posterior]
        c, c_error = quad(distr, -1, 1, args=(n, T1, T2))
        return c

    def fiducial_orig_2_new(self, rho, n, T1, T2):
        if type(rho) == type(np.array([0.1])) and len(rho) == 1:
            rho = rho[0]
        if type(rho) == type(np.array([])):
            rho[rho <= -1] = 0
            rho[rho >= 1] = 0
        elif (rho <= (-1) or rho >= (1)):
            return 0


        S1 = np.sum((T1+T2)**2)
        S2 = np.sum((T1-T2)**2)

        temp = (S1+S2)*rho**2-2*(S1-S2)*rho+(S1+S2)

        return np.sqrt(temp)*(1-rho**2)**(-n/2-1)*np.exp(-1/4*(S1/(1+rho)+S2/(1-rho)))

    def fiducial_orig_infinity_new(self, rho, n, T1, T2):
        if type(rho) == type(np.array([0.1])) and len(rho) == 1:
            rho = rho[0]
        if type(rho) == type(np.array([])):
            rho[rho <= -1] = 0
            rho[rho >= 1] = 0
            temp1 = np.outer(np.ones(len(rho)),T1)-np.outer(rho,T2)
            temp2 = np.outer(np.ones(len(rho)),T2)-np.outer(rho,T1)
            temp = np.sum(np.abs(temp1),axis=1)+np.sum(np.abs(temp2))
        else:
            if (rho <= (-1) or rho >= (1)):
                return 0
            temp = np.sum(np.abs(T1-rho*T2))+np.sum(np.abs(T2-rho*T1))

        S1 = np.sum((T1 + T2) ** 2)
        S2 = np.sum((T1 - T2) ** 2)

        return temp * (1 - rho ** 2) ** (-n / 2 - 1) * np.exp(-1 / 4 * (S1 / (1 + rho) + S2 / (1 - rho)))

    def fiducial_orig_2(self, rho, n, T1, T2):
        if type(rho) == type(np.array([0.1])) and len(rho) == 1:
            rho = rho[0]
        if type(rho) == type(np.array([])):
            rho[rho <= -1] = 0
            rho[rho >= 1] = 0
        elif (rho <= (-1) or rho >= (1)):
            return 0

        temp = np.sum(T1**2)+np.sum(T2**2)*rho**2-2*np.sum(T1*T2)*rho

        S1 = np.sum((T1+T2)**2)
        S2 = np.sum((T1-T2)**2)

        return np.sqrt(temp)*(1-rho**2)**(-n/2-1)*np.exp(-1/4*(S1/(1+rho)+S2/(1-rho)))

    def fiducial_orig_infinity(self, rho, n, T1, T2):
        if type(rho) == type(np.array([0.1])) and len(rho) == 1:
            rho = rho[0]
        if type(rho) == type(np.array([])):
            rho[rho <= -1] = 0
            rho[rho >= 1] = 0
            temp1 = np.outer(np.ones(len(rho)),T1)-np.outer(rho,T2)
            temp = np.sum(np.abs(temp1),axis=1)
        else:
            if (rho <= (-1) or rho >= (1)):
                return 0
            temp = np.sum(np.abs(T1-rho*T2))


        S1 = np.sum((T1 + T2) ** 2)
        S2 = np.sum((T1 - T2) ** 2)

        return temp * (1 - rho ** 2) ** (-n / 2 - 1) * np.exp(-1 / 4 * (S1 / (1 + rho) + S2 / (1 - rho)))

    def fiducial_2(self, rho, n, T1, T2):
        if type(rho) == type(np.array([0.1])) and len(rho) == 1:
            rho = rho[0]
        if type(rho) == type(np.array([])):
            rho[rho <= -1] = 0
            rho[rho >= 1] = 0
        elif (rho <= (-1) or rho >= (1)):
            return 0
        S1 = T1+2*T2
        S2 = T1-2*T2

        return np.sqrt(S1**2*(1+rho)**(-2)+S2**2*(1-rho)**(-2))*(1-rho**2)**(-n/2)*np.exp(-1/4*(S1/(1+rho)+S2/(1-rho)))

    def fiducial_infinity(self, rho, n, T1, T2):
        if type(rho) == type(np.array([0.1])) and len(rho) == 1:
            rho = rho[0]
        temp = 1 - rho ** 2
        if type(rho) == type(np.array([])):
            rho[rho <= -1] = 0
            rho[rho >= 1] = 0
        elif (rho <= (-1) or rho >= (1)):
            return 0
        S1 = T1+2*T2
        S2 = T1-2*T2

        return (S1/(1+rho)+S2/(1-rho))*(1-rho**2)**(-n/2)*np.exp(-1/4*(S1/(1+rho)+S2/(1-rho)))

    def uniformprior(self,rho, n, T1, T2):
        if type(rho) == type(np.array([0.1])) and len(rho) == 1:
            rho = rho[0]
        temp = 1 - rho ** 2
        if type(rho) == type(np.array([])):
            rho[rho <= -1] = 0
            rho[rho >= 1] = 0
        elif (rho <= (-1) or rho >= (1)):
            return 0
        return temp ** (-n / 2) * np.exp(-T1 / (2 * temp) + rho * T2 / temp)

    def jeffreys(self,rho, n, T1, T2):
        if type(rho) == type(np.array([0.1])) and len(rho) == 1:
            rho = rho[0]
        temp = 1 - rho ** 2
        if type(rho) == type(np.array([])):
            rho[rho <= -1] = 0
            rho[rho >= 1] = 0
        elif (rho <= (-1) or rho >= (1)):
            return 0
        return np.sqrt(1 + rho ** 2) / (temp ** (n / 2 + 1)) * np.exp(
            -T1 / (2 * temp) + rho * T2 / temp)

    def PC(self,rho, n, T1, T2, lam):
        if type(rho) == type(np.array([0.1])) and len(rho) == 1:
            rho = rho[0]
        if type(rho) == type(np.array([])):
            rho[rho <= -1] = 0
            rho[rho >= 1] = 0
            rho[rho == 0] = lam
        elif (rho <= (-1) or rho >= (1)):
            return 0
        elif rho == 0:
            return lam
        temp = 1 - rho ** 2
        return np.abs(rho) / (temp ** (n / 2 + 1) * np.sqrt(-np.log(temp))) * \
               np.exp(-T1 / (2 * temp) + rho * T2 / temp - lam * np.sqrt(-np.log(temp)))

    def arcsine(self,rho, n, T1, T2):
        if type(rho) == type(np.array([0.1])) and len(rho) == 1:
            rho = rho[0]
        if type(rho) == type(np.array([])):
            rho[rho <= -1] = 0
            rho[rho >= 1] = 0
        elif (rho <= (-1) or rho >= (1)):
            return 0
        temp = 1 - rho ** 2
        return 1 / (temp ** (n / 2 + 1 / 2)) * np.exp(-T1 / (2 * temp) + rho * T2 / temp)

    def arctanh(self, rho, n, T1, T2):
        if type(rho) == type(np.array([0.1])) and len(rho) == 1:
            rho = rho[0]
        if type(rho) == type(np.array([])):
            rho[rho <= -1] = 0
            rho[rho >= 1] = 0
        elif (rho <= (-1) or rho >= (1)):
            return 0
        temp = 1 - rho ** 2
        return 1 / (temp ** (n / 2 + 1)) * np.exp(-T1 / (2 * temp) + rho * T2 / temp)

if __name__ == "__main__":
    import matplotlib.pyplot as plt

    rhos = np.linspace(-1,1,100)

    plt.figure()
    plt.plot(rhos, np.sqrt(1+rhos**2)/(1-rhos**2))
    plt.show()









