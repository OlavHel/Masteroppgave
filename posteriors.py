import numpy as np
from scipy.integrate import quad


class Posterior:
    def __init__(self,posterior=None,lam=10**(-1)):
        self.posterior = posterior
        self.lam = lam
        self.distr = {
            "jeffrey":self.jeffreys,
            "PC":lambda x,n,T1,T2: self.PC(x,n,T1,T2,self.lam),
            "uniform":self.uniformprior,
            "arcsine":self.arcsine,
            "new":self.new_one,
            "fiduc_2": self.fiducial_2,
            "fiduc_infty": self.fiducial_infinity,
            "test": self.testprior
        }

    def get_list_of_posteriors(self):
        return [
            [0,"jeffrey"],
            [1,"PC"],
            [2,"uniform"],
            [3,"arcsine"],
            [4,"new"],
            [5,"fiducial 2"],
            [6,"fiducial infinity"],
            [7,"test"]
        ]

    def set_posterior(self,i,lam=10**(-1)):
        self.posterior = {
            0:"jeffrey",
            1:"PC",
            2:"uniform",
            3:"arcsine",
            4:"new",
            5:"fiduc_2",
            6:"fiduc_infty",
            7:"test"
        }[i]

    def distribution(self,rho,n,T1,T2):
        return self.distr[self.posterior](rho,n,T1,T2)

    def norm_distribution(self,rho,n,T1,T2):
        return self.distribution(rho,n,T1,T2)/self.normalization(n,T1,T2)

    def normalization(self,n,T1,T2):
        distr = self.distr[self.posterior]
        c, c_error = quad(distr, -1, 1, args=(n, T1, T2))
        return c

    def fiducial_2(self, rho, n, T1, T2):
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


    def testprior(self, rho, n, T1, T2):
        if type(rho) == type(np.array([0.1])) and len(rho) == 1:
            rho = rho[0]
        if type(rho) == type(np.array([])):
            rho[rho <= -1] = 0
            rho[rho >= 1] = 0
        elif (rho <= (-1) or rho >= (1)):
            return 0
        temp = 1 - rho ** 2
        return rho**2*temp ** (-n / 2) * np.exp(-T1 / (2 * temp) + rho * T2 / temp)

    def uniformprior(self,rho, n, T1, T2):
        if type(rho) == type(np.array([0.1])) and len(rho) == 1:
            rho = rho[0]
        temp = 1 - rho ** 2
        if type(rho) == type(np.array([])):
            rho[rho <= -1] = 0
            rho[rho >= 1] = 0
        elif (rho <= (-1) or rho >= (1)):
            return 0
        return np.exp(n / 1.1) * temp ** (-n / 2) * np.exp(-T1 / (2 * temp) + rho * T2 / temp)

    def jeffreys(self,rho, n, T1, T2):
        if type(rho) == type(np.array([0.1])) and len(rho) == 1:
            rho = rho[0]
        temp = 1 - rho ** 2
        if type(rho) == type(np.array([])):
            rho[rho <= -1] = 0
            rho[rho >= 1] = 0
        elif (rho <= (-1) or rho >= (1)):
            return 0
        return np.sqrt(1 + rho ** 2) / ((2 * np.pi) ** n * temp ** (n / 2 + 1)) * np.exp(
            -T1 / (2 * temp) + rho * T2 / temp)

    def PC(self,rho, n, T1, T2, lam): ## noe galt med PC
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
        return lam * np.abs(rho) / (temp ** (n / 2 + 1) * np.sqrt(-np.log(temp))) * \
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
        return np.exp(n) / (temp ** (n / 2 + 1 / 2)) * np.exp(-T1 / (2 * temp) + rho * T2 / temp)

    def new_one(self, rho, n, T1, T2):
        if type(rho) == type(np.array([0.1])) and len(rho) == 1:
            rho = rho[0]
        if type(rho) == type(np.array([])):
            rho[rho <= -1] = 0
            rho[rho >= 1] = 0
        elif (rho <= (-1) or rho >= (1)):
            return 0
        temp = 1 - rho ** 2
        return np.exp(n) / (temp ** (n / 2 + 1)) * np.exp(-T1 / (2 * temp) + rho * T2 / temp)

if __name__ == "__main__":
    import matplotlib.pyplot as plt

    rhos = np.linspace(-1,1,100)

    plt.figure()
    plt.plot(rhos, np.sqrt(1+rhos**2)/(1-rhos**2))
    plt.show()









