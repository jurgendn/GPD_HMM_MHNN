import numpy as np

import PHMMs as pm

theta = np.array([[1/4 for _ in range(4)] for __ in range(4)])
delta = np.ones(4)/0.25
lambdas = np.array([1,1,1])
seqs = [0, 3, 2, 9, 9, 0, 1, 1, 1, 4, 5, 1, 1, 1, 1, 0, 2, 3, 2, 3, 1, 5, 5, 4, 6, 1, 6, 1, 1, 1, 5, 1, 5, 0, 8, 1, 4, 4, 2, 2, 2, 0, 8, 1, 2, 6, 0, 7, 8, 7,
        2, 3, 2, 10, 1, 7, 1, 4, 2, 1, 8, 1, 2, 10, 5, 4, 2, 6, 0, 2, 2, 8, 5, 5, 8, 6, 2, 5, 1, 1, 0, 3, 6, 2, 0, 9, 0, 1, 0, 8, 13, 10, 2, 1, 2, 3, 9, 4, 3, 5]
test = pm.PHMMs(delta, theta, lambdas, seqs, epsi=0.0001)
test.Baum_Welch()
print("_________________")
print("_Infomation Test_")
print(test.AIC())
print(test.BIC())
print("_________________")
print(np.exp(test.log_init_trans_matrix))
print(test.set_paramPoisson)
print("_________________")
