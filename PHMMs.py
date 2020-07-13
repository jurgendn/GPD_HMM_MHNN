import numpy as np
import scipy as sp

from numpy import seterr
from scipy import stats
from scipy.special import logsumexp


class PHMMs:

    """
        Poisson Hidden Markov Model:
        ----------------
        Hidden Markov Model with emission probability is Poisson Distribution.
        --
                        - Input: Observation sequence
                        - Output: Hidden Markov Model, include:
                                        - Initial probability distribution: Pi
                                        - Transition matrix: A
                                        - Emission distribution parameters: B
        ----------------
    Initial model parameters:
        - nber_states: 				number of states in hidden layer
        - log_init_distribution: 	log of initial distribution
        - log_init_trans_matrix:	log of initial transition matrix of Markov chain
        - set_paramPoisson:			initialize parameter for emission distribution
        - epsi:						stopping criterion
        - ob_seqs:					observation sequence
    """

    def __init__(self, init_ditri, init_trans_matrix, set_paramPoisson, ob_seqs, epsi):
        seterr(divide='ignore')
        self.nber_states = len(init_ditri)
        self.log_init_ditri = np.log(init_ditri)
        self.log_init_trans_matrix = np.log(init_trans_matrix)
        self.set_paramPoisson = np.array(set_paramPoisson)
        self.epsi = epsi
        self.ob_seqs = ob_seqs
        seterr(divide='warn')

        # log[P(X = val), X is Poisson(mean)]
    def log_prob_Poisson(self, mean, val):
        return stats.poisson(mean).logpmf(val)

    """
	Generate matrix of alpha_t(i), which is a n x T matrix satisfy:
	- Element at row t col i represent log(\alpha_t(i))
	"""

    def matrix_alpha(self):
        seterr(divide='ignore')
        log_bi_o1 = [self.log_prob_Poisson(
            self.set_paramPoisson[i], self.ob_seqs[0]) for i in range(self.nber_states)]
        log_pibi_o1 = np.add(self.log_init_ditri, log_bi_o1)
        alpha_matrix = [log_pibi_o1]
        for i in range(1, len(self.ob_seqs)):
            log_alpha_i = []
            for j in range(self.nber_states):
                log_alpha_prev = alpha_matrix[-1]
                log_prev_mulA = np.add(
                    log_alpha_prev, self.log_init_trans_matrix[::, j])
                sumlog = logsumexp(log_prev_mulA)
                alpha_ij = sumlog + \
                    self.log_prob_Poisson(
                        self.set_paramPoisson[j], self.ob_seqs[i])
                log_alpha_i.append(alpha_ij)
            alpha_matrix.append(log_alpha_i)
        seterr(divide='warn')
        return np.array(alpha_matrix)

    def matrix_beta(self):
        seterr(divide='ignore')
        # log_b_oT=[self.log_prob_Poisson(self.set_paramPoisson[i],self.ob_seqs[-1]) for i in range(self.nber_states) ]
        log_b_oT = [0 for i in range(self.nber_states)]
        beta_matrix = [log_b_oT]
        for i in range(len(self.ob_seqs)-2, -1, -1):  # i o day la t
            log_beta_i = []
            for j in range(self.nber_states):  # j ow day la i trong cong thuc
                log_beta_prev = beta_matrix[-1]
                log_prev_mulA = np.add(
                    self.log_init_trans_matrix[j], log_beta_prev)
                emit_prob = [self.log_prob_Poisson(
                    self.set_paramPoisson[k], self.ob_seqs[i+1]) for k in range(self.nber_states)]
                log_prev_mulA1 = np.add(log_prev_mulA, emit_prob)
                sumlog = logsumexp(log_prev_mulA1)
                beta_ij = sumlog
                log_beta_i.append(beta_ij)
            beta_matrix.append(log_beta_i)
        beta_matrix.reverse()
        seterr(divide='warn')
        return np.array(beta_matrix)

    """
	Calculate L_T by 2 method
	- By forward algorithm
	- By backward algorithm
	If they dont have significant different, return true
	"""

    def CDLL(self, tol=1e-100):  # P(O|lambda)=prob_O
        forward_probability = self.matrix_alpha()
        CDLL = np.exp(logsumexp(forward_probability[-1]))
        return CDLL

    def viterbi(self):
        v_n = [0.0 for _ in range(self.nber_states)]
        vlst = [v_n]
        wlst = []
        for i in range(len(self.ob_seqs)-1, 0, -1):
            v_i = []
            w_i = []
            for j in range(self.nber_states):
                all_v_ij = []
                for k in range(self.nber_states):
                    temp = self.log_init_trans_matrix[j, k] + self.log_prob_Poisson(
                        self.set_paramPoisson[k], self.ob_seqs[i])
                    temp += vlst[-1][k]
                    all_v_ij.append(temp)
                v_i.append(max(all_v_ij))
                w_i.append(np.argmax(all_v_ij))
            vlst.append(v_i)
            wlst.append(w_i)
        wlst.reverse()
        first_prob = [self.log_prob_Poisson(
            self.set_paramPoisson[i], self.ob_seqs[0]) for i in range(self.nber_states)]
        first_prob = np.add(first_prob, self.log_init_ditri)
        first_prob = np.add(first_prob, vlst[-1])
        h_1 = np.argmax(first_prob)
        statelst = [h_1]
        for i in range(len(wlst)):
            statelst.append(wlst[i][statelst[-1]])
        return statelst

    def numerator_update_trans(self, i, j):
        temp1 = self.matrix_alpha()
        temp2 = self.matrix_beta()
        C = 0
        for t in range(len(self.ob_seqs)-1):
            A = temp1[t][i]+self.log_init_trans_matrix[i, j]+self.log_prob_Poisson(
                self.set_paramPoisson[j], self.ob_seqs[t+1])+temp2[t+1][j]
            B = np.exp(A)
            C = C+B
        return np.log(C)

    def denominator_update(self, i):
        temp1 = self.matrix_alpha()
        temp2 = self.matrix_beta()
        C = 0
        for t in range(len(self.ob_seqs)-1):
            A = temp1[t][i]+temp2[t][i]
            B = np.exp(A)
            C = C+B
        return np.log(C)

    def numerator_update_lambda(self, i):
        temp1 = self.matrix_alpha()
        temp2 = self.matrix_beta()
        C = 0
        for t in range(len(self.ob_seqs)-1):
            A = temp1[t][i]+temp2[t][i]
            B = np.exp(A)*self.ob_seqs[t]
            C = C+B
        return C

    def update_init_ditri(self):
        temp1 = self.matrix_alpha()
        temp2 = self.matrix_beta()
        L_T = self.CDLL()
        for i in range(self.nber_states):
            self.log_init_ditri[i] = temp1[0][i]+temp2[0][i]-np.log(L_T)
        return True

    def Baum_Welch(self, max_iter=100):
        for _ in range(max_iter):
            pre_L_T = self.CDLL()
            # update trans_matrix
            for i in range(self.nber_states):
                for j in range(self.nber_states):
                    self.log_init_trans_matrix[i, j] = self.numerator_update_trans(
                        i, j)-self.denominator_update(i)
            # update paramPoisson
            for i in range(self.nber_states):
                self.set_paramPoisson[i] = self.numerator_update_lambda(
                    i)/(np.exp(self.denominator_update(i)))
            # update init_ditri
            self.update_init_ditri()
            current_L_T = self.CDLL()
            print("Current CDLL: ", current_L_T)
            if current_L_T < self.epsi*pre_L_T and current_L_T > pre_L_T:
                break
        return True

    def AIC(self):
        return (2*(self.nber_states**2+self.nber_states) - 2*self.CDLL())

    def BIC(self):
        return ((self.nber_states**2+self.nber_states)*np.log(len(self.ob_seqs)) - 2*self.CDLL())
