import scipy as sp
import numpy as np
from scipy import stats
from scipy.special import logsumexp
from numpy import seterr

class PHMMs:
	def __init__(self,init_ditri,init_trans_matrix,set_paramPoisson,ob_seqs,epsi):
		seterr(divide='ignore')
		self.nber_states=len(init_ditri)
		self.log_init_ditri=np.log(init_ditri)
		self.log_init_trans_matrix=np.log(init_trans_matrix)
		self.set_paramPoisson=np.array(set_paramPoisson)
		self.epsi=epsi
		self.ob_seqs=ob_seqs
		seterr(divide='warn')

	def log_prob_Poisson(self, mean, val):
		return stats.poisson(mean).logpmf(val)


	def matrix_alpha(self):#output la matrix alpha voi hang la tu 0 den N-1, cot la tu 0 den T-1
		seterr(divide='ignore')
		log_bi_o1=[self.log_prob_Poisson(self.set_paramPoisson[i],self.ob_seqs[0]) for i in range(self.nber_states) ]
		log_pibi_o1=np.add(self.log_init_ditri,log_bi_o1)
		alpha_matrix=[log_pibi_o1]
		for i in range(1,len(self.ob_seqs)):
			log_alpha_i=[]
			for j in range(self.nber_states):
				log_alpha_prev=alpha_matrix[-1]
				log_prev_mulA=np.add(log_alpha_prev,self.log_init_trans_matrix[::,j])
				sumlog=logsumexp(log_prev_mulA)
				alpha_ij=sumlog+self.log_prob_Poisson(self.set_paramPoisson[j],self.ob_seqs[i])
				log_alpha_i.append(alpha_ij)
			alpha_matrix.append(log_alpha_i)
		alpha_T=alpha_matrix[-1]
		seterr(divide='warn')
		print("kich thuoc ma tran alpha:",len(alpha_matrix),len(alpha_matrix[0]))
		return np.array(alpha_matrix)

	def matrix_beta(self):#output laf ma tran beta voi beta[i] la tai t=i, 1 hang tu 1 den N(nberstates)
		seterr(divide='ignore')
		#log_b_oT=[self.log_prob_Poisson(self.set_paramPoisson[i],self.ob_seqs[-1]) for i in range(self.nber_states) ]
		log_b_oT=np.zeros((1,self.nber_states))
		print("log_b_oT",log_b_oT)
		beta_matrix=[log_b_oT]
		for i in range(len(self.ob_seqs)-2,-1,-1):
			log_beta_i=[]
			for j in range(self.nber_states):
				log_beta_prev=beta_matrix[-1]
				log_prev_mulA=np.add(self.log_init_trans_matrix[j],log_beta_prev)
				sumlog=logsumexp(log_prev_mulA)
				beta_ij=self.log_prob_Poisson(self.set_paramPoisson[j],self.ob_seqs[i])+sumlog
				log_beta_i.append(beta_ij)
			beta_matrix.append(log_beta_i)
		beta_matrix.reverse()
		seterr(divide='warn')
		print("beta_T",beta_matrix[-1])
		print("kick thuoc ma tran beta",len(beta_matrix))
		return np.array(beta_matrix)
	def check(self):#P(O|lambda)=prob_O
		temp1=self.matrix_alpha()
		temp2=self.matrix_beta()
		prob_O_directly=np.exp(logsumexp(temp1[-1]))
		print("alpha_T:",temp1[-1])
	#tinh theo 2 ma tran beta and alpha
		#t=np.random.randint(len(self.ob_seqs))#tu chon de check, mien la thuoc (0,T)
		t=3
		alpha_t=temp1[t]
		beta_t=temp2[t]
		log_result=np.add(alpha_t,beta_t)
		print("thourgh alpha, beta:",log_result)#ghi nho xichma beta_T=1 voi moi i 
		prob_O_through_albe=np.exp(logsumexp(log_result))
		print(prob_O_through_albe)
		print(prob_O_directly)
		if (prob_O_directly)==(prob_O_through_albe):
			print("the result of problem 1 of HMMs is probability:",prob_O_directly)
			return prob_O_directly
		else:
			print("have problems to compute matrix alpha and beta",prob_O_directly,prob_O_through_albe)
			return False
	def vterbi(self):
		v_n=[0.0 for _ in range(self.nber_states)]
		vlst=[v_n]
		wlst=[]
		for i in range(len(self.ob_seqs)):
			v_i = []
			w_i = []
			for j in range(self.nber_states):
				all_v_ij = []
				for k in range(self.nber_states):
					temp = self.log_init_trans_matrix[j,k] + self.log_prob_Poisson(self.set_paramPoisson[k], self.ob_seqs[i])
					temp += vlst[-1][k]
					all_v_ij.append(temp)
				v_i.append(max(all_v_ij))
				w_i.append(np.argmax(all_v_ij))
			vlst.append(v_i)
			wlst.append(w_i)
		wlst.reverse()
		first_prob = [self.log_prob_Poisson(self.set_paramPoisson[i], self.ob_seqs[0]) for i in range(self.nber_states)]
		first_prob = np.add(first_prob, self.log_init_ditri)
		first_prob=np.add(first_prob, vlst[-1])
		h_1 = np.argmax(first_prob)
		statelst = [h_1]
		for i in range(len(wlst)):
			statelst.append(wlst[i][statelst[-1]])
		return statelst
	
	
	def numerator_update_trans(self,i,j):
		temp1=self.matrix_alpha()
		temp2=self.matrix_beta()
		C=0
		for t in range(len(self.ob_seqs)-1):
			A=temp1[t][i]+self.log_init_trans_matrix[i,j]+self.log_prob_Poisson(self.set_paramPoisson[j],self.ob_seqs[t+1])+temp2[t+1][j]
			B=np.exp(A)
			C=C+B
		return np.log(C)
	def denominator_update(self,i):
		temp1=self.matrix_alpha()
		temp2=self.matrix_beta()
		C=0
		for t in range(len(self.ob_seqs)-1):
			A=temp1[t][i]+temp2[t][i]
			B=np.exp(A)
			C=C+B
		return np.log(C)

	def numerator_update_lambda(self,i):
		temp1=self.matrix_alpha()
		temp2=self.matrix_beta()
		C=0
		for t in range(len(self.ob_seqs)-1):
			A=temp1[t][i]+temp2[t][i]
			B=np.exp(A)*self.ob_seqs[t]
			C=C+B
		return C

	def update_init_ditri(self):
		temp1=self.matrix_alpha()
		temp2=self.matrix_beta()
		L_T=self.check()
		for i in range(self.nber_states):
			self.log_init_ditri[i]=temp1[0][i]+temp2[0][i]-np.log(L_T)
		return True

		



	def Baum_Welch(self,max_iter=150):
		for itere in range(max_iter):
			pre_L_T=self.check()
			#update trans_matrix
			for i in range(self.nber_states):
				for j in range(self.nber_states):
					self.log_init_trans_matrix[i,j]=self.numerator_update_trans(i,j)-self.denominator_update(i)
			#update paramPoisson
			for i in range(self.nber_states):
				self.set_paramPoisson[i]=self.numerator_update_lambda(i)/(np.exp(self.denominator_update(i)))
			#update init_ditri
			self.update_init_ditri()
			current_L_T=self.check()		
			if (current_L_T) < np.exp(self.epsi)*pre_L_T:
				break
		return True		








			
theta = np.array([[0.3, 0.7],[0.7, 0.3]]) 
delta = np.array([1.0, 0.0])
lambdas = np.array([0.5, 6.7])
epsi1=0.0001
seqs=np.array([1, 1, 6, 5, 6, 10, 0, 9, 1, 0, 0, 8, 1, 1, 7, 5, 1, 1, 1, 0, 2, 3, 5, 5, 0, 2, 1, 0, 5, 3, 4, 5, 1, 6, 0, 7, 2, 0, 5, 6, 3, 9, 2, 3, 0, 1, 0, 1, 0, 4, 2, 4, 2, 9, 3, 0, 0, 4, 1, 1, 2, 8, 5, 1, 0, 9, 2, 0, 2, 3, 0, 7, 6, 7, 1, 0, 4, 1, 0, 2, 1, 6, 4, 6, 1, 7, 7, 6, 2, 6, 0, 1, 8, 6, 0, 1, 9, 9, 4, 1, 1, 8, 3, 1, 6, 3, 7, 0, 6, 1, 1, 1, 8, 2, 7, 1, 1, 2, 3, 5, 0, 0, 3, 2, 0, 5, 0, 2, 1, 1, 1, 2, 2, 0, 1, 2, 1, 4, 2, 1, 0, 1, 0, 0, 0, 4, 2, 3, 2, 5, 7, 1, 3, 3, 0, 1, 1, 8, 1, 0, 1, 1, 1, 7, 4, 2, 5, 4, 0, 2, 6, 2, 3, 1, 3, 0, 4, 2, 1, 0, 3, 4, 3, 4, 2, 3, 5, 1, 6, 2, 3, 3, 5, 0, 0, 5, 1, 6, 5, 1, 0, 6, 0, 3, 2, 2, 1, 1, 7, 0, 8, 1, 1, 1, 3, 2, 4, 2, 1, 4, 5, 1, 1, 3, 7, 1, 1, 1, 1, 4, 7, 0, 2, 5, 0, 4, 8, 8, 3, 1, 6, 6, 6, 1, 1, 3, 1, 3, 1, 7, 5, 9, 1, 2, 7, 1, 7, 10, 0, 2, 0, 0, 0, 7, 1, 1, 2, 4, 2, 2, 0, 7, 4, 2, 7, 0, 0, 0, 2, 4, 7, 0, 4, 0, 1, 1, 1, 1, 2, 6, 0, 2, 7, 1, 1, 7, 6, 1, 6, 4, 0, 1, 0, 6, 6, 0, 5, 1, 3, 1, 9, 5, 3, 1, 8, 1, 0, 6, 1, 7, 6, 2, 3, 7, 2, 4, 1, 2, 2, 5, 5, 1, 4, 2, 4, 0, 5, 3, 1, 1, 3, 1, 1, 7, 0, 4, 4, 6, 5, 0, 9, 1, 0, 5, 4, 2, 0, 1, 7, 3, 3, 1, 6, 2, 0, 1, 0, 3, 4, 0, 6, 3, 2, 0, 4, 1, 0, 0, 10, 4, 1, 0, 6, 1, 2, 0, 5, 10, 2, 1, 6, 5, 1, 2, 0, 3, 3, 7, 1, 1, 1, 1, 5, 5, 3, 3, 1, 1, 2, 0, 1, 5, 0, 0, 1, 6, 1, 7, 5, 3, 1, 3, 1, 3, 0, 0, 3, 6, 2, 3, 5, 5, 0, 1, 4, 0, 2, 3, 0, 7, 0, 5, 4, 5, 1, 6, 10, 3, 5, 0, 1, 3, 5, 1, 1, 10, 4, 1, 1, 1, 2, 6, 1, 0, 6, 2, 0, 5, 6, 6, 5, 1, 0, 7, 3, 11, 6, 1, 2, 2, 2, 3, 2, 0, 1, 0, 0, 1, 2, 7, 4, 1, 7, 9, 5, 4, 4, 4, 1, 2, 1, 1, 1, 3, 1, 1, 7, 0, 7, 1, 0, 4, 12, 6, 5, 4, 2, 12, 1, 4, 6, 2, 4, 8, 4, 0, 8, 3, 4, 6, 2, 1, 0, 0, 1, 1, 1, 13, 4, 8, 4, 2, 6, 3, 1, 2, 4, 7, 5, 1, 2, 2, 3, 4, 5, 7, 3, 3, 3, 7, 1, 0, 0, 5, 1, 4, 8, 5, 2, 2, 1, 5, 3, 5, 5, 0, 0, 1, 2, 2, 3, 9, 3, 1, 1, 8, 0, 1, 0, 1, 7, 1, 1, 0, 4, 1, 2, 5, 9, 3, 1, 0, 2, 1, 2, 9, 1, 5, 5, 6, 1, 2, 9, 2, 2, 2, 6, 0, 0, 0, 1, 1, 4, 1, 3, 1, 3, 7, 2, 1, 7, 1, 0, 3, 1, 3, 3, 1, 7, 1, 4, 7, 4, 5, 0, 3, 0, 5, 5, 6, 0, 0, 2, 0, 5, 9, 0, 0, 0, 6, 9, 1, 3, 2, 3, 5, 7, 5, 0, 7, 3, 13, 0, 1, 7, 0, 0, 3, 1, 1, 6, 2, 3, 2, 0, 1, 3, 3, 9, 4, 8, 1, 1, 0, 3, 4, 3, 0, 0, 0, 0, 1, 11, 4, 1, 5, 0, 1, 2, 6, 1, 1, 0, 1, 0, 1, 9, 3, 3, 9, 1, 5, 4, 9, 8, 0, 2, 5, 6, 1, 4, 2, 1, 1, 2, 10, 1, 3, 1, 3, 3, 0, 1, 1, 1, 6, 2, 4, 0, 7, 3, 4, 2, 5, 1, 2, 0, 3, 6, 1, 5, 2, 0, 1, 0, 4, 1, 0, 0, 0, 0, 1, 4, 7, 1, 1, 2, 0, 1, 6, 1, 1, 10, 5, 6, 4, 3, 0, 0, 4, 8, 1, 6, 4, 10, 2, 6, 1, 5, 2, 1, 2, 7, 5, 3, 2, 0, 0, 5, 7, 3, 1, 0, 1, 4, 6, 6, 5, 0, 1, 1, 1, 2, 3, 3, 0, 1, 12, 6, 1, 0, 5, 2, 1, 1, 0, 1, 5, 0, 1, 4, 2, 2, 3, 1, 0, 0, 4, 5, 3, 4, 5, 10, 4, 0, 2, 0, 1, 4, 7, 1, 1, 1, 2, 2, 0, 3, 1, 6, 4, 1, 3, 6, 1, 0, 2, 3, 7, 2, 6, 6, 4, 3, 1, 2, 6, 6, 1, 4, 0, 2, 0, 0, 2, 1, 5, 4, 5, 5, 0, 2, 4, 2, 11, 3, 0, 3, 2, 3, 1, 7, 7, 6, 4, 5, 2, 10, 1, 4, 3, 0, 2, 7, 5, 4, 0, 0, 6, 3, 1, 7, 12, 0, 5, 9, 3, 0, 0, 2, 6, 0, 2, 5, 4, 1, 9, 0, 5, 5, 8, 1, 7, 3, 3, 3, 1, 2, 1, 0, 1, 5, 3, 1, 3, 2, 1, 0, 3, 1, 6, 1, 10, 6, 7, 5, 7, 0, 7, 0, 1, 7, 0, 1, 2, 1, 1, 2, 1, 2, 2, 2, 3, 4, 5, 2, 3, 5, 7, 10, 3, 1, 7, 2, 2, 12, 4, 3, 3, 2, 1, 2, 2, 2, 13, 1, 4, 5, 0, 5, 1, 6, 0, 16, 2, 0, 3, 1, 6, 1, 0, 5, 4, 1, 2, 2, 2, 0, 5, 2, 9, 1, 2, 4, 3, 6, 8, 6, 6, 6, 2, 0, 6, 3, 9, 6, 0, 7, 0, 7, 1, 3, 1, 1, 2, 0, 6, 1, 7, 6, 1, 7, 0, 4, 1, 1, 0, 1, 4, 2, 0, 0, 4, 3, 4, 3, 7, 6, 0, 1, 9, 1, 0, 1, 1, 8, 4, 10, 0, 8, 6, 2, 2, 2, 7, 1, 4, 1, 8, 1, 8, 5, 6, 0, 2, 6, 1, 4, 1, 3, 4, 10, 8, 1, 7, 5, 0, 6, 0, 5, 1, 1, 5, 4, 1, 5, 0, 1, 4, 2, 1, 4, 8, 2, 12, 1, 3, 0, 0, 0, 0, 1, 2, 8, 4, 1, 3, 1, 4, 0, 0, 1, 0, 1, 2, 5, 2, 1, 2, 1, 1, 1, 3, 5, 2, 0, 7, 6, 3, 4, 6, 1, 5, 7, 1, 2, 3, 1, 0, 0, 6, 4, 2, 9, 1, 1, 1, 2, 5, 3, 4, 2, 4, 5, 1, 1, 1, 6, 0, 3, 2, 6, 2, 5, 4, 3, 1, 1, 5, 6, 8, 0, 1, 8, 0, 9, 0, 0, 3, 0, 2, 1, 1, 4, 1, 6, 4, 1, 4, 6, 1, 1, 0, 9, 0, 0, 2, 1, 5, 7, 5, 5, 7, 1, 5, 1, 4, 1, 1, 1, 1, 1, 1, 9, 2, 0, 1, 3, 6, 2, 3, 3, 0, 2, 1, 0, 9, 3, 11, 2, 2, 6, 1, 1, 1, 3, 1, 5, 0, 2, 1, 2, 1, 6, 7, 2, 0, 9, 0, 1, 2, 1, 3, 1, 2, 6, 0, 6, 0, 0, 5, 6, 0, 6, 1, 4, 10, 3, 0, 1, 0, 2, 5, 0, 0, 4, 5, 0, 8, 1, 0, 3, 1, 0, 1, 7, 8, 7, 0, 10, 0, 7, 5, 5, 6, 7, 3, 1, 10, 0, 5, 5, 8, 5, 1, 6, 1, 5, 1, 1, 5, 5, 3, 6, 0, 0, 0, 5, 6, 8, 1, 2, 0, 0, 0, 7, 7, 2, 0, 2, 0, 4, 8, 4, 8, 7, 2, 6, 0, 9, 5, 0, 4, 2, 1, 5, 1, 0, 3, 2, 1, 2, 0, 4, 0, 0, 8, 1, 2, 6, 2, 7, 0, 0, 0, 0, 7, 1, 0, 8, 0, 3, 0, 0, 0, 0, 1, 7, 7, 6, 0, 0, 6, 0, 5, 2, 6, 11, 1, 2, 6, 8, 5, 8, 2, 1, 5, 4, 3, 2, 0, 6, 0, 5, 0, 1, 0, 9, 5, 4, 1, 4, 5, 4, 3, 2, 1, 6, 1, 1, 1, 2, 8, 0, 3, 0, 7, 0, 3, 1, 2, 1, 2, 2, 1, 2, 0, 5, 8, 3, 3, 8, 6, 4, 1, 2, 6, 5, 5, 4, 7, 2, 6, 9, 4, 6, 4, 1, 0, 6, 1, 3, 2, 8, 1, 0, 0, 9, 4, 2, 2, 0, 4, 1, 6, 7, 1, 10, 0, 0, 4, 2, 0, 10, 1, 4, 0, 4, 5, 4, 0, 2, 2, 4, 1, 8, 4, 1, 7, 1, 7, 0, 0, 5, 6, 2, 4, 5, 6, 3, 9, 0, 0, 3, 9, 8, 1, 11, 9, 2, 1, 9, 5, 4, 2, 1, 1, 7, 10, 6, 0, 0, 4, 1, 9, 0, 4, 2, 0, 4, 5, 2, 7, 1, 0, 3, 0, 4, 5, 9, 5, 0, 5, 2, 5, 1, 1, 0, 6, 6, 0, 8, 6, 2, 6, 1, 4, 6, 7, 0, 2, 0, 0, 1, 4, 1, 2, 1, 4, 1, 1, 6, 9, 6, 7, 4, 0, 3, 4, 4, 1, 6, 1, 0, 1, 2, 4, 0, 8, 10, 10, 2, 1, 8, 0, 10, 2, 1, 4, 3, 3, 5, 5, 9, 0, 6, 1, 1, 3, 1, 0, 7, 10, 0, 1, 0, 0, 1, 6, 1, 6, 2, 2, 3, 4, 5, 1, 0, 1, 5, 3, 1, 1, 2, 3, 6, 0, 0, 1, 6, 1, 0, 7, 2, 0, 3, 5, 3, 1, 6, 2, 1, 1, 5, 2, 8, 0, 9, 3, 7, 4, 2, 1, 6, 4, 0, 4, 8, 0, 0, 5, 1, 1, 6, 10, 1, 11, 0, 2, 8, 8, 10, 2, 5, 6, 1, 1, 10, 4, 3, 2, 1, 5, 8, 5, 4, 1, 6, 1, 6, 3, 0, 0, 1, 0, 6, 5, 1, 4, 5, 5, 1, 4, 6, 2, 0, 1, 0, 5, 6, 2, 4, 2, 6, 1, 2, 1, 7, 5, 3, 5, 7, 0, 11, 3, 2, 2, 3, 7, 9, 3, 0, 9, 2, 11, 3, 4, 9, 3, 0, 0, 6, 6, 3, 6, 4, 10, 1, 0, 6, 6, 3, 5, 0, 4, 0, 2, 2, 0, 6, 5, 1, 2, 2, 5, 2, 0, 5, 1, 8, 1, 0, 3, 9, 1, 1, 5, 3, 8, 0, 5, 2, 3, 5, 6, 5, 0, 1, 1, 7, 5, 3, 3, 2, 0, 0, 7, 8, 10, 0, 9, 2, 6, 4, 8, 0, 1, 3, 2, 0, 1, 1, 6, 6, 6, 1, 0, 7, 3, 1, 1, 6, 8, 4, 1, 0, 0, 7, 1, 7, 1, 1, 6, 6, 2, 1, 6, 8, 1, 2, 1, 4, 0, 2, 1, 1, 3, 2, 10, 6, 0, 1, 1, 8, 6, 7, 6, 3, 8, 0, 0, 0, 4, 0, 2, 0, 6, 8, 0, 6, 3, 5, 0, 1, 2, 5, 2, 5, 1, 9, 7, 1, 5, 10, 3, 1, 6, 2, 5, 6, 0, 0, 0, 1, 4, 2, 6, 9, 6, 2, 11, 5, 0, 4, 3, 2, 6, 2, 0, 6, 0, 1, 1, 1, 0, 2, 4, 0, 1, 0, 2, 4, 7, 9, 0, 1, 6, 1, 2, 2, 1])
#seqs=np.array([1,2,0,2,3,4,5,12,3,3,1,2,3,4,1])
test=PHMMs(delta,theta,lambdas,seqs,epsi1)
test.check()
test.Baum_Welch()

print(test.set_paramPoisson)
print(np.exp(test.log_init_trans_matrix))






	

