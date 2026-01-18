import numpy as np
import scipy as sc

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

    def __init__(self, initial_distribution, initial_transition_matrix,
                 emission_parameters):
        self.pi = initial_distribution
        self.A = list(map(lambda x: np.log(x), initial_transition_matrix))
        self.B = emission_parameters
        seterr(divide='ignore')

    def _log_emission_probs(self, mean, val):
        return sc.stats.poisson(mean).logpmf(val)

    def forward_prob(self):
        seterr(divide='ignore')
