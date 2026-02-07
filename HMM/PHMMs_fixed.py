import logging
import numpy as np
from scipy import stats
from scipy.special import logsumexp
from numpy import seterr

logger = logging.getLogger(__name__)

class PHMMs:
    """
    Optimized Poisson Hidden Markov Model implementation.

    Notes:
    - Internally works in log-space for transition/forward/backward recursions.
    - Emission log-probabilities are computed vectorized as a T x N matrix.
    - Baum-Welch updates are vectorized using log-sum-exp for stability.

    Constructor signature is kept compatible with existing callers:
      PHMMs(init_ditri, init_trans_matrix, set_paramPoisson, ob_seqs, epsi)
    """

    def __init__(
        self,
        init_ditri: np.ndarray,
        init_trans_matrix: np.ndarray,
        set_paramPoisson: np.ndarray,
        ob_seqs: np.ndarray,
        epsi: float,
    ) -> None:
        # avoid warnings during log of zeros
        seterr(divide='ignore')
        self.nber_states = int(len(init_ditri))
        self.log_init_ditri = np.log(np.asarray(init_ditri, dtype=float))
        self.log_init_trans_matrix = np.log(np.asarray(init_trans_matrix, dtype=float))
        self.set_paramPoisson = np.asarray(set_paramPoisson, dtype=float)
        self.epsi = float(epsi)
        self.ob_seqs = np.asarray(ob_seqs, dtype=float)
        # verbosity flag - default False to avoid noisy prints
        self.verbose = False
        seterr(divide='warn')

    def _log_emission_matrix(self) -> np.ndarray:
        """Return T x N matrix of log P(y_t | state=i)."""
        # stats.poisson.logpmf accepts broadcasting: (k, mu)
        return stats.poisson.logpmf(self.ob_seqs[:, None], self.set_paramPoisson[None, :])

    def matrix_alpha(self) -> np.ndarray:
        """Vectorized forward recursion (log-space). Returns T x N array."""
        T = len(self.ob_seqs)
        N = self.nber_states
        log_b = self._log_emission_matrix()
        alpha = np.full((T, N), -np.inf, dtype=float)
        alpha[0, :] = self.log_init_ditri + log_b[0]
        for t in range(1, T):
            # previous alpha is shape (N,), add logA to get (N, N), then logsumexp over axis=0
            alpha[t, :] = logsumexp(alpha[t - 1][:, None] + self.log_init_trans_matrix, axis=0) + log_b[t]
        return alpha

    def matrix_beta(self) -> np.ndarray:
        """Vectorized backward recursion (log-space). Returns T x N array."""
        T = len(self.ob_seqs)
        N = self.nber_states
        log_b = self._log_emission_matrix()
        beta = np.full((T, N), -np.inf, dtype=float)
        beta[-1, :] = 0.0
        for t in range(T - 2, -1, -1):
            # For each i: beta[t,i] = logsumexp_j( logA[i,j] + log_b[t+1,j] + beta[t+1,j] )
            beta[t, :] = logsumexp(self.log_init_trans_matrix + (log_b[t + 1] + beta[t + 1])[None, :], axis=1)
        return beta

    def matrix_emission(self) -> np.ndarray:
        """Return emission probabilities (not log) as T x N array."""
        return np.exp(self._log_emission_matrix())

    def check(self) -> float:
        alpha = self.matrix_alpha()
        return float(logsumexp(alpha[-1]))

    def viterbi(self) -> list[int]:
        T = len(self.ob_seqs)
        N = self.nber_states
        log_b = self._log_emission_matrix()
        delta = np.full((T, N), -np.inf, dtype=float)
        psi = np.zeros((T, N), dtype=int)
        delta[0, :] = self.log_init_ditri + log_b[0]
        for t in range(1, T):
            scores = delta[t - 1][:, None] + self.log_init_trans_matrix
            psi[t, :] = np.argmax(scores, axis=0)
            delta[t, :] = np.max(scores, axis=0) + log_b[t]
        states = np.zeros(T, dtype=int)
        states[-1] = int(np.argmax(delta[-1]))
        for t in range(T - 2, -1, -1):
            states[t] = int(psi[t + 1, states[t + 1]])
        return states.tolist()

    def Baum_Welch(self, max_iter: int = 200) -> None:
        T = len(self.ob_seqs)
        N = self.nber_states
        if T == 0:
            return

        for itere in range(max_iter):
            # E-step
            alpha = self.matrix_alpha()
            beta = self.matrix_beta()
            log_b = self._log_emission_matrix()
            loglik_before = float(logsumexp(alpha[-1]))

            # gamma in log-space then normalized
            gamma_log = alpha + beta
            loglik = float(logsumexp(gamma_log[0])) if T == 1 else float(logsumexp(alpha[-1]))
            gamma_log = gamma_log - loglik  # now log of posterior probabilities

            # xi: shape (T-1, N, N)
            if T > 1:
                xi_log = (
                    alpha[:-1][:, :, None]
                    + self.log_init_trans_matrix[None, :, :]
                    + log_b[1:][:, None, :]
                    + beta[1:][:, None, :]
                )
                # normalize by loglik
                xi_log = xi_log - loglik

                # update transition matrix: numerator = logsumexp over t of xi_log (axis=0)
                numer = logsumexp(xi_log, axis=0)
                denom = logsumexp(gamma_log[:-1], axis=0)  # shape (N,)
                new_logA = numer - denom[:, None]
            else:
                # Degenerate sequence length 1: keep transitions unchanged
                new_logA = self.log_init_trans_matrix.copy()

            # update lambdas using gamma in probability space
            gamma = np.exp(gamma_log)
            numer_lambda = (gamma * self.ob_seqs[:, None]).sum(axis=0)
            denom_lambda = gamma.sum(axis=0)
            # avoid division by zero
            with np.errstate(divide='ignore', invalid='ignore'):
                new_lambdas = np.where(denom_lambda > 0, numer_lambda / denom_lambda, self.set_paramPoisson)

            # update initial distribution
            new_log_pi = gamma_log[0]
            new_log_pi = new_log_pi - logsumexp(new_log_pi)

            # M-step assignment
            self.log_init_trans_matrix = new_logA
            self.set_paramPoisson = new_lambdas
            self.log_init_ditri = new_log_pi

            # Convergence check
            # recompute forward to get fresh log-likelihood after M-step
            alpha_new = self.matrix_alpha()
            loglik_after = float(logsumexp(alpha_new[-1]))
            if self.verbose:
                logger.info("Iter %d: loglik_before=%f, loglik_after=%f, lambdas=%s", itere, loglik_before, loglik_after, np.array2string(new_lambdas, precision=4))

            if 0 < loglik_after - loglik_before < self.epsi:
                break

    def AIC(self) -> float:
        return float(2 * (self.nber_states ** 2 + self.nber_states) - 2 * self.check())

    def BIC(self) -> float:
        return float((self.nber_states ** 2 + self.nber_states) * np.log(len(self.ob_seqs)) - 2 * self.check())
