"""Optimized Poisson Hidden Markov Model (PHMM) implementation.

This module provides the PHMMs class which implements a Poisson HMM with
log-space computations for numerical stability, vectorized forward/backward
algorithms, Viterbi decoding, and Baum-Welch parameter estimation.
"""

import logging

import numpy as np
from numpy import seterr
from scipy import stats
from scipy.special import logsumexp

logger = logging.getLogger(__name__)


class PHMMs:
    """Optimized Poisson Hidden Markov Model implementation.

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
        set_param_poisson: np.ndarray,
        ob_seqs: np.ndarray,
        epsi: float,
    ) -> None:
        """Initialize Poisson Hidden Markov Model.

        Args:
            init_ditri: Initial state distribution.
            init_trans_matrix: State transition matrix.
            set_param_poisson: Poisson rate parameters for each state.
            ob_seqs: Observation sequence.
            epsi: Convergence threshold for Baum-Welch.
        """
        # avoid warnings during log of zeros
        seterr(divide="ignore")
        self.nber_states = len(init_ditri)
        self.log_init_ditri = np.log(np.asarray(init_ditri, dtype=float))
        self.log_init_trans_matrix = np.log(np.asarray(init_trans_matrix, dtype=float))
        self.set_paramPoisson = np.asarray(set_param_poisson, dtype=float)
        self.epsi = float(epsi)
        self.ob_seqs = np.asarray(ob_seqs, dtype=float)
        # verbosity flag - default False to avoid noisy prints
        self.verbose = False
        seterr(divide="warn")

    def _log_emission_matrix(self) -> np.ndarray:
        """Return T x N matrix of log P(y_t | state=i)."""
        # stats.poisson.logpmf accepts broadcasting: (k, mu)
        return stats.poisson.logpmf(self.ob_seqs[:, None], self.set_paramPoisson[None, :])

    def matrix_alpha(self) -> np.ndarray:
        """Vectorized forward recursion (log-space). Returns T x N array."""
        t = len(self.ob_seqs)
        n = self.nber_states
        log_b = self._log_emission_matrix()
        alpha = np.full((t, n), -np.inf, dtype=float)
        alpha[0, :] = self.log_init_ditri + log_b[0]
        for time_idx in range(1, t):
            # previous alpha is shape (n,), add logA to get (n, n), then logsumexp over axis=0
            alpha[time_idx, :] = (
                logsumexp(alpha[time_idx - 1][:, None] + self.log_init_trans_matrix, axis=0)
                + log_b[time_idx]
            )
        return alpha

    def matrix_beta(self) -> np.ndarray:
        """Vectorized backward recursion (log-space). Returns T x N array."""
        t = len(self.ob_seqs)
        n = self.nber_states
        log_b = self._log_emission_matrix()
        beta = np.full((t, n), -np.inf, dtype=float)
        beta[-1, :] = 0.0
        for time_idx in range(t - 2, -1, -1):
            # For each i: beta[t,i] = logsumexp_j( logA[i,j] + log_b[t+1,j] + beta[t+1,j] )
            beta[time_idx, :] = logsumexp(
                self.log_init_trans_matrix
                + (log_b[time_idx + 1] + beta[time_idx + 1])[None, :],
                axis=1,
            )
        return beta

    def matrix_emission(self) -> np.ndarray:
        """Return emission probabilities (not log) as T x N array."""
        return np.exp(self._log_emission_matrix())

    def check(self) -> float:
        """Compute complete data log-likelihood."""
        alpha = self.matrix_alpha()
        return float(logsumexp(alpha[-1]))

    def viterbi(self) -> list[int]:
        """Compute Viterbi path via dynamic programming in log-space."""
        t = len(self.ob_seqs)
        n = self.nber_states
        log_b = self._log_emission_matrix()
        delta = np.full((t, n), -np.inf, dtype=float)
        psi = np.zeros((t, n), dtype=int)
        delta[0, :] = self.log_init_ditri + log_b[0]
        for time_idx in range(1, t):
            scores = delta[time_idx - 1][:, None] + self.log_init_trans_matrix
            psi[time_idx, :] = np.argmax(scores, axis=0)
            delta[time_idx, :] = np.max(scores, axis=0) + log_b[time_idx]
        states = np.zeros(t, dtype=int)
        states[-1] = int(np.argmax(delta[-1]))
        for time_idx in range(t - 2, -1, -1):
            states[time_idx] = int(psi[time_idx + 1, states[time_idx + 1]])
        return states.tolist()

    def baum_welch(self, max_iter: int = 200) -> None:
        """Fit parameters via Baum-Welch EM algorithm.

        Args:
            max_iter: Maximum number of iterations.
        """
        t = len(self.ob_seqs)
        if t == 0:
            return

        for iteration in range(max_iter):
            # E-step
            alpha = self.matrix_alpha()
            beta = self.matrix_beta()
            log_b = self._log_emission_matrix()
            loglik_before = float(logsumexp(alpha[-1]))

            # gamma in log-space then normalized
            gamma_log = alpha + beta
            loglik = float(logsumexp(gamma_log[0])) if t == 1 else float(logsumexp(alpha[-1]))
            gamma_log = gamma_log - loglik  # now log of posterior probabilities

            # xi: shape (t-1, n, n)
            if t > 1:
                xi_log = (
                    alpha[:-1][:, :, None]
                    + self.log_init_trans_matrix[None, :, :]
                    + log_b[1:][:, None, :]
                    + beta[1:][:, None, :]
                )
                # normalize by loglik
                xi_log = xi_log - loglik

                # update transition matrix: numerator = logsumexp over time of xi_log (axis=0)
                numer = logsumexp(xi_log, axis=0)
                denom = logsumexp(gamma_log[:-1], axis=0)  # shape (n,)
                new_loga = numer - denom[:, None]
            else:
                # Degenerate sequence length 1: keep transitions unchanged
                new_loga = self.log_init_trans_matrix.copy()

            # update lambdas using gamma in probability space
            gamma = np.exp(gamma_log)
            numer_lambda = (gamma * self.ob_seqs[:, None]).sum(axis=0)
            denom_lambda = gamma.sum(axis=0)
            # avoid division by zero
            with np.errstate(divide="ignore", invalid="ignore"):
                new_lambdas = np.where(
                    denom_lambda > 0, numer_lambda / denom_lambda, self.set_paramPoisson,
                )

            # update initial distribution
            new_log_pi = gamma_log[0]
            new_log_pi = new_log_pi - logsumexp(new_log_pi)

            # M-step assignment
            self.log_init_trans_matrix = new_loga
            self.set_paramPoisson = new_lambdas
            self.log_init_ditri = new_log_pi

            # Convergence check
            # recompute forward to get fresh log-likelihood after M-step
            alpha_new = self.matrix_alpha()
            loglik_after = float(logsumexp(alpha_new[-1]))
            if self.verbose:
                logger.info(
                    "Iter %d: loglik_before=%f, loglik_after=%f, lambdas=%s",
                    iteration,
                    loglik_before,
                    loglik_after,
                    np.array2string(new_lambdas, precision=4),
                )

            if 0 < loglik_after - loglik_before < self.epsi:
                break

    def aic(self) -> float:
        """Compute Akaike Information Criterion."""
        return float(2 * (self.nber_states**2 + self.nber_states) - 2 * self.check())

    def bic(self) -> float:
        """Compute Bayesian Information Criterion."""
        return float(
            (self.nber_states**2 + self.nber_states) * np.log(len(self.ob_seqs)) - 2 * self.check(),
        )
