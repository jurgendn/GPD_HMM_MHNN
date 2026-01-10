# Poisson Hidden Markov Models for Over-Dispersed Counts

[![Python](https://img.shields.io/badge/Python-3.9%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![NumPy](https://img.shields.io/badge/NumPy-required-013243?logo=numpy&logoColor=white)](https://numpy.org/)
[![SciPy](https://img.shields.io/badge/SciPy-required-8CAAE6?logo=scipy&logoColor=white)](https://scipy.org/)
[![Matplotlib](https://img.shields.io/badge/Matplotlib-required-11557C)](https://matplotlib.org/)

This repository accompanies the report on modeling over-dispersed count time series with Poisson Hidden Markov Models (PHMMs). It contains a minimal Python implementation used to fit traffic-count data, select the number of hidden states, and visualize regime-dependent dynamics.

## What this project does
- Implements a PHMM with Poisson emissions and EM/Baum–Welch training. The core code lives in [HMM/PHMMs_fixed.py](HMM/PHMMs_fixed.py).
- Provides a runnable script, [exec_model.py](exec_model.py), that trains models with 1..m states, compares AIC/BIC, decodes hidden regimes with Viterbi, and saves plots.
- Ships a sample traffic-count series (see [data](data)) extracted from NYC roadway sensors to illustrate overdispersion and regime switching.

## Background (from the report)
- Overdispersion: many count series satisfy $\operatorname{Var}(X) > \operatorname{E}[X]$, violating the Poisson equality $\operatorname{Var}(X)=\operatorname{E}[X]$.
- PHMM idea: combine a discrete latent Markov chain with state-specific Poisson rates. The resulting Poisson mixture naturally inflates variance and captures temporal regime changes.
- Key algorithms: Forward/Backward for likelihood, Viterbi for decoding, Baum–Welch (EM) for parameter updates. Model selection relies on AIC/BIC; dwell times come from self-transition probabilities.
- Empirical finding: 4–7 states fit the traffic data well; a 7-state model balances likelihood, information criteria, and interpretability (distinct congestion regimes).

## Mathematical models

### Poisson Hidden Markov Model (PHMM)

Let $S_t \in \{1,\dots,N\}$ be a hidden Markov chain and $Y_t \in \mathbb{N}_0$ be the observed count at time $t$.

- Initial distribution: $\pi_i = \Pr(S_1=i)$
- Transition matrix: $A_{ij} = \Pr(S_t=j\mid S_{t-1}=i)$
- Poisson emissions: $Y_t\mid (S_t=i) \sim \text{Poisson}(\lambda_i)$, so
  $$\Pr(Y_t=y\mid S_t=i)=\frac{e^{-\lambda_i}\lambda_i^y}{y!}$$

The complete parameter set is $\theta = (\pi, A, \lambda)$.

### Why PHMMs explain overdispersion

Even though each state-wise Poisson has $\operatorname{Var}(Y_t\mid S_t)=\operatorname{E}(Y_t\mid S_t)=\lambda_{S_t}$, the marginal variance includes a mixture term:

$$\operatorname{Var}(Y_t)=\operatorname{E}[\operatorname{Var}(Y_t\mid S_t)]+\operatorname{Var}(\operatorname{E}[Y_t\mid S_t])
=\operatorname{E}[\lambda_{S_t}] + \operatorname{Var}(\lambda_{S_t})$$

So whenever multiple regimes have different rates (i.e., $\operatorname{Var}(\lambda_{S_t})>0$), the marginal variance exceeds the marginal mean.

### Inference algorithms used here

The implementation in [HMM/PHMMs_fixed.py](HMM/PHMMs_fixed.py) works in log-space for numerical stability.

- Forward recursion (log):
  - $\alpha_1(j)=\log \pi_j + \log p(y_1\mid j)$
  - $\alpha_t(j)=\log p(y_t\mid j) + \log\sum_i \exp(\alpha_{t-1}(i)+\log A_{ij})$

- Backward recursion (log): analogous for $\beta_t(i)$.

- EM / Baum–Welch updates (conceptually):
  - Transition probabilities: $A_{ij} \leftarrow \frac{\sum_{t=1}^{T-1} \xi_t(i,j)}{\sum_{t=1}^{T-1} \gamma_t(i)}$
  - Poisson rates: $\lambda_i \leftarrow \frac{\sum_{t=1}^{T} \gamma_t(i)\, y_t}{\sum_{t=1}^{T} \gamma_t(i)}$
  where $\gamma_t(i)=\Pr(S_t=i\mid y_{1:T})$ and $\xi_t(i,j)=\Pr(S_t=i,S_{t+1}=j\mid y_{1:T})$.

### Model selection: AIC/BIC

This repo reports AIC/BIC across $N=1..m$ states.

- With $T$ observations and log-likelihood $\ell$, the code uses a parameter count of $k=N^2+N$ (transition matrix + Poisson rates):
  - $\text{AIC}=2k-2\ell$
  - $\text{BIC}=k\log T-2\ell$

### Regime duration (dwell time)

If $A_{ii}$ is the self-transition probability, the expected number of consecutive steps spent in state $i$ is:

$$\operatorname{E}[\text{dwell}_i] = \frac{1}{1-A_{ii}}$$

This is the quantity printed by [exec_model.py](exec_model.py) after training.

### Related count-data models (for comparison/extension)

Depending on your data and goals, you may also consider:

- Negative Binomial (NB): handles overdispersion via an extra dispersion parameter, e.g. $Y\sim\text{NB}(r,p)$ with $\operatorname{Var}(Y)=\mu+\mu^2/r$.
- NB-HMM: same latent Markov structure as PHMM but with NB emissions (often more robust when overdispersion is not mainly “regime switching”).
- Zero-inflated Poisson (ZIP): mixture for excess zeros, $Y\sim\begin{cases}0 & \text{w.p. }\psi\\ \text{Poisson}(\lambda) & \text{w.p. }1-\psi\end{cases}$.
- Poisson regression / GLM: $Y_t\sim\text{Poisson}(\lambda_t)$ with $\log \lambda_t = x_t^\top\beta$ (covariate-driven intensity).
- Markov-modulated Poisson process (MMPP): continuous-time analogue of PHMM-like regime switching.
- Self-exciting processes (e.g., Hawkes): for event clustering/contagion dynamics rather than piecewise-stationary regimes.

## Repository layout
- [exec_model.py](exec_model.py) — CLI entry point; trains PHMMs, computes metrics, plots results.
- [HMM/PHMMs_fixed.py](HMM/PHMMs_fixed.py) — primary PHMM implementation (forward/backward, Viterbi, Baum–Welch, AIC/BIC).
- [HMM/phmm.py](HMM/phmm.py) and [HMM/PHMMs.py](HMM/PHMMs.py) — earlier/experimental versions kept for reference.
- [HMM/update.py](HMM/update.py) — duplicate of fixed PHMM with verbose logging.
- data/ — traffic count CSVs and saved arrays used in the accompanying experiments.

## Getting started
1) Environment
- Python 3.9+ recommended.
- Install dependencies:
  ```bash
  pip install numpy scipy matplotlib
  ```

2) Run the demo
- Execute the script to train PHMMs with up to *m* states (default 4) for *iter* EM iterations each:
  ```bash
  python exec_model.py --data "data/traffic count.csv" --m 7 --iter 30
  ```

- You can also pass a headerless single-column series (one number per line), e.g.:
  ```bash
  python exec_model.py --data "data/covid_19_us.csv" --m 4 --iter 20
  ```

3) Outputs
- `AIC-BIC.png` — information criteria across state counts.
- `CDLL.png` — log-likelihood trace per model.
- `vsdata.png` — raw observation series.
- `visualized.png` — training fit: observed vs. regime-conditional means.
- `TSC.png` — test-segment states vs. new observations.
- Console prints: AIC/BIC arrays, dwell-time estimates (1/(1 - p_ii)), estimated Poisson rates, transition matrix.

## How the script works (high level)
- Loads the 1D traffic series `OBS_SERIES` embedded in [exec_model.py](exec_model.py).
- Splits into train/test (indices 0–179 for training, 181+ for testing).
- For each state count `n` in 1..m: random initialization → Baum–Welch for `iter` iterations → compute AIC/BIC and log-likelihood.
- Picks a base model (defaults to index 4 if available) for visualization and dwell-time reporting.
- Uses Viterbi decoding to generate regime paths and samples Poisson means for plotted predictions.

## Notes and limitations
- Initialization is random; results can vary. Increase iterations or rerun to stabilize estimates.
- Data are preloaded; replace `OBS_SERIES` or wire your own CSV loader to fit new sequences.
- The implementation is pedagogical and not optimized for very long sequences or batching.

## Extending the work
- Add covariates or non-homogeneous transitions to capture seasonality/external effects.
- Swap in Bayesian inference for uncertainty over the number of states and parameters.
- Compare against negative binomial or generalized Poisson baselines on held-out data.

## Citation
If you use this code or replicate the experiments, please cite the accompanying report: “Sử dụng Poisson Hidden Markov Model để mô hình hóa dữ liệu đếm bị phân tán.”
