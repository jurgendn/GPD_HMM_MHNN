# Auto-generated from exec_model.ipynb on 2026-01-10
# Execution order mirrors the notebook cells.

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import scipy as sc

# Use the fixed PHMM implementation provided in the repo
from HMM.PHMMs_fixed import PHMMs


def load_observation_series(csv_path: str) -> np.ndarray:
    """Load a 1D count series from a CSV in ./data.

    Supported formats:
    - Hourly time series table with a column named 'Hourly_Counts' (e.g. data/Sensor_1.csv)
    - Daily rows with 24 hourly columns and a 'Date' column (e.g. data/traffic count.csv)
    - Headerless single-column series (one value per line), e.g. data/covid_19_us.csv
    """

    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    # Fast path: headerless single-column numeric series (one value per line).
    # Many datasets in this repo are stored like this (e.g., data/covid_19_us.csv).
    with path.open("r", encoding="utf-8-sig") as f_plain:
        lines = [ln.strip() for ln in f_plain.readlines() if ln.strip() != ""]
    if lines:
        def _parse_numeric_token(token: str) -> int:
            return int(float(token))

        plain_values: list[int] = []
        plain_ok = True
        for ln in lines:
            # Allow optional commas/spaces; we only take the first token.
            token = ln.split(",")[0].strip()
            try:
                plain_values.append(_parse_numeric_token(token))
            except Exception:
                plain_ok = False
                break
        if plain_ok:
            return np.asarray(plain_values, dtype=int)

    with path.open("r", newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError(f"CSV has no header row: {csv_path}")

        fieldnames = [name.strip() if name is not None else "" for name in reader.fieldnames]

        # Case 1: single count column
        if "Hourly_Counts" in fieldnames:
            values: list[int] = []
            for row in reader:
                raw = row.get("Hourly_Counts", "")
                if raw is None or str(raw).strip() == "":
                    continue
                values.append(int(float(raw)))
            return np.asarray(values, dtype=int)

        # Case 2: date + multiple hour columns
        if "Date" in fieldnames:
            hour_cols = [c for c in fieldnames if c != "Date"]
            if not hour_cols:
                raise ValueError(f"No hourly columns found in CSV: {csv_path}")

            values = []
            for row in reader:
                for col in hour_cols:
                    raw = row.get(col, "")
                    if raw is None or str(raw).strip() == "":
                        continue
                    values.append(int(float(raw)))
            return np.asarray(values, dtype=int)

    raise ValueError(
        "Unsupported CSV schema. Expected a 'Hourly_Counts' column or a 'Date' column with hourly columns. "
        f"Got headers: {fieldnames}"
    )


def split_observations(series, train_end=180, test_start=181):
    """Return train/test splits while matching the original notebook offsets."""

    if len(series) == 0:
        raise ValueError("Observation series is empty")

    train_end = max(1, min(int(train_end), len(series)))
    test_start = max(0, min(int(test_start), len(series)))

    train = series[:train_end]
    test = series[test_start:]
    return train, test

def generate_init(n):
    theta = np.random.uniform(0, 1, size=(n*n))
    theta = np.reshape(theta, newshape=(n, n))
    for i in range(n):
        s = sum(theta[i])
        for j in range(n):
            theta[i][j] = theta[i][j]/s        
    delta = np.random.uniform(0, 1, size=n)
    s = sum(delta)
    for i in range(n):
        delta[i] = delta[i]/s        
    lambdas = np.ones(n)
    return theta, delta, lambdas

def phmm(n, obs, iterations=50):
    # Random parameters
    theta, delta, lambdas = generate_init(n)
    # obs is [sequence]; PHMMs expects a 1D array in the instance
    seq = np.array(obs[0])
    model = PHMMs(delta, theta, lambdas, seq, 1e-4)
    model.Baum_Welch(iterations)
    return model

def run_with_m(m, obs, iterations=50):
    model = []
    for nstate in range(1, m+1):
        model.append(phmm(nstate, obs, iterations))
    return model

def compute_aic_bic(model_list):
    aic = [m.AIC() for m in model_list]
    bic = [m.BIC() for m in model_list]
    return aic, bic


def compute_cdll(model_list):
    return [mdl.check() for mdl in model_list]

def generate_graph_data(state_sequence, rate_vector):
    expected_mean = []
    for state in state_sequence:
        expected_mean.append(rate_vector[state])
    return expected_mean

def configure_style():
    try:
        plt.style.use("seaborn-v0_8")
    except Exception:
        try:
            import seaborn as sns

            sns.set_theme()
        except Exception:
            plt.style.use("ggplot")


def generate_predict(state_sequence, rate_vector):
    expected_mean = []
    for state in state_sequence:
        expected_mean.append(sc.stats.poisson(rate_vector[state]).rvs())
    return expected_mean


def plot_aic_bic(AIC, BIC):
    plt.figure()
    plt.plot(list(range(1, len(AIC) + 1)), AIC, c="g")
    plt.plot(list(range(1, len(BIC) + 1)), BIC, c="r")
    plt.savefig("AIC-BIC.png", dpi=300, pad_inches=0, bbox_inches="tight")


def plot_cdll(cdll_values):
    plt.figure()
    plt.plot(list(range(len(cdll_values))), cdll_values)
    plt.savefig("CDLL.png", dpi=300, pad_inches=0, bbox_inches="tight")


def plot_raw_data(observations):
    plt.figure()
    plt.plot(list(range(len(observations))), observations)
    plt.savefig("vsdata.png", dpi=300, pad_inches=0, bbox_inches="tight")


def plot_training_prediction(model, obs_series, time_axis):
    model.ob_seqs = np.array(obs_series)
    path = model.viterbi()
    expected_mean_train = generate_predict(path, model.set_paramPoisson)
    plt.figure()
    plt.plot(time_axis, obs_series, c="r")
    plt.plot(time_axis, expected_mean_train, c="b")
    plt.legend(["Rate", "Predicted"], frameon=True)
    plt.savefig("visualized.png", dpi=300, pad_inches=0, bbox_inches="tight")


def plot_test_states(model, obs_series):
    model.ob_seqs = np.array(obs_series)
    path = model.viterbi()
    expected_mean_test = generate_predict(path, model.set_paramPoisson)
    plt.figure()
    plt.plot(list(range(len(expected_mean_test))), generate_graph_data(path, model.set_paramPoisson))
    plt.plot(list(range(len(obs_series))), obs_series, c="r", linestyle="--")
    plt.legend(["State", "New Observations"], frameon=True)
    plt.savefig("TSC.png", dpi=300, pad_inches=0, bbox_inches="tight")


def select_base_model(models, preferred_index=4):
    """Pick a model index safely, defaulting to preferred_index when available."""

    return min(preferred_index, len(models) - 1)


def report_dwell_times(matrix):
    for i in range(min(5, matrix.shape[0])):
        print(1 / (1 - matrix[i][i]))


def parse_args():
    parser = argparse.ArgumentParser(description="Run PHMM models and visualize results")
    parser.add_argument(
        "--data",
        type=str,
        default=str(Path("data") / "traffic count.csv"),
        help="Path to CSV containing the count series (default: data/traffic count.csv)",
    )
    parser.add_argument("--m", type=int, default=4, help="Number of models/states to train (1..m)")
    parser.add_argument("--iter", type=int, default=10, help="Baum-Welch iterations per model")
    parser.add_argument(
        "--train-end",
        type=int,
        default=180,
        help="Training split end index (default matches the original notebook)",
    )
    parser.add_argument(
        "--test-start",
        type=int,
        default=181,
        help="Test split start index (default matches the original notebook)",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    configure_style()

    obs_series = load_observation_series(args.data)
    obs_train, obs_test = split_observations(obs_series, train_end=args.train_end, test_start=args.test_start)

    train_seq = [obs_train]
    test_seq = [obs_test]

    model = run_with_m(args.m, train_seq, iterations=args.iter)
    AIC, BIC = compute_aic_bic(model)
    CDLL = compute_cdll(model)

    print(AIC)
    print(BIC)

    time_axis = list(range(len(obs_train)))

    plot_aic_bic(AIC, BIC)
    plot_cdll(CDLL)
    plot_raw_data(obs_series)

    base_idx = select_base_model(model)
    matrix = np.around(np.exp(model[base_idx].log_init_trans_matrix), 4)

    report_dwell_times(matrix)

    plot_training_prediction(model[base_idx], obs_train, time_axis)
    if len(obs_test) == 0:
        print(
            "Warning: test split is empty (series too short for --test-start). "
            "Skipping test-state plot."
        )
    else:
        plot_test_states(model[base_idx], obs_test)

    print(np.around(model[base_idx].set_paramPoisson, 2))
    print(np.around(np.exp(model[base_idx].log_init_trans_matrix), 4))


if __name__ == "__main__":
    main()
