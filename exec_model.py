"""Simple CLI for running PHMM demos.

This module is refactored to delegate core utilities to HMM.utils and the
PHMM implementation to HMM.PHMMs_fixed. Behavior preserved from original
notebook-generated script but simplified for researcher use.
"""

import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import scipy as sc

from HMM.utils import (
    load_observation_series,
    split_observations,
    run_with_m,
    compute_aic_bic,
    compute_cdll,
    select_base_model,
)


def generate_graph_data(state_sequence: list[int], rate_vector: np.ndarray) -> list[float]:
    return [rate_vector[state] for state in state_sequence]


def configure_style() -> None:
    try:
        plt.style.use("seaborn-v0_8")
    except Exception:
        try:
            import seaborn as sns

            sns.set_theme()
        except Exception:
            plt.style.use("ggplot")


def generate_predict(state_sequence: list[int], rate_vector: np.ndarray) -> list[int]:
    return [sc.stats.poisson(rate_vector[state]).rvs() for state in state_sequence]


def plot_aic_bic(AIC: list[float], BIC: list[float]) -> None:
    plt.figure()
    plt.plot(list(range(1, len(AIC) + 1)), AIC, c="g")
    plt.plot(list(range(1, len(BIC) + 1)), BIC, c="r")
    plt.savefig("AIC-BIC.png", dpi=300, pad_inches=0, bbox_inches="tight")


def plot_cdll(cdll_values: list[float]) -> None:
    plt.figure()
    plt.plot(list(range(len(cdll_values))), cdll_values)
    plt.savefig("CDLL.png", dpi=300, pad_inches=0, bbox_inches="tight")


def plot_raw_data(observations: np.ndarray) -> None:
    plt.figure()
    plt.plot(list(range(len(observations))), observations)
    plt.savefig("vsdata.png", dpi=300, pad_inches=0, bbox_inches="tight")


def plot_training_prediction(model, obs_series: np.ndarray, time_axis: list[int]) -> None:
    model.ob_seqs = np.array(obs_series)
    path = model.viterbi()
    expected_mean_train = generate_predict(path, model.set_paramPoisson)
    plt.figure()
    plt.plot(time_axis, obs_series, c="r")
    plt.plot(time_axis, expected_mean_train, c="b")
    plt.legend(["Rate", "Predicted"], frameon=True)
    plt.savefig("visualized.png", dpi=300, pad_inches=0, bbox_inches="tight")


def plot_test_states(model, obs_series: np.ndarray) -> None:
    model.ob_seqs = np.array(obs_series)
    path = model.viterbi()
    expected_mean_test = generate_predict(path, model.set_paramPoisson)
    plt.figure()
    plt.plot(list(range(len(expected_mean_test))), generate_graph_data(path, model.set_paramPoisson))
    plt.plot(list(range(len(obs_series))), obs_series, c="r", linestyle="--")
    plt.legend(["State", "New Observations"], frameon=True)
    plt.savefig("TSC.png", dpi=300, pad_inches=0, bbox_inches="tight")


def report_dwell_times(matrix: np.ndarray) -> None:
    for i in range(min(5, matrix.shape[0])):
        print(1 / (1 - matrix[i][i]))


def parse_args() -> argparse.Namespace:
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


def main() -> None:
    args = parse_args()
    configure_style()

    obs_series = load_observation_series(args.data)
    obs_train, obs_test = split_observations(obs_series, train_end=args.train_end, test_start=args.test_start)

    train_seq = [obs_train]
    test_seq = [obs_test]

    models = run_with_m(args.m, train_seq, iterations=args.iter)
    AIC, BIC = compute_aic_bic(models)
    CDLL = compute_cdll(models)

    print(AIC)
    print(BIC)

    time_axis = list(range(len(obs_train)))

    plot_aic_bic(AIC, BIC)
    plot_cdll(CDLL)
    plot_raw_data(obs_series)

    base_idx = select_base_model(models)
    matrix = np.around(np.exp(models[base_idx].log_init_trans_matrix), 4)

    report_dwell_times(matrix)

    plot_training_prediction(models[base_idx], obs_train, time_axis)
    if len(obs_test) == 0:
        print(
            "Warning: test split is empty (series too short for --test-start). "
            "Skipping test-state plot."
        )
    else:
        plot_test_states(models[base_idx], obs_test)

    print(np.around(models[base_idx].set_paramPoisson, 2))
    print(np.around(np.exp(models[base_idx].log_init_trans_matrix), 4))


if __name__ == "__main__":
    main()
