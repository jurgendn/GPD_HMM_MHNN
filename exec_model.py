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

# ============================================================================
# Configuration & Styling
# ============================================================================
FONTSIZE_LABEL = 24
FONTSIZE_TITLE = 24
FONTSIZE_LEGEND = 18
DPI = 300
PAD_INCHES = 0.1


def configure_style() -> None:
    try:
        plt.style.use("seaborn-v0_8")
    except Exception:
        try:
            import seaborn as sns
            sns.set_theme()
        except Exception:
            plt.style.use("ggplot")


# ============================================================================
# Model Computation Helpers
# ============================================================================
def generate_graph_data(state_sequence: list[int], rate_vector: np.ndarray) -> list[float]:
    """Map hidden state sequence to rate values."""
    return [rate_vector[state] for state in state_sequence]


def generate_predict(state_sequence: list[int], rate_vector: np.ndarray) -> list[int]:
    """Generate Poisson samples from state sequence."""
    return [sc.stats.poisson(rate_vector[state]).rvs() for state in state_sequence]


def get_model_predictions(model, obs_series: np.ndarray) -> tuple[list[int], list[int], list[float]]:
    """Extract Viterbi path and predictions for a model."""
    model.ob_seqs = np.array(obs_series)
    path = model.viterbi()
    predictions = generate_predict(path, model.set_paramPoisson)
    state_means = generate_graph_data(path, model.set_paramPoisson)
    return path, predictions, state_means


# ============================================================================
# Core Plot Functions
# ============================================================================
def plot_aic_bic(AIC: list[float], BIC: list[float]) -> None:
    """Plot combined AIC/BIC, plus individual plots."""
    x_vals = list(range(1, len(AIC) + 1))
    
    plt.figure(figsize=(10, 6))
    plt.plot(x_vals, AIC, c="g", linewidth=1.5, marker="o", label="AIC", markersize=5, alpha=0.6)
    plt.plot(x_vals, BIC, c="r", linewidth=1.5, marker="p", label="BIC", markersize=5, alpha=0.6)
    plt.xlabel("Number of States", fontsize=FONTSIZE_LABEL, fontweight="bold")
    plt.ylabel("Value", fontsize=FONTSIZE_LABEL, fontweight="bold")
    plt.title("Akaike/Bayesian Information Criterion (AIC/BIC)", fontsize=FONTSIZE_TITLE, fontweight="bold")
    plt.legend(fontsize=FONTSIZE_LEGEND, loc="best", frameon=True, shadow=True)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("images/AIC-BIC.png", dpi=DPI, pad_inches=PAD_INCHES, bbox_inches="tight")
    
    plt.figure(figsize=(10, 6))
    plt.plot(x_vals, AIC, c="g", linewidth=1.5, marker="o", label="AIC", markersize=5, alpha=0.6)
    plt.xlabel("Number of States", fontsize=FONTSIZE_LABEL, fontweight="bold")
    plt.ylabel("AIC", fontsize=FONTSIZE_LABEL, fontweight="bold")
    plt.title("Akaike Information Criterion (AIC)", fontsize=FONTSIZE_TITLE, fontweight="bold")
    plt.legend(fontsize=FONTSIZE_LEGEND, loc="best", frameon=True, shadow=True)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("images/AIC.png", dpi=DPI, pad_inches=PAD_INCHES, bbox_inches="tight")
    
    plt.figure(figsize=(10, 6))
    plt.plot(x_vals, BIC, c="r", linewidth=1.5, marker="p", label="BIC", markersize=5, alpha=0.6)
    plt.xlabel("Number of States", fontsize=FONTSIZE_LABEL, fontweight="bold")
    plt.ylabel("BIC", fontsize=FONTSIZE_LABEL, fontweight="bold")
    plt.title("Bayesian Information Criterion (BIC)", fontsize=FONTSIZE_TITLE, fontweight="bold")
    plt.legend(fontsize=FONTSIZE_LEGEND, loc="best", frameon=True, shadow=True)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("images/BIC.png", dpi=DPI, pad_inches=PAD_INCHES, bbox_inches="tight")


def plot_cdll(cdll_values: list[float]) -> None:
    """Plot training convergence via complete data log-likelihood."""
    plt.figure(figsize=(10, 6))
    plt.plot(list(range(len(cdll_values))), cdll_values, linewidth=1.5, color="steelblue", label="CDLL", alpha=0.6)
    plt.xlabel("Iteration", fontsize=FONTSIZE_LABEL, fontweight="bold")
    plt.ylabel("Complete Data Log-Likelihood", fontsize=FONTSIZE_LABEL, fontweight="bold")
    plt.title("Training Progress: CDLL Over Iterations", fontsize=FONTSIZE_TITLE, fontweight="bold")
    plt.legend(fontsize=FONTSIZE_LEGEND, loc="best", frameon=True, shadow=True)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("images/CDLL.png", dpi=DPI, pad_inches=PAD_INCHES, bbox_inches="tight")


def plot_raw_data(observations: np.ndarray) -> None:
    """Plot raw observation time series."""
    plt.figure(figsize=(12, 6))
    plt.plot(list(range(len(observations))), observations, linewidth=1.5, color="darkblue", label="Observations", alpha=0.6)
    plt.xlabel("Time Index", fontsize=FONTSIZE_LABEL, fontweight="bold")
    plt.ylabel("Count Value", fontsize=FONTSIZE_LABEL, fontweight="bold")
    plt.title("Raw Observation Series", fontsize=FONTSIZE_TITLE, fontweight="bold")
    plt.legend(fontsize=FONTSIZE_LEGEND, loc="best", frameon=True, shadow=True)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("images/vsdata.png", dpi=DPI, pad_inches=PAD_INCHES, bbox_inches="tight")


# ============================================================================
# Single Model Plots
# ============================================================================
def plot_training_prediction(model, obs_series: np.ndarray, time_axis: list[int], phase: str = "train") -> None:
    """Plot observed vs predicted for a single model on a phase."""
    path, predictions, _ = get_model_predictions(model, obs_series)
    
    plt.figure(figsize=(12, 6))
    plt.plot(time_axis, obs_series, c="r", linewidth=1.5, label="Observed", alpha=0.6)
    plt.plot(time_axis, predictions, c="b", linewidth=1.5, label="Predicted", alpha=0.6)
    plt.xlabel("Time Index", fontsize=FONTSIZE_LABEL, fontweight="bold")
    plt.ylabel("Count Value", fontsize=FONTSIZE_LABEL, fontweight="bold")
    plt.title("Training Fit: Observed vs Predicted", fontsize=FONTSIZE_TITLE, fontweight="bold")
    plt.legend(fontsize=FONTSIZE_LEGEND, loc="best", frameon=True, shadow=True)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"images/visualized_{phase}.png", dpi=DPI, pad_inches=PAD_INCHES, bbox_inches="tight")


def plot_test_states(model, idx: int, obs_series: np.ndarray, phase: str = "test") -> None:
    """Plot hidden state means vs observations for a single model."""
    path, _, state_means = get_model_predictions(model, obs_series)
    
    plt.figure(figsize=(12, 6))
    plt.plot(list(range(len(state_means))), state_means, linewidth=1.5, label="State Mean", color="steelblue", alpha=0.6)
    plt.plot(list(range(len(obs_series))), obs_series, c="r", linestyle="--", linewidth=1.5, label="Observations", alpha=0.6)
    plt.xlabel("Time Index", fontsize=FONTSIZE_LABEL, fontweight="bold")
    plt.ylabel("Count Value", fontsize=FONTSIZE_LABEL, fontweight="bold")
    plt.title(f"Test States vs New Observations (Model {idx})", fontsize=FONTSIZE_TITLE, fontweight="bold")
    plt.legend(fontsize=FONTSIZE_LEGEND, loc="best", frameon=True, shadow=True)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"images/TSC_{idx}_{phase}.png", dpi=DPI, pad_inches=PAD_INCHES, bbox_inches="tight")


# ============================================================================
# Multi-Model Comparison Plots (single plot)
# ============================================================================
def plot_test_comparisons(models: list, idx_list: list | None, obs_series: np.ndarray, phase: str = "test") -> None:
    """Plot predictions from multiple models overlaid on a single plot."""
    if idx_list is None:
        idx_list = list(range(len(models)))
    
    plt.figure(figsize=(12, 6))
    for idx in idx_list:
        _, predictions, _ = get_model_predictions(models[idx], obs_series)
        plt.plot(list(range(len(predictions))), predictions, linewidth=1.5, label=f"Model {idx}", alpha=0.6)
    
    plt.plot(list(range(len(obs_series))), obs_series, c="r", linestyle="--", linewidth=1.5, label="Observations", alpha=0.6)
    plt.xlabel("Time Index", fontsize=FONTSIZE_LABEL, fontweight="bold")
    plt.ylabel("Count Value", fontsize=FONTSIZE_LABEL, fontweight="bold")
    plt.title("Test State Predictions vs New Observations", fontsize=FONTSIZE_TITLE, fontweight="bold")
    plt.legend(fontsize=FONTSIZE_LEGEND, loc="best", frameon=True, shadow=True)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"images/TS_comparisons_{phase}.png", dpi=DPI, pad_inches=PAD_INCHES, bbox_inches="tight")


# ============================================================================
# Multi-Model Comparison Plots (subplots)
# ============================================================================
def plot_multi_model_subplots(
    models: list, idx_list: list | None, obs_series: np.ndarray, plot_type: str = "predictions", phase: str = "test"
) -> None:
    """Plot predictions or hidden states from multiple models in subplots.
    
    Args:
        plot_type: "predictions" (model predictions) or "hidden_states" (state means as horizontal lines)
    """
    if idx_list is None:
        idx_list = list(range(len(models)))
    
    num_plots = len(idx_list)
    cols = max(2, (num_plots + 1) // 2)
    rows = (num_plots + cols - 1) // cols
    fig, axs = plt.subplots(rows, cols, figsize=(30, 6 * rows))
    
    for plot_idx, idx in enumerate(idx_list):
        row = plot_idx // cols
        col = plot_idx % cols
        ax = axs[row, col] if rows > 1 else axs[col]
        
        path, predictions, state_means = get_model_predictions(models[idx], obs_series)
        
        if plot_type == "predictions":
            ax.plot(list(range(len(predictions))), predictions, linewidth=1.5, label=f"Model {idx}", alpha=0.6)
        elif plot_type == "hidden_states":
            for mean in models[idx].set_paramPoisson:
                ax.hlines(mean, xmin=0, xmax=len(obs_series), colors="b", linestyles="solid", linewidth=2.0, alpha=0.9)
        
        ax.plot(list(range(len(obs_series))), obs_series, c="r", linestyle="--", linewidth=1.5, label="Observations", alpha=0.6)
        ax.set_xlabel("Time Index", fontsize=FONTSIZE_LABEL, fontweight="bold")
        ax.set_ylabel("Count Value", fontsize=FONTSIZE_LABEL, fontweight="bold")
        ax.set_title(f"Model {idx} ({plot_type.title()})", fontsize=FONTSIZE_TITLE, fontweight="bold")
        ax.legend(fontsize=FONTSIZE_LEGEND, loc="best", frameon=True, shadow=True)
        ax.grid(True, alpha=0.3)
    
    for empty_idx in range(num_plots, rows * cols):
        row = empty_idx // cols
        col = empty_idx % cols
        fig.delaxes(axs[row, col] if rows > 1 else axs[col])
    
    plt.tight_layout()
    plt.savefig(f"images/TS_comparisons_subplots_{phase}.png" if plot_type == "predictions" else f"images/hidden_state_comparisons_subplots_{phase}.png", 
                dpi=DPI, pad_inches=PAD_INCHES, bbox_inches="tight")


def plot_test_comparisons_subplots(models: list, idx_list: list | None, obs_series: np.ndarray, phase: str = "test") -> None:
    """Plot predictions from multiple models in separate subplots."""
    plot_multi_model_subplots(models, idx_list, obs_series, plot_type="predictions", phase=phase)


def plot_hidden_state_comparisons_subplots(models: list, idx_list: list | None, obs_series: np.ndarray, phase: str = "test") -> None:
    """Plot hidden state means from multiple models in separate subplots."""
    plot_multi_model_subplots(models, idx_list, obs_series, plot_type="hidden_states", phase=phase)


# ============================================================================
# Reporting Utilities
# ============================================================================
def report_dwell_times(matrix: np.ndarray) -> None:
    """Print expected dwell times (iterations in each state) from transition matrix."""
    for i in range(min(5, matrix.shape[0])):
        dwell_time = 1 / (1 - matrix[i][i])
        print(f"State {i} dwell time: {dwell_time:.3f}")


def report_model_summary(model, model_idx: int) -> None:
    """Print summary of model parameters."""
    print(f"\n{'='*60}")
    print(f"Model {model_idx} Summary")
    print(f"{'='*60}")
    print("Poisson Rates (lambdas):")
    print(np.around(model.set_paramPoisson, 2))
    print("\nTransition Matrix:")
    print(np.around(np.exp(model.log_init_trans_matrix), 4))
    print("Expected Dwell Times (iterations per state):")
    trans_matrix = np.around(np.exp(model.log_init_trans_matrix), 4)
    report_dwell_times(trans_matrix)


# ============================================================================
# CLI & Main
# ============================================================================
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

    # Load and split data
    obs_series = load_observation_series(args.data)
    obs_train, obs_test = split_observations(obs_series, train_end=args.train_end, test_start=args.test_start)

    # Train models with 1..m states
    print(f"Training models with 1..{args.m} states, {args.iter} iterations each...")
    models = run_with_m(args.m, [obs_train], iterations=args.iter)
    
    # Compute information criteria
    AIC, BIC = compute_aic_bic(models)
    CDLL = compute_cdll(models)
    
    print(f"\nAIC: {AIC}")
    print(f"BIC: {BIC}")

    # ========================================================================
    # Model Selection & Diagnostics
    # ========================================================================
    base_idx = select_base_model(models)
    print(f"\nSelected base model: {base_idx} states")
    
    # ========================================================================
    # Generate Overview Plots
    # ========================================================================
    print("Generating overview plots...")
    plot_aic_bic(AIC, BIC)
    plot_cdll(CDLL)
    plot_raw_data(obs_series)

    # ========================================================================
    # Generate Single-Model Plots (base model)
    # ========================================================================
    print(f"Generating single-model plots for model {base_idx}...")
    plot_training_prediction(models[base_idx], obs_train, list(range(len(obs_train))), phase="train")
    plot_training_prediction(models[base_idx], obs_test, list(range(len(obs_test))), phase="test")
    
    if len(obs_test) > 0:
        plot_test_states(models[base_idx], base_idx, obs_test, phase="test")
    else:
        print("Warning: test split is empty; skipping test-state plot.")

    # ========================================================================
    # Generate Multi-Model Comparison Plots
    # ========================================================================
    model_indices = [5, 6, 7, 8]  # Compare a subset of models
    print(f"Generating multi-model comparison plots for models {model_indices}...")
    
    # Single plot with all models overlaid
    plot_test_comparisons(models, model_indices, obs_series, phase="full")
    plot_test_comparisons(models, model_indices, obs_train, phase="train")
    plot_test_comparisons(models, model_indices, obs_test, phase="test")
    
    # Subplots: predictions
    plot_test_comparisons_subplots(models, model_indices, obs_series, phase="full")
    plot_test_comparisons_subplots(models, model_indices, obs_train, phase="train")
    plot_test_comparisons_subplots(models, model_indices, obs_test, phase="test")
    
    # Subplots: hidden state means
    plot_hidden_state_comparisons_subplots(models, model_indices, obs_series, phase="full")
    plot_hidden_state_comparisons_subplots(models, model_indices, obs_train, phase="train")
    plot_hidden_state_comparisons_subplots(models, model_indices, obs_test, phase="test")

    # ========================================================================
    # Report Base Model Summary
    # ========================================================================
    report_model_summary(models[base_idx], base_idx)


if __name__ == "__main__":
    main()
