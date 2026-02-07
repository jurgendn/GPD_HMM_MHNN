"""Lightweight helper utilities for the PHMM package.

These utilities were extracted from exec_model.py to provide a concise
researcher-friendly API: loading CSV series, splits, initialization, model
construction and small helpers used in the demo.

Only functionality observed in the repository was reimplemented here — no new
behaviour introduced.
"""

import csv
from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np


def load_observation_series(csv_path: str) -> np.ndarray:
    """Load a 1D count series from a CSV file.

    Supported formats (observed in repo):
    - Headerless single-column series (one numeric value per line)
    - CSV with 'Hourly_Counts' column
    - CSV with 'Date' column and remaining hourly columns (flattened row-wise)

    Raises FileNotFoundError or ValueError on unsupported schema.
    """
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    # Fast path: headerless single-column numeric series
    with path.open("r", encoding="utf-8-sig") as f_plain:
        lines = [ln.strip() for ln in f_plain.readlines() if ln.strip() != ""]
    if lines:

        def _parse_numeric_token(token: str) -> int:
            return int(float(token))

        plain_values: List[int] = []
        plain_ok = True
        for ln in lines:
            token = ln.split(",")[0].strip()
            try:
                plain_values.append(_parse_numeric_token(token))
            except Exception:
                plain_ok = False
                break
        if plain_ok:
            return np.asarray(plain_values, dtype=int)

    # Otherwise try csv.DictReader and observed header schemas
    with path.open("r", newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError(f"CSV has no header row: {csv_path}")
        fieldnames = [name.strip() if name is not None else "" for name in reader.fieldnames]

        if "Hourly_Counts" in fieldnames:
            values: List[int] = []
            for row in reader:
                raw = row.get("Hourly_Counts", "")
                if raw is None or str(raw).strip() == "":
                    continue
                values.append(int(float(raw)))
            return np.asarray(values, dtype=int)

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
        "Unsupported CSV schema. Expected a headerless numeric series, a 'Hourly_Counts' column,"
        f" or a 'Date' column with hourly columns. Got headers: {fieldnames}",
    )


def split_observations(
    series: np.ndarray, train_end: int = 180, test_start: int = 181,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return (train, test) splits using the observed notebook offsets."""
    if len(series) == 0:
        raise ValueError("Observation series is empty")
    train_end = max(1, min(int(train_end), len(series)))
    test_start = max(0, min(int(test_start), len(series)))
    train = series[:train_end]
    test = series[test_start:]
    return train, test


def generate_init(n: int, seed: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate random (transition matrix, initial distribution, lambdas).

    Uses numpy's default_rng when seed provided for reproducibility.
    """
    rng = np.random.default_rng(seed) if seed is not None else np.random
    theta = rng.uniform(0, 1, size=(n, n))
    theta = theta / theta.sum(axis=1, keepdims=True)
    delta = rng.uniform(0, 1, size=n)
    delta = delta / delta.sum()
    lambdas = np.ones(n, dtype=float)
    return theta, delta, lambdas


def phmm(n: int, obs: List[np.ndarray], iterations: int = 50, seed: Optional[int] = None):
    """Construct and train a PHMM instance using the observed API.

    This mirrors the behaviour previously present in exec_model.phmm().
    """
    from .PHMMs_fixed import PHMMs

    theta, delta, lambdas = generate_init(n, seed=seed)
    seq = np.array(obs[0])
    model = PHMMs(delta, theta, lambdas, seq, 1e-4)
    model.Baum_Welch(iterations)
    return model


def run_with_m(
    m: int, obs: List[np.ndarray], iterations: int = 50, seed: Optional[int] = None,
) -> List:
    models = []
    for nstate in range(1, m + 1):
        models.append(phmm(nstate, obs, iterations, seed=seed))
    return models


def compute_aic_bic(model_list: List) -> Tuple[List[float], List[float]]:
    aic = [m.AIC() for m in model_list]
    bic = [m.BIC() for m in model_list]
    return aic, bic


def compute_cdll(model_list: List) -> List[float]:
    return [mdl.check() for mdl in model_list]


def select_base_model(models: List, preferred_index: int = 4) -> int:
    return min(preferred_index, len(models) - 1)


def report_dwell_times(matrix: np.ndarray) -> None:
    for i in range(min(5, matrix.shape[0])):
        print(1 / (1 - matrix[i][i]))
