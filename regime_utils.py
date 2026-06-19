"""Shared path helpers and numerical utilities for market regime analysis."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import logsumexp
from scipy.stats import entropy
from sklearn.preprocessing import StandardScaler

REPO_ROOT = Path(__file__).resolve().parent
DATA_DIR = REPO_ROOT / "data"

RANDOM_STATE = 42
N_COMPONENTS = 3


def data_path(name: str) -> Path:
    return DATA_DIR / name


def read_csv(name: str, **kwargs) -> pd.DataFrame:
    return pd.read_csv(data_path(name), **kwargs)


def write_csv(df: pd.DataFrame, name: str, **kwargs) -> None:
    df.to_csv(data_path(name), **kwargs)


def regime_durations(labels) -> np.ndarray:
    """Run lengths for each contiguous regime block."""
    values = np.asarray(labels)
    if values.size == 0:
        return np.array([], dtype=int)

    durations = []
    current = values[0]
    length = 1
    for label in values[1:]:
        if label == current:
            length += 1
        else:
            durations.append(length)
            current = label
            length = 1
    durations.append(length)
    return np.asarray(durations, dtype=int)


def mean_regime_duration(labels) -> float:
    durations = regime_durations(labels)
    if durations.size == 0:
        return float("nan")
    return float(np.mean(durations))


def full_sample_zscore(features: pd.DataFrame) -> pd.DataFrame:
    """Z-score each column using full-sample mean and standard deviation."""
    scaler = StandardScaler()
    scaled = scaler.fit_transform(features)
    return pd.DataFrame(scaled, index=features.index, columns=features.columns)


def expanding_zscore(features: pd.DataFrame, min_periods: int = 52) -> pd.DataFrame:
    """Z-score each row using only past observations (expanding window)."""
    out = pd.DataFrame(index=features.index, columns=features.columns, dtype=float)
    for col in features.columns:
        mean = features[col].expanding(min_periods=min_periods).mean()
        std = features[col].expanding(min_periods=min_periods).std()
        out[col] = (features[col] - mean) / std
    return out.dropna()


def forward_filter_labels(model, X: np.ndarray) -> np.ndarray:
    """Causal MAP decode: argmax P(state_t | y_1:t)."""
    framelogprob = model._compute_log_likelihood(X)
    log_start = np.log(model.startprob_ + 1e-300)
    log_trans = np.log(model.transmat_ + 1e-300)
    n_samples, n_components = framelogprob.shape
    log_alpha = np.empty((n_samples, n_components))
    log_alpha[0] = log_start + framelogprob[0]
    for t in range(1, n_samples):
        log_alpha[t] = framelogprob[t] + logsumexp(
            log_alpha[t - 1] + log_trans.T, axis=1
        )
    log_norm = logsumexp(log_alpha, axis=1, keepdims=True)
    filtered_probs = np.exp(log_alpha - log_norm)
    return filtered_probs.argmax(axis=1)


def transition_matrix(regimes) -> pd.DataFrame:
    states = np.unique(regimes)
    mat = pd.DataFrame(0, index=states, columns=states)
    values = regimes.values if hasattr(regimes, "values") else np.asarray(regimes)
    for i in range(len(values) - 1):
        mat.loc[values[i], values[i + 1]] += 1
    return mat.div(mat.sum(axis=1), axis=0)


def transition_entropy(tm: pd.DataFrame) -> np.ndarray:
    return entropy(tm.values, axis=1)


def max_drawdown(series: pd.Series) -> float:
    """Max drawdown on non-contiguous weeks labeled with the same regime."""
    cum = (1 + series).cumprod()
    peak = cum.cummax()
    return float(((cum - peak) / peak).min())
