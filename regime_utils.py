"""Shared path helpers and numerical utilities for market regime analysis."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM
from scipy.special import logsumexp
from scipy.stats import entropy
from sklearn.metrics import adjusted_rand_score, cohen_kappa_score
from sklearn.mixture import GaussianMixture

REPO_ROOT = Path(__file__).resolve().parent
DATA_DIR = REPO_ROOT / "data"
TABLES_DIR = REPO_ROOT / "tables"

MODEL_FEATURES = ["SP500_Return", "VIX", "Yield_Spread"]
RANDOM_STATE = 42
N_COMPONENTS = 3

NBER_RECESSIONS = [
    ("2001-03-01", "2001-11-30", "Dot-com"),
    ("2007-12-01", "2009-06-30", "GFC"),
    ("2020-02-01", "2020-04-30", "COVID"),
]
STRESS_LABEL = "Stressful"
CALM_LABEL = "Calm"
TRANSITIONAL_LABEL = "Transitional"
VIX_THRESHOLD = 20

# Legacy full-sample maps — integer labels permute across std variants; use
# label_map_from_mean_vix() for any fit (rolling, expanding, or full).
GMM_LABEL_MAP = {0: TRANSITIONAL_LABEL, 1: CALM_LABEL, 2: STRESS_LABEL}
HMM_LABEL_MAP = {0: STRESS_LABEL, 1: CALM_LABEL, 2: TRANSITIONAL_LABEL}


def data_path(name: str) -> Path:
    return DATA_DIR / name


def read_csv(name: str, **kwargs) -> pd.DataFrame:
    return pd.read_csv(data_path(name), **kwargs)


def write_csv(df: pd.DataFrame, name: str, **kwargs) -> None:
    kwargs.setdefault("index", False)
    df.to_csv(data_path(name), **kwargs)


def commit_table(df: pd.DataFrame, name: str, atol: float = 1e-6) -> None:
    """Numbers gate for a published table: first run commits it under tables/,
    every later run must reproduce it (within atol) or raise.

    Extends the reproduction gate (which only covers committed regime labels)
    to every number the README quotes, so a stale or hand-edited figure fails
    loudly on the next run instead of silently drifting from the pipeline.
    """
    path = TABLES_DIR / f"{name}.csv"
    path.parent.mkdir(exist_ok=True)
    if path.exists():
        ref = pd.read_csv(path, index_col=0)
        np.testing.assert_allclose(
            df.values.astype(float), ref.values.astype(float), atol=atol
        )
        print(f"{name}: matches committed")
    else:
        df.to_csv(path)
        print(f"{name}: committed")


def _load_dated_std(name: str) -> pd.DataFrame:
    df = read_csv(name, parse_dates=["Date"])
    if "Date" in df.columns:
        return df.set_index("Date")
    return df


def load_raw_features() -> pd.DataFrame:
    df = read_csv("market_features_weekly.csv", parse_dates=["Date"])
    if "Unnamed: 0" in df.columns:
        df = df.drop(columns=["Unnamed: 0"])
    if "Date" in df.columns:
        return df.set_index("Date")
    return df


def load_model_features() -> pd.DataFrame:
    return load_raw_features()[MODEL_FEATURES]


def load_full_sample_std() -> pd.DataFrame:
    """Load committed full-sample z-scores (sklearn ddof=0). Do not recompute."""
    return _load_dated_std("market_features_weekly_std.csv")


def load_rolling_std() -> pd.DataFrame:
    """Load rolling-window z-scores written by 02_standardize."""
    return _load_dated_std("market_features_weekly_std_rolling.csv")


def load_expanding_std() -> pd.DataFrame:
    """Load expanding-window z-scores written by 02_standardize."""
    return _load_dated_std("market_features_weekly_std_expanding.csv")


def rolling_zscore(
    features: pd.DataFrame, window: int = 52, min_periods: int = 52
) -> pd.DataFrame:
    """Causal z-score using a rolling window (ddof=0, matches sklearn baseline)."""
    out = pd.DataFrame(index=features.index, columns=features.columns, dtype=float)
    for col in features.columns:
        roll = features[col].rolling(window=window, min_periods=min_periods)
        out[col] = (features[col] - roll.mean()) / roll.std(ddof=0)
    return out.dropna()


def expanding_zscore(features: pd.DataFrame, min_periods: int = 52) -> pd.DataFrame:
    """Causal z-score using expanding mean/std (ddof=0, matches sklearn baseline)."""
    out = pd.DataFrame(index=features.index, columns=features.columns, dtype=float)
    for col in features.columns:
        mean = features[col].expanding(min_periods=min_periods).mean()
        std = features[col].expanding(min_periods=min_periods).std(ddof=0)
        out[col] = (features[col] - mean) / std
    return out.dropna()


def fit_gmm(X: np.ndarray) -> GaussianMixture:
    return GaussianMixture(
        n_components=N_COMPONENTS,
        covariance_type="full",
        init_params="k-means++",
        n_init=10,
        random_state=RANDOM_STATE,
    ).fit(X)


def fit_hmm(X: np.ndarray, seeds: tuple[int, ...] = tuple(range(RANDOM_STATE, RANDOM_STATE + 10))) -> GaussianHMM:
    """Multi-restart EM: fit at each seed, keep the highest-likelihood result.

    Deterministic because the seed tuple is fixed, so the reproduction gate
    still means something. Single-seed EM (the old behavior) can land on a
    mediocre local optimum — see the seed diagnostic in 03_models.ipynb.
    """
    fits = [
        GaussianHMM(
            n_components=N_COMPONENTS,
            covariance_type="full",
            n_iter=1000,
            random_state=s,
        ).fit(X)
        for s in seeds
    ]
    return max(fits, key=lambda m: m.score(X))


def viterbi_decode(hmm: GaussianHMM, X: np.ndarray) -> np.ndarray:
    """Global MAP path — uses the full sequence (look-ahead)."""
    return hmm.predict(X)


def smoothed_decode(hmm: GaussianHMM, X: np.ndarray) -> np.ndarray:
    """Smoothed MAP — P(state_t | y_1:T) via forward-backward (look-ahead)."""
    _, posteriors = hmm.score_samples(X)
    return posteriors.argmax(axis=1)


def filtered_decode(hmm: GaussianHMM, X: np.ndarray) -> np.ndarray:
    """Causal MAP — P(state_t | y_1:t) via forward filtering only."""
    framelogprob = hmm._compute_log_likelihood(X)
    log_start = np.log(hmm.startprob_ + 1e-300)
    log_trans = np.log(hmm.transmat_ + 1e-300)
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


def regime_spells(labels) -> list[tuple[object, int, int]]:
    """Contiguous spells as (label, start, end_exclusive)."""
    values = np.asarray(labels)
    spells, start = [], 0
    for i in range(1, len(values) + 1):
        if i == len(values) or values[i] != values[i - 1]:
            spells.append((values[start], start, i))
            start = i
    return spells


def regime_durations(labels) -> np.ndarray:
    values = np.asarray(labels)
    if values.size == 0:
        return np.array([], dtype=int)
    durations = np.asarray([e - s for _, s, e in regime_spells(values)], dtype=int)
    assert durations.sum() == len(values)
    return durations


def run_length_mean(labels) -> float:
    durations = regime_durations(labels)
    if durations.size == 0:
        return float("nan")
    return float(np.mean(durations))


def transition_matrix(regimes) -> pd.DataFrame:
    states = np.unique(regimes)
    mat = pd.DataFrame(0, index=states, columns=states)
    values = regimes.values if hasattr(regimes, "values") else np.asarray(regimes)
    for i in range(len(values) - 1):
        mat.loc[values[i], values[i + 1]] += 1
    return mat.div(mat.sum(axis=1), axis=0)


def transition_entropy(labels) -> pd.Series:
    """Per-state Shannon entropy (bits) of the empirical next-state distribution."""
    tm = transition_matrix(labels)
    ent = entropy(tm.values, axis=1, base=2)
    return pd.Series(ent, index=tm.index)


def cross_model_ari(a, b) -> float:
    return float(adjusted_rand_score(a, b))


def max_drawdown(log_returns: pd.Series) -> float:
    """Max drawdown of one contiguous LOG-return path."""
    wealth = np.exp(log_returns.cumsum())
    peak = wealth.cummax()
    return float(((wealth - peak) / peak).min())


def max_drawdown_by_spell(df, label_col, return_col="SP500_Return") -> pd.Series:
    """Worst peak-to-trough decline inside any single contiguous spell per regime."""
    out = {}
    for label, s, e in regime_spells(df[label_col].values):
        dd = max_drawdown(df[return_col].iloc[s:e])
        out[label] = min(out.get(label, 0.0), dd)
    return pd.Series(out)


def conditional_drawdown(df, label_col, return_col="SP500_Return") -> pd.Series:
    """Deepest market underwater level during weeks carrying each label."""
    wealth = np.exp(df[return_col].cumsum())
    dd = (wealth - wealth.cummax()) / wealth.cummax()
    return dd.groupby(df[label_col]).min()


def apply_label_map(labels, mapping: dict) -> pd.Series:
    return pd.Series(labels).map(mapping)


def label_map_from_mean_vix(
    integer_labels,
    vix: pd.Series | np.ndarray,
) -> dict[int, str]:
    """Map integer cluster/state IDs to names by ascending mean VIX within each label."""
    labels = np.asarray(integer_labels)
    vix_arr = np.asarray(vix)
    if labels.shape[0] != vix_arr.shape[0]:
        raise ValueError("integer_labels and vix must have the same length")
    mean_vix = (
        pd.DataFrame({"label": labels, "vix": vix_arr})
        .groupby("label", sort=False)["vix"]
        .mean()
        .sort_values()
    )
    names = (CALM_LABEL, TRANSITIONAL_LABEL, STRESS_LABEL)
    return {int(lbl): names[i] for i, lbl in enumerate(mean_vix.index)}


def build_regime_frame(
    raw: pd.DataFrame,
    integer_labels: np.ndarray,
    integer_col: str,
    label_map: dict | None = None,
) -> pd.DataFrame:
    """Attach integer and named regime labels aligned to raw's index.

    Names are assigned per fit from mean VIX (lowest→Calm, highest→Stressful)
    unless an explicit label_map is passed.
    """
    out = raw.copy()
    if "Unnamed: 0" in out.columns:
        out = out.drop(columns=["Unnamed: 0"])
    out[integer_col] = integer_labels
    if label_map is None:
        label_map = label_map_from_mean_vix(integer_labels, out["VIX"])
    out["Regime_label"] = apply_label_map(integer_labels, label_map).values
    return out


def load_regime_labels(name: str) -> pd.DataFrame:
    """Load committed regime CSV with Date index."""
    df = read_csv(name, parse_dates=["Date"])
    if "Unnamed: 0" in df.columns:
        df = df.drop(columns=["Unnamed: 0"])
    return df.set_index("Date")


def in_recession(date) -> bool:
    ts = pd.Timestamp(date)
    for start, end, _ in NBER_RECESSIONS:
        if pd.Timestamp(start) <= ts <= pd.Timestamp(end):
            return True
    return False


def nber_stress_rates(df: pd.DataFrame, regime_col: str) -> pd.Series:
    rec = df[df["NBER_recession"]]
    non = df[~df["NBER_recession"]]
    return pd.Series(
        {
            "Recession weeks (% Stressful)": (rec[regime_col] == STRESS_LABEL).mean(),
            "Non-recession weeks (% Stressful)": (non[regime_col] == STRESS_LABEL).mean(),
            "Recession weeks (n)": len(rec),
        }
    )


def vix_baseline_metrics(df: pd.DataFrame, regime_col: str) -> pd.Series:
    model_stress = df[regime_col] == STRESS_LABEL
    base_stress = df["VIX"] > VIX_THRESHOLD
    return pd.Series(
        {
            "Agreement with VIX>20 (%)": (model_stress == base_stress).mean(),
            "Cohen's kappa (Stress vs rest)": cohen_kappa_score(model_stress, base_stress),
            "Model Stress recall vs baseline": (model_stress & base_stress).sum()
            / base_stress.sum(),
            "Model Stress precision vs baseline": (model_stress & base_stress).sum()
            / model_stress.sum(),
        }
    )
