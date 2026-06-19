# Unsupervised Discovery of Market Regimes

This repository implements an in-sample comparison of **Gaussian Mixture Models (GMM)** and **Hidden Markov Models (HMM)** for discovering latent weekly market regimes—**Calm**, **Transitional**, and **Stressful**—from S&P 500 log returns, VIX, and the 10Y–2Y yield spread.

The sample is **1,564 weekly observations** from **14 January 1994** through **29 December 2023**. The goal is descriptive: how static clustering and temporal modeling differ in persistence, label agreement, and crisis-period behavior. This is **not** a deployable risk-management or trading system.

---

## What we did

The analysis is organized as a numbered notebook pipeline (`01` → `05`), with shared utilities in [`regime_utils.py`](regime_utils.py). Each stage builds on committed artifacts in [`data/`](data/).

| Step | Notebook | What it does |
|------|----------|--------------|
| 1 | [`01_features.ipynb`](notebooks/01_features.ipynb) | Resample daily S&P 500, VIX, and T10Y2Y to **weekly (W-FRI)**; compute log returns; merge into `market_features_weekly.csv` |
| 2 | [`02_standardize.ipynb`](notebooks/02_standardize.ipynb) | Load committed full-sample z-scores; compute **rolling** (52-week) and **expanding** causal standardizations |
| 3 | [`03_models.ipynb`](notebooks/03_models.ipynb) | Fit 3-component GMM (`k-means++`) and HMM on each std variant; **reproduction gate** (ARI ≥ 0.99 vs committed CSVs); write regime labels |
| 4 | [`04_decode_durations.ipynb`](notebooks/04_decode_durations.ipynb) | Compare decode methods (GMM pointwise, HMM Viterbi/smoothed/filtered) across std variants; durations, transition entropy, cross-model ARI |
| 5 | [`05_external_validation.ipynb`](notebooks/05_external_validation.ipynb) | **NBER recession overlap** and **VIX > 20** baseline on look-ahead labels (Section A), decode-corrected filtered HMM (Section B), and **fully causal** rolling/expanding std + filtered (Section C) |
| 6 | [`06_robustness.ipynb`](notebooks/06_robustness.ipynb) | Placeholder for walk-forward refit (not implemented) |

### Core methodological choices

- **Three features:** `SP500_Return` (weekly log return), `VIX`, `Yield_Spread` (10Y–2Y).
- **Three regimes:** integer cluster/state IDs are mapped to names **per fit** by ascending mean VIX within each label (lowest → Calm, middle → Transitional, highest → Stressful). Integer IDs permute across standardization variants; names are comparable.
- **GMM:** `GaussianMixture(n_components=3, covariance_type="full", init_params="k-means++", n_init=10, random_state=42)`. Labels each week independently via `predict`—no temporal decode step.
- **HMM:** `GaussianHMM(n_components=3, covariance_type="full", n_iter=1000, random_state=42)`. Temporal structure enters through the transition matrix and decode method.
- **Committed regime CSVs** (`gmm_regimes.csv`, `hmm_regimes.csv`): fit on **full-sample** standardized features; HMM labels use **global Viterbi** decode (`hmm.predict`). These are the look-ahead baseline used in Section A of external validation.

---

## Data

### Inputs

Daily series in [`data/`](data/): `sp500.csv`, `vix.csv`, `t10y2y.csv`.

### Weekly feature table

[`data/market_features_weekly.csv`](data/market_features_weekly.csv): **1,564 rows × 4 columns** (`SP500`, `SP500_Return`, `VIX`, `Yield_Spread`), no missing values after merge.

### Standardized features

| Variant | File | Rows | Causal? | Notes |
|---------|------|------|---------|-------|
| Full-sample | `market_features_weekly_std.csv` | 1,564 | No | Committed baseline; sklearn population std (ddof=0). **Do not recompute in notebooks.** |
| Rolling 52-week | `market_features_weekly_std_rolling.csv` | 1,513 | Yes | `(x − roll_mean_52) / roll_std_52`; first 51 weeks dropped |
| Expanding | `market_features_weekly_std_expanding.csv` | 1,513 | Yes | Expanding mean/std with `min_periods=52` |

Full-sample standardized features (committed):

| Feature | Mean | Std | Min | Max |
|---------|------|-----|-----|-----|
| SP500_Return | 0.00 | 1.00 | −8.37 | 4.66 |
| VIX | 0.00 | 1.00 | −1.29 | 7.29 |
| Yield_Spread | 0.00 | 1.00 | −2.24 | 2.07 |

---

## HMM decoding

HMM regime labels depend on how latent states are decoded from the fitted model. GMM has no equivalent step.

| Method | Function | Uses future data? | Role in this repo |
|--------|----------|-------------------|-------------------|
| **Viterbi** | `viterbi_decode` | Yes — global MAP path over full sample | Committed HMM CSVs; look-ahead baseline |
| **Smoothed MAP** | `smoothed_decode` | Yes — forward–backward | Duration comparison only |
| **Forward-filtered** | `filtered_decode` | **No** — causal MAP, \(P(s_t \mid y_{1:t})\) | Honest deployable HMM labels; Section B validation |

The paper's central causal correction is switching from Viterbi to **forward-filtered** decode. Viterbi inflates persistence by smoothing states using information from the entire series; filtered decode only uses data available up to each week.

---

## Results

All numbers below are reproduced by running notebooks `03`–`05` with `random_state=42` and the committed data files.

### Model fitting and reproduction (Notebook 03)

Fits on all three standardization variants. Integer label IDs differ by variant (e.g. full-sample GMM starts `[1,1,1,…]`; rolling starts `[0,0,0,…]`), but VIX-based naming makes regimes comparable.

**Reproduction gate** (refit on full-sample std vs committed CSVs):

| Model | ARI vs committed | Gate |
|-------|------------------|------|
| GMM | 1.000 | PASS |
| HMM (Viterbi) | 1.000 | PASS |

Committed label coverage: **1,564 / 1,564** non-null named labels.

### Regime composition (committed labels, full-sample std)

| Regime | GMM share | HMM (Viterbi) share |
|--------|-----------|---------------------|
| Calm | 57.5% | 30.0% |
| Transitional | 29.5% | 44.6% |
| Stressful | 13.0% | 25.4% |

HMM (Viterbi) assigns roughly **twice** the Stressful share of GMM and a much larger Transitional share, reflecting smoother, more persistent paths.

### Feature profiles by regime (committed labels)

Mean raw features within each named regime:

**GMM**

| Regime | SP500_Return | VIX | Yield_Spread |
|--------|--------------|-----|--------------|
| Calm | +0.004 | 14.8 | 1.13 |
| Transitional | −0.001 | 23.8 | 0.29 |
| Stressful | −0.001 | 31.8 | 2.09 |

**HMM (Viterbi)**

| Regime | SP500_Return | VIX | Yield_Spread |
|--------|--------------|-----|--------------|
| Calm | +0.003 | 15.9 | 0.29 |
| Transitional | +0.002 | 17.3 | 1.74 |
| Stressful | −0.001 | 28.2 | 0.56 |

Both models isolate a high-VIX Stressful regime. GMM Stressful has the highest mean VIX (~32) and elevated yield spread; HMM spreads Calm/Transitional more evenly on VIX.

**Max drawdown within regime** (cumulative return path inside each regime label):

| Regime | GMM | HMM (Viterbi) |
|--------|-----|---------------|
| Calm | −9.2% | −10.4% |
| Transitional | −78.5% | −25.7% |
| Stressful | −56.4% | −77.3% |

GMM Transitional captures severe drawdown episodes that HMM assigns to longer Stressful runs.

### Cross-model agreement (full-sample std, integer labels)

| Metric | GMM vs HMM Viterbi | GMM vs HMM filtered |
|--------|--------------------|---------------------|
| Adjusted Rand Index | 0.254 | 0.274 |
| Raw label agreement | 28.6% | — |

| Subsample | ARI (GMM vs HMM Viterbi) |
|-----------|--------------------------|
| Pre-2005 | 0.270 |
| Post-2005 | 0.240 |

Models agree on roughly **29%** of weeks overall and mainly converge during stress episodes, not on Calm vs Transitional boundaries.

### Persistence and transition dynamics (Notebook 04)

**Mean regime duration (weeks)** by standardization × decode method:

| Std variant | GMM pointwise | HMM Viterbi | HMM smoothed | HMM filtered |
|-------------|---------------|-------------|--------------|--------------|
| Full-sample | 9.0 | **50.5** | 44.7 | **25.6** |
| Rolling | 6.2 | 18.9 | 17.2 | 13.5 |
| Expanding | 7.3 | 26.5 | 25.6 | 16.4 |

Key takeaways:

- GMM durations (~6–9 weeks) are **honest**—no temporal smoothing.
- HMM Viterbi on full-sample std (~50 weeks) is **inflated by global look-ahead**; this was the headline persistence gap in early analysis.
- **Forward-filtered decode cuts full-sample HMM duration roughly in half** (50.5 → 25.6 weeks) while remaining causal.
- Causal standardization (rolling/expanding) further reduces HMM Viterbi durations (18.9 and 26.5 weeks).

**Transition entropy** (bits; per-state Shannon entropy of empirical next-state distribution, full-sample std, integer labels):

| State | GMM | HMM Viterbi | HMM filtered |
|-------|-----|-------------|--------------|
| 0 | 0.618 | 0.237 | 0.412 |
| 1 | 0.524 | 0.112 | 0.213 |
| 2 | 0.643 | 0.121 | 0.184 |
| **Average** | **0.595** | **0.157** | **0.270** |

HMM Viterbi paths are the most **sticky** (lowest entropy). Filtered decode increases transition entropy toward GMM-like reactivity while preserving temporal structure.

### External validation — Section A: committed look-ahead labels (Notebook 05)

NBER recessions in sample: **Dot-com** (Mar–Nov 2001), **GFC** (Dec 2007–Jun 2009), **COVID** (Feb–Apr 2020) — **134 recession weeks** total.

**Aggregate NBER overlap** (% of weeks labeled Stressful):

| | GMM | HMM (Viterbi) |
|--|-----|---------------|
| Recession weeks | 52.2% | 59.7% |
| Non-recession weeks | 9.3% | 22.2% |

**Per-recession Stressful share:**

| Episode | GMM | HMM (Viterbi) |
|---------|-----|---------------|
| Dot-com | 30.0% | 57.5% |
| GFC | 65.9% | 58.5% |
| COVID | 33.3% | 75.0% |

HMM flags more Stressful weeks outside NBER recessions (22.2% vs 9.3%), consistent with broader, smoother stress classification.

**VIX > 20 baseline** (Stressful vs all other weeks):

| Metric | GMM | HMM (Viterbi) |
|--------|-----|---------------|
| Agreement with VIX>20 | 72.9% | 78.3% |
| Cohen's κ | 0.350 | 0.513 |
| Stress recall vs VIX>20 | 31.8% | 54.9% |
| Stress precision vs VIX>20 | 95.1% | 83.7% |

GMM Stressful labels are **high precision, low recall** relative to VIX>20 (few false alarms). HMM trades precision for recall.

### External validation — Section B: causal filtered decode (Notebook 05)

Refit on full-sample std; compare GMM (already causal), HMM Viterbi, and HMM **forward-filtered**.

**Decode shift (Viterbi → filtered):**

| Metric | Value |
|--------|-------|
| Weeks with any label change | 86 / 1,564 (**5.5%**) |
| Stress label agreement (Viterbi vs filtered) | **95.1%** |
| Filtered recall vs Viterbi Stress | 0.945 |
| Filtered precision vs Viterbi Stress | 0.872 |

Causal filtering changes labels on only ~5% of weeks; Stressful assignments are highly stable.

**NBER overlap with causal filtered decode:**

| | GMM (pointwise) | HMM Viterbi | HMM filtered |
|--|-----------------|-------------|--------------|
| Recession weeks (% Stressful) | 52.2% | 59.7% | **61.9%** |
| Non-recession weeks (% Stressful) | 9.3% | 22.2% | 24.3% |

**Per-recession Stressful share (causal comparison):**

| Episode | GMM | HMM Viterbi | HMM filtered |
|---------|-----|-------------|--------------|
| Dot-com | 30.0% | 57.5% | **60.0%** |
| GFC | 65.9% | 58.5% | **61.0%** |
| COVID | 33.3% | 75.0% | **75.0%** |

**VIX > 20 baseline (causal comparison):**

| Metric | GMM | HMM Viterbi | HMM filtered |
|--------|-----|-------------|--------------|
| Agreement with VIX>20 | 72.9% | 78.3% | **80.1%** |
| Cohen's κ | 0.350 | 0.513 | **0.556** |
| Stress recall vs VIX>20 | 31.8% | 54.9% | **59.8%** |
| Stress precision vs VIX>20 | 95.1% | 83.7% | **84.2%** |

**Central external-validation finding (Section B):** switching to causal filtered decode does **not** break alignment with NBER recessions or the VIX>20 rule. Overlap is preserved or slightly **improved** versus Viterbi, while persistence is honestly reduced (see duration table above).

### External validation — Section C: fully causal variants (Notebook 05)

Section B fixes decode look-ahead but still fits on full-sample z-scores. Section C refits on **rolling** and **expanding** causal standardizations with HMM **forward-filtered** decode — every step is deployable (1,513 weeks).

**NBER overlap — HMM filtered across std variants:**

| | Full-sample (B) | Rolling | Expanding |
|--|-----------------|---------|-----------|
| Recession weeks (% Stressful) | **61.9%** | **61.2%** | **63.4%** |
| Non-recession weeks (% Stressful) | 24.3% | 29.2% | 33.4% |

Non-recession Stressful rates creep up as expected when both features and decode are causal; recession overlap stays strong (61–63%).

**Per-recession Stressful share — HMM filtered:**

| Episode | Full-sample | Rolling | Expanding |
|---------|-------------|---------|-----------|
| Dot-com | 60.0% | 52.5% | 47.5% |
| GFC | 61.0% | 59.8% | 69.5% |
| COVID | 75.0% | 100.0% | 75.0% |

**VIX > 20 baseline — HMM filtered:**

| Metric | Full-sample | Rolling | Expanding |
|--------|-------------|---------|-----------|
| Agreement with VIX>20 | 80.1% | 69.5% | 79.6% |
| Cohen's κ | 0.556 | 0.342 | 0.569 |
| Stress recall vs VIX>20 | 59.8% | 51.8% | 69.6% |
| Stress precision vs VIX>20 | 84.2% | 64.9% | 77.3% |

**Fully causal external-validation finding:** every causal variant validates. Recession Stressful rates stay in the 61–63% band; non-recession rates creep to 29–33% on rolling/expanding std (vs 24% on decode-corrected full-sample); Cohen's κ remains 0.34–0.57. External alignment is not an artifact of look-ahead in decode or features.

---

## Key findings (summary)

1. **Persistence is a decode choice, not just a model choice.** HMM Viterbi on full-sample data yields ~50-week regimes; forward-filtered decode yields ~26 weeks; GMM ~9 weeks. Headline HMM–GMM persistence gaps shrink substantially once look-ahead is removed.

2. **Models disagree on most weeks.** Cross-model ARI ≈ 0.25–0.27; raw agreement ≈ 29%. Agreement is driven mainly by stress co-occurrence, not by shared Calm/Transitional boundaries.

3. **GMM is reactive; HMM is smoother.** GMM flips regimes frequently with high transition entropy. HMM Viterbi produces sticky paths and assigns more Stressful weeks overall.

4. **External validity holds for every causal variant.** Decode-corrected filtered HMM labels (Section B) preserve NBER and VIX>20 alignment; fully causal rolling/expanding std + filtered (Section C) still show 61–63% recession Stressful rates and κ = 0.34–0.57, with non-recession Stressful rates rising only modestly (29–33%).

5. **Standardization matters for magnitudes.** Rolling and expanding z-scores shorten regime durations further and are the right variants for strictly causal feature construction; full-sample std remains the committed baseline for reproduction.

---

## Limitations

- **In-sample fitting and labeling** on the full 1994–2023 window. No walk-forward refit (notebook 06 is a placeholder).
- **Full-sample z-scores** in committed CSVs use future mean/std (look-ahead in features, not just decode). Rolling/expanding variants address this for sensitivity analysis.
- **Three fixed components** and hand-crafted feature set; no model selection over \(K\) or alternative macro variables.
- **NBER and VIX checks are descriptive**, not formal hypothesis tests. VIX>20 is a simple rule-of-thumb baseline, not an oracle.
- **Not out-of-sample validation** of regime stability or economic utility.

---

## Repository structure

| Path | Contents |
|------|----------|
| [`data/`](data/) | Weekly features, standardized features (3 variants), regime label CSVs |
| [`regime_utils.py`](regime_utils.py) | Paths, fits, decode methods, validation helpers |
| [`notebooks/01_features.ipynb`](notebooks/01_features.ipynb) | Weekly resample + feature merge |
| [`notebooks/02_standardize.ipynb`](notebooks/02_standardize.ipynb) | Full-sample, rolling, expanding standardization |
| [`notebooks/03_models.ipynb`](notebooks/03_models.ipynb) | GMM + HMM fit; reproduction gate |
| [`notebooks/04_decode_durations.ipynb`](notebooks/04_decode_durations.ipynb) | Std × decode duration table; transition entropy |
| [`notebooks/05_external_validation.ipynb`](notebooks/05_external_validation.ipynb) | NBER + VIX validation (look-ahead and causal) |
| [`notebooks/06_robustness.ipynb`](notebooks/06_robustness.ipynb) | Walk-forward placeholder |

---

## Setup and reproduction

```bash
pip install -r requirements.txt
```

**Dependencies:** Python 3.9+, `numpy`, `pandas`, `scikit-learn`, `scipy`, `hmmlearn`, `matplotlib`, `jupyter`.

Run notebooks **01 → 05** in order from the `notebooks/` directory. All I/O goes through [`regime_utils.py`](regime_utils.py) into [`data/`](data/).

**Reproduction requirements:**

- GMM must use `init_params="k-means++"` (enforced in `fit_gmm`).
- Load `market_features_weekly_std.csv` for the full-sample baseline — do not recompute.
- Notebook 03 checks **ARI ≥ 0.99** against committed `gmm_regimes.csv` and `hmm_regimes.csv` before overwriting them.

---

## Citation

If you use this code or results, please cite the associated working paper. When reporting persistence or external-validation numbers, specify:

- HMM decode method (**Viterbi** vs **forward-filtered**)
- Standardization variant (**full-sample**, **rolling**, or **expanding**)

Headline results in this README use **full-sample** standardization unless noted otherwise.
