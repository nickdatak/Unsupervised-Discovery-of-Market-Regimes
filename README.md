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
| 5 | [`05_external_validation.ipynb`](notebooks/05_external_validation.ipynb) | **NBER recession overlap** and **VIX > 20** baseline on look-ahead labels (Section A), decode-corrected filtered HMM (Section B), **fully causal** rolling/expanding std + filtered (Section C), and pipeline-generated regime profile + drawdown stats (Section D) |
| 6 | [`06_robustness.ipynb`](notebooks/06_robustness.ipynb) | Placeholder for walk-forward refit (not implemented) |

### Core methodological choices

- **Three features:** `SP500_Return` (weekly log return), `VIX`, `Yield_Spread` (10Y–2Y).
- **Three regimes:** integer cluster/state IDs are mapped to names **per fit** by ascending mean VIX within each label (lowest → Calm, middle → Transitional, highest → Stressful). Integer IDs permute across standardization variants; names are comparable.
- **GMM:** `GaussianMixture(n_components=3, covariance_type="full", init_params="k-means++", n_init=10, random_state=42)`. Labels each week independently via `predict`—no temporal decode step.
- **HMM:** `GaussianHMM(n_components=3, covariance_type="full", n_iter=1000)`, fit with a **10-seed restart** (`random_state` 42–51) and the highest-likelihood fit kept (`fit_hmm`). Temporal structure enters through the transition matrix and decode method.
  - *Why a restart:* a single EM run at `random_state=42` is one local optimum among several with materially different log-likelihoods. A seed-stability diagnostic in [`notebooks/03_models.ipynb`](notebooks/03_models.ipynb) sweeps seeds 42–61 on the full-sample std: seed 42 alone sits 86 nats below the best fit found (seed 48), and that better fit has ARI 0.55 against the single-seed labels — a different clustering, not a relabeling. `fit_hmm` now restarts across a fixed 10-seed tuple and keeps the best log-likelihood, so the fit is still deterministic (same seeds every run ⇒ same winner) but is a genuine best-of-10 optimum rather than whatever `random_state=42` happened to land on. This changed the committed HMM labels and every table derived from them; GMM is unaffected (`n_init=10` k-means++ already restarts internally).
- **Committed regime CSVs** (`gmm_regimes.csv`, `hmm_regimes.csv`): fit on **full-sample** standardized features; HMM labels use **global Viterbi** decode (`hmm.predict`) on the restarted fit above. These are the look-ahead baseline used in Section A of external validation.

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
| Rolling 52-week | `market_features_weekly_std_rolling.csv` | 1,513 | Yes | `(x − roll_mean_52) / roll_std_52`, `ddof=0`; first 51 weeks dropped |
| Expanding | `market_features_weekly_std_expanding.csv` | 1,513 | Yes | Expanding mean/std with `min_periods=52`, `ddof=0` |

All three variants use the same population-std estimator (`ddof=0`) — rolling and expanding used to default to pandas' `ddof=1`, a ~1% difference in z-space that's now unified with the committed full-sample baseline.

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

All numbers below are reproduced by running notebooks `03`–`05` with the committed data files. GMM uses `random_state=42`; HMM uses the 10-seed restart (42–51) described above.

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
| Calm | 57.5% | 29.7% |
| Transitional | 29.5% | 37.1% |
| Stressful | 13.0% | 33.2% |

HMM (Viterbi) assigns roughly **2.5×** the Stressful share of GMM, reflecting smoother, more persistent paths.

### Regime profile (committed labels, pipeline-generated)

Mean raw features, composition share, and two drawdown statistics per regime, produced by `regime_profile()` in [`notebooks/05_external_validation.ipynb`](notebooks/05_external_validation.ipynb) (Section D). This replaces a previously hand-typed "max drawdown" table that turned out to be miscomputed (see note below) — the whole table is now a pipeline output, checked against the committed CSVs.

- **Max DD (within spell):** worst peak-to-trough decline computed on log returns *inside* a single contiguous run of one label — resets at every label change (`max_drawdown_by_spell`).
- **Mkt DD while in regime:** deepest the whole-market wealth curve is underwater versus its all-time-to-date peak, restricted to weeks carrying that label — does not reset at label changes, so it can be large for a Calm week that follows a crash (`conditional_drawdown`).

**GMM**

| Regime | VIX | Yield_Spread | Mean weekly return | Share | Max DD (within spell) | Mkt DD while in regime |
|---|---|---|---|---|---|---|
| Calm | 14.8 | 1.13 | +0.004 | 57.5% | −7.5% | −41.2% |
| Transitional | 23.8 | 0.29 | −0.001 | 29.5% | −25.8% | −28.9% |
| Stressful | 31.8 | 2.09 | −0.001 | 13.0% | −45.6% | −56.2% |

**HMM (Viterbi)**

| Regime | VIX | Yield_Spread | Mean weekly return | Share | Max DD (within spell) | Mkt DD while in regime |
|---|---|---|---|---|---|---|
| Calm | 13.3 | 0.48 | +0.003 | 29.7% | −6.8% | −25.2% |
| Transitional | 18.8 | 1.93 | +0.002 | 37.1% | −15.6% | −45.7% |
| Stressful | 26.3 | 0.44 | −0.001 | 33.2% | −45.6% | −56.2% |

Both models isolate a high-VIX Stressful regime with the deepest within-spell drawdown of any label for either model (−45.6%, since both models' Stressful spells span the GFC). GMM's Calm regime looks tame within any single spell (−7.5%) but the market was **41.2%** underwater during weeks GMM calls Calm — those are 2003 and post-2009 recovery weeks that are calm *going forward* while still below the pre-crash peak (notebook 05 confirms the GMM-Calm underwater minimum falls on 2003-04-25, the whole-sample minimum on 2009-03-06, i.e. the post-dot-com and post-GFC bottoms).

**Note — corrected drawdown definition:** an earlier version of this table reported GMM Transitional at −78.5%, worse than GMM Stressful (−56.4%), and concluded *"GMM Transitional captures severe drawdown episodes that HMM assigns to longer Stressful runs."* That number was wrong: the original `max_drawdown` compounded returns across the label's matched rows using `.cumprod()` over simple (non-log) returns without first restricting to contiguous spells, so it built a synthetic path splicing together dozens of disjoint Transitional weeks as if they occurred back-to-back — a path that never existed on any real trading calendar. With drawdown computed correctly (contiguous spells only, and on log returns via `exp(cumsum)`), GMM Transitional's within-spell drawdown is −25.8%, not the worst of the three regimes, and the "GMM buries crashes in Transitional" reading does not survive the fix.

### Cross-model agreement (full-sample std)

| Metric | GMM vs HMM Viterbi | GMM vs HMM filtered |
|--------|--------------------|---------------------|
| Adjusted Rand Index (integer labels) | 0.348 | 0.332 |
| Raw label agreement (named) | 35.7% | — |

| Subsample | ARI (GMM vs HMM Viterbi, integer labels) |
|-----------|--------------------------|
| Pre-2005 | 0.551 |
| Post-2005 | 0.260 |

Models agree on roughly **36%** of weeks overall (up from 28.6% under the single-seed HMM fit). The 3×3 co-classification (named labels; see [`notebooks/04_decode_durations.ipynb`](notebooks/04_decode_durations.ipynb)) shows why: of GMM's 900 Calm weeks, 441 (49%) are also HMM Calm; of GMM's 203 Stressful weeks, 83 (41%) are also HMM Stressful. The larger pattern is a one-way absorption, not mutual agreement — 403 of GMM's 461 Transitional weeks (87%) are labeled HMM Stressful, consistent with HMM's wider Stressful share (33.2% vs GMM's 13.0%). Pre-2005 agreement (ARI 0.55) is much stronger than post-2005 (ARI 0.26); the two models diverge more in the second half of the sample.

### Persistence and transition dynamics (Notebook 04)

**Mean regime duration (weeks)** by standardization × decode method:

| Std variant | GMM pointwise | HMM Viterbi | HMM smoothed | HMM filtered |
|-------------|---------------|-------------|--------------|--------------|
| Full-sample | 9.0 | **55.9** | 55.9 | **41.2** |
| Rolling | 6.2 | 18.9 | 17.2 | 13.5 |
| Expanding | 7.3 | 50.4 | 50.4 | 37.8 |

Key takeaways:

- GMM durations (~6–9 weeks) are **honest**—no temporal smoothing.
- HMM Viterbi on full-sample std (~56 weeks) is **inflated by global look-ahead**. This is *worse* than the ~50-week figure from the original single-seed fit: the multi-restart best-likelihood optimum (Fix 3) turned out to be a stickier model, not a less sticky one.
- **Forward-filtered decode reduces full-sample HMM duration by about a quarter** (55.9 → 41.2 weeks) while remaining causal — a smaller correction than the roughly-half reduction reported under the original single-seed fit (50.5 → 25.6).
- Causal standardization's effect on duration is **not uniform**: rolling std still shows a large reduction from full-sample (18.9 vs. 55.9 weeks, unchanged by the restart fix — seed 42 was already rolling's best-likelihood seed), but expanding std's best-likelihood fit is nearly as persistent as full-sample (50.4 vs. 55.9 weeks), reversing the earlier reading that expanding std reliably shortens HMM duration.

**Transition entropy** (bits; per-regime Shannon entropy of empirical next-state distribution, full-sample std, states named by mean VIX before computing entropy — integer state IDs are not comparable across models, since GMM's and HMM's index-to-regime mapping differ):

| Regime | GMM | HMM Viterbi | HMM filtered |
|--------|-----|-------------|--------------|
| Calm | 0.524 | 0.100 | 0.155 |
| Transitional | 0.618 | 0.130 | 0.141 |
| Stressful | 0.643 | 0.180 | 0.233 |
| **Average** | **0.595** | **0.136** | **0.177** |

HMM Viterbi paths are the most **sticky** (lowest entropy). Filtered decode increases transition entropy toward GMM-like reactivity while preserving temporal structure.

### External validation — Section A: committed look-ahead labels (Notebook 05)

NBER recessions in sample: **Dot-com** (Mar–Nov 2001), **GFC** (Dec 2007–Jun 2009), **COVID** (Feb–Apr 2020) — **134 recession weeks** total.

**Aggregate NBER overlap** (% of weeks labeled Stressful):

| | GMM | HMM (Viterbi) |
|--|-----|---------------|
| Recession weeks | 52.2% | 55.2% |
| Non-recession weeks | 9.3% | 31.2% |

**Per-recession Stressful share:**

| Episode | GMM | HMM (Viterbi) |
|---------|-----|---------------|
| Dot-com | 30.0% | 52.5% |
| GFC | 65.9% | 53.7% |
| COVID | 33.3% | 75.0% |

HMM flags more Stressful weeks outside NBER recessions (31.2% vs 9.3%), consistent with broader, smoother stress classification — the gap widened versus the single-seed HMM (was 22.2%), since the restarted fit's Stressful regime is broader (33.2% share vs. 25.4%).

**VIX > 20 baseline** (Stressful vs all other weeks):

| Metric | GMM | HMM (Viterbi) |
|--------|-----|---------------|
| Agreement with VIX>20 | 72.9% | 79.2% |
| Cohen's κ | 0.350 | 0.551 |
| Stress recall vs VIX>20 | 31.8% | 66.1% |
| Stress precision vs VIX>20 | 95.1% | 77.1% |

GMM Stressful labels are **high precision, low recall** relative to VIX>20 (few false alarms). HMM trades precision for recall, more so than under the original single-seed fit (recall rose from 54.9% to 66.1%; precision fell from 83.7% to 77.1%).

### External validation — Section B: causal filtered decode (Notebook 05)

Refit on full-sample std; compare GMM (already causal), HMM Viterbi, and HMM **forward-filtered**.

**Decode shift (Viterbi → filtered):**

| Metric | Value |
|--------|-------|
| Weeks with any label change | 51 / 1,564 (**3.3%**) |
| Stress label agreement (Viterbi vs filtered) | **97.4%** |
| Filtered recall vs Viterbi Stress | 0.981 |
| Filtered precision vs Viterbi Stress | 0.944 |

Causal filtering changes labels on only ~3% of weeks (fewer than the 5.5% under the original single-seed fit); Stressful assignments are highly stable.

**NBER overlap with causal filtered decode:**

| | GMM (pointwise) | HMM Viterbi | HMM filtered |
|--|-----------------|-------------|--------------|
| Recession weeks (% Stressful) | 52.2% | 55.2% | **58.2%** |
| Non-recession weeks (% Stressful) | 9.3% | 31.2% | 32.3% |

**Per-recession Stressful share (causal comparison):**

| Episode | GMM | HMM Viterbi | HMM filtered |
|---------|-----|-------------|--------------|
| Dot-com | 30.0% | 52.5% | **55.0%** |
| GFC | 65.9% | 53.7% | **57.3%** |
| COVID | 33.3% | 75.0% | **75.0%** |

**VIX > 20 baseline (causal comparison):**

| Metric | GMM | HMM Viterbi | HMM filtered |
|--------|-----|-------------|--------------|
| Agreement with VIX>20 | 72.9% | 79.2% | **79.3%** |
| Cohen's κ | 0.350 | 0.551 | **0.556** |
| Stress recall vs VIX>20 | 31.8% | 66.1% | **67.9%** |
| Stress precision vs VIX>20 | 95.1% | 77.1% | **76.3%** |

**Central external-validation finding (Section B):** switching to causal filtered decode does **not** break alignment with NBER recessions or the VIX>20 rule. Overlap is preserved or slightly **improved** versus Viterbi, while persistence is honestly reduced (see duration table above). This conclusion is unchanged by the HMM rebase — filtered decode remains a small, stabilizing correction on top of whatever HMM fit it's applied to.

### External validation — Section C: fully causal variants (Notebook 05)

Section B fixes decode look-ahead but still fits on full-sample z-scores. Section C refits on **rolling** and **expanding** causal standardizations with HMM **forward-filtered** decode — every step is deployable (1,513 weeks).

**NBER overlap — HMM filtered across std variants:**

| | Full-sample (B) | Rolling | Expanding |
|--|-----------------|---------|-----------|
| Recession weeks (% Stressful) | **58.2%** | **61.2%** | **44.0%** |
| Non-recession weeks (% Stressful) | 32.3% | 29.2% | 35.4% |

Rolling std is unaffected by the HMM rebase (its best-likelihood seed was already 42) and still validates cleanly. **Expanding std no longer does:** its multi-restart HMM optimum has lower recession overlap (44.0%, down from 63.4% under the original single-seed fit) and only a modest edge over its own non-recession rate (35.4%) — a materially weaker signal than full-sample or rolling. This is a real consequence of fitting expanding std properly (best of 10 seeds) rather than accepting whatever `random_state=42` produced; it is not restored by any further change in this pipeline revision.

**Per-recession Stressful share — HMM filtered:**

| Episode | Full-sample | Rolling | Expanding |
|---------|-------------|---------|-----------|
| Dot-com | 55.0% | 52.5% | 20.0% |
| GFC | 57.3% | 59.8% | 51.2% |
| COVID | 75.0% | 100.0% | 75.0% |

**VIX > 20 baseline — HMM filtered:**

| Metric | Full-sample | Rolling | Expanding |
|--------|-------------|---------|-----------|
| Agreement with VIX>20 | 79.3% | 69.5% | 75.9% |
| Cohen's κ | 0.556 | 0.342 | 0.489 |
| Stress recall vs VIX>20 | 67.9% | 51.8% | 65.0% |
| Stress precision vs VIX>20 | 76.3% | 64.9% | 72.0% |

**Fully causal external-validation finding (revised):** rolling std still validates cleanly (recession overlap 61.2%, unchanged) and full-sample filtered decode remains strong (58.2%). Expanding std's external alignment weakened once its HMM fit was corrected to a genuine best-of-10 optimum — recession overlap fell to 44.0%, and Dot-com Stressful share fell to 20%. External validity is **not uniform across causal standardization choices**: it holds for rolling std but is materially weaker for expanding std under the corrected fitting procedure. This replaces the earlier claim that "every causal variant validates," which was an artifact of the expanding-std HMM having been fit at a single arbitrary seed.

---

## Key findings (summary)

1. **Persistence is a decode choice, not just a model choice — but the effect is smaller than first measured.** HMM Viterbi on full-sample data yields ~56-week regimes; forward-filtered decode yields ~41 weeks; GMM ~9 weeks. Filtered decode still cuts persistence, but by about a quarter rather than the roughly-half reduction reported under the original single-seed HMM fit (a multi-restart fit, see below, found a *stickier* optimum, not a less sticky one).

2. **Models disagree on most weeks, but less than first measured.** Cross-model ARI ≈ 0.33–0.35 (was 0.25–0.27); raw agreement ≈ 36% (was 29%). Agreement is concentrated in Calm (49% of GMM Calm weeks) and Stressful (41% of GMM Stressful weeks); the largest single pattern is disagreement — 87% of GMM's Transitional weeks are labeled HMM Stressful.

3. **GMM is reactive; HMM is smoother.** GMM flips regimes frequently with high transition entropy (avg. 0.595 bits vs. HMM Viterbi's 0.136). HMM Viterbi produces sticky paths and assigns more Stressful weeks overall (33.2% vs. GMM's 13.0%).

4. **External validity holds for full-sample and rolling std, but not uniformly across causal standardization choices.** Decode-corrected filtered HMM labels (Section B, full-sample std) preserve NBER and VIX>20 alignment. In Section C, rolling std (whose HMM fit is unaffected by the multi-restart correction) still validates cleanly (61.2% recession Stressful rate), but expanding std's corrected HMM fit has materially weaker external alignment (44.0% recession Stressful rate, down from 63.4% under the original single-seed fit). "Every causal variant validates" does not survive fitting the HMM properly.

5. **Standardization's effect on persistence is not uniform once the HMM fit is corrected.** Rolling std still shortens HMM duration sharply relative to full-sample (18.9 vs. 55.9 weeks); expanding std's best-likelihood fit is nearly as persistent as full-sample (50.4 vs. 55.9 weeks). Full-sample std remains the committed baseline for reproduction.

6. **A single-seed HMM fit is not a stable baseline.** The committed HMM labels originally came from one EM run at `random_state=42`. A seed-stability sweep (42–61) found fits up to 86 nats higher in log-likelihood with materially different label assignments (ARI as low as 0.16 against the seed-42 labels). `fit_hmm` now restarts across 10 fixed seeds and keeps the best log-likelihood — still fully deterministic, but no longer at the mercy of one arbitrary seed. This one change moved most of the numbers in this README; see [`notebooks/03_models.ipynb`](notebooks/03_models.ipynb) for the diagnostic.

---

## Limitations

- **In-sample fitting and labeling** on the full 1994–2023 window. No walk-forward refit (notebook 06 is a placeholder).
- **Full-sample z-scores** in committed CSVs use future mean/std (look-ahead in features, not just decode). Rolling/expanding variants address this for sensitivity analysis.
- **Three fixed components** and hand-crafted feature set; no model selection over \(K\) or alternative macro variables.
- **NBER and VIX checks are descriptive**, not formal hypothesis tests. VIX>20 is a simple rule-of-thumb baseline, not an oracle.
- **Not out-of-sample validation** of regime stability or economic utility.
- **Expanding-std HMM external validity is weak.** Once fit with a proper multi-restart (Section C), the expanding-std HMM's recession overlap (44.0%) is well below full-sample (58.2%) and rolling (61.2%). Rolling std is the more reliable fully-causal variant in this sample; expanding std's apparent validity in earlier analysis depended on an arbitrary single-seed HMM fit.

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
- HMM must use the 10-seed restart in `fit_hmm` (seeds 42–51, highest log-likelihood kept) — a single-seed fit will not reproduce the committed labels.
- Load `market_features_weekly_std.csv` for the full-sample baseline — do not recompute.
- Notebook 03 checks **ARI ≥ 0.99** against committed `gmm_regimes.csv` and `hmm_regimes.csv` before overwriting them.

---

## Citation

If you use this code or results, please cite the associated working paper. When reporting persistence or external-validation numbers, specify:

- HMM decode method (**Viterbi** vs **forward-filtered**)
- Standardization variant (**full-sample**, **rolling**, or **expanding**)

Headline results in this README use **full-sample** standardization unless noted otherwise.
