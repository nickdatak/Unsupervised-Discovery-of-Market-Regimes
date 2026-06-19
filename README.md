# Unsupervised Discovery of Market Regimes

This repository implements an in-sample comparison of **Gaussian Mixture Models (GMM)** and **Hidden Markov Models (HMM)** for discovering latent weekly market regimes (Calm, Transitional, Stressful) from S&P 500 returns, VIX, and the 10Y–2Y yield spread. The sample is **1,564 weekly observations from January 1994 through December 2023**.

The goal is descriptive: how static clustering and temporal modeling differ in persistence, label agreement, and crisis-period behavior—not a deployable risk-management system.

> A fuller methodological revision (causal HMM decoding, expanding standardization, and updated paper text) is in progress; see the evaluation notebooks below for current robustness checks.

---

## Key findings (calibrated)

**Persistence differs by construction.** On the original pipeline (full-sample z-scoring, HMM labels via global Viterbi decode), mean regime run length is about **50 weeks (HMM)** vs **9 weeks (GMM)**. That gap partly reflects different information sets: GMM assigns each week independently; HMM Viterbi uses the full series. With **forward-filtered (causal) HMM decoding**, mean duration falls to about **26 weeks**—still above GMM, but roughly half the headline Viterbi gap. Details: [`notebooks/evaluation/causal_robustness_analysis.ipynb`](notebooks/evaluation/causal_robustness_analysis.ipynb).

**Models agree mainly in stress, not overall.** Adjusted Rand Index (GMM vs HMM) ≈ **0.25**; raw label agreement ≈ **29%**. Stress weeks are more identifiable than Calm or Transitional (see comparison and external-validation notebooks).

**Crisis labeling is model-dependent.** HMM assigns most of the 2008 GFC window (Sep 2008–Mar 2009) to Stressful (**28/30 weeks**). GMM labels almost that entire window as Transitional (**1/30** Stressful). Both models largely flag 2020 COVID stress. HMM also labels much of 2022–23 (inverted curve, slower grind) as Stressful; GMM does not as uniformly.

**Methodological trade-off.** GMM reacts week-to-week; HMM paths are smoother and more persistent, especially under Viterbi. “Stress” is not the same economic object across models (e.g., yield-spread patterns differ by regime). External checks (NBER recession overlap, VIX > 20 baseline) are in [`external_validation.ipynb`](notebooks/evaluation/external_validation.ipynb).

**Scope limits (stated upfront).** Features are z-scored with full-sample statistics (look-ahead for any real-time use). There is no walk-forward refit; pre/post-2005 ARI splits (~0.27 / ~0.24) measure in-sample cross-model agreement only. Regime validation uses the same inputs as clustering plus basic drawdown summaries—not a full out-of-sample test.

---

## Repository structure

| Path | Contents |
|------|----------|
| [`data/`](data/) | Weekly features, standardized features, regime label CSVs |
| [`figures/`](figures/) | Exploration, model, and comparison plots |
| [`regime_utils.py`](regime_utils.py) | Shared path helpers and numerical utilities |
| [`notebooks/features/`](notebooks/features/) | Data assembly and standardization |
| [`notebooks/models/`](notebooks/models/) | GMM and HMM fitting |
| [`notebooks/evaluation/`](notebooks/evaluation/) | Comparison metrics, causal robustness, external validation |

Primary analysis notebooks:

- [`regime_comparison_analysis.ipynb`](notebooks/evaluation/regime_comparison_analysis.ipynb) — durations, transitions, ARI, drawdowns
- [`causal_robustness_analysis.ipynb`](notebooks/evaluation/causal_robustness_analysis.ipynb) — filtered HMM vs Viterbi; expanding z-score
- [`external_validation.ipynb`](notebooks/evaluation/external_validation.ipynb) — NBER recessions, VIX-threshold baseline

---

## Setup

```bash
pip install -r requirements.txt
```

All notebooks read from and write to [`data/`](data/) via [`regime_utils.py`](regime_utils.py).

---

## Citation

If you use this code or results, please cite the associated working paper and note that headline persistence numbers depend on HMM decoding choice (Viterbi vs filtered).
