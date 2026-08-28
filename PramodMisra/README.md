# AI and Individual Life Insurance — ML281 Challenge Submission

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Society-of-actuaries-research-institute/ML281-AI-and-Individual-Life-Insurance-Challenge-Pramod-Misra/blob/main/notebooks/ML281_reproduce.ipynb)
[![reproduce](https://github.com/Society-of-actuaries-research-institute/ML281-AI-and-Individual-Life-Insurance-Challenge-Pramod-Misra/actions/workflows/reproduce.yml/badge.svg)](https://github.com/Society-of-actuaries-research-institute/ML281-AI-and-Individual-Life-Insurance-Challenge-Pramod-Misra/actions/workflows/reproduce.yml)
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](LICENSE)

Reproduction code for the **SOA ML281 AI and Individual Life Insurance
Challenge** submission by **Pramod Misra**, selected for publication following
review committee comment and a revised resubmission.

**Author:** Pramod Misra — Independent Researcher, Georgia Institute of
Technology, Atlanta · [pmisra30@gatech.edu](mailto:pmisra30@gatech.edu)

---

## What this is

The submission asks whether machine learning improves on a standard actuarial
mortality table for individual life insurance, using the SOA's ILEC 2012–2019
experience study. This repository holds everything needed to regenerate the
answer: the notebook, the data path, the seven published figures, and a numeric
ledger tying every reported number to the file that produced it.

The short version of the answer is uncomfortable, and is the reason the work is
worth reading: **out of time, the static VBT 2015 table forecasts better than any
of the machine-learned models tested.**

| Finding | Evidence |
|---|---|
| Out of time (fit ≤ 2017, score 2018–19), static VBT 2015 lands closest to A/E = 1.0 at **1.023**; the best ML model reaches 0.960 and the rest under-predict further | [Figure 7](figures/fig7_out_of_time.png) · [`results/out_of_time.json`](results/out_of_time.json) |
| The gradient-boosted model is **not** just relearning the table — deleting the VBT feature leaves test R² unchanged (0.775 vs 0.759) | [Figure 4](figures/fig4_ablation_offset.png) · [`results/ablation.json`](results/ablation.json) |
| Differences between GLM, neural net, random forest and XGBoost sit **inside fold-to-fold noise** — the model class is not where the leverage is | [Figure 3](figures/fig3_multimodel.png) · [`results/multimodel.json`](results/multimodel.json) |
| Formulating the problem properly as a count model with a VBT log-offset beats the squared-error model on deviance (0.151 vs 0.179) and calibrates to A/E **0.994** | [Figure 4](figures/fig4_ablation_offset.png) · [`results/poisson_offset.json`](results/poisson_offset.json) |
| Frequency and severity are **negatively** dependent (ρ = −0.366); the simulated aggregate tail load is ≈1.18–1.20× the mean | [Figure 6](figures/fig6_copula_aggregate.png) · [`results/copula.json`](results/copula.json) |

Full numbers, reported against reproduced: **[RESULTS.md](RESULTS.md)**.

> **These are aggregated exposure cells, not individual policies.** Each row is a
> cell of homogeneous risk characteristics with summed exposure and summed deaths.
> A model prediction is an expected death count for a cell, never a statement
> about a person. This framing governs how every result here should be read; see
> [`data/README.md`](data/README.md).

## Reproduce it

### Option 1 — Colab, nothing to install

Click the badge above and choose **Runtime → Run all**. The notebook downloads its
own data (about 6 MB, from a public Hugging Face dataset, no account needed),
verifies the SHA-256, and writes the figures and the results ledger. Between 3 and
10 minutes depending on the machine — the multi-model cross-validation is the slow
cell. A GPU does not help.

If you see `ImportError: cannot import name '_center'`, that is a stale-numpy
issue in a live kernel — **Runtime → Restart session**, then Run all again.

### Option 2 — Locally, in the exact reference environment

```bash
git clone https://github.com/Society-of-actuaries-research-institute/ML281-AI-and-Individual-Life-Insurance-Challenge-Pramod-Misra.git
cd ML281-AI-and-Individual-Life-Insurance-Challenge-Pramod-Misra

python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\Activate.ps1
pip install -r requirements.txt

python scripts/download_data.py    # fetch + checksum-verify the input
python -m pytest                   # data contract + published-value checks
python scripts/run_notebook.py     # full run, 3-8 min, writes figures/ and results/
python scripts/verify_results.py   # reported vs reproduced, exits non-zero on drift
```

`verify_results.py` prints one row per published value. On the reference
environment all 18 checks pass, 17 of them within ±0.0005.

### Option 3 — Fast smoke run (what CI does)

```bash
ML281_SMOKE=1 python scripts/run_notebook.py
```

Runs the whole pipeline in about 90 seconds against the small committed fixture in
`data/sample/`. Output goes to `.smoke/`, so it can never overwrite the published
figures or results. **Smoke numbers are not the published results** — the fixture
deliberately over-samples death-bearing cells so that every code path has signal.

## Layout

```
notebooks/ML281_reproduce.ipynb   the analysis, sections 1-10, Colab-ready
data/                             provenance, checksums, aggregates, CI fixture
scripts/                          download, fixture build, headless run, verify
figures/                          the seven figures, at the published dpi 170
results/                          numeric ledger (one JSON per section) + run_report.html
tests/                            data contract + published-value regression tests
docs/DATA_PROVENANCE.md           where the data came from, and the repro limits
RESULTS.md                        every reported number vs its reproduction
```

## Method, in brief

The target is `Death_Count` per exposure cell. Features are the policy-year
structure (issue age, attained age, duration), the categoricals (sex, smoker
status, plan), log exposure, and — for the "anchored" variant — the VBT expected
mortality rate. A shared 60/20/20 split and seed 42 run through everything.

Six analyses, one notebook section each: a VBT-feature ablation; a regularized
early-stopped fit with 5-fold CV and error-by-cell-size bands; a four-model
comparison under one shared CV split; Poisson count models carrying an exposure or
VBT log-offset; a Gaussian copula over cell frequency and severity feeding an
aggregate-loss simulation; and an out-of-time validation that fits on ≤ 2017 and
scores 2018–19.

Expensive results are computed once and reused, so the figure section plots stored
numbers rather than retraining.

## Reproducibility notes

- **Seeded throughout.** Seed 42 reaches both splits, every estimator,
  `KFold(shuffle=True)`, the XGBoost `seed` parameter and the copula RNG.
- **Inputs are checksum-verified** before use, by the notebook and by
  `download_data.py`. A mismatch is a hard error, not a warning.
- **Every run records its own environment** to `results/environment.json` and
  warns when resolved package versions differ from the reference set.
- **Tree ensembles and SHAP are not bit-identical across library versions or
  platforms.** Expect agreement to about three decimals. The one genuinely
  unstable number — the copula TVaR ratio — is documented as a range rather than
  as a constant.

The limits are written down plainly in
[`docs/DATA_PROVENANCE.md`](docs/DATA_PROVENANCE.md), including the one step that
is documented rather than executable here: the warehouse aggregation that turned
the 45.5M-cell ILEC table into the two public files. Everything downstream of that
point runs from the verified public data.

## Data

Derived from the **SOA Individual Life Experience Committee (ILEC) 2012–2019**
study. This repository does not redistribute the ILEC dataset; it provides two
small derived files, free of personally identifiable information, and documents
how they were built. See [`data/README.md`](data/README.md) for the schema, the
licensing position and download instructions.

## Licence and citation

Code in this repository is licensed **GPL-3.0** — see [`LICENSE`](LICENSE). The
submission report itself is published by the Society of Actuaries.

Citation metadata is in [`CITATION.cff`](CITATION.cff); GitHub renders it as
"Cite this repository" in the sidebar.
