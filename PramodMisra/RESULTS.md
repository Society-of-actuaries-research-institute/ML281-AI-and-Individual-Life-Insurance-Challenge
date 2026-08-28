# Results

Every number the ML281 submission reports, the file it comes from, and the value a
verified reproduction produces.

**Reproduction status:** all 18 checked values reproduce within tolerance on the
reference environment — 17 of them to within ±0.0005. Verified 2026-08-24,
Python 3.11.9, seed 42, input `soa_sample.parquet`
(`sha256 9bcd09b4…93c2e9a`). Full environment record: [`results/environment.json`](results/environment.json).

Re-check it yourself:

```bash
python scripts/download_data.py
python scripts/run_notebook.py        # full run
python scripts/verify_results.py      # prints the table below, exits non-zero on drift
```

Machine-readable reference values and per-metric tolerances live in
[`tests/published_reference.json`](tests/published_reference.json); CI asserts the
committed ledger against them on every push.

---

## Population experience (full 45.5M-cell table)

Source: [`results/full_table_ae_summary.json`](results/full_table_ae_summary.json),
computed from the committed aggregates in `data/full_table_ae.json`.

| Quantity | Reported | Reproduced |
|---|---|---|
| Exposure cells in the ILEC 2012–2019 table | 45,501,036 | property of the source data (not recomputed) |
| Actual deaths | 4,552,009 | 4,552,008 |
| Expected deaths, VBT 2015 w/ MI | 4,229,015 | 4,229,014.7 |
| **Overall A/E** | **1.0764** | **1.0764** |

The one-death difference is a transcription artifact in the report; the A/E is
identical. All four groupings in `full_table_ae.json` (smoker status, sex, year,
age band) sum to the same totals — asserted in `tests/test_data_contract.py`.

## Q1 — Does the model just relearn the VBT table?

Source: [`results/ablation.json`](results/ablation.json) · Figure 4 (left)

| Model | Reported test R² | Reproduced |
|---|---|---|
| Anchored (VBT feature included) | 0.759 | 0.7591 |
| Ablated (VBT feature removed) | 0.775 | 0.7746 |

Removing the VBT feature does not reduce performance — it marginally improves it.
The model is not merely re-deriving the table. `tests/test_results_match_published.py`
asserts the direction of this finding, not just the values.

## Q2 — Generalization

Source: [`results/generalization.json`](results/generalization.json)

| Quantity | Reported | Reproduced |
|---|---|---|
| XGBoost 5-fold CV R² | 0.868 ± 0.015 | 0.8677 ± 0.0153 |
| Regularized + early-stopped test R² | — | 0.7808 (best iteration 189) |

Error rises steeply with cell size — RMSE 0.12 in the zero-death band against
14.47 in the 11+ band. The full band table is in the ledger.

## Multi-model comparison (5-fold CV)

Source: [`results/multimodel.json`](results/multimodel.json) · Figure 3

| Model | Reported CV R² | Reproduced |
|---|---|---|
| Neural Net (64-32-16) | 0.915 ± 0.011 | 0.9149 ± 0.0112 |
| GLM (Poisson) | 0.906 ± 0.012 | 0.9058 ± 0.0115 |
| Random Forest | 0.893 ± 0.020 | 0.8925 ± 0.0204 |
| XGBoost | 0.868 ± 0.015 | 0.8677 ± 0.0153 |

The spread across models is comparable to the fold-to-fold standard deviation
within any one of them. This is the revised finding: on this data the model class
is not where the leverage is. An earlier draft of the submission reported
multi-model numbers that were not reproducible; these are, and the notebook cell
that computes them is the one that produces this table.

## Q4 — Proper count formulation with an offset

Source: [`results/poisson_offset.json`](results/poisson_offset.json) · Figure 4 (right)

| Model | Test RMSE | Test Poisson deviance | Test A/E |
|---|---|---|---|
| Squared-error (as submitted) | — | 0.1786 | — |
| Poisson + exposure offset | 0.4048 | 0.1523 | 1.0196 |
| **Poisson + VBT offset** | **0.4033** (reported 0.403) | **0.1507** (reported 0.151) | **0.9936** (reported 0.994) |

Carrying the VBT expected count as a log offset gives the best-calibrated model on
the held-out split, and both count models beat the submitted squared-error model
on deviance.

## Q6 — Frequency–severity copula and aggregate loss

Source: [`results/copula.json`](results/copula.json) · Figure 6

| Quantity | Reported | Reproduced |
|---|---|---|
| Copula ρ (frequency vs severity) | −0.366 | −0.3659 |
| Cells with both a death and a claim | — | 5,059 |
| Simulations | 2,500 | 2,500 |
| Aggregate mean | — | \$1.010 B |
| VaR99 | — | \$1.160 B |
| TVaR99 | — | \$1.210 B |
| **TVaR99 / mean load** | **≈1.18×** | **1.1979×** |

> **Read the TVaR ratio as ≈1.18–1.20, not as a constant.** It is a 1-in-100 tail
> statistic from a finite simulation and it is the least stable number in the
> submission: 1.1804 / 1.1909 / 1.1979 across seeds at 2,500 simulations, and
> 1.1837 at 5,000 simulations with quantile interpolation. The report's ≈1.18×
> sits inside that band; the shipped seed-42 configuration gives 1.198. The
> tolerance in `tests/published_reference.json` is widened to 0.03 for this metric
> alone, deliberately. See [`docs/DATA_PROVENANCE.md`](docs/DATA_PROVENANCE.md).
>
> ρ is computed from the data rather than simulated, and reproduces exactly.

## Out-of-time validation (fit ≤ 2017, score 2018–19)

Source: [`results/out_of_time.json`](results/out_of_time.json) · Figure 7

| Model | Reported A/E | Reproduced | \|A/E − 1\| |
|---|---|---|---|
| **VBT 2015 (static table)** | **1.023** | **1.0231** | **0.0231** |
| Poisson VBT-offset | 0.960 | 0.9596 | 0.0404 |
| XGBoost (anchored) | 0.922 | 0.9217 | 0.0783 |
| XGBoost (no VBT) | 0.897 | 0.8973 | 0.1027 |

Train 87,020 cells, test 44,902 cells. The static actuarial table is the
best-calibrated forecaster out of time; every machine-learned model
under-predicts. This is the submission's central practitioner finding, and the
test suite asserts the ranking holds.

## Figures

The seven figures are committed in [`figures/`](figures/) and are regenerated by
section 9 of the notebook. What is committed is the output of the verified
reference run, so re-running on the reference environment leaves `git status`
clean.

### Fidelity against the figures in the SOA report

Compared pixel-by-pixel at the published raster (dpi 170) against the versions
that appear in the submission report:

| Figure | Dimensions | Byte-identical | Mean absolute pixel difference (0–255) |
|---|---|---|---|
| 1 — EDA A/E by segment | match | **yes** | 0.0000 |
| 2 — XGBoost performance | match | no | 0.51 |
| 3 — Multi-model CV | match | no | 1.11 |
| 4 — Ablation + offset | match | no | 0.14 |
| 5 — SHAP summary | match | **yes** | 0.0000 |
| 6 — Copula aggregate loss | match | no | 4.81 |
| 7 — Out-of-time A/E | match | no | 1.77 |

Figures 1 and 5 reproduce byte for byte. The other five differ imperceptibly —
the printed bar labels move in the fourth decimal, which shifts a handful of
anti-aliased pixels. Figure 6 shows the largest difference because it plots the
Monte-Carlo aggregate-loss histogram, the one genuinely simulation-dependent
output; see the TVaR note above.

> **If you change the figure DPI, every figure changes size.** The notebook pins
> `figure.dpi = savefig.dpi = 170` to match the committed PNGs. An earlier variant
> of the notebook used 150, which is why a run at that setting produces rasters
> about 12% smaller with otherwise identical content.

| Figure | File | Content |
|---|---|---|
| 1 | `fig1_eda_ae_segments.png` | A/E by smoker status, sex, attained-age band; deaths by age band (full table) |
| 2 | `fig2_xgb_performance.png` | Performance by split, feature importance, actual-vs-predicted, residuals |
| 3 | `fig3_multimodel.png` | 5-fold CV R² and RMSE across four model classes, ±1 SD |
| 4 | `fig4_ablation_offset.png` | VBT ablation (left), Poisson deviance comparison (right) |
| 5 | `fig5_shap.png` | SHAP summary for the anchored XGBoost model |
| 6 | `fig6_copula_aggregate.png` | Simulated aggregate-loss distribution with mean, VaR99, TVaR99 |
| 7 | `fig7_out_of_time.png` | Out-of-time A/E by model against the 1.0 line |

An executed copy of the full run, with all output inline, is at
[`results/run_report.html`](results/run_report.html).
