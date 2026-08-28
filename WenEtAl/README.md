# ML281-AI-and-Individual-Life-Insurance-Challenge-Wen-et-al
AI and Life Insurance Challenge - Submission from Wen, et al.

# AI-Assisted Actuarial Intelligence Framework for Mortality Analysis
**SOA AI & Individual Life Insurance Challenge — Code Submission**
*Cathay Life Insurance Co., Ltd. | January 30, 2026*

---

## Overview

This repository contains the full source code for the framework described in:

> *Designing Trustworthy AI for Actuarial Mortality Experience Analysis: A Governed, Explainable, and Scalable Framework*

The system implements an end-to-end actuarial workflow that integrates LightGBM mortality modeling, SHAP-based explainability, deterministic segmentation, and a governed LLM reporting layer — all orchestrated under formal actuarial governance (ASOP 23/25/41/56).

---

## Prerequisites

**Python 3.9+**

```bash
pip install -r requirements.txt
```

The full pinned dependency list is in `requirements.txt` (versions match the original development environment).

**Gemini API key** — Create a `.env` file in the project root (a template is provided in `.env.example`):
```
GOOGLE_API_KEY=your_key_here
```

---

## Data Setup (Not Included — Too Large)

The raw dataset is the **SOA Individual Life Experience Committee (ILEC) dataset (2012–2019)**, available from the SOA Research Institute:
- Source: https://www.soa.org/resources/experience-studies/2021/ilec-data/
- File expected at: `code/data/soa_data_2012_19_20240429.txt` (~12 GB)
- Data dictionary: available from the same SOA download page

Once the raw data is in place, run the data pipeline to generate the cleaned parquet files used by the rest of the system:

```bash
python code/data_pipeline/build_parquet.py
```

This produces:
- `data/ilec_cleaned.parquet` — cleaned, feature-engineered dataset
- `data/ilec_with_predictions.parquet` — dataset with model predictions appended (generated after model training)

---

## Model Training

The **primary trained model is included** in this package (`models/lgbm_mortality_offset_poisson.txt`, ~4.9 MB), together with `models/label_encoders.pkl` and `models/year_factors_offset.csv` — so the report pipeline below runs without any retraining.

Only the 7 expanding-window backtest variants (`models/lgbm_cutoff_2012.txt` … `lgbm_cutoff_2018.txt`) are not included due to size constraints; they are needed only for backtest reproduction.

To regenerate all model files from the raw data, run the core modeling notebook:

```bash
jupyter notebook code/lgbm_mortality_offset.ipynb
```

This trains the two-stage LightGBM Poisson model and saves:
- `models/lgbm_mortality_offset_poisson.txt` — primary model (full 2012–2019 training)
- `models/lgbm_cutoff_<year>.txt` — one model per cutoff year (expanding-window backtesting)
- `models/label_encoders.pkl` — categorical encoders (already included)
- `models/year_factors_offset.csv` — Stage 2 year adjustment factors (already included)

---

## Generating the Knowledge Base

The `knowledge_base/` directory is **pre-populated** with all outputs used in the paper (calibration stats, SHAP rankings, segment definitions, model contract, etc.). No regeneration is needed to run the report pipeline.

If you wish to regenerate it from scratch (requires the parquet data and trained model), run the notebooks in order:

```bash
jupyter notebook code/knowledge_generators/01_generate_contracts.ipynb
jupyter notebook code/knowledge_generators/02_generate_eda.ipynb
jupyter notebook code/knowledge_generators/03_generate_calibration.ipynb
jupyter notebook code/knowledge_generators/04_segments_semantic.ipynb
jupyter notebook code/knowledge_generators/05_generate_shap.ipynb
jupyter notebook code/knowledge_generators/06_generate_external.ipynb
jupyter notebook code/knowledge_generators/07_model_validation.ipynb
```

---

## Running the Report Pipeline

The unified entry point is `code/report_generator/run_pipeline.py`.

**Policy Interpretation Report (S3)**
```bash
python code/report_generator/run_pipeline.py --report-type policy_report --use-test-data
```

**Spotlight / Exception Report (S2)**
```bash
python code/report_generator/run_pipeline.py --report-type spotlight_report --top-n-spotlight 5
```

**Experience Trend Pack (S1)**
```bash
python code/report_generator/run_pipeline.py --report-type trend_pack
```

**Validation Diagnostic Report**
```bash
python code/report_generator/run_pipeline.py --report-type validation_diagnostic
```

All outputs are written to `data/reports/`.

> **Note:** The `--use-test-data` flag runs the policy report against a hardcoded example policy
> (the same policy used in Supplementary S3 of the paper). Without this flag, the pipeline
> expects a running data source.

---

## File Structure

```
soa_framework_release/
├── README.md                          # This file
├── ARCHITECTURE.md                    # Detailed system architecture and module reference
├── requirements.txt                   # Pinned Python dependencies
├── .env.example                       # API key template (copy to .env)
│
├── code/
│   ├── lgbm_mortality_offset.ipynb    # Core model training (Stage 1 + Stage 2)
│   ├── lgbm_oot_validation.ipynb      # Out-of-time validation
│   ├── data_cleaning_lgbm.ipynb       # Data cleaning and feature engineering
│   ├── analyze_auc_roc.py             # AUC/ROC analysis utilities
│   ├── evaluate_expanding_window.py   # Expanding-window backtesting evaluator
│   ├── run_expanding_window.py        # Entry point for backtesting runs
│   ├── verify_exposure.py             # Exposure sanity check
│   │
│   ├── data_pipeline/
│   │   └── build_parquet.py           # Raw data to cleaned parquet
│   │
│   ├── report_generator/              # Core LLM reporting pipeline
│   │   ├── run_pipeline.py            # Unified CLI entry point
│   │   ├── pipeline.py                # Orchestration logic
│   │   ├── router.py                  # Report type routing
│   │   ├── validation.py              # Input and LLM output validation
│   │   ├── prompt_builder.py          # System prompt with governance constraints
│   │   ├── llm.py                     # Gemini API calls and two-pass self-review
│   │   ├── knowledge.py               # Knowledge base loader
│   │   ├── predictor.py               # LightGBM inference and SHAP
│   │   ├── matcher.py                 # Segment rule matching
│   │   ├── schemas.py                 # Pydantic output schemas
│   │   ├── renderer.py                # Markdown/HTML/JSON rendering
│   │   ├── policy_report.py           # S3 data builder
│   │   ├── spotlight_report.py        # S2 data builder
│   │   ├── diagnostic_report.py       # S1 data builder
│   │   ├── run_control.py             # Run control and logging
│   │   ├── config.py                  # Configuration
│   │   ├── demo_structured.py         # Demo: policy report
│   │   └── demo_trend_pack.py         # Demo: trend pack
│   │
│   ├── knowledge_generators/          # Notebooks to populate knowledge_base/
│   │   ├── 01_generate_contracts.ipynb
│   │   ├── 02_generate_eda.ipynb
│   │   ├── 03_generate_calibration.ipynb
│   │   ├── 04_segments_semantic.ipynb
│   │   ├── 05_generate_shap.ipynb
│   │   ├── 06_generate_external.ipynb
│   │   ├── 07_model_validation.ipynb
│   │   ├── generate_evidence_charts.py
│   │   ├── refine_segments.py
│   │   └── refine_segments_with_stability.py
│   │
│   ├── chart_generators/              # Chart generation scripts (for S1 figures)
│   │   ├── generate_charts_tested.py
│   │   ├── generate_backtest_chart.py
│   │   └── snippet_*.py               # Individual chart snippet scripts
│   │
│   ├── tests/                         # Governance validation tests (Appendix A and B)
│   │   ├── test_constraint_checker.py
│   │   ├── test_self_review_injection.py
│   │   └── _run_baseline.py
│   │
│   └── tools/
│       └── md_to_report.py            # Markdown to HTML/PDF converter
│
├── knowledge_base/                    # Pre-populated structured knowledge stores
│   ├── contracts/                     # Model contract (JSON) — governance and ASOP alignment
│   ├── calibration/                   # Overall and yearly A/E ratios (JSON)
│   ├── eda/                           # Numerical/categorical summaries (JSON)
│   ├── shap/                          # Global SHAP rankings and example plots (JSON, PNG)
│   ├── segments/                      # Coverage tree and spotlight segment definitions
│   │   └── spotlight/                 # Individual spotlight segment files (SPOT_*.json)
│   ├── external/                      # External context store (2018 anomaly, events)
│   └── templates/                     # Actuarial report narrative template
│
├── models/
│   ├── lgbm_mortality_offset_poisson.txt  # Primary trained model (included)
│   ├── year_factors_offset.csv        # Stage 2 year adjustment factors (pre-computed)
│   └── label_encoders.pkl             # Categorical label encoders (pre-computed)
│   # lgbm_cutoff_*.txt backtest variants not included (7 files, backtesting only)
│   # Regenerate by running code/lgbm_mortality_offset.ipynb
│
└── appendix/                          # Committee response appendix artifacts
    ├── AppendixA_constraint_checker_validation.py   # LLM constraint validation code
    ├── AppendixA_llm_self_review.py                 # Self-review injection logic
    ├── AppendixB_injection_test_script.py            # Injection robustness test runner
    ├── AppendixB_constraint_checker_test.py          # Constraint checker unit tests
    ├── AppendixB_Injection_Test_Results.json         # Test results (68 injection tests)
    ├── AppendixC_S1_*.json                           # S1 run logs, constraint checks, diffs
    └── AppendixF_S2_*.json / AppendixF_S3_*.json    # S2/S3 run logs and governance diffs
```

---

## Key Design Decisions

| Decision | Rationale |
|---|---|
| Two-stage LightGBM (Stage 1 + year factors) | Mirrors actuarial practice of base table + improvement scale |
| SHAP for explainability | Satisfies Efficiency, Symmetry, Linearity, Dummy axioms — auditable at policy level |
| Model contract (JSON) | Enforces ASOP 56; injected into LLM system prompt as governance context |
| Two-pass LLM self-review | Draft then constraint-check then corrected JSON; prevents causal overstatement |
| Deterministic spotlight thresholds | \|A/E - 1\| > 0.15 and Exposure >= 1,000 life-years; reproducible, not ML-generated |

See `ARCHITECTURE.md` for detailed module-level documentation.
