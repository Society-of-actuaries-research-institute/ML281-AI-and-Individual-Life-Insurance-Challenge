# System Architecture (SOA Actuarial Intelligence Framework)

This document provides a detailed, end-to-end view of the system so readers can quickly understand how data flows through modeling, validation, explainability, and LLM reporting.

---

## 1) Pipeline Architecture (Data -> Model -> Validation -> SHAP -> LLM)

```
                   ┌─────────────────────────────────────────┐
                   │           Knowledge Base (KB)           │
                   │  - EDA summaries / calibration store    │
                   │  - Coverage & spotlight segments        │
                   │  - SHAP global importance               │
                   │  - Model contract & governance rules    │
                   └───────────────────┬─────────────────────┘
                                       │
                                       ▼
┌──────────────┐   ┌──────────────────┐   ┌────────────────────┐
│ Raw / Clean  │   │  Predictor (LGBM)│   │ Validation Layer    │
│ Data (ILEC)  │─▶ │ Poisson + offset │─▶│ A/E + backtesting    │
└──────────────┘   └──────────────────┘   └────────────────────┘
                            │                     │
                            ▼                     ▼
                   ┌────────────────┐     ┌────────────────────┐
                   │  SHAP Explain  │     │ Segmentation Layer │
                   │  (global/local)│     │ Coverage + Spotlight│
                   └────────────────┘     └────────────────────┘
                            │                     │
                            └──────────┬──────────┘
                                       ▼
                            ┌────────────────────┐
                            │ LLM w/ Guardrails  │
                            │ Prompt + Contract  │
                            │ + Self-Review Pass │
                            └────────────────────┘
                                       │
                                       ▼
                            ┌────────────────────┐
                            │ Reports (S1–S3)    │
                            └────────────────────┘
```

**Key Modules**
- Pipeline/Router: `code/report_generator/pipeline.py`, `code/report_generator/router.py`
- Model Inference + SHAP: `code/report_generator/predictor.py`
- Knowledge Base Loader: `code/report_generator/knowledge.py`
- LLM + Prompt: `code/report_generator/llm.py`, `code/report_generator/prompt_builder.py`
- Report Rendering: `code/report_generator/renderer.py`

**Document-aligned flow highlights**
- The framework is explicitly modular: modeling, validation, explanation, segmentation, and reporting are separate stages.
- Validation includes both deterministic A/E monitoring and expanding-window backtesting to detect drift.
- Reporting is governed by a model contract and constrained LLM rules, with actuarial review emphasized as the final control point.

---

## 2) Key Model Settings (LightGBM, Poisson, offset)

**Model Type**
- LightGBM Poisson regression (death count as target)
- offset = log(Policies_Exposed)
- Target is **mortality rate**, not death probability

**Model Contract (authoritative source)**
File: `knowledge_base/contracts/model_contract.json`

Key definitions (simplified):
- Objective: Poisson
- Formula: log(E[Death_Count]) = log(Exposure) + f(X)
- Two-stage: base mortality + year factors (`models/year_factors_offset.csv`)

**Inference**
File: `code/report_generator/predictor.py`
- `predict()`: returns `mortality_rate` + `SHAP values`
- If model/SHAP unavailable -> mock fallback

---

## 3) Validation / A/E Computation

### A/E Definitions (from KB)
Sources:
- `knowledge_base/calibration/overall_ae.json`
- `knowledge_base/calibration/yearly_ae.json`

Example formula:
```
Overall A/E = total_actual_deaths / total_expected_deaths
```

### Validation Diagnostic (S1)
Builders:
- `code/report_generator/diagnostic_report.py`
- `code/report_generator/renderer.py` (_render_validation_markdown)

Checks:
- Overall A/E within expected range (0.95-1.05)
- Yearly A/E availability (2012–2019 in the paper)
- Model Contract presence (governance and intended use/limitations)
- EDA / SHAP / Segments presence in KB

### Backtesting (Expanding Window)
The paper describes expanding-window backtesting that simulates real-time deployment:
- Train on an initial window, predict future years, then expand the training window.
- Used to detect drift or structural change (e.g., a one-time A/E elevation).
- This logic is represented by the deterministic A/E store and can be extended in code to mirror production cycles.

### Input / LLM Output Validation
File: `code/report_generator/validation.py`
1. Input checks: required fields, numeric ranges, categorical normalization
2. LLM output checks: causal language red flags (review required)

---

## 4) Spotlight Rule Deterministic Logic

### Spotlight Sources
Loaded from KB:
- `knowledge_base/segments/spotlight_summary.json`
- `knowledge_base/segments/spotlight/*.json`

### Deterministic Ranking (Spotlight Report S2)
File: `code/report_generator/spotlight_report.py`

Ranking rule:
```
score = Exposure × |A/E − 1|
Take Top-N anomalies
```

**Thresholds from the paper (conceptual)**
- |A/E − 1| > 0.15
- Exposure ≥ 1,000 life-years
These thresholds are applied in the spotlight report definition and are used as deterministic filters.

### Rule Matching (Policy Report, heuristic)
File: `code/report_generator/policy_report.py`

Rule evaluation based on rule_list:
- Duration, Attained_Age, Issue_Age, Face_Amount_Band
- All rules satisfied -> matched segment

---

## 5) How the LLM Is Constrained (prompt + rules + model contract)

### System Prompt Structure
File: `code/report_generator/prompt_builder.py`

**Primary constraints**
1. Use evidence only (KB + user prompt)
2. Do not invent numbers or sources
3. Use associative (non-causal) language
4. Comply with ASOP 23/25/41/56
5. Output JSON only (schema-bound)

### Model Contract Injection
Source: `knowledge_base/contracts/model_contract.json`
Includes:
- Intended use / scope
- Limitations (e.g., no COVID-19 period)
- Governance metadata

### Two-pass LLM Self-Review
File: `code/report_generator/llm.py`

Flow:
1. Draft JSON generated from context
2. Self-review pass checks constraints and returns corrected JSON-only output

### Actuarial Review (Human Oversight)
The document emphasizes that the actuary reviews the LLM narrative for tone, emphasis, and professional judgment.  
This is represented in the architecture as a final review control after the governed LLM output.

---

## 6) Report Types (S1-S3)

| Report | Current Output | Module |
|------|----------|------|
| S1 Validation Diagnostic | `validation_diagnostic.md/.json` | `diagnostic_report.py` |
| S2 Spotlight / Exception | `spotlight_report.md/.json` | `spotlight_report.py` |
| S3 Policy Interpretation | `actuarial_report.md/.json` | `policy_report.py` |
| Experience Trend Pack | `experience_trend_pack.md/.json` | `demo_trend_pack.py` |

**Document mapping**
- S1: Portfolio-level experience trend pack (yearly A/E, age curves, product mix shifts, drift notes).
- S2: Deterministic spotlight anomalies with exposure/credibility thresholds and stability indicators.
- S3: Policy-level interpretation report with SHAP + coverage/spotlight segments + calibration notes.

---

## 7) Pipeline Entry Point

File: `code/report_generator/run_pipeline.py`

Examples:
```
python code/report_generator/run_pipeline.py --report-type policy_report --use-test-data
python code/report_generator/run_pipeline.py --report-type spotlight_report --top-n-spotlight 5
python code/report_generator/run_pipeline.py --report-type trend_pack
python code/report_generator/run_pipeline.py --report-type validation_diagnostic
```

---

## 8) Module View Summary

```
Pipeline
├── Router (report type decision)
├── Validation (input + LLM output)
├── Predictor (LightGBM + offset)
├── SHAP (XAI explanation)
├── LLM (prompt + contract + KB)
└── Renderer (MD/JSON, optional PDF)
```

---
