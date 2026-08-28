# ILEC Mortality Experience Analysis Report

**Report Type**: Individual Life Insurance Mortality Study  
**Report Date**: {{ report_date }}  
**Version**: 1.0  

---

## Executive Summary

### Purpose and Intended Use
This report presents mortality experience analysis for individual life insurance using the SOA ILEC dataset (2012-2019). It is intended for actuarial experience studies and internal analysis.

> **Limitations**: This report is not intended for regulatory capital calculation (Solvency II/RBC) or IFRS 17 best estimate without additional review.

### Key Findings
| Metric | Value | Status |
|:-------|:------|:-------|
| Overall A/E Ratio | {{ overall_ae }} | {{ ae_status }} |
| Model Type | LightGBM (Poisson) | |
| Observation Period | 2012-2019 | |
| Total Exposure | {{ total_exposure }} policy-years | |
| Total Deaths | {{ total_deaths }} | |

### Key Risk Segments (Spotlight Alerts)
{{ spotlight_summary }}

---

## 1. Scope & Context

### 1.1 Business Scope
- **Products**: Individual life insurance (Term, Whole Life, UL)
- **Geography**: United States
- **Data Source**: SOA ILEC Dataset

### 1.2 Observation Period
- **Experience Period**: January 1, 2012 - December 31, 2019
- **Estimation Window**: Same as experience period

### 1.3 Event Definition
- **Death**: Date of death (not report date)
- **Censoring**: Lapse, maturity, or study end

---

## 2. Data Quality (ASOP 23)

### 2.1 Data Source
Data provided by Society of Actuaries Experience Studies.

### 2.2 Quality Metrics
| Metric | Value |
|:-------|:------|
| Total Records | {{ total_records }} |
| Missing Rate | < 0.1% |
| Duplicate Rate | 0% |

### 2.3 Exposure Calculation
Central exposure method: fraction of year policy was in force at each age.

**Reference**: `methodology/data_specification.md`

---

## 3. Experience Study

### 3.1 Overall Experience
| Metric | Value |
|:-------|:------|
| Total Policies Exposed | {{ total_exposure }} |
| Total Deaths | {{ total_deaths }} |
| Crude Death Rate | {{ crude_rate }} |
| Overall A/E | {{ overall_ae }} |

### 3.2 A/E by Year
{{ yearly_ae_table }}

### 3.3 A/E by Key Segments
{{ segment_ae_table }}

---

## 4. Model Specification (ASOP 56)

### 4.1 Model Overview
| Item | Value |
|:-----|:------|
| Algorithm | LightGBM |
| Objective | Poisson with exposure offset |
| Features | 11 (3 numerical, 8 categorical) |

### 4.2 Feature Importance (SHAP)
{{ shap_importance_table }}

### 4.3 Parameter Selection Rationale
| Parameter | Value | Rationale |
|:----------|:------|:----------|
| max_depth | 6 | Balance complexity vs interpretability |
| min_child_samples | 100 | Credibility threshold |
| n_estimators | 100 | Prevent overfitting |

**Reference**: `methodology/model_specification.md`

---

## 5. Model Validation

### 5.1 Out-of-Time Validation
| Metric | Value |
|:-------|:------|
| Train Period | 2012-2017 |
| Test Period | 2018-2019 |
| Holdout A/E | {{ holdout_ae }} |
| Status | {{ holdout_status }} |

### 5.2 Calibration by Decile
{{ calibration_table }}

### 5.3 ML vs Traditional Comparison
{{ ml_vs_trad_table }}

---

## 6. Assumptions & Limitations

### 6.1 Key Assumptions
1. Mortality trends remain stable beyond 2019
2. COVID-19 impact requires separate adjustment
3. Year factors capture temporal variation

### 6.2 Known Limitations
1. Not suitable for ages > 95 (limited data)
2. COVID-19 period not modeled
3. Assumes homogeneous mortality within segments

---

## 7. Risk Assessment

### 7.1 Anomalous Segments (Spotlight)
{{ spotlight_details }}

### 7.2 Sensitivity Analysis
| Assumption Change | Impact on A/E |
|:------------------|:--------------|
| +10% mortality | {{ sens_plus_10 }} |
| -10% mortality | {{ sens_minus_10 }} |

---

## 8. Governance

### 8.1 Version Control
| Item | Version | Date |
|:-----|:--------|:-----|
| Data | ILEC 2023 | 2023-06 |
| Model | v1.0 | 2026-01-05 |
| Report | v1.0 | {{ report_date }} |

### 8.2 Peer Review
| Reviewer | Date | Status |
|:---------|:-----|:-------|
| {{ reviewer }} | {{ review_date }} | {{ review_status }} |

---

## 9. Disclosures

### 9.1 Standard Disclaimer
This report is prepared for informational purposes and represents the actuary's professional judgment based on available data. Actual experience may differ from projected values.

### 9.2 Deviations from Standard Practice
None.

### 9.3 Dependencies
- SOA ILEC dataset quality
- Year factor adjustments for temporal effects

---

*Report prepared in accordance with ASOP No. 23, 25, 41, and 56.*

---

## Appendix A: Technical Details
- Full model specification: `methodology/model_specification.md`
- Data specification: `methodology/data_specification.md`
- Validation details: `methodology/validation_framework.md`

## Appendix B: SHAP Analysis
- Global importance: `shap/global_importance.json`
- Summary plot: `shap/shap_summary_plot.png`
