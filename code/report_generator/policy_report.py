# -*- coding: utf-8 -*-
"""
Policy report builder

Creates ReportData for an individual policy interpretation report.
"""

import re
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, Callable
from datetime import datetime

import pandas as pd

from .schemas import ReportData, PredictionData, SHAPData, SegmentData, EvidenceChart
from .config import FEATURES


TEST_SAMPLE = {
    # Age-30 male non-smoker permanent (WL) policy, 30 years in force
    # Uses actual ILEC data category values (Insurance_Plan='Perm', not 'WL')
    "Attained_Age": 30,
    "Issue_Age": 0,
    "Duration": 30,
    "Sex": "M",
    "Smoker_Status": "NS",
    "Insurance_Plan": "Perm",           # ILEC uses 'Perm', not 'WL'
    "Face_Amount_Band": "07: 500,000 - 999,999",  # ILEC band numbering
    "Preferred_Class": "U",             # Unclassified (most common for perm)
    "SOA_Post_Lvl_Ind": "WLT",         # Whole Life Term indicator
    "SOA_Antp_Lvl_TP": "N/A (Not Term)",
    "SOA_Guar_Lvl_TP": "N/A (Not Term)",
}


def load_sample(use_test_data: bool = True) -> Tuple[Optional[dict], Optional[pd.DataFrame]]:
    data_path = Path(__file__).parent.parent.parent / "data" / "ilec_cleaned.parquet"
    if not data_path.exists():
        print(f"[WARN] Data file not found: {data_path}")
        return None, None

    df = pd.read_parquet(data_path)
    if use_test_data:
        return TEST_SAMPLE.copy(), df
    return df.sample(1).iloc[0].to_dict(), df


def _match_coverage_by_rules(input_record: dict, kb) -> Optional[dict]:
    """
    Heuristic match to coverage segments using rule_text/rule_list.
    """
    if not kb.coverage_registry:
        return None

    age = input_record.get("Attained_Age", 0)
    duration = input_record.get("Duration", 0)
    plan = input_record.get("Insurance_Plan", "Term")
    smoker = input_record.get("Smoker_Status", "NS")
    sex = input_record.get("Sex", "M")

    smoker_known = smoker in ["NS", "S"]
    is_perm = plan in ["WL", "UL", "VUL", "ULSG"]

    for seg in kb.coverage_registry.get("segments", []):
        rules = seg.get("rule_list", [])
        if not rules:
            continue

        match = True
        for rule in rules:
            rule_lower = rule.lower()

            if "duration" in rule_lower:
                if "<=" in rule and ">" not in rule:
                    threshold = float(rule.split("<=")[1].strip())
                    if not (duration <= threshold):
                        match = False
                elif ">" in rule and "<=" not in rule:
                    threshold = float(rule.split(">")[1].strip())
                    if not (duration > threshold):
                        match = False
                elif "<" in rule and ">" in rule:
                    parts = rule.replace("Duration", "").strip().split("<")
                    lower = float(parts[0].strip())
                    upper = float(parts[1].replace("=", "").strip())
                    if not (lower < duration <= upper):
                        match = False

            elif "attained_age" in rule_lower or "age" in rule_lower:
                if "<=" in rule and ">" not in rule:
                    threshold = float(rule.split("<=")[1].strip())
                    if not (age <= threshold):
                        match = False
                elif ">" in rule and "<=" not in rule:
                    threshold = float(rule.split(">")[1].strip())
                    if not (age > threshold):
                        match = False
                elif "<" in rule and ">" in rule:
                    parts = rule.split("<")
                    lower = float(parts[0].split(">")[-1].strip())
                    upper = float(parts[1].replace("=", "").replace("Attained_Age", "").strip())
                    if not (lower < age <= upper):
                        match = False

            elif "smoker" in rule_lower:
                if "is unknown" in rule_lower and smoker_known:
                    match = False
                if "is known" in rule_lower and not smoker_known:
                    match = False

            elif "insurance plan" in rule_lower or "plan is" in rule_lower:
                if "is perm" in rule_lower and not is_perm:
                    match = False
                elif "is not perm" in rule_lower and is_perm:
                    match = False
                elif "is term" in rule_lower and plan != "Term":
                    match = False
                elif "is not term" in rule_lower and plan == "Term":
                    match = False

            elif "sex is" in rule_lower:
                if "female" in rule_lower and sex != "F":
                    match = False
                elif "male" in rule_lower and sex != "M":
                    match = False

        if match:
            return seg

    segments = kb.coverage_registry.get("segments", [])
    return segments[0] if segments else None


def _parse_rule(rule: str) -> Callable[[dict], bool]:
    """
    Parse a rule string into a callable predicate over an input_record dict.

    Supports numeric comparisons: Feature > N, Feature >= N, Feature < N, Feature <= N
    Supports categorical: Feature is VALUE, Feature is NOT VALUE
    Unknown rule format returns always-True (permissive fallback).
    """
    _NUMERIC = re.compile(
        r'([A-Za-z][A-Za-z_0-9]*)\s*([<>]=?)\s*([\d.]+)'
    )
    _IS_NOT = re.compile(r'([A-Za-z][A-Za-z_0-9]*)\s+is NOT\s+(.+)', re.IGNORECASE)
    _IS = re.compile(r'([A-Za-z][A-Za-z_0-9]*)\s+is\s+(.+)', re.IGNORECASE)

    m_num = _NUMERIC.search(rule)
    if m_num:
        feat, op, val_str = m_num.group(1), m_num.group(2), float(m_num.group(3))
        ops = {'<=': lambda a, b: a <= b, '>=': lambda a, b: a >= b,
               '<': lambda a, b: a < b, '>': lambda a, b: a > b}
        fn = ops.get(op, lambda a, b: True)
        return lambda rec, f=feat, o=fn, v=val_str: o(float(rec.get(f, 0)), v)

    m_not = _IS_NOT.search(rule)
    if m_not:
        feat, val_str = m_not.group(1), m_not.group(2).strip()
        return lambda rec, f=feat, v=val_str: str(rec.get(f, "")) != v

    m_is = _IS.search(rule)
    if m_is:
        feat, val_str = m_is.group(1), m_is.group(2).strip()
        return lambda rec, f=feat, v=val_str: str(rec.get(f, "")) == v

    return lambda rec: True  # unknown rule — do not filter


def _match_spotlight_by_rules(input_record: dict, kb) -> Optional[dict]:
    if not kb.spotlight_summary:
        return None

    anomalies = kb.spotlight_summary.get("anomalies", [])
    if not anomalies:
        return None

    for anom in anomalies:
        rules = anom.get("rule_list", [])
        if not rules:
            continue
        predicates = [_parse_rule(r) for r in rules]
        if all(p(input_record) for p in predicates):
            return anom

    return anomalies[0] if anomalies else None


def build_policy_report_data(sample: Dict[str, Any], df, predictor, kb) -> ReportData:
    if df is None:
        raise ValueError(
            "DataFrame is required for policy report. "
            "Run build_parquet.py first to generate data/ilec_cleaned.parquet."
        )

    input_record = {f: sample[f] for f in FEATURES}

    mortality_rate, shap_dict = predictor.predict(input_record)
    overall_rate = df["Death_Count"].sum() / df["Policies_Exposed"].sum()

    prediction = PredictionData(
        mortality_rate=mortality_rate,
        overall_death_rate=overall_rate,
        interpretation="higher" if mortality_rate > overall_rate else "lower",
    )

    shap_contributions = []
    sorted_shap = sorted(shap_dict.items(), key=lambda x: abs(x[1]), reverse=True)
    for feat, shap_val in sorted_shap[:5]:
        direction = "Increases" if shap_val > 0 else "Decreases"
        shap_contributions.append(
            SHAPData(
                feature=feat,
                value=input_record.get(feat, "N/A"),
                shap_value=shap_val,
                direction=direction,
            )
        )

    percentile_context = []
    for feat in ["Attained_Age", "Issue_Age", "Duration"]:
        val = sample[feat]
        pct = int((df[feat] < val).mean() * 100)
        percentile_context.append(
            {
                "feature": feat,
                "value": val,
                "percentile": pct,
                "description": f"{feat}={val} is at the {pct}th percentile",
            }
        )

    category_context = []
    for feat in ["Sex", "Smoker_Status", "Insurance_Plan"]:
        val = sample[feat]
        pct = round((df[feat] == val).mean() * 100, 1)
        category_context.append(
            {
                "feature": feat,
                "value": val,
                "population_pct": pct,
                "description": f"{feat}={val} ({pct}% of population)",
            }
        )

    coverage_segment = None
    cov_seg = _match_coverage_by_rules(input_record, kb)
    if cov_seg:
        coverage_segment = SegmentData(
            segment_id=cov_seg.get("segment_id"),
            rule_text=cov_seg.get("rule_text"),
            ae_ratio=cov_seg.get("statistics", {}).get("ae_ratio", 1.0),
            credibility="high"
            if cov_seg.get("statistics", {}).get("exposure", 0) > 50000
            else "medium",
        )

    spotlight_segment = None
    spot_match = _match_spotlight_by_rules(input_record, kb)
    if spot_match:
        spotlight_segment = SegmentData(
            segment_id=spot_match.get("segment_id", "SPOT_UNKNOWN"),
            rule_text=" AND ".join(spot_match.get("rule_list", [])),
            ae_ratio=spot_match.get("ae_ratio", 1.0),
            credibility=str(spot_match.get("credibility", "unknown")),
            risk_level="high_risk" if spot_match.get("ae_ratio", 1.0) > 1.5 else "elevated",
        )

    evidence_charts = []
    charts_dir = Path(__file__).parent.parent.parent / "data" / "reports"
    if (charts_dir / "shap_waterfall.png").exists():
        evidence_charts.append(
            EvidenceChart(
                name="SHAP Waterfall",
                path="shap_waterfall.png",
                description="Feature contributions to prediction",
            )
        )

    overall_ae = kb.overall_ae.get("overall_ae", 0.9989) if kb.overall_ae else 0.9989

    return ReportData(
        report_date=datetime.now().strftime("%Y-%m-%d"),
        report_version="1.0",
        input_record=input_record,
        prediction=prediction,
        shap_contributions=shap_contributions,
        percentile_context=percentile_context,
        category_context=category_context,
        coverage_segment=coverage_segment,
        spotlight_segment=spotlight_segment,
        overall_ae=overall_ae,
        evidence_charts=evidence_charts,
        disclaimers=[
            "This prediction is based on historical data from 2012-2019 and is for reference only.",
            "This model does not include data from periods outside the 2012-2019 training window; predictions outside this window require external adjustment.",
            "This report complies with ASOP 23/25/41/56 standards.",
        ],
    )
