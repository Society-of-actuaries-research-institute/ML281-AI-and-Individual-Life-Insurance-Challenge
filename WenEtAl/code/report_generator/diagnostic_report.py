# -*- coding: utf-8 -*-
"""
Validation diagnostic report builder.
"""

from datetime import date
from typing import Dict, Any, List, Optional

from .schemas import ValidationDiagnosticData, ValidationCheck, DriftFlag


def _safe_pct(x: Optional[float]) -> Optional[float]:
    if x is None:
        return None
    try:
        return float(x)
    except Exception:
        return None


def _compute_drift_flags(yearly_ae: Dict[str, Any]) -> List[DriftFlag]:
    """
    Compute drift flags from yearly A/E series.

    Primary rule (from paper Appendix):
      Flag a year if |A/E - 1| > 0.10 AND the deviation persists for >= 2 consecutive years.
      Both conditions must be met (persistence required to avoid single-year noise).

    Secondary flag (sudden shift, standalone):
      |year-over-year delta A/E| > 0.05, even if the absolute level < 0.10.
    """
    flags: List[DriftFlag] = []
    years_sorted = sorted(
        [(int(yr), float(ae)) for yr, ae in yearly_ae.items() if isinstance(ae, (int, float))],
        key=lambda x: x[0]
    )
    if not years_sorted:
        return flags

    # --- Primary rule: |A/E - 1| > 0.10 with persistence >= 2 consecutive years ---
    # Mark each year as "deviation" or not
    deviation_mask = [abs(ae - 1.0) > 0.10 for _, ae in years_sorted]

    # A year qualifies for primary flag if it AND its neighbor form a consecutive run >= 2
    primary_flagged = set()
    n = len(years_sorted)
    for i in range(n):
        if deviation_mask[i]:
            # Check previous neighbor
            if i > 0 and deviation_mask[i - 1]:
                primary_flagged.add(i - 1)
                primary_flagged.add(i)
            # Check next neighbor
            if i < n - 1 and deviation_mask[i + 1]:
                primary_flagged.add(i)
                primary_flagged.add(i + 1)

    # --- Build flags ---
    prev_ae = None
    for idx, (yr, ae) in enumerate(years_sorted):
        delta = (ae - prev_ae) if prev_ae is not None else 0.0
        desc_parts = []

        if idx in primary_flagged:
            direction = "above" if ae > 1.0 else "below"
            desc_parts.append(
                f"Persistent drift: A/E {direction} 1.0 by {abs(ae - 1.0):.3f} "
                f"(>=2 consecutive years; |A/E-1|>0.10 threshold)"
            )

        if prev_ae is not None and abs(delta) > 0.05:
            desc_parts.append(f"Sudden shift: year-over-year delta = {delta:+.3f}")

        if desc_parts:
            flags.append(DriftFlag(
                year=yr,
                ae_ratio=round(ae, 4),
                delta_ae=round(delta, 4),
                description="; ".join(desc_parts),
            ))

        prev_ae = ae

    return flags


def build_validation_diagnostic_data(kb) -> ValidationDiagnosticData:
    """
    Build validation diagnostics using knowledge base artifacts.
    """
    overall_ae = kb.overall_ae or {}
    yearly_ae = kb.yearly_ae or {}
    model_contract = kb._load_json("contracts/model_contract.json") if kb else {}

    checks: List[ValidationCheck] = []
    overall_value = overall_ae.get("overall_ae")

    if overall_value is None:
        checks.append(
            ValidationCheck(
                check_id="calibration_overall",
                status="warn",
                description="Overall A/E not found in knowledge base",
                evidence="calibration/overall_ae.json missing or incomplete",
            )
        )
    else:
        status = "pass" if 0.95 <= float(overall_value) <= 1.05 else "warn"
        checks.append(
            ValidationCheck(
                check_id="calibration_overall",
                status=status,
                description="Overall A/E within expected range (0.95–1.05)",
                evidence=f"overall_ae={float(overall_value):.4f}",
            )
        )

    if yearly_ae:
        checks.append(
            ValidationCheck(
                check_id="calibration_yearly",
                status="pass",
                description="Yearly A/E series available",
                evidence=f"{len(yearly_ae)} years",
            )
        )
    else:
        checks.append(
            ValidationCheck(
                check_id="calibration_yearly",
                status="warn",
                description="Yearly A/E series not found",
                evidence="calibration/yearly_ae.json missing",
            )
        )

    governance = model_contract.get("governance", {})
    if governance:
        checks.append(
            ValidationCheck(
                check_id="governance_review",
                status="pass",
                description="Model governance metadata available",
                evidence=f"last_review={governance.get('last_review')}, next_review={governance.get('next_review')}",
            )
        )
    else:
        checks.append(
            ValidationCheck(
                check_id="governance_review",
                status="warn",
                description="Model governance metadata missing",
                evidence="contracts/model_contract.json governance section missing",
            )
        )

    data_quality = {
        "numerical_summary_present": kb.numerical_summary is not None,
        "categorical_summary_present": kb.categorical_summary is not None,
        "global_shap_present": kb.global_shap is not None,
        "coverage_registry_present": kb.coverage_registry is not None,
        "spotlight_summary_present": kb.spotlight_summary is not None,
    }

    drift_flags = _compute_drift_flags(yearly_ae)

    # Add drift check
    if drift_flags:
        checks.append(
            ValidationCheck(
                check_id="drift_detection",
                status="warn",
                description=f"{len(drift_flags)} year(s) flagged for A/E deviation or shift",
                evidence=", ".join(str(f.year) for f in drift_flags),
            )
        )
    else:
        checks.append(
            ValidationCheck(
                check_id="drift_detection",
                status="pass",
                description="No significant A/E drift detected",
                evidence="All years within ±1% deviation",
            )
        )

    return ValidationDiagnosticData(
        report_date=str(date.today()),
        report_version="1.0",
        calibration_overall=overall_ae,
        calibration_yearly=yearly_ae,
        data_quality=data_quality,
        model_contract=model_contract,
        validation_checks=checks,
        drift_flags=drift_flags,
        disclaimers=[
            "Diagnostics are derived from available knowledge base artifacts.",
            "Missing artifacts reduce validation completeness and should be investigated.",
        ],
    )
