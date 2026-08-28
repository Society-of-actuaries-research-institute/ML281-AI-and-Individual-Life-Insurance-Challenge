# -*- coding: utf-8 -*-
"""
Self-Review Injection Test
==========================
Systematically injects known constraint violations into LLM output drafts
and measures whether the two-pass self-review mechanism detects and corrects them.

Usage:
    python code/tests/test_self_review_injection.py

Output:
    Prints per-case results and writes summary to data/reports/injection_test_results.json

Test design:
    - 20 injection cases across 3 violation categories
    - Detection is measured by comparing draft vs. reviewed output
    - Uses validate_llm_text() post-generation scan as second-layer check

Note: Requires a valid LLM API key (GOOGLE_API_KEY or ANTHROPIC_API_KEY).
      If LLM is unavailable, only the validate_llm_text() layer is tested.
"""

import sys
import json
import copy
from pathlib import Path
from datetime import datetime

# Add project root to path
ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))

from code.report_generator.validation import validate_llm_text
from code.report_generator.llm import create_llm_client, LLMUnavailableError
from code.report_generator.schemas import ActuarialInsights, SHAPExplanation


# =============================================================================
# Injection Case Definitions
# =============================================================================

CLEAN_DRAFT = {
    "summary": (
        "The predicted mortality rate of 0.000691 per year is lower than the "
        "population average, consistent with a non-smoker permanent policy holder "
        "at attained age 30. The overall portfolio A/E ratio of 0.9989 indicates "
        "near-perfect model calibration against the VBT2015 standard table."
    ),
    "risk_interpretation": (
        "This policy segment is associated with below-average mortality risk. "
        "The age and non-smoker status are the primary factors driving lower-than-average rates."
    ),
    "shap_explanations": [
        {
            "feature": "Attained_Age",
            "value": "30",
            "direction": "Decreases",
            "explanation": "Younger age is associated with lower mortality rates."
        },
        {
            "feature": "Smoker_Status",
            "value": "NS",
            "direction": "Decreases",
            "explanation": "Non-smoker status may reflect reduced mortality risk."
        },
    ],
    "calibration_note": (
        "The model A/E of 0.9989 indicates the portfolio mortality experience "
        "closely tracks the VBT2015 expected values."
    ),
    "segment_insight": None,
    "recommendations": [
        "Monitor mortality trends for the 30-40 age band annually.",
        "Consider credibility weighting for emerging data beyond 2019.",
    ],
}

# 20 injection cases: (case_id, category, description, injected_field, injected_value, detection_hint)
INJECTION_CASES = [
    # --- Category A: Causal Language (8 cases) ---
    (1,  "causal", "Direct 'causes' in summary",
     "summary",
     "The young age causes the lower mortality rate observed in this segment.",
     "causes"),

    (2,  "causal", "'proves' in risk interpretation",
     "risk_interpretation",
     "The NS status proves this policyholder is low-risk.",
     "proves"),

    (3,  "causal", "'results in' in calibration note",
     "calibration_note",
     "Good underwriting results in the A/E of 0.9989.",
     "results in"),

    (4,  "causal", "Chinese causal '導致' in summary",
     "summary",
     "The portfolio's good experience 導致 the low A/E ratio.",
     "導致"),

    (5,  "causal", "'proves' in SHAP explanation",
     "shap_explanations[0].explanation",
     "Young attained age proves the policy is low-risk.",
     "proves"),

    (6,  "causal", "'cause' (noun form) in recommendation",
     "recommendations[0]",
     "Investigate the cause of the low mortality in this segment.",
     "cause"),

    (7,  "causal", "Chinese '因此' in summary",
     "summary",
     "The model A/E is 0.9989, 因此 the model is well calibrated.",
     "因此"),

    (8,  "causal", "'causal' as adjective in risk interpretation",
     "risk_interpretation",
     "There is a causal relationship between NS status and reduced mortality.",
     "causal"),

    # --- Category B: Fabricated Numbers (6 cases) ---
    (9,  "fabrication", "Wrong mortality rate in summary (0.0005 vs correct 0.000691)",
     "summary",
     "The predicted mortality rate of 0.0005 per year is lower than the population average.",
     "0.000691"),

    (10, "fabrication", "Invented A/E ratio in summary (1.05 vs correct 0.9989)",
     "summary",
     "The overall portfolio A/E ratio of 1.05 indicates slight over-prediction.",
     "0.9989"),

    (11, "fabrication", "Wrong mortality rate AND wrong A/E in summary",
     "summary",
     "The predicted rate of 0.0003 and A/E of 0.95 both suggest underpricing.",
     "0.000691"),

    (12, "fabrication", "Invented ASOP number (ASOP 99)",
     "calibration_note",
     "Per ASOP 99, the model calibration is adequate.",
     "ASOP 99"),

    (13, "fabrication", "Wrong ASOP for modeling (ASOP 30 instead of 56)",
     "recommendations[0]",
     "Per ASOP 30 (modeling), annual monitoring is recommended.",
     "ASOP 30"),

    (14, "fabrication", "Wrong ASOP for data quality (ASOP 10 instead of 23)",
     "calibration_note",
     "Data quality review per ASOP 10 is advised.",
     "ASOP 10"),

    # --- Category C: Schema / Format Violations (6 cases) ---
    (15, "schema", "More than 3 recommendations (4 items)",
     "recommendations",
     [
         "Monitor trends annually.",
         "Review underwriting criteria.",
         "Update VBT2015 assumptions.",
         "Commission a credibility study.",
     ],
     "max 3 recommendations"),

    (16, "schema", "Empty summary string",
     "summary",
     "",
     "empty summary"),

    (17, "schema", "Markdown in summary (code fence)",
     "summary",
     "```json\n{\"rate\": 0.000691}\n```\nThe rate is 0.000691.",
     "markdown"),

    (18, "schema", "Definitive language in hypothesis (no conditional)",
     "risk_interpretation",
     "The model will perform better on non-smoker segments.",
     "will"),

    (19, "schema", "Summary missing the predicted mortality rate entirely",
     "summary",
     "This policy exhibits below-average mortality risk based on demographic factors.",
     "0.000691"),

    (20, "schema", "Recommendation not actionable (vague)",
     "recommendations[0]",
     "Things should be done about mortality.",
     "actionable"),
]


# =============================================================================
# Helpers
# =============================================================================

def _apply_injection(draft: dict, field: str, value) -> dict:
    """Apply an injected value to a nested field path."""
    result = copy.deepcopy(draft)

    if field == "shap_explanations[0].explanation":
        result["shap_explanations"][0]["explanation"] = value
    elif field == "recommendations[0]":
        result["recommendations"][0] = value
    elif field == "recommendations":
        result["recommendations"] = value
    else:
        result[field] = value

    return result


def _to_actuarial_insights(d: dict) -> ActuarialInsights:
    shap_list = [SHAPExplanation(**s) for s in d.get("shap_explanations", [])]
    return ActuarialInsights(
        summary=d.get("summary", ""),
        risk_interpretation=d.get("risk_interpretation", ""),
        shap_explanations=shap_list,
        calibration_note=d.get("calibration_note", ""),
        segment_insight=d.get("segment_insight"),
        recommendations=d.get("recommendations", []),
    )


def _text_contains(text: str, hint: str) -> bool:
    """Check if hint keyword appears in text (case-insensitive for ASCII, exact for CJK)."""
    if any(ord(c) > 127 for c in hint):
        return hint in text  # CJK: exact match
    return hint.lower() in text.lower()


def _has_causal_language(insights: ActuarialInsights, hint: str) -> bool:
    """Check if any text field in insights contains the causal hint."""
    fields = [
        insights.summary,
        insights.risk_interpretation,
        insights.calibration_note or "",
        insights.segment_insight or "",
    ]
    fields += list(insights.recommendations or [])
    fields += [s.explanation for s in (insights.shap_explanations or [])]
    return any(_text_contains(f, hint) for f in fields)


def _insights_to_fulltext(insights: ActuarialInsights) -> str:
    parts = [insights.summary, insights.risk_interpretation, insights.calibration_note or ""]
    parts += list(insights.recommendations or [])
    parts += [s.explanation for s in (insights.shap_explanations or [])]
    return " ".join(parts)


# =============================================================================
# Layer-1: validate_llm_text() scan
# =============================================================================

def run_layer1_scan(draft_insights: ActuarialInsights) -> bool:
    """Return True if validate_llm_text() flags a causal language warning."""
    full_text = _insights_to_fulltext(draft_insights)
    result = validate_llm_text(full_text)
    return len(result.warnings) > 0


# =============================================================================
# Layer-2: LLM self-review
# =============================================================================

def run_layer2_review(llm_client, draft_insights: ActuarialInsights, system_prompt: str, user_prompt: str) -> ActuarialInsights:
    """Run _self_review() and return corrected insights."""
    reviewed = llm_client._self_review(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        draft=draft_insights,
        output_schema=ActuarialInsights,
    )
    return reviewed if reviewed is not None else draft_insights


# =============================================================================
# Main test runner
# =============================================================================

MINIMAL_SYSTEM_PROMPT = (
    "You are an actuarial AI assistant. Generate structured JSON matching the "
    "ActuarialInsights schema. The predicted mortality rate is 0.000691. "
    "The overall A/E is 0.9989. Use conditional language for hypotheses."
)

MINIMAL_USER_PROMPT = (
    "Policy: Age-30, NS, Perm, Face Band 07. Predicted rate: 0.000691. "
    "Overall A/E: 0.9989. Generate actuarial insights."
)


def run_tests(use_llm: bool = True):
    print("=" * 70)
    print("SOA Self-Review Injection Test")
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print(f"LLM self-review: {'enabled' if use_llm else 'DISABLED (layer-1 only)'}")
    print("=" * 70)

    llm_client = None
    if use_llm:
        try:
            llm_client = create_llm_client()
            if not llm_client.is_available():
                print("[WARN] LLM not available - running layer-1 (validate_llm_text) only")
                llm_client = None
            else:
                print(f"[OK] LLM: {llm_client.model_version}")
        except Exception as e:
            print(f"[WARN] LLM init failed: {e} - running layer-1 only")
            llm_client = None

    results = []
    category_counts = {"causal": {"detected": 0, "total": 0},
                       "fabrication": {"detected": 0, "total": 0},
                       "schema": {"detected": 0, "total": 0}}

    for case_id, category, description, field, value, detection_hint in INJECTION_CASES:
        print(f"\n[{case_id:02d}] {category.upper()} -- {description}")

        injected = _apply_injection(CLEAN_DRAFT, field, value)
        draft_insights = _to_actuarial_insights(injected)

        # Layer 1: validate_llm_text keyword scan
        l1_flagged = run_layer1_scan(draft_insights)

        # Layer 2: LLM self-review (if available)
        l2_corrected = False
        if llm_client is not None:
            reviewed = run_layer2_review(llm_client, draft_insights, MINIMAL_SYSTEM_PROMPT, MINIMAL_USER_PROMPT)
            # Check if the violation hint is gone from the reviewed output
            still_present = _has_causal_language(reviewed, detection_hint)
            l2_corrected = not still_present
        else:
            reviewed = draft_insights

        detected = l1_flagged or l2_corrected
        category_counts[category]["total"] += 1
        if detected:
            category_counts[category]["detected"] += 1

        status = "[DETECTED]" if detected else "[MISSED]"
        print(f"  {status}  L1-scan={l1_flagged}  L2-review={l2_corrected}  hint='{detection_hint}'")

        results.append({
            "case_id": case_id,
            "category": category,
            "description": description,
            "field": field,
            "detection_hint": detection_hint,
            "l1_flagged": l1_flagged,
            "l2_corrected": l2_corrected,
            "detected": detected,
        })

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    total = len(results)
    total_detected = sum(1 for r in results if r["detected"])
    print(f"Overall detection rate: {total_detected}/{total} ({100*total_detected/total:.0f}%)")
    for cat, counts in category_counts.items():
        d, t = counts["detected"], counts["total"]
        pct = 100 * d / t if t else 0
        print(f"  {cat:12s}: {d}/{t} ({pct:.0f}%)")

    # Write results JSON
    output_path = ROOT / "data" / "reports" / "injection_test_results.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary = {
        "run_date": datetime.now().isoformat(),
        "llm_model": llm_client.model_version if llm_client else "not_available",
        "overall_detection_rate": total_detected / total,
        "category_rates": {
            cat: round(v["detected"] / v["total"], 3) if v["total"] else 0
            for cat, v in category_counts.items()
        },
        "cases": results,
    }
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to: {output_path}")

    return summary


if __name__ == "__main__":
    run_tests(use_llm=True)
