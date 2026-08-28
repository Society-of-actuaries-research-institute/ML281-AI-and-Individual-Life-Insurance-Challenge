# -*- coding: utf-8 -*-
"""
Validation utilities

Input validation for policy records and lightweight checks for LLM outputs.
"""

from typing import Any, Dict, List, Optional
from dataclasses import dataclass

from .config import FEATURES, NUMERICAL_FEATURES, CATEGORICAL_FEATURES


@dataclass
class ValidationResult:
    is_valid: bool
    errors: List[str]
    warnings: List[str]
    normalized_record: Optional[Dict[str, Any]] = None


def _is_number(val: Any) -> bool:
    try:
        float(val)
        return True
    except Exception:
        return False


def validate_input_record(record: Dict[str, Any]) -> ValidationResult:
    """
    Validate and normalize a policy input record.

    This is intentionally conservative and does not reject unknown categories,
    but flags them for review.
    """
    errors: List[str] = []
    warnings: List[str] = []

    if not isinstance(record, dict):
        return ValidationResult(False, ["Input record must be a dict"], [], None)

    normalized = dict(record)

    # Required features
    missing = [f for f in FEATURES if f not in record]
    if missing:
        errors.append(f"Missing required features: {', '.join(missing)}")

    # Numeric ranges
    for feat in NUMERICAL_FEATURES:
        if feat not in record:
            continue
        val = record.get(feat)
        if not _is_number(val):
            errors.append(f"{feat} must be numeric")
            continue
        val = float(val)
        normalized[feat] = val
        if feat == "Attained_Age" and not (0 <= val <= 120):
            warnings.append("Attained_Age out of expected range (0-120)")
        if feat == "Issue_Age" and not (0 <= val <= 120):
            warnings.append("Issue_Age out of expected range (0-120)")
        if feat == "Duration" and not (0 <= val <= 100):
            warnings.append("Duration out of expected range (0-100)")

    # Categorical type normalization
    for feat in CATEGORICAL_FEATURES:
        if feat not in record:
            continue
        val = record.get(feat)
        if val is None:
            warnings.append(f"{feat} is null; treating as 'Unknown'")
            normalized[feat] = "Unknown"
        else:
            normalized[feat] = str(val)

    return ValidationResult(len(errors) == 0, errors, warnings, normalized)


def validate_llm_text(text: str) -> ValidationResult:
    """
    Lightweight validation on LLM-generated text for policy/spotlight narratives.
    """
    errors: List[str] = []
    warnings: List[str] = []

    if not text:
        warnings.append("LLM output is empty")
        return ValidationResult(True, errors, warnings, None)

    # Causality red flags (heuristic)
    causal_tokens = ["cause", "causes", "causal", "proves", "prove", "因此", "造成", "導致", "證明"]
    lowered = text.lower()
    if any(tok in lowered for tok in causal_tokens):
        warnings.append("Potential causal language detected; review required")

    return ValidationResult(True, errors, warnings, None)


def merge_validation_notes(*results: ValidationResult) -> List[str]:
    notes: List[str] = []
    for r in results:
        for e in r.errors:
            notes.append(f"ERROR: {e}")
        for w in r.warnings:
            notes.append(f"WARN: {w}")
    return notes
