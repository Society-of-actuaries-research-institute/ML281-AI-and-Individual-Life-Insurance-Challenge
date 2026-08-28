# -*- coding: utf-8 -*-
"""
Run Control Module
==================
Automatically handles all governance controls on every pipeline execution:

  1. LLM Run Log        -- records run_id, input_hash, prompt_version, draft,
                           self_review_result, constraint_check, final output
  2. Narrative Versioning -- never overwrites; writes timestamped copies
  3. Constraint Checker  -- deterministic post-review checks (numbers, disclaimers, scope)
  4. Evidence Mapping    -- links narrative claims back to structured data sources
  5. Approval Gate       -- marks reports as ai_draft / self_reviewed / actuary_approved
  6. Reproducibility Pkg -- bundles all artifacts per run into data/run_logs/<run_id>/
  7. Narrative Diff      -- compares current final output with previous run

Usage (called from pipeline.py):
    rc = RunController(output_dir)
    rc.begin_run(input_record, prompt_version)
    rc.record_draft(draft_insights)
    rc.record_review(reviewed_insights)
    rc.record_constraint_check(check_result)
    rc.record_final(final_report_md, final_report_json)
    diff = rc.finish_run()   # saves run log + reproducibility package + diff
"""

import hashlib
import json
import re
import shutil
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional


# =============================================================================
# Data structures
# =============================================================================

@dataclass
class ConstraintCheckResult:
    """Results of deterministic constraint checking on LLM output."""
    causal_violations: List[str] = field(default_factory=list)
    number_mismatches: List[str] = field(default_factory=list)
    missing_disclaimers: List[str] = field(default_factory=list)
    out_of_scope_claims: List[str] = field(default_factory=list)
    passed: bool = True

    def all_violations(self) -> List[str]:
        return (
            self.causal_violations
            + self.number_mismatches
            + self.missing_disclaimers
            + self.out_of_scope_claims
        )

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class EvidenceMap:
    """Maps key narrative claims back to their structured data sources."""
    mortality_rate_source: Optional[str] = None    # e.g. "predictor.predict() = 0.000691"
    overall_ae_source: Optional[str] = None        # e.g. "kb.overall_ae['overall_ae'] = 0.9989"
    shap_sources: List[str] = field(default_factory=list)  # top SHAP feature -> value
    segment_sources: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class RunLog:
    """Complete audit record for one pipeline execution."""
    run_id: str
    run_timestamp: str
    report_type: str
    input_hash: str
    prompt_version: str
    llm_model_version: str
    approval_status: str                # ai_draft | self_reviewed | actuary_approved
    draft_snapshot: Optional[str]       # JSON string of draft before self-review
    self_review_applied: bool
    constraint_check: Optional[dict]    # ConstraintCheckResult.to_dict()
    evidence_map: Optional[dict]        # EvidenceMap.to_dict()
    final_output_files: List[str]
    total_tokens: Optional[int]
    cost_usd: Optional[float]
    run_duration_sec: Optional[float]
    validation_notes: List[str]

    def to_dict(self) -> dict:
        return asdict(self)


# =============================================================================
# Deterministic Constraint Checker  (v2 — precision-focused)
# =============================================================================

# Causal tokens: context-sensitive, avoid false positives from disclaimers
_CAUSAL_TOKENS = [
    "causes", "causal", "proves", "proven",
    "results in", "leads to", "directly responsible",
    "因此", "造成", "導致", "證明",
]
# "cause" and "prove" need word-boundary checks to avoid "because", "disprove"
_CAUSAL_WORD_TOKENS = ["cause", "prove"]

# Required disclaimer keyword groups — checked against the FULL combined text
_REQUIRED_DISCLAIMER_GROUPS = [
    (["historical data", "2012", "2019"], "data period disclaimer"),
    (["asop", "actuarial standard"], "ASOP compliance disclaimer"),
    (["not a guarantee", "for reference", "reference only", "not intended"], "limitation disclaimer"),
]

# Out-of-scope patterns (refined to avoid false positives from compliant disclaimers)
_OUT_OF_SCOPE_PATTERNS = [
    # COVID reference in the narrative (but not in a data-limitation context)
    (r"\bcovid\b(?!\s+(?:not|excluded|outside|beyond))", "COVID-19 reference detected; model covers 2012-2019 only"),
    # Post-2019 projections (exclude "2012-2019" range mentions which are compliant)
    (r"\b20(?:2[0-9])\b(?!\s*[-–]\s*20(?:1[0-9]))", "Post-2019 year reference; projections outside model scope"),
    # Explicit guarantee claims — NOT triggered by "not a guarantee" disclaimers
    (r"\b(?:provides?|offers?|gives?|is)\s+(?:a\s+)?guarantee", "Explicit guarantee claim; model gives predictions, not guarantees"),
    # Pricing / underwriting decisions (forward-looking) — NOT historical underwriting mentions
    (r"\bpric(?:e|ing)\b(?!\s+(?:band|amount|range|history|effect))",
     "Pricing analysis is outside model scope (model supports mortality experience, not pricing)"),
    (r"\bpremium\b(?!\s+(?:amount|band|payment|history))",
     "Premium/pricing reference detected; model scope is mortality experience analysis only"),
    (r"\bunderwr(?:it(?:e|ing|ten)|iters?)\s+(?:decision|model|criteria|adjustment|recommendation|change)\b",
     "Underwriting decision/model reference; this model does not support underwriting recommendations"),
]

# ASOP number validation: map function → correct ASOP number
_ASOP_CORRECT = {
    "data": 23, "credibility": 25, "communication": 41, "modeling": 56,
}
_ASOP_WRONG_PATTERNS = [
    # Pattern: "ASOP <number>" where the number is neither 23/25/41/56
    (re.compile(r'\basop\s+(\d+)', re.IGNORECASE), {23, 25, 41, 56},
     "Unrecognized ASOP number {n} — valid actuarial references are ASOP 23/25/41/56"),
]

# Schema / format constraints applied to narrative text
_MARKDOWN_FENCE = re.compile(r'```')
_DEFINITIVE_FUTURE = re.compile(
    r'\b(?:will\s+(?:perform|be\s+better|outperform|show|result|increase|decrease|confirm|guarantee))',
    re.IGNORECASE,
)
_VAGUE_ACTION = re.compile(
    r'\b(?:things?\s+should\s+be\s+done|something\s+should\s+(?:be\s+)?done|action\s+(?:should\s+be\s+)?taken)\b',
    re.IGNORECASE,
)


def run_constraint_check(
    text: str,
    context_numbers: Optional[Dict[str, float]] = None,
    n_recommendations: Optional[int] = None,
) -> ConstraintCheckResult:
    """
    Deterministic constraint check on LLM-generated text (v2).

    Args:
        text: Combined LLM narrative + disclaimer text to check.
        context_numbers: {label: value} — numbers the LLM must cite accurately.
                         e.g. {"mortality_rate": 0.000691, "overall_ae": 0.9989}
        n_recommendations: If provided, flags if this exceeds 3.

    Returns:
        ConstraintCheckResult with per-category violation lists.
    """
    result = ConstraintCheckResult()
    lowered = text.lower()

    # ------------------------------------------------------------------
    # 1. Causal language (phrase-level and word-boundary tokens)
    # ------------------------------------------------------------------
    for tok in _CAUSAL_TOKENS:
        if tok.lower() in lowered:
            result.causal_violations.append(f"Causal token: '{tok}'")

    for tok in _CAUSAL_WORD_TOKENS:
        # Use word boundary to avoid "because", "disprove"
        if re.search(rf'\b{tok}\b', lowered):
            result.causal_violations.append(f"Causal word (boundary-matched): '{tok}'")

    # ------------------------------------------------------------------
    # 2. Number accuracy (two complementary checks)
    # ------------------------------------------------------------------
    if context_numbers:
        numbers_in_text = set(float(m) for m in re.findall(r'\b\d+\.\d+\b', text))

        for label, expected in context_numbers.items():
            tolerance = max(abs(expected) * 0.005, 1e-7)        # ±0.5% OK
            proximity_band = max(abs(expected) * 0.50, 1e-6)    # ±50% suspicious zone

            # Check A: close-but-wrong numbers (±0.5%–±50% of expected)
            suspicious_close = [
                n for n in numbers_in_text
                if tolerance < abs(n - expected) <= proximity_band
            ]
            if suspicious_close:
                result.number_mismatches.append(
                    f"{label}: expected ~{expected}, "
                    f"suspicious neighbor(s) in text: {[round(s, 6) for s in suspicious_close[:3]]}"
                )

            # Check B: context number not found AND no number in the proximity band
            # (suggests the value was simply fabricated with a wrong order of magnitude)
            exact_found = any(abs(n - expected) <= tolerance for n in numbers_in_text)
            any_neighbor = bool(suspicious_close)
            if not exact_found and not any_neighbor and numbers_in_text:
                # Numbers exist in text but none are near expected — check if there are
                # numbers in text that are in the same order of magnitude (within 10x)
                same_magnitude = [n for n in numbers_in_text
                                  if 0.1 * expected <= n <= 10 * expected]
                if same_magnitude:
                    result.number_mismatches.append(
                        f"{label}: expected ~{expected}, "
                        f"same-magnitude values in text {[round(n, 6) for n in same_magnitude[:3]]} "
                        f"do not match (possible fabrication)"
                    )

    # ------------------------------------------------------------------
    # 3. ASOP number validation
    # ------------------------------------------------------------------
    for pattern, valid_set, msg_template in _ASOP_WRONG_PATTERNS:
        for m in pattern.finditer(lowered):
            n = int(m.group(1))
            if n not in valid_set:
                result.number_mismatches.append(
                    msg_template.format(n=n)
                )

    # ------------------------------------------------------------------
    # 4. Missing disclaimers (full combined text, including fixed sections)
    # ------------------------------------------------------------------
    for keywords, label in _REQUIRED_DISCLAIMER_GROUPS:
        if not any(kw.lower() in lowered for kw in keywords):
            result.missing_disclaimers.append(f"Missing {label}")

    # ------------------------------------------------------------------
    # 5. Out-of-scope claims (refined patterns)
    # ------------------------------------------------------------------
    for pattern, message in _OUT_OF_SCOPE_PATTERNS:
        if re.search(pattern, lowered):
            result.out_of_scope_claims.append(message)

    # ------------------------------------------------------------------
    # 6. Schema / format constraints
    # ------------------------------------------------------------------
    if _MARKDOWN_FENCE.search(text):
        result.out_of_scope_claims.append(
            "Markdown code fence detected in narrative; output must be plain text"
        )

    if _DEFINITIVE_FUTURE.search(text):
        result.out_of_scope_claims.append(
            "Definitive future-tense claim detected (e.g., 'will perform better'); "
            "use conditional language: 'may', 'might', 'could'"
        )

    if _VAGUE_ACTION.search(text):
        result.out_of_scope_claims.append(
            "Vague non-actionable recommendation detected; "
            "recommendations must specify who does what by when"
        )

    if n_recommendations is not None and n_recommendations > 3:
        result.out_of_scope_claims.append(
            f"Too many recommendations: {n_recommendations} (maximum allowed: 3)"
        )

    # ------------------------------------------------------------------
    # 7. Key number citation check (summary must reference key numbers)
    # ------------------------------------------------------------------
    if context_numbers:
        numbers_in_text = set(float(m) for m in re.findall(r'\b\d+\.\d+\b', text))
        for label, expected in context_numbers.items():
            tolerance = max(abs(expected) * 0.005, 1e-7)
            exact_found = any(abs(n - expected) <= tolerance for n in numbers_in_text)
            if not exact_found:
                result.number_mismatches.append(
                    f"{label} ({expected}) not found in text — "
                    f"key context numbers must be explicitly cited"
                )

    result.passed = not result.all_violations()
    return result


# =============================================================================
# Evidence Mapper
# =============================================================================

def build_evidence_map(data, insights) -> EvidenceMap:
    """
    Build EvidenceMap linking narrative claims to structured data sources.

    Works with ReportData (S3) only; returns empty map for other report types.
    """
    em = EvidenceMap()

    # Mortality rate source
    pred = getattr(data, "prediction", None)
    if pred is not None:
        em.mortality_rate_source = (
            f"predictor.predict() = {pred.mortality_rate:.6f} "
            f"({'higher' if pred.interpretation == 'higher' else 'lower'} than avg "
            f"{pred.overall_death_rate:.6f})"
        )

    # Overall A/E source
    overall_ae = getattr(data, "overall_ae", None)
    if overall_ae is not None:
        em.overall_ae_source = (
            f"kb.overall_ae = {overall_ae:.4f} "
            f"(Death_Count / ExpDth_VBT2015_Cnt; portfolio vs VBT2015)"
        )

    # SHAP sources
    shap_contribs = getattr(data, "shap_contributions", [])
    for s in shap_contribs[:5]:
        em.shap_sources.append(
            f"{s.feature}={s.value}: SHAP={s.shap_value:+.4f} ({s.direction})"
        )

    # Segment sources
    cov = getattr(data, "coverage_segment", None)
    if cov is not None:
        em.segment_sources.append(
            f"Coverage: {cov.segment_id} A/E={cov.ae_ratio:.3f} "
            f"credibility={cov.credibility}"
        )
    spot = getattr(data, "spotlight_segment", None)
    if spot is not None:
        em.segment_sources.append(
            f"Spotlight: {spot.segment_id} A/E={spot.ae_ratio:.3f} "
            f"risk={spot.risk_level}"
        )

    return em


# =============================================================================
# Narrative Diff
# =============================================================================

def _extract_numbers(text: str) -> List[float]:
    """Extract all numeric values from text."""
    return [float(m) for m in re.findall(r'-?\d+\.\d+', text)]


def _detect_causal_language(text: str) -> List[str]:
    return [tok for tok in _CAUSAL_TOKENS if tok.lower() in text.lower()]


def generate_narrative_diff(prev_text: str, curr_text: str) -> Dict[str, Any]:
    """
    Compare two versions of report narrative text.

    Returns a dict summarizing:
    - Added / removed lines
    - Number changes (values that changed between versions)
    - Causal language changes
    - Disclaimer changes
    """
    if not prev_text or not curr_text:
        return {"status": "no_previous_version", "diff": None}

    prev_lines = set(prev_text.splitlines())
    curr_lines = set(curr_text.splitlines())

    added = [l for l in curr_lines - prev_lines if l.strip()]
    removed = [l for l in prev_lines - curr_lines if l.strip()]

    prev_nums = set(round(n, 6) for n in _extract_numbers(prev_text))
    curr_nums = set(round(n, 6) for n in _extract_numbers(curr_text))
    new_numbers = sorted(curr_nums - prev_nums)
    removed_numbers = sorted(prev_nums - curr_nums)

    prev_causal = set(_detect_causal_language(prev_text))
    curr_causal = set(_detect_causal_language(curr_text))

    # Disclaimer keywords present/absent
    disclaimer_kws = ["asop", "historical data", "not a guarantee", "reference only"]
    disclaimer_changes = {}
    for kw in disclaimer_kws:
        was_present = kw.lower() in prev_text.lower()
        is_present = kw.lower() in curr_text.lower()
        if was_present != is_present:
            disclaimer_changes[kw] = "added" if is_present else "removed"

    return {
        "status": "diff_computed",
        "lines_added": len(added),
        "lines_removed": len(removed),
        "sample_added": added[:5],
        "sample_removed": removed[:5],
        "numbers_added": new_numbers[:10],
        "numbers_removed": removed_numbers[:10],
        "causal_language_added": sorted(curr_causal - prev_causal),
        "causal_language_removed": sorted(prev_causal - curr_causal),
        "disclaimer_changes": disclaimer_changes,
        "risk_level_changed": (
            ("higher" in curr_text.lower()) != ("higher" in prev_text.lower())
        ),
    }


# =============================================================================
# Run Controller
# =============================================================================

class RunController:
    """
    Central controller that automatically executes all governance controls
    on every pipeline run.
    """

    def __init__(self, output_dir: Path, report_type: str = "unknown"):
        self.output_dir = output_dir
        self.report_type = report_type
        self.run_id = self._generate_run_id()
        self.run_dir = output_dir / "run_logs" / self.run_id
        self.run_dir.mkdir(parents=True, exist_ok=True)

        self._start_time = datetime.now()
        self._input_hash: str = "unknown"
        self._prompt_version: str = "unknown"
        self._llm_model_version: str = "unknown"
        self._draft_snapshot: Optional[str] = None
        self._self_review_applied: bool = False
        self._constraint_check: Optional[ConstraintCheckResult] = None
        self._evidence_map: Optional[EvidenceMap] = None
        self._final_files: List[str] = []
        self._total_tokens: Optional[int] = None
        self._cost_usd: Optional[float] = None
        self._validation_notes: List[str] = []
        self._final_md: str = ""

    # ------------------------------------------------------------------
    # Run ID & Hashing
    # ------------------------------------------------------------------

    def _generate_run_id(self) -> str:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        rnd = hashlib.sha256(ts.encode()).hexdigest()[:6]
        return f"{ts}_{rnd}"

    @staticmethod
    def compute_input_hash(input_record: dict) -> str:
        """Stable SHA-256 hash of canonical input JSON."""
        canonical = json.dumps(input_record, sort_keys=True, ensure_ascii=True)
        return hashlib.sha256(canonical.encode()).hexdigest()[:16]

    @staticmethod
    def compute_prompt_version(system_prompt: str) -> str:
        """Short hash of the system prompt content (detects prompt changes)."""
        return hashlib.sha256(system_prompt.encode()).hexdigest()[:8]

    # ------------------------------------------------------------------
    # Step recording (called from pipeline.py)
    # ------------------------------------------------------------------

    def begin_run(
        self,
        input_record: dict,
        system_prompt: str,
        llm_model_version: str,
    ):
        self._input_hash = self.compute_input_hash(input_record)
        self._prompt_version = self.compute_prompt_version(system_prompt)
        self._llm_model_version = llm_model_version

        # Save input record
        (self.run_dir / "input_record.json").write_text(
            json.dumps(input_record, indent=2, ensure_ascii=False), encoding="utf-8"
        )

    def record_draft(self, draft_insights):
        """Save the LLM draft before self-review."""
        self._self_review_applied = True
        try:
            if hasattr(draft_insights, "model_dump"):
                snapshot = json.dumps(draft_insights.model_dump(), indent=2, ensure_ascii=False)
            else:
                snapshot = json.dumps(draft_insights, indent=2, ensure_ascii=False)
        except Exception:
            snapshot = str(draft_insights)
        self._draft_snapshot = snapshot
        (self.run_dir / "draft_insights.json").write_text(snapshot, encoding="utf-8")

    def record_reviewed(self, reviewed_insights):
        """Save the reviewed (post-self-review) insights."""
        try:
            if hasattr(reviewed_insights, "model_dump"):
                content = json.dumps(reviewed_insights.model_dump(), indent=2, ensure_ascii=False)
            else:
                content = json.dumps(reviewed_insights, indent=2, ensure_ascii=False)
        except Exception:
            content = str(reviewed_insights)
        (self.run_dir / "reviewed_insights.json").write_text(content, encoding="utf-8")

    def record_constraint_check(
        self,
        text: str,
        context_numbers: Optional[Dict[str, float]] = None,
        full_report_text: str = "",
        n_recommendations: Optional[int] = None,
    ) -> ConstraintCheckResult:
        """
        Run deterministic constraint check and save result.
        full_report_text is used for disclaimer checks (includes fixed sections).
        n_recommendations: pass len(insights.recommendations) to enforce max-3 rule.
        """
        combined = text + "\n" + full_report_text
        result = run_constraint_check(combined, context_numbers, n_recommendations=n_recommendations)
        self._constraint_check = result
        (self.run_dir / "constraint_check.json").write_text(
            json.dumps(result.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8"
        )
        return result

    def record_evidence_map(self, data, insights):
        """Build and save evidence map linking narrative to structured sources."""
        em = build_evidence_map(data, insights)
        self._evidence_map = em
        (self.run_dir / "evidence_map.json").write_text(
            json.dumps(em.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8"
        )
        return em

    def record_usage(self, total_tokens: int, cost_usd: float):
        self._total_tokens = total_tokens
        self._cost_usd = cost_usd

    def record_validation_notes(self, notes: List[str]):
        self._validation_notes = notes

    def record_final_output(
        self,
        report_md: str,
        report_json: str,
        system_prompt: str,
        shap_values: Optional[dict] = None,
        llm_raw_response: Optional[str] = None,
    ):
        """
        Save all final artifacts to the run directory (reproducibility package).
        Uses versioned filenames — never overwrites.
        """
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Final report (versioned)
        md_path = self.run_dir / f"final_report_{ts}.md"
        md_path.write_text(report_md, encoding="utf-8")
        self._final_md = report_md
        self._final_files.append(str(md_path))

        json_path = self.run_dir / f"final_report_{ts}.json"
        json_path.write_text(report_json, encoding="utf-8")
        self._final_files.append(str(json_path))

        # System prompt snapshot
        prompt_path = self.run_dir / "system_prompt.txt"
        prompt_path.write_text(system_prompt, encoding="utf-8")

        # SHAP values
        if shap_values:
            shap_path = self.run_dir / "shap_values.json"
            shap_path.write_text(
                json.dumps(shap_values, indent=2, ensure_ascii=False), encoding="utf-8"
            )
            self._final_files.append(str(shap_path))

        # LLM raw response (before structured output parsing)
        if llm_raw_response:
            raw_path = self.run_dir / "llm_raw_response.txt"
            raw_path.write_text(llm_raw_response, encoding="utf-8")

        # Approval gate stub (default: ai_draft)
        approval_path = self.run_dir / "approval_gate.json"
        approval_gate = {
            "run_id": self.run_id,
            "approval_status": "ai_draft",
            "approved_by": None,
            "approval_timestamp": None,
            "note": (
                "This report has NOT been reviewed by a qualified actuary. "
                "It must not be used as a formal actuarial opinion or for pricing/reserving decisions."
            ),
        }
        approval_path.write_text(
            json.dumps(approval_gate, indent=2, ensure_ascii=False), encoding="utf-8"
        )

    # ------------------------------------------------------------------
    # Narrative diff against previous run
    # ------------------------------------------------------------------

    def _find_previous_run_md(self) -> Optional[str]:
        """Find the most recent final_report_*.md from a previous run."""
        run_logs_dir = self.output_dir / "run_logs"
        if not run_logs_dir.exists():
            return None

        candidates = []
        for run_subdir in sorted(run_logs_dir.iterdir(), reverse=True):
            if run_subdir.name == self.run_id:
                continue
            for f in sorted(run_subdir.glob("final_report_*.md"), reverse=True):
                candidates.append(f)
                break
        if not candidates:
            return None
        try:
            return candidates[0].read_text(encoding="utf-8")
        except Exception:
            return None

    # ------------------------------------------------------------------
    # Finish run: write run log + narrative diff
    # ------------------------------------------------------------------

    def finish_run(self) -> Dict[str, Any]:
        """
        Called at the end of every pipeline run.
        Writes run_log.json, computes narrative diff, returns summary.
        """
        duration = (datetime.now() - self._start_time).total_seconds()

        run_log = RunLog(
            run_id=self.run_id,
            run_timestamp=self._start_time.isoformat(),
            report_type=self.report_type,
            input_hash=self._input_hash,
            prompt_version=self._prompt_version,
            llm_model_version=self._llm_model_version,
            approval_status="ai_draft",
            draft_snapshot=self._draft_snapshot,
            self_review_applied=self._self_review_applied,
            constraint_check=self._constraint_check.to_dict() if self._constraint_check else None,
            evidence_map=self._evidence_map.to_dict() if self._evidence_map else None,
            final_output_files=self._final_files,
            total_tokens=self._total_tokens,
            cost_usd=self._cost_usd,
            run_duration_sec=round(duration, 2),
            validation_notes=self._validation_notes,
        )

        (self.run_dir / "run_log.json").write_text(
            json.dumps(run_log.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8"
        )

        # Ensure approval gate exists even if record_final_output() was not called
        gate_path = self.run_dir / "approval_gate.json"
        if not gate_path.exists():
            approval_gate = {
                "run_id": self.run_id,
                "approval_status": "ai_draft",
                "approved_by": None,
                "approval_timestamp": None,
                "note": (
                    "This report has NOT been reviewed by a qualified actuary. "
                    "It must not be used as a formal actuarial opinion or for pricing/reserving decisions."
                ),
            }
            gate_path.write_text(
                json.dumps(approval_gate, indent=2, ensure_ascii=False), encoding="utf-8"
            )

        # Narrative diff
        prev_md = self._find_previous_run_md()
        diff = generate_narrative_diff(prev_md or "", self._final_md)
        (self.run_dir / "narrative_diff.json").write_text(
            json.dumps(diff, indent=2, ensure_ascii=False), encoding="utf-8"
        )

        return {
            "run_id": self.run_id,
            "run_dir": str(self.run_dir),
            "constraint_passed": self._constraint_check.passed if self._constraint_check else None,
            "approval_status": "ai_draft",
            "narrative_diff": diff,
        }


# =============================================================================
# Approval Gate (standalone utility — called separately by actuary)
# =============================================================================

def approve_run(
    output_dir: Path,
    run_id: str,
    approver_id: str,
    notes: str = "",
) -> Dict[str, Any]:
    """
    Mark a run as actuary-approved.
    Must be called explicitly by the actuary — never called automatically.

    Args:
        output_dir: Base reports directory.
        run_id: The run to approve.
        approver_id: Actuary identifier (e.g., "JL/FSA").
        notes: Optional review notes.

    Returns:
        Updated approval gate dict.
    """
    gate_path = output_dir / "run_logs" / run_id / "approval_gate.json"
    if not gate_path.exists():
        raise FileNotFoundError(f"Run {run_id} not found at {gate_path}")

    gate = json.loads(gate_path.read_text(encoding="utf-8"))
    if gate["approval_status"] == "actuary_approved":
        raise ValueError(f"Run {run_id} is already approved by {gate['approved_by']}")

    gate["approval_status"] = "actuary_approved"
    gate["approved_by"] = approver_id
    gate["approval_timestamp"] = datetime.now().isoformat()
    gate["note"] = notes or "Reviewed and approved by qualified actuary."

    gate_path.write_text(json.dumps(gate, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[OK] Run {run_id} approved by {approver_id}")
    return gate
