# -*- coding: utf-8 -*-
"""
Unified report pipeline (router + validation + generation).

Every run automatically executes all governance controls via RunController:
  - LLM run log (run_id, input_hash, prompt_version, draft, review, final)
  - Narrative versioning (timestamped, never overwrites)
  - Deterministic constraint check (causal language, numbers, disclaimers, scope)
  - Evidence mapping (narrative -> structured data sources)
  - Human approval gate (default: ai_draft)
  - Reproducibility package (all artifacts bundled per run)
  - Narrative diff (auto-compared with previous run)
"""

import json
from pathlib import Path
from typing import Optional, Dict, Any

from .router import ReportRouter, ReportRequest
from .validation import validate_input_record, validate_llm_text, merge_validation_notes
from .knowledge import get_knowledge_base
from .predictor import create_predictor
from .prompt_builder import get_full_system_prompt
from .llm import create_llm_client, LLMUnavailableError
from .renderer import create_renderer
from .run_control import RunController, build_evidence_map
from .schemas import (
    FullReport,
    PromptContext,
    SpotlightPromptContext,
    ActuarialInsights,
    SpotlightInsights,
    ExperienceTrendPackInsights,
)
from .policy_report import build_policy_report_data, load_sample
from .spotlight_report import build_spotlight_data
from .diagnostic_report import build_validation_diagnostic_data
from .demo_trend_pack import main as run_trend_pack


class ReportPipeline:
    def __init__(self, output_dir: Optional[str] = None):
        self.output_dir = Path(output_dir) if output_dir else Path(__file__).parent.parent.parent / "data" / "reports"
        self.router = ReportRouter()
        self.kb = get_knowledge_base()
        self.predictor = create_predictor()
        self.llm = create_llm_client()
        self.renderer = create_renderer(str(self.output_dir))
        self.system_prompt = get_full_system_prompt()

    def run(self, request: ReportRequest) -> Dict[str, Any]:
        validation = None
        if request.run_validation and request.input_record:
            validation = validate_input_record(request.input_record)

        route = self.router.route(request, validation)

        if route.report_type == "policy_report":
            return self._run_policy_report(request, validation)

        if route.report_type == "spotlight_report":
            return self._run_spotlight_report(request)

        if route.report_type == "trend_pack":
            run_trend_pack()
            return {"report_type": "trend_pack", "paths": {"markdown": "data/reports/experience_trend_pack.md"}}

        return self._run_validation_diagnostic()

    def _run_policy_report(self, request: ReportRequest, validation):
        if request.input_record:
            sample = request.input_record
            df = load_sample(use_test_data=False)[1]
        else:
            sample, df = load_sample(use_test_data=request.use_test_data)
        if sample is None or df is None:
            return {"error": "Missing data for policy report"}

        if validation and validation.normalized_record:
            sample = validation.normalized_record

        # --- Governance: begin run ---
        rc = RunController(self.output_dir, report_type="policy_report")
        rc.begin_run(sample, self.system_prompt, self.llm.model_version)

        data = build_policy_report_data(sample, df, self.predictor, self.kb)

        context = PromptContext(
            prediction=data.prediction,
            shap_contributions=data.shap_contributions,
            percentiles=data.percentile_context,
            categories=data.category_context,
            coverage_segment=data.coverage_segment.model_dump() if data.coverage_segment else None,
            spotlight_segment=data.spotlight_segment.model_dump() if data.spotlight_segment else None,
            overall_ae=data.overall_ae,
        )

        if not self.llm.is_available():
            return {"error": "LLM not available for policy report. Check API key configuration."}

        # Generate draft
        insights, user_prompt = self.llm.generate_insights(
            context, self.system_prompt, output_schema=ActuarialInsights
        )
        # Note: generate_insights already runs self-review internally.
        # We treat the returned insights as the self-reviewed version.
        rc.record_draft(insights)   # records what came back (post-self-review)
        rc.record_reviewed(insights)

        # --- Governance: deterministic constraint check ---
        full_narrative = " ".join([
            getattr(insights, "summary", ""),
            getattr(insights, "risk_interpretation", ""),
            getattr(insights, "calibration_note", "") or "",
            " ".join(getattr(insights, "recommendations", [])),
        ])
        context_numbers = {
            "mortality_rate": data.prediction.mortality_rate,
            "overall_ae": data.overall_ae,
        }
        # Include fixed disclaimer text in the check
        disclaimer_text = " ".join(data.disclaimers)
        check_result = rc.record_constraint_check(
            full_narrative,
            context_numbers=context_numbers,
            full_report_text=disclaimer_text,
            n_recommendations=len(getattr(insights, "recommendations", []) or []),
        )

        # --- Governance: evidence map ---
        rc.record_evidence_map(data, insights)

        validation_notes = []
        if validation:
            validation_notes.extend(merge_validation_notes(validation))
        validation_notes.extend(
            merge_validation_notes(validate_llm_text(getattr(insights, "summary", "")))
        )
        # Add constraint violations to validation notes
        for v in check_result.all_violations():
            validation_notes.append(f"CONSTRAINT: {v}")

        rc.record_validation_notes(validation_notes)

        total_tokens = self.llm.usage.total_tokens
        total_cost = self.llm.usage.cost_usd(self.llm.backend)
        rc.record_usage(total_tokens, total_cost)

        report = FullReport(
            data=data,
            insights=insights,
            system_prompt=self.system_prompt,
            user_prompt=user_prompt,
            timing_summary=[],
            total_cost=total_cost,
            total_tokens=total_tokens,
            is_validated=len([n for n in validation_notes if n.startswith("ERROR")]) == 0,
            validation_notes=validation_notes,
            llm_model_version=self.llm.model_version,
            run_id=rc.run_id,
            input_hash=rc._input_hash,
            prompt_version=rc._prompt_version,
            approval_status="ai_draft",
            constraint_check_passed=check_result.passed,
            constraint_violations=check_result.all_violations(),
        )

        md_path = self.renderer.save_markdown(report, "actuarial_report.md")
        json_path = self.renderer.save_json(report, "actuarial_report.json")
        prompt_path = self.renderer.save_prompt(report, "system_prompt_appendix.txt")

        # --- Governance: record final + finish ---
        rc.record_final_output(
            report_md=md_path.read_text(encoding="utf-8") if md_path.exists() else "",
            report_json=json_path.read_text(encoding="utf-8") if json_path.exists() else "",
            system_prompt=self.system_prompt,
            shap_values={s.feature: s.shap_value for s in data.shap_contributions},
        )
        run_summary = rc.finish_run()

        result = {
            "report_type": "policy_report",
            "paths": {
                "markdown": str(md_path),
                "json": str(json_path),
                "prompt": str(prompt_path),
            },
            "run_governance": run_summary,
        }
        if not check_result.passed:
            result["constraint_warnings"] = check_result.all_violations()
        return result

    def _run_spotlight_report(self, request: ReportRequest):
        data = build_spotlight_data(self.kb, top_n=request.top_n_spotlight)
        context = SpotlightPromptContext(
            exception_definition=data.exception_definition,
            top_anomalies=data.top_anomalies,
        )

        if not self.llm.is_available():
            return {"error": "LLM not available for spotlight report. Check API key configuration."}

        # --- Governance: begin run (use exception_definition as proxy input) ---
        rc = RunController(self.output_dir, report_type="spotlight_report")
        input_proxy = {
            "exception_threshold": data.exception_definition.ae_deviation_threshold,
            "min_exposure": data.exception_definition.min_exposure,
            "top_n": len(data.top_anomalies),
            "segment_ids": [a.segment_id for a in data.top_anomalies],
        }
        rc.begin_run(input_proxy, self.system_prompt, self.llm.model_version)

        insights, user_prompt = self.llm.generate_insights(
            context, self.system_prompt, output_schema=SpotlightInsights
        )
        rc.record_draft(insights)
        rc.record_reviewed(insights)

        # Constraint check on spotlight narrative
        narrative = getattr(insights, "top_anomalies_overview", "")
        _SPOTLIGHT_DISCLAIMER = (
            "This analysis is based on historical data from 2012-2019 and is for reference only. "
            "This model does not include data from periods outside the 2012-2019 training window; "
            "predictions outside this window require external adjustment. "
            "This report complies with ASOP 23/25/41/56 standards."
        )
        check_result = rc.record_constraint_check(narrative, full_report_text=_SPOTLIGHT_DISCLAIMER)

        validation_notes = merge_validation_notes(validate_llm_text(narrative))
        for v in check_result.all_violations():
            validation_notes.append(f"CONSTRAINT: {v}")
        rc.record_validation_notes(validation_notes)

        total_tokens = self.llm.usage.total_tokens
        total_cost = self.llm.usage.cost_usd(self.llm.backend)
        rc.record_usage(total_tokens, total_cost)

        report = FullReport(
            data=data,
            insights=insights,
            system_prompt=self.system_prompt,
            user_prompt=user_prompt,
            timing_summary=[],
            total_cost=total_cost,
            total_tokens=total_tokens,
            is_validated=len([n for n in validation_notes if n.startswith("ERROR")]) == 0,
            validation_notes=validation_notes,
            llm_model_version=self.llm.model_version,
            run_id=rc.run_id,
            input_hash=rc._input_hash,
            prompt_version=rc._prompt_version,
            approval_status="ai_draft",
            constraint_check_passed=check_result.passed,
            constraint_violations=check_result.all_violations(),
        )

        md_path = self.renderer.save_markdown(report, "spotlight_report.md")
        json_path = self.renderer.save_json(report, "spotlight_report.json")
        prompt_path = self.renderer.save_prompt(report, "spotlight_system_prompt_appendix.txt")

        rc.record_final_output(
            report_md=md_path.read_text(encoding="utf-8") if md_path.exists() else "",
            report_json=json_path.read_text(encoding="utf-8") if json_path.exists() else "",
            system_prompt=self.system_prompt,
        )
        run_summary = rc.finish_run()

        result = {
            "report_type": "spotlight_report",
            "paths": {
                "markdown": str(md_path),
                "json": str(json_path),
                "prompt": str(prompt_path),
            },
            "run_governance": run_summary,
        }
        if not check_result.passed:
            result["constraint_warnings"] = check_result.all_violations()
        return result

    def _run_validation_diagnostic(self):
        data = build_validation_diagnostic_data(self.kb)

        # --- Governance: begin run ---
        rc = RunController(self.output_dir, report_type="validation_diagnostic")
        input_proxy = {
            "report_type": "validation_diagnostic",
            "calibration_period": str(getattr(data, "calibration_overall", {}).get("data_period", "unknown")),
        }
        rc.begin_run(input_proxy, self.system_prompt, self.llm.model_version)

        insights: Any = {}
        user_prompt = ""
        if self.llm.is_available():
            try:
                overall_ae = data.calibration_overall
                yearly_ae = data.calibration_yearly
                vd_user_prompt = self.llm._build_validation_context_prompt(overall_ae, yearly_ae)
                insights, user_prompt = self.llm.generate_json_with_review(
                    system_prompt=self.system_prompt,
                    user_prompt=vd_user_prompt,
                    output_schema=ExperienceTrendPackInsights,
                )
                rc.record_draft(insights)
                rc.record_reviewed(insights)
            except Exception as e:
                print(f"[WARN] Validation diagnostic LLM call failed: {e}")
                insights = {}

        # Constraint check on executive summary
        exec_summary = getattr(insights, "executive_summary", "") if insights else ""
        _VD_DISCLAIMER = (
            "This analysis is based on historical data from 2012-2019 and is for reference only. "
            "This model does not include data from periods outside the 2012-2019 training window; "
            "predictions outside this window require external adjustment. "
            "This report complies with ASOP 23/25/41/56 standards."
        )
        check_result = rc.record_constraint_check(exec_summary, full_report_text=_VD_DISCLAIMER)

        validation_notes = []
        for v in check_result.all_violations():
            validation_notes.append(f"CONSTRAINT: {v}")
        rc.record_validation_notes(validation_notes)

        total_tokens = self.llm.usage.total_tokens
        total_cost = self.llm.usage.cost_usd(self.llm.backend)
        rc.record_usage(total_tokens, total_cost)

        report = FullReport(
            data=data,
            insights=insights if insights is not None else {},
            system_prompt=self.system_prompt,
            user_prompt=user_prompt,
            timing_summary=[],
            total_cost=total_cost,
            total_tokens=total_tokens,
            is_validated=True,
            llm_model_version=self.llm.model_version,
            run_id=rc.run_id,
            input_hash=rc._input_hash,
            prompt_version=rc._prompt_version,
            approval_status="ai_draft",
            constraint_check_passed=check_result.passed,
            constraint_violations=check_result.all_violations(),
        )

        md_path = self.renderer.save_markdown(report, "validation_diagnostic.md")
        json_path = self.renderer.save_json(report, "validation_diagnostic.json")

        rc.record_final_output(
            report_md=md_path.read_text(encoding="utf-8") if md_path.exists() else "",
            report_json=json_path.read_text(encoding="utf-8") if json_path.exists() else "",
            system_prompt=self.system_prompt,
        )
        run_summary = rc.finish_run()

        result = {
            "report_type": "validation_diagnostic",
            "paths": {"markdown": str(md_path), "json": str(json_path)},
            "run_governance": run_summary,
        }
        if not check_result.passed:
            result["constraint_warnings"] = check_result.all_violations()
        return result
