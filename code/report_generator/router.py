# -*- coding: utf-8 -*-
"""
Report router

Decides which report type to generate based on request and validation context.
"""

from dataclasses import dataclass
from typing import Optional, Literal

from .validation import ValidationResult

ReportType = Literal["policy_report", "spotlight_report", "trend_pack", "validation_diagnostic"]


@dataclass
class ReportRequest:
    report_type: Optional[ReportType] = None
    input_record: Optional[dict] = None
    use_test_data: bool = False
    run_validation: bool = True
    top_n_spotlight: int = 5


@dataclass
class ReportRoute:
    report_type: ReportType
    reason: str
    needs_prediction: bool = False
    needs_llm: bool = True


class ReportRouter:
    def route(self, request: ReportRequest, validation: Optional[ValidationResult] = None) -> ReportRoute:
        """
        Simple deterministic routing:
        - Explicit report_type wins
        - If input_record present -> policy_report
        - Fallback -> validation_diagnostic
        """
        if request.report_type:
            return ReportRoute(
                report_type=request.report_type,
                reason=f"Explicit report_type={request.report_type}",
                needs_prediction=request.report_type == "policy_report",
                needs_llm=request.report_type in ("policy_report", "spotlight_report", "trend_pack"),
            )

        if request.input_record:
            return ReportRoute(
                report_type="policy_report",
                reason="Input record provided",
                needs_prediction=True,
                needs_llm=True,
            )

        return ReportRoute(
            report_type="validation_diagnostic",
            reason="Default fallback",
            needs_prediction=False,
            needs_llm=False,
        )
