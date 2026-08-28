# -*- coding: utf-8 -*-
"""
Pydantic Schemas for Structured Report Output

Defines the exact structure for LLM-generated insights and fixed report data.
"""

from typing import List, Optional, Dict, Any, Literal, Union
from datetime import date

# Check for pydantic
try:
    from pydantic import BaseModel, Field
    HAS_PYDANTIC = True
except ImportError:
    raise ImportError(
        "pydantic is required for report generation. "
        "Install with: pip install pydantic"
    )


# =============================================================================
# LLM-Generated Insights (Structured Output)
# =============================================================================

class SHAPExplanation(BaseModel):
    """Explanation for a single SHAP feature contribution."""
    feature: str = Field(description="Feature name")
    value: str = Field(description="Feature value for this policy")
    direction: str = Field(description="Direction of mortality impact: Increases or Decreases")
    explanation: str = Field(description="One-sentence explanation of this feature's mortality impact")


class ActuarialInsights(BaseModel):
    """
    LLM-generated interpretive content only.

    Defines the scope of content the LLM may produce.
    Fixed data (tables, charts, disclaimers) are not included here.
    """

    # Executive Summary (2-3 sentences)
    summary: str = Field(description="2-3 sentence summary: prediction result, key risk drivers, risk level")

    # Risk Interpretation
    risk_interpretation: str = Field(description="Explanation of this policy's risk level; must cite the A/E Ratio value")

    # SHAP Explanations (top 5 features)
    shap_explanations: List[SHAPExplanation] = Field(description="Interpretations of the top 5 most important SHAP features")

    # Model Calibration Interpretation
    calibration_note: str = Field(description="Explanation of A/E Ratio meaning and model calibration status")

    # Segment-Specific Insight
    segment_insight: Optional[str] = Field(default=None, description="If a Spotlight segment applies, describe its risk characteristics")

    # Recommendations (max 3)
    recommendations: List[str] = Field(description="Up to 3 actionable recommendations based on the analysis")


# =============================================================================
# Fixed Report Data (Not LLM-generated)
# =============================================================================

class PredictionData(BaseModel):
    """固定的預測數據。"""
    mortality_rate: float
    overall_death_rate: float
    interpretation: str


class SHAPData(BaseModel):
    """固定的 SHAP 數據。"""
    feature: str
    value: Any
    shap_value: float
    direction: str


class SegmentData(BaseModel):
    """固定的區段數據。"""
    segment_id: str
    rule_text: str
    ae_ratio: float
    credibility: str
    risk_level: Optional[str] = None


class EvidenceChart(BaseModel):
    """證據圖表引用。"""
    name: str
    path: str
    description: str


class ReportData(BaseModel):
    """
    固定的報告數據（非 LLM 生成）。
    包含所有數據表格、圖表路徑、免責聲明。
    """
    
    # Metadata
    report_date: str
    report_version: str = "1.0"
    
    # Input Record
    input_record: Dict[str, Any]
    
    # Prediction
    prediction: PredictionData
    
    # SHAP (top 5)
    shap_contributions: List[SHAPData]
    
    # Population Context
    percentile_context: List[Dict[str, Any]]
    category_context: List[Dict[str, Any]]
    
    # Segments
    coverage_segment: Optional[SegmentData] = None
    spotlight_segment: Optional[SegmentData] = None
    
    # Calibration
    overall_ae: float
    yearly_ae: Optional[Dict[str, float]] = None
    
    # Evidence Charts (paths for PDF inclusion)
    evidence_charts: List[EvidenceChart] = Field(default_factory=list)
    
    # Fixed Disclaimers
    disclaimers: List[str]

# =============================================================================
# Prompt Context (Input to LLM)
# =============================================================================

class PromptContext(BaseModel):
    """
    傳給 LLM 的上下文資訊。
    LLM 根據此資訊生成 ActuarialInsights。
    """
    
    # Core prediction
    prediction: PredictionData
    
    # SHAP data for interpretation
    shap_contributions: List[SHAPData]
    
    # Context
    percentiles: List[Dict[str, Any]]
    categories: List[Dict[str, Any]]
    
    # Segment info
    coverage_segment: Optional[Dict[str, Any]] = None
    spotlight_segment: Optional[Dict[str, Any]] = None
    
    # Calibration
    overall_ae: float

# =========================
# Spotlight / Exception Report
# =========================

class ExceptionDefinition(BaseModel):
    ae_deviation_threshold: float
    min_exposure: int
    notes: Optional[str] = None


class SpotlightAnomaly(BaseModel):
    segment_id: str
    rule_text: str
    ae_ratio: float
    exposure: float
    deaths: Optional[float] = None
    expected_deaths: Optional[float] = None
    years_present: Optional[List[int]] = None
    concentrated_years: Optional[List[int]] = None
    # Structured stability metrics (from KB stability dict)
    stability_score: Optional[float] = Field(default=None, description="Fraction of observed years where anomaly persists (0-1)")
    persistence: Optional[float] = Field(default=None, description="Exposure-weighted persistence of anomaly direction (0-1)")
    log_ae_sd: Optional[float] = Field(default=None, description="Standard deviation of log(A/E) across years — proxy for volatility")
    stability_note: Optional[str] = None
    credibility_score: Optional[float] = Field(default=None, description="Bühlmann credibility score (0-1) based on exposure size")
    # Financial impact: net death count deviation (positive = over-mortality loss)
    deaths_delta: Optional[float] = Field(default=None, description="Actual minus expected deaths (positive = excess mortality)")


class SpotlightReportData(BaseModel):
    report_date: str
    report_version: str = "1.0"

    exception_definition: ExceptionDefinition
    top_anomalies: List[SpotlightAnomaly]

    overall_ae: Optional[float] = None
    yearly_ae: Optional[Dict[str, float]] = None

    disclaimers: List[str] = Field(default_factory=list)


class SpotlightPromptContext(BaseModel):
    exception_definition: ExceptionDefinition
    top_anomalies: List[SpotlightAnomaly]
    overall_ae: Optional[float] = None
    yearly_ae: Optional[Dict[str, float]] = None


class QuadrantAttribution(BaseModel):
    data_related: List[str]
    model_related: List[str]
    external_factors: List[str]
    definition_segmentation: List[str]


class SpotlightInsightItem(BaseModel):
    segment_id: str
    summary: str
    temporal_stability_interpretation: str
    why_might_happen: QuadrantAttribution
    investigation_path: List[str]
    recommended_actions: List[str]


class SpotlightInsights(BaseModel):
    exception_definition_restate: str
    top_anomalies_overview: str
    items: List[SpotlightInsightItem]
    overall_recommended_actions: List[str] = Field(default_factory=list)

class ExperienceTrendPackData(BaseModel):
    report_type: Literal["trend_pack"] = "trend_pack"
    report_date: str
    report_version: str = "1.0"

    # fixed definitions (deterministic)
    definitions: Dict[str, str] = Field(default_factory=dict)

    # deterministic tables & charts (relative paths from data/reports/)
    tables: Dict[str, str] = Field(default_factory=dict)
    charts: Dict[str, List[str]] = Field(default_factory=dict)

class ExperienceTrendPackInsights(BaseModel):
    # make all optional/default to avoid structured output brittleness
    executive_summary: str = ""
    yearly_compare_notes: str = ""
    age_curve_notes: str = ""
    mix_shift_notes: str = ""
    drift_flags_notes: str = ""
    hypothesis_directions: List[str] = Field(default_factory=list)

# =============================================================================
# Validation Diagnostic Report (S1)
# =============================================================================

class ValidationCheck(BaseModel):
    check_id: str
    status: Literal["pass", "warn", "fail"]
    description: str
    evidence: Optional[str] = None


class DriftFlag(BaseModel):
    year: int
    ae_ratio: float
    delta_ae: float
    description: str


class ValidationDiagnosticData(BaseModel):
    report_type: Literal["validation_diagnostic"] = "validation_diagnostic"
    report_date: str
    report_version: str = "1.0"

    calibration_overall: Dict[str, Any] = Field(default_factory=dict)
    calibration_yearly: Dict[str, Any] = Field(default_factory=dict)
    data_quality: Dict[str, Any] = Field(default_factory=dict)
    model_contract: Dict[str, Any] = Field(default_factory=dict)
    validation_checks: List[ValidationCheck] = Field(default_factory=list)
    drift_flags: List[DriftFlag] = Field(default_factory=list)
    disclaimers: List[str] = Field(default_factory=list)
    
# =============================================================================
# Full Report (Combined)
# =============================================================================

class FullReport(BaseModel):
    """
    完整報告 = 固定數據 + LLM 解釋。
    
    這個結構可以直接渲染為 Markdown 或 PDF。
    """
    data: Union[ReportData, SpotlightReportData, ExperienceTrendPackData, ValidationDiagnosticData]
    insights: Union[ActuarialInsights, SpotlightInsights, ExperienceTrendPackInsights, Dict[str, Any]]
    # # Fixed data
    # data: ReportData
    
    # # LLM-generated insights
    # insights: ActuarialInsights
    
    # System prompt used (for transparency)
    system_prompt: Optional[str] = None
    
    # User prompt sent to LLM (for transparency)
    user_prompt: Optional[str] = None
    
    # Timing and cost tracking
    timing_summary: Optional[List[Dict[str, Any]]] = None
    total_cost: Optional[float] = None
    total_tokens: Optional[int] = None
    
    # Validation status
    is_validated: bool = False
    validation_notes: List[str] = Field(default_factory=list)

    # Audit trail: which LLM model produced this report
    llm_model_version: Optional[str] = None

    # Run governance fields (populated by RunController)
    run_id: Optional[str] = None
    input_hash: Optional[str] = None
    prompt_version: Optional[str] = None
    # Approval gate: ai_draft -> self_reviewed -> actuary_approved
    approval_status: str = "ai_draft"
    constraint_check_passed: Optional[bool] = None
    constraint_violations: List[str] = Field(default_factory=list)
