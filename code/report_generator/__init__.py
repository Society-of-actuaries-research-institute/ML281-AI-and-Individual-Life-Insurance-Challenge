# -*- coding: utf-8 -*-
"""
Report Generator Package

Generates structured explanation reports for mortality predictions.
"""

from .config import (
    KNOWLEDGE_BASE_DIR,
    FEATURES,
    CATEGORICAL_FEATURES,
    NUMERICAL_FEATURES,
    REPORT_CONFIG,
)

from .knowledge import KnowledgeBase, get_knowledge_base
from .matcher import SegmentMatcher, create_matcher
from .predictor import MortalityPredictor, create_predictor
from .llm import LLMClient, create_llm_client
from .renderer import ReportRenderer, create_renderer
from .prompt_builder import get_full_system_prompt, build_system_prompt
from .router import ReportRouter, ReportRequest
from .pipeline import ReportPipeline

# Import schemas (optional pydantic)
from .schemas import (
    ActuarialInsights, ReportData, FullReport,
    PredictionData, SegmentData,
    HAS_PYDANTIC
)

__all__ = [
    # Config
    'KNOWLEDGE_BASE_DIR', 'FEATURES', 'CATEGORICAL_FEATURES', 
    'NUMERICAL_FEATURES', 'REPORT_CONFIG',
    # Knowledge
    'KnowledgeBase', 'get_knowledge_base',
    # Matcher
    'SegmentMatcher', 'create_matcher',
    # Predictor
    'MortalityPredictor', 'create_predictor',
    # LLM
    'LLMClient', 'create_llm_client',
    # Renderer
    'ReportRenderer', 'create_renderer',
    # Prompt
    'get_full_system_prompt', 'build_system_prompt',
    # Router / Pipeline
    'ReportRouter', 'ReportRequest', 'ReportPipeline',
    # Schemas
    'ActuarialInsights', 'ReportData', 'FullReport',
    'PredictionData', 'SegmentData',
    'HAS_PYDANTIC',
]
