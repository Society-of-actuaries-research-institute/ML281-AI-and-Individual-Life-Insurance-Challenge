# -*- coding: utf-8 -*-
"""
Configuration for Report Generator
"""

from pathlib import Path

# =============================================================================
# Paths
# =============================================================================

# Base paths
ROOT_DIR = Path(__file__).parent.parent.parent
CODE_DIR = ROOT_DIR / 'code'
DATA_DIR = ROOT_DIR / 'data'
MODELS_DIR = ROOT_DIR / 'models'
KNOWLEDGE_BASE_DIR = ROOT_DIR / 'knowledge_base'

# Knowledge base sub-paths
EDA_DIR = KNOWLEDGE_BASE_DIR / 'eda'
CALIBRATION_DIR = KNOWLEDGE_BASE_DIR / 'calibration'
SEGMENTS_DIR = KNOWLEDGE_BASE_DIR / 'segments'
SHAP_DIR = KNOWLEDGE_BASE_DIR / 'shap'

# Model files
LGBM_MODEL_PATH = MODELS_DIR / 'lgbm_mortality_offset_poisson.txt'
YEAR_FACTORS_PATH = MODELS_DIR / 'year_factors_offset.csv'
ILEC_CLEANED_PATH = DATA_DIR / 'ilec_cleaned.parquet'
ILEC_WITH_PRED_PATH = DATA_DIR / 'ilec_with_predictions.parquet'
LABEL_ENCODERS_PATH = MODELS_DIR / 'label_encoders.pkl'

# Raw data (outside SOA_Challenge-main; override via env var RAW_DATA_PATH)
import os as _os
RAW_DATA_PATH = _os.environ.get(
    "RAW_DATA_PATH",
    str(ROOT_DIR.parent / 'code' / 'data' / 'soa_data_2012_19_20240429.txt')
)

# =============================================================================
# Model Features
# =============================================================================

FEATURES = [
    'Attained_Age', 'Issue_Age', 'Duration', 'Sex', 'Smoker_Status',
    'Insurance_Plan', 'Face_Amount_Band', 'Preferred_Class',
    'SOA_Post_Lvl_Ind', 'SOA_Antp_Lvl_TP', 'SOA_Guar_Lvl_TP'
]

CATEGORICAL_FEATURES = [
    'Sex', 'Smoker_Status', 'Insurance_Plan', 'Face_Amount_Band',
    'Preferred_Class', 'SOA_Post_Lvl_Ind', 'SOA_Antp_Lvl_TP', 'SOA_Guar_Lvl_TP'
]

NUMERICAL_FEATURES = ['Attained_Age', 'Issue_Age', 'Duration']

# =============================================================================
# LLM Configuration
# =============================================================================

LLM_BACKEND = _os.environ.get("LLM_BACKEND", "gemini")  # "gemini" | "claude"
LLM_TIMEOUT_SECONDS = 60
LLM_MAX_RETRIES = 3

# =============================================================================
# Report Settings
# =============================================================================

REPORT_CONFIG = {
    "top_shap_features": 5,
    "ae_anomaly_threshold": 0.15,  # |A/E - 1| > 15%
    "ae_calibrated_range": (0.95, 1.05),
}
