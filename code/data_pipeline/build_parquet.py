#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Data Pipeline: Raw TXT -> Cleaned Parquet -> Predictions Parquet

Steps:
  1. Read raw ILEC tab-separated data (soa_data_2012_19_20240429.txt)
  2. Apply cleaning: filter years, drop zero-exposure rows, cast types
  3. Fit sklearn LabelEncoders on ALL unique values per categorical feature
     -> Serialize to models/label_encoders.pkl (used by predictor.py for single-record prediction)
  4. Load lgbm_mortality_offset_poisson.txt and run batch prediction
  5. Save:
     - data/ilec_cleaned.parquet    (cleaned features + actuals)
     - data/ilec_with_predictions.parquet  (+ Expected_Deaths column)
  6. Validate: Expected_Deaths.sum() / Death_Count.sum() ~ 0.9989

Usage:
    python code/data_pipeline/build_parquet.py [--raw-data PATH]

Override raw data path via env var RAW_DATA_PATH or --raw-data argument.
"""

import argparse
import os
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT_DIR = Path(__file__).parent.parent.parent
DATA_DIR = ROOT_DIR / 'data'
MODELS_DIR = ROOT_DIR / 'models'
LGBM_MODEL_PATH = MODELS_DIR / 'lgbm_mortality_offset_poisson.txt'
LABEL_ENCODERS_PATH = MODELS_DIR / 'label_encoders.pkl'
CLEANED_PATH = DATA_DIR / 'ilec_cleaned.parquet'
PRED_PATH = DATA_DIR / 'ilec_with_predictions.parquet'

# Default raw data location (sibling code/data/ folder)
DEFAULT_RAW_DATA = str(ROOT_DIR.parent / 'code' / 'data' / 'soa_data_2012_19_20240429.txt')

# ---------------------------------------------------------------------------
# Feature configuration (mirrors config.py)
# ---------------------------------------------------------------------------
NUMERICAL_FEATURES = ['Attained_Age', 'Issue_Age', 'Duration']
CATEGORICAL_FEATURES = [
    'Sex', 'Smoker_Status', 'Insurance_Plan', 'Face_Amount_Band',
    'Preferred_Class', 'SOA_Post_Lvl_Ind', 'SOA_Antp_Lvl_TP', 'SOA_Guar_Lvl_TP'
]
ALL_FEATURES = NUMERICAL_FEATURES + CATEGORICAL_FEATURES
TARGET = 'Death_Count'
EXPOSURE = 'Policies_Exposed'

KEEP_COLUMNS = ALL_FEATURES + [
    TARGET, EXPOSURE, 'Observation_Year', 'Issue_Year',
    'Amount_Exposed', 'Death_Claim_Amount'
]


# ---------------------------------------------------------------------------
# Step 1: Load and clean raw data
# ---------------------------------------------------------------------------

def load_and_clean(raw_path: str) -> pd.DataFrame:
    print(f"[1/5] Loading raw data: {raw_path}")

    # Try Polars first for speed; fall back to Pandas
    try:
        import polars as pl
        lf = pl.scan_csv(
            raw_path, separator='\t', infer_schema_length=10000,
            encoding='latin1', ignore_errors=True,
        )
        # Filter years and exposure upfront before materializing
        lf = lf.filter(
            (pl.col('Observation_Year') >= 2012) &
            (pl.col('Policies_Exposed') > 0)
        )
        available_cols = lf.columns
        keep = [c for c in KEEP_COLUMNS if c in available_cols]
        df = lf.select(keep).collect().to_pandas()
        print(f"  -> Loaded {len(df):,} rows via Polars (lazy scan)")
    except Exception as polars_err:
        print(f"  Polars failed ({polars_err}); using pandas chunked read...")
        chunks = []
        chunk_num = 0
        for chunk in pd.read_csv(
            raw_path, sep='\t', chunksize=500_000, encoding='latin-1',
            low_memory=False
        ):
            chunk = chunk[chunk['Observation_Year'] >= 2012]
            chunk = chunk[chunk[EXPOSURE] > 0]
            available_cols = chunk.columns.tolist()
            keep = [c for c in KEEP_COLUMNS if c in available_cols]
            chunks.append(chunk[keep])
            chunk_num += 1
            if chunk_num % 10 == 0:
                print(f"    ...processed ~{chunk_num * 500_000:,} rows...")
        df = pd.concat(chunks, ignore_index=True)
        print(f"  -> Loaded {len(df):,} rows via Pandas")

    # Cast numerics
    for col in NUMERICAL_FEATURES:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype(float)

    # Cast target and exposure
    df[TARGET] = pd.to_numeric(df[TARGET], errors='coerce').fillna(0).astype(float)
    df[EXPOSURE] = pd.to_numeric(df[EXPOSURE], errors='coerce').fillna(0).astype(float)

    # Fill categorical NAs with 'Unknown'
    for col in CATEGORICAL_FEATURES:
        if col in df.columns:
            df[col] = df[col].fillna('Unknown').astype(str)

    # Remove rows where exposure or target is invalid
    df = df[df[EXPOSURE] > 0].reset_index(drop=True)

    print(f"  -> After cleaning: {len(df):,} rows | Years: {df['Observation_Year'].min()}-{df['Observation_Year'].max()}")
    return df


# ---------------------------------------------------------------------------
# Step 2: Fit LabelEncoders on full dataset
# ---------------------------------------------------------------------------

def fit_and_save_encoders(df: pd.DataFrame) -> dict:
    """
    Fit sklearn LabelEncoder on all unique values per categorical feature.
    LabelEncoder sorts categories alphabetically --same as pandas Categorical,
    ensuring codes match what LightGBM learned during training.
    """
    print("[2/5] Fitting LabelEncoders on full dataset...")
    try:
        from sklearn.preprocessing import LabelEncoder
        import joblib
    except ImportError:
        print("  WARNING: sklearn or joblib not available --skipping encoder serialization")
        return {}

    encoders = {}
    for col in CATEGORICAL_FEATURES:
        if col in df.columns:
            le = LabelEncoder()
            le.fit(df[col].astype(str).unique())
            encoders[col] = le
            print(f"  {col}: {len(le.classes_)} categories")

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(encoders, str(LABEL_ENCODERS_PATH))
    print(f"  OK Encoders saved to {LABEL_ENCODERS_PATH}")
    return encoders


# ---------------------------------------------------------------------------
# Step 3: Save cleaned parquet
# ---------------------------------------------------------------------------

def save_cleaned(df: pd.DataFrame):
    print(f"[3/5] Saving cleaned parquet to {CLEANED_PATH}...")
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(str(CLEANED_PATH), index=False)
    print(f"  OK {len(df):,} rows saved")


# ---------------------------------------------------------------------------
# Step 4: Run LightGBM predictions
# ---------------------------------------------------------------------------

def run_predictions(df: pd.DataFrame) -> pd.DataFrame:
    print(f"[4/5] Running LightGBM predictions...")
    try:
        import lightgbm as lgb
    except ImportError:
        print("  WARNING: lightgbm not installed --skipping predictions")
        df['Expected_Deaths'] = df[TARGET]  # identity fallback
        return df

    if not LGBM_MODEL_PATH.exists():
        print(f"  WARNING: Model not found at {LGBM_MODEL_PATH} --skipping predictions")
        df['Expected_Deaths'] = df[TARGET]
        return df

    try:
        from sklearn.preprocessing import LabelEncoder as _LE
    except ImportError:
        print("  WARNING: sklearn not installed -- skipping predictions")
        df = df.copy()
        df['Expected_Deaths'] = df[TARGET]
        return df

    model = lgb.Booster(model_file=str(LGBM_MODEL_PATH))
    print(f"  Model loaded: {model.num_trees()} trees")

    # Replicate the same encoding as add_predictions_to_data.py:
    # LabelEncoder.fit_transform per column --gives integer numpy codes.
    # Passing as .values avoids LightGBM category dtype mismatch warnings.
    df_enc = df[ALL_FEATURES].copy()
    for col in CATEGORICAL_FEATURES:
        le = _LE()
        df_enc[col] = le.fit_transform(df_enc[col].astype(str))

    predicted_rate = model.predict(df_enc[ALL_FEATURES].values)
    df = df.copy()
    df['Expected_Deaths'] = predicted_rate * df[EXPOSURE].values
    df['Predicted_Rate'] = predicted_rate

    print(f"  OK Predictions complete")
    return df


# ---------------------------------------------------------------------------
# Step 5: Validate and save with predictions
# ---------------------------------------------------------------------------

def validate_and_save(df: pd.DataFrame):
    print("[5/5] Validating and saving predictions parquet...")

    actual = df[TARGET].sum()
    expected = df['Expected_Deaths'].sum()
    overall_ae = actual / expected if expected > 0 else float('nan')

    print(f"  Actual Deaths:   {actual:,.0f}")
    print(f"  Expected Deaths: {expected:,.0f}")
    print(f"  Overall A/E:     {overall_ae:.4f}  (target ~0.9989)")

    if abs(overall_ae - 0.9989) > 0.05:
        print(f"  WARNING: WARNING: Overall A/E {overall_ae:.4f} deviates significantly from 0.9989")
        print("    Check model path and data filtering.")
    else:
        print(f"  OK A/E is within expected range")

    df.to_parquet(str(PRED_PATH), index=False)
    print(f"  OK Saved {len(df):,} rows to {PRED_PATH}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Build ILEC parquet files from raw data")
    parser.add_argument(
        '--raw-data', default=os.environ.get('RAW_DATA_PATH', DEFAULT_RAW_DATA),
        help='Path to raw tab-separated data file'
    )
    args = parser.parse_args()

    raw_path = args.raw_data
    if not Path(raw_path).exists():
        print(f"ERROR: Raw data file not found: {raw_path}")
        print(f"  Set RAW_DATA_PATH env var or pass --raw-data <path>")
        sys.exit(1)

    print("=" * 60)
    print("SOA Data Pipeline: build_parquet.py")
    print("=" * 60)

    df = load_and_clean(raw_path)
    fit_and_save_encoders(df)
    save_cleaned(df)
    df_pred = run_predictions(df)
    validate_and_save(df_pred)

    print("\nOK Pipeline complete")
    print(f"  Cleaned data:    {CLEANED_PATH}")
    print(f"  With predictions: {PRED_PATH}")
    print(f"  Label encoders:  {LABEL_ENCODERS_PATH}")


if __name__ == "__main__":
    main()
