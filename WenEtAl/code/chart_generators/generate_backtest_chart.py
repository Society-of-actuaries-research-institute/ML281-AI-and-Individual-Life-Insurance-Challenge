#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Expanding Window Backtest Chart Generator

Loads cutoff-year LightGBM models (lgbm_cutoff_2012 through lgbm_cutoff_2018),
generates out-of-sample A/E predictions for each subsequent year, and produces:
  - expanding_window_ae_by_year.png  (line chart)
  - expanding_window_ae_by_year.csv  (deterministic table)

Usage:
    python code/chart_generators/generate_backtest_chart.py

Reuses logic from code/run_expanding_window.py — no duplication of training code.
"""

import os
import sys
import warnings
from pathlib import Path

import pandas as pd
import numpy as np

warnings.filterwarnings('ignore')

# ---------------------------------------------------------------------------
# Paths (resolve relative to project root)
# ---------------------------------------------------------------------------
ROOT_DIR = Path(__file__).parent.parent.parent
DATA_PATH = ROOT_DIR / 'data' / 'ilec_cleaned.parquet'
MODELS_DIR = ROOT_DIR / 'models'
PLOTS_DIR = ROOT_DIR / 'data' / 'plots'
REPORTS_DIR = ROOT_DIR / 'data' / 'reports'

CUTOFF_YEARS = [2012, 2013, 2014, 2015, 2016, 2017, 2018]
NUMERICAL_FEATURES = ['Attained_Age', 'Issue_Age', 'Duration']
CATEGORICAL_FEATURES = [
    'Sex', 'Smoker_Status', 'Insurance_Plan', 'Face_Amount_Band',
    'Preferred_Class', 'SOA_Post_Lvl_Ind', 'SOA_Antp_Lvl_TP', 'SOA_Guar_Lvl_TP'
]
TARGET = 'Death_Count'
EXPOSURE = 'Policies_Exposed'


def prepare_features(data: pd.DataFrame) -> pd.DataFrame:
    X = data[NUMERICAL_FEATURES + CATEGORICAL_FEATURES].copy()
    for col in CATEGORICAL_FEATURES:
        X[col] = X[col].astype('category')
    return X


def compute_ae_by_year(df: pd.DataFrame, model, cutoff: int) -> pd.DataFrame:
    """
    For rows with Observation_Year > cutoff, predict expected deaths and compute A/E.
    Returns DataFrame with columns: [cutoff_year, Observation_Year, actual, expected, ae_ratio].
    """
    test_df = df[df['Observation_Year'] > cutoff].copy()
    if test_df.empty:
        return pd.DataFrame()

    X_test = prepare_features(test_df)
    exposure = test_df[EXPOSURE].values
    offset = np.log(exposure + 1e-10)

    preds = model.predict(X_test) * exposure

    result = test_df[['Observation_Year', TARGET]].copy()
    result['expected'] = preds
    result['cutoff_year'] = cutoff

    agg = result.groupby(['cutoff_year', 'Observation_Year']).agg(
        actual=(TARGET, 'sum'),
        expected=('expected', 'sum')
    ).reset_index()
    agg['ae_ratio'] = agg['actual'] / agg['expected'].clip(lower=1e-10)
    return agg


def run_backtest() -> pd.DataFrame:
    """Load data and cutoff models, compute A/E by year for each cutoff."""
    try:
        import lightgbm as lgb
    except ImportError:
        print("ERROR: lightgbm not installed. Run: pip install lightgbm")
        sys.exit(1)

    if not DATA_PATH.exists():
        print(f"ERROR: Data file not found at {DATA_PATH}")
        print("Run build_parquet.py first to generate ilec_cleaned.parquet.")
        sys.exit(1)

    print(f"Loading data from {DATA_PATH}...")
    df = pd.read_parquet(DATA_PATH)
    print(f"  Rows: {len(df):,} | Years: {df['Observation_Year'].min()}–{df['Observation_Year'].max()}")

    all_results = []
    for cutoff in CUTOFF_YEARS:
        model_path = MODELS_DIR / f'lgbm_cutoff_{cutoff}.txt'
        if not model_path.exists():
            print(f"  ⚠ Model not found: {model_path} — skipping cutoff {cutoff}")
            continue

        print(f"  Loading model: {model_path.name}")
        model = lgb.Booster(model_file=str(model_path))
        result = compute_ae_by_year(df, model, cutoff)
        if not result.empty:
            all_results.append(result)

    if not all_results:
        print("No results generated — check model files and data path.")
        return pd.DataFrame()

    combined = pd.concat(all_results, ignore_index=True)
    return combined


def plot_backtest(combined: pd.DataFrame, out_dir: Path) -> str:
    """Generate expanding window A/E line chart."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import matplotlib.cm as cm
    except ImportError:
        print("ERROR: matplotlib not installed.")
        sys.exit(1)

    out_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(12, 6))

    colors = cm.tab10.colors
    for i, cutoff in enumerate(sorted(combined['cutoff_year'].unique())):
        sub = combined[combined['cutoff_year'] == cutoff].sort_values('Observation_Year')
        ax.plot(
            sub['Observation_Year'], sub['ae_ratio'],
            marker='o', linewidth=1.8, markersize=5,
            color=colors[i % len(colors)],
            label=f'Train ≤ {cutoff}'
        )

    ax.axhline(1.0, color='black', linewidth=1, linestyle='--', label='A/E = 1.0')
    ax.axhspan(0.95, 1.05, alpha=0.07, color='green', label='±5% band')
    ax.set_xlabel('Observation Year', fontsize=12)
    ax.set_ylabel('A/E Ratio', fontsize=12)
    ax.set_title('Expanding Window Backtesting: Out-of-Sample A/E by Year', fontsize=13)
    ax.legend(fontsize=9, ncol=2)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0.85, 1.20)

    out_path = out_dir / 'expanding_window_ae_by_year.png'
    fig.savefig(str(out_path), dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Chart saved: {out_path}")
    return str(out_path)


def main():
    print("=== Expanding Window Backtest Chart Generator ===")

    combined = run_backtest()
    if combined.empty:
        return

    # Save CSV
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = REPORTS_DIR / 'expanding_window_ae_by_year.csv'
    combined.to_csv(str(csv_path), index=False)
    print(f"  CSV saved: {csv_path}")

    # Plot
    plot_backtest(combined, PLOTS_DIR)

    # Summary
    print("\nBacktest A/E Summary:")
    summary = combined.groupby('Observation_Year')['ae_ratio'].agg(['mean', 'min', 'max'])
    print(summary.to_string())


if __name__ == "__main__":
    main()
