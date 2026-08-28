# -*- coding: utf-8 -*-
"""
Tested Chart Generation Logic (Optimized using Pre-calculated Predictions)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from pathlib import Path
import os
import matplotlib.cm as cm
import gc
import json

# DATA_PATH: allow override via environment variable
DATA_PATH = Path(os.getenv("ILEC_WITH_PRED_PATH", "../data/ilec_with_predictions.parquet"))
INTERACTIVE = False 
# Trend Pack output dirs: allow override (default to data/reports/trend_pack)
TREND_PACK_DIR = Path(os.getenv("TREND_PACK_OUTDIR", "../data/reports/trend_pack"))
ASSETS_DIR = TREND_PACK_DIR / "assets"
TABLES_DIR = TREND_PACK_DIR / "tables"
ASSETS_DIR.mkdir(parents=True, exist_ok=True)
TABLES_DIR.mkdir(parents=True, exist_ok=True)

# Note: lightgbm and sklearn are no longer needed for chart generation!

# =========================================================================
# CELL 1: SETUP
# =========================================================================

# Constants & Paths
# Updated to use the dataset with pre-calculated predictions
PLOTS_DIR = Path('../data/plots')
print(PLOTS_DIR)
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

# Mappings
GROUP_MAP = {
    '01: < 10,000': '0 - 99,999',
    '02: 10,000 - 24,999': '0 - 99,999',
    '03: 25,000 - 49,999': '0 - 99,999',
    '04: 50,000 - 99,999': '0 - 99,999',
    '05: 100,000 - 249,999': '100,000 - 249,999',
    '06: 250,000 - 499,999': '250,000 - 499,999',
    '07: 500,000 - 999,999': '500,000+',
    '08: 1,000,000 - 2,499,999': '500,000+',
    '09: 2,500,000+': '500,000+'
}

# Plot Settings
plt.style.use('default')
plt.rcParams['font.family'] = 'sans-serif'

def _savefig_or_show(fig_path=None):
    if fig_path:
        plt.savefig(fig_path, dpi=150, bbox_inches="tight")
        plt.close()
    else:
        plt.show()


def _default_csv_path(out_png: Path, tables_dir: Path) -> Path:
    """Map an output PNG path to a CSV path in tables_dir using the same stem."""
    return tables_dir / (out_png.stem + ".csv")

def _write_csv(df: pd.DataFrame, csv_path: Path) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_path, index=False)
def load_data_optimized():
    """
    Load parquet data with Polars (3-5x faster than pandas for 45M rows).
    Falls back to pandas if Polars is not installed.
    """
    print('Loading data (with pre-calculated predictions)...')

    try:
        import polars as pl
        lf = pl.scan_parquet(str(DATA_PATH))
        df = lf.collect().to_pandas()
        print(f'Records: {len(df):,}  [loaded via Polars]')
    except ImportError:
        print('  Polars not installed — using pandas (install with: pip install polars)')
        df = pd.read_parquet(DATA_PATH)
        print(f'Records: {len(df):,}  [loaded via Pandas]')

    # Feature Engineering (Face Group needed for Trellis Plot)
    if 'Face_Group' not in df.columns:
        df['Face_Group'] = df['Face_Amount_Band'].map(GROUP_MAP)

    # Note: 'Expected_Deaths' is already in the dataframe!
    print('Done.')
    return df


def ae_by_year_duckdb(data_path: str = None) -> pd.DataFrame:
    """
    Compute yearly A/E directly from parquet using DuckDB (no full load into memory).
    Returns DataFrame with columns: [Observation_Year, ae_ratio, actual_deaths, expected_deaths].
    """
    path = data_path or str(DATA_PATH)
    try:
        import duckdb
        con = duckdb.connect()
        result = con.execute(f"""
            SELECT
                Observation_Year,
                SUM(Death_Count)::DOUBLE / NULLIF(SUM(Expected_Deaths), 0) AS ae_ratio,
                SUM(Death_Count)::BIGINT AS actual_deaths,
                SUM(Expected_Deaths)::DOUBLE AS expected_deaths
            FROM read_parquet('{path}')
            GROUP BY Observation_Year
            ORDER BY Observation_Year
        """).df()
        con.close()
        return result
    except ImportError:
        print("  DuckDB not installed — falling back to pandas groupby (install with: pip install duckdb)")
        df = pd.read_parquet(path, columns=['Observation_Year', 'Death_Count', 'Expected_Deaths'])
        agg = df.groupby('Observation_Year').agg(
            actual_deaths=('Death_Count', 'sum'),
            expected_deaths=('Expected_Deaths', 'sum')
        ).reset_index()
        agg['ae_ratio'] = agg['actual_deaths'] / agg['expected_deaths'].clip(lower=1e-10)
        return agg

def trendpack_age_curve(df, fig_path=None, csv_path=None):
    """
    Age curve: empirical vs model
    """
    result = (
        df.groupby("Attained_Age")
          .agg(
              exposure=("Policies_Exposed", "sum"),
              deaths=("Death_Count", "sum"),
              expected=("Expected_Deaths", "sum")
          )
          .reset_index()
          .sort_values("Attained_Age")
    )

    result["empirical_rate"] = result["deaths"] / result["exposure"]
    result["model_rate"] = result["expected"] / result["exposure"]

    if csv_path:
        result.to_csv(csv_path, index=False)

    plt.figure(figsize=(8, 5))
    plt.plot(result["Attained_Age"], result["empirical_rate"], label="Empirical", marker="o")
    plt.plot(result["Attained_Age"], result["model_rate"], label="Model", linestyle="--")
    plt.legend()
    plt.grid(True)
    # plt.title("Age Curve: Empirical vs Model")

    _savefig_or_show(fig_path)

    return result


def trendpack_yearly_compare_year_sex(df, fig_path=None, csv_path=None):
    """
    Year × Sex high-level comparison
    Deterministic output for Trend Pack
    """
    result = (
        df.groupby(["Observation_Year", "Sex"])
          .agg(
              exposure=("Policies_Exposed", "sum"),
              deaths=("Death_Count", "sum"),
              expected=("Expected_Deaths", "sum")
          )
          .reset_index()
    )

    result["A/E"] = result["deaths"] / result["expected"]
    result["RR"] = result["A/E"] / result["A/E"].mean()

    if csv_path:
        result.to_csv(csv_path, index=False)

    # === plotting (keep your original plotting logic here) ===
    plt.figure(figsize=(8, 5))
    for sex, g in result.groupby("Sex"):
        plt.plot(g["Observation_Year"], g["A/E"], marker="o", label=f"Sex={sex}")

    plt.axhline(1.0, linestyle="--", color="grey")
    # plt.title("Yearly A/E by Sex")
    plt.legend()
    plt.grid(True)

    _savefig_or_show(fig_path)

    return result

def trendpack_product_mix_shift(df, fig_path=None, csv_path=None):
    """
    Product mix shift over time
    """
    result = (
        df.groupby(["Observation_Year", "Insurance_Plan"])
          .agg(exposure=("Policies_Exposed", "sum"))
          .reset_index()
    )

    total = result.groupby("Observation_Year")["exposure"].transform("sum")
    result["exposure_share"] = result["exposure"] / total

    if csv_path:
        result.to_csv(csv_path, index=False)

    plt.figure(figsize=(9, 5))
    for prod, g in result.groupby("Insurance_Plan"):
        plt.plot(g["Observation_Year"], g["exposure_share"], marker="o", label=prod)

    # plt.title("Product Mix Shift (Exposure Share)")
    plt.legend()
    plt.grid(True)
    _savefig_or_show(fig_path)

    return result


def trendpack_drift_flags_from_year_sex(df, csv_path=None):
    """
    Detect structural drift using Year × Sex A/E change
    """
    if df is None:
        raise ValueError("df is required for drift flag computation")

    base = (
        df.groupby(["Observation_Year", "Sex"])
          .agg(
              deaths=("Death_Count", "sum"),
              expected=("Expected_Deaths", "sum")
          )
          .reset_index()
    )

    base["A/E"] = base["deaths"] / base["expected"]

    flags = []
    for sex, g in base.groupby("Sex"):
        g = g.sort_values("Observation_Year")
        g["delta_ae"] = g["A/E"].diff()

        for _, r in g.iterrows():
            if abs(r["delta_ae"]) > 0.15:
                flags.append({
                    "Observation_Year": r["Observation_Year"],
                    "Sex": sex,
                    "A/E": r["A/E"],
                    "delta_A/E": r["delta_ae"],
                    "flag": "Potential structural break"
                })

    result = pd.DataFrame(flags)

    if csv_path:
        result.to_csv(csv_path, index=False)

    return result


def plot_chart_1(df, save_path=None, tables_dir: Path | None = None):
    # =========================================================================
    # CELL 3: A/E BY OBSERVATION YEAR
    # =========================================================================
    
    # Aggregation
    yearly = df.groupby('Observation_Year').agg({  # Note: Column name might be Observation_Year in cleaned file
        'Death_Count': 'sum',
        'Expected_Deaths': 'sum'
    }).reset_index()
    
    # Handle column name difference if necessary (Year vs Observation_Year)
    year_col = 'Observation_Year' if 'Observation_Year' in yearly.columns else 'Year'

    yearly['AE_Count'] = yearly['Death_Count'] / yearly['Expected_Deaths']

    # Plot
    fig, ax1 = plt.subplots(figsize=(10, 6))

    # Colors
    color_ae = '#005f9e'    # Blue for A/E
    color_bar = '#004b7d'   # Dark Blue for bars

    # Setup Dual Axis
    ax2 = ax1.twinx()

    # Bars (Deaths)
    bars = ax2.bar(yearly[year_col], yearly['Death_Count'], color=color_bar, alpha=0.3, label='Number of Deaths')
    ax2.set_ylabel('Number of Deaths', fontsize=12, color=color_bar)
    ax2.tick_params(axis='y', labelcolor=color_bar)
    ax2.set_ylim(0, yearly['Death_Count'].max() * 2.5)  # Make bars occupy bottom ~40%

    # Line (A/E)
    line1, = ax1.plot(yearly[year_col], yearly['AE_Count'], color=color_ae, linewidth=3, label='A/E Deaths: Count')

    # Reference Line (100%)
    ax1.axhline(1.0, color='gray', linestyle='-', linewidth=1, alpha=0.5)

    # Formatting
    ax1.set_ylabel('Actual vs Expected', fontsize=12)
    ax1.set_ylim(0.8, 1.25)
    ax1.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    # plt.title('A/E BY OBSERVATION YEAR (CORE DATA)', fontsize=14, loc='left', pad=15, color='#003366', fontweight='bold')

    # Combine Legend
    lines = [line1, bars]
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='upper right', frameon=True)

    plt.tight_layout()
    # Save the plotting data as CSV (one-to-one with the PNG stem)
    if save_path and tables_dir is not None:
        _write_csv(yearly, _default_csv_path(Path(save_path), tables_dir))
    plt.grid(True)
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
    else:
        plt.show()


def plot_chart_2(df, save_path=None, tables_dir: Path | None = None):
    # =========================================================================
    # CELL 4: A/E BY FACE AMOUNT & YEAR (TRELLIS)
    # =========================================================================
    
    # Handle column name difference
    year_col = 'Observation_Year' if 'Observation_Year' in df.columns else 'Year'
    
    # Aggregation
    grouped = df.groupby(['Face_Group', year_col]).agg({
        'Death_Count': 'sum',
        'Expected_Deaths': 'sum'
    }).reset_index()

    grouped['AE_Count'] = grouped['Death_Count'] / grouped['Expected_Deaths']

    # Trellis Plot
    groups = ['0 - 99,999', '100,000 - 249,999', '250,000 - 499,999', '500,000+']

    fig, axes = plt.subplots(1, 4, figsize=(20, 6), sharey=False)

    color_ae = '#005f9e'
    color_bar = '#004b7d'
    ae_ylim = (0.75, 1.25)

    for i, group in enumerate(groups):
        ax1 = axes[i]
        subset = grouped[grouped['Face_Group'] == group]
        
        # Dual Axis
        ax2 = ax1.twinx()
        
        # Bars (Deaths)
        max_deaths = subset['Death_Count'].max()
        ax2.set_ylim(0, max_deaths * 2.5)
        bars = ax2.bar(subset[year_col], subset['Death_Count'], color=color_bar, alpha=0.3, label='Number of Deaths')
        
        # Line (A/E)
        line1, = ax1.plot(subset[year_col], subset['AE_Count'], color=color_ae, linewidth=2.5, label='AE Deaths: Count')
        
        # Reference
        ax1.axhline(1.0, color='gray', linestyle='-', linewidth=1, alpha=0.5)
        
        # Formatting
        ax1.set_title(group, fontsize=14, fontweight='bold', color='#444444')
        ax1.set_ylim(ae_ylim)
        ax1.grid(True, alpha=0.3)
        
        # Clean up ticks
        if i == 0:
            ax1.set_ylabel('Actual vs Expected', fontsize=12)
            ax2.set_yticks([])
        elif i == 3:
            ax2.set_ylabel('Number of Deaths', fontsize=12, color=color_bar)
            ax2.tick_params(axis='y', labelcolor=color_bar)
            ax1.set_yticklabels([])
        else:
            ax1.set_yticklabels([])
            ax2.set_yticks([])
            
        # Legend
        if i == 3:
            lines = [line1, bars]
            labels = [l.get_label() for l in lines]
            ax1.legend(lines, labels, loc='upper right', frameon=True, fontsize=9)

    # plt.suptitle('A/E BY FACE AMOUNT AND OBSERVATION YEAR (CORE DATA)', fontsize=16, color='#003366', fontweight='bold', y=1.05)
    plt.tight_layout()
    # Save the plotting data as CSV (one-to-one with the PNG stem)
    if save_path and tables_dir is not None:
        _write_csv(grouped, _default_csv_path(Path(save_path), tables_dir))
    plt.grid(True)
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
    else:
        plt.show()

def plot_chart_10_deaths_product_issue_year(df, save_path=None, tables_dir: Path | None = None):
    # =========================================================================
    # CELL 13: DEATHS BY PRODUCT AND ISSUE YEAR (1980-2019)
    # =========================================================================

    # Handle Issue Year
    if 'Issue_Year' not in df.columns:
        if 'Observation_Year' in df.columns and 'Duration' in df.columns:
            df['Issue_Year'] = df['Observation_Year'] - df['Duration'] + 1
        elif 'Year' in df.columns and 'Duration' in df.columns:
            df['Issue_Year'] = df['Year'] - df['Duration'] + 1

    # Filter Data for Issue Years 1980-2019
    df_issue = df[(df['Issue_Year'] >= 1980) & (df['Issue_Year'] <= 2019)].copy()

    # Aggregation by Issue Year and Product - Using DEATH COUNT
    grouped_issue = df_issue.groupby(['Issue_Year', 'Insurance_Plan'])['Death_Count'].sum().reset_index()

    # Pivot for Stacked Bar
    pivot_issue = grouped_issue.pivot(index='Issue_Year', columns='Insurance_Plan', values='Death_Count').fillna(0)

    # Calculate Percentages for 100% Stacked Bar
    pivot_pct = pivot_issue.div(pivot_issue.sum(axis=1), axis=0)

    # Colors
    color_map_issue = {
        'Perm': '#003366',    # Dark Blue
        'Term': '#7d3f98',    # Purple
        'UL': '#5bc0de',      # Light Blue/Cyan
        'ULSG': '#d62728',    # Red
        'VL': '#ffdd05',      # Yellow
        'VLSG': '#77ac30',    # Green (if exists)
        'Other': '#001f3f'    # Very Dark Blue
    }

    # Plot: 2 Subplots (Top: %, Bottom: Count)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True, gridspec_kw={'height_ratios': [1, 1], 'hspace': 0.05})

    # --- Top Plot: 100% Stacked Bar (Deaths %) ---
    bottom = np.zeros(len(pivot_pct))
    for i, col in enumerate(pivot_pct.columns):
        values = pivot_pct[col].values
        color = color_map_issue.get(col, None) 
        if color is None: color = plt.cm.tab20(i)
            
        ax1.bar(pivot_pct.index, values, bottom=bottom, label=col, color=color, alpha=0.9, width=0.8)
        bottom += values

    ax1.set_ylabel('% of Deaths', fontsize=12)
    ax1.set_ylim(0, 1.0)
    ax1.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    # ax1.set_title('DEATHS BY PRODUCT AND ISSUE YEARS 1980-2019', fontsize=14, loc='left', pad=15, color='#003366', fontweight='bold')
    ax1.legend(loc='upper left', bbox_to_anchor=(1, 1), title='Insurance Plan')
    ax1.grid(True, alpha=0.3, axis='y')

    # --- Bottom Plot: Stacked Bar (Death Count) ---
    bottom = np.zeros(len(pivot_issue))
    for i, col in enumerate(pivot_issue.columns):
        values = pivot_issue[col].values
        color = color_map_issue.get(col, None) 
        if color is None: color = plt.cm.tab20(i)
        
        ax2.bar(pivot_issue.index, values, bottom=bottom, label=col, color=color, alpha=0.9, width=0.8)
        bottom += values

    ax2.set_ylabel('Number of Deaths', fontsize=12)
    ax2.set_xlabel('Issue Year', fontsize=12)
    ax2.grid(True, alpha=0.3, axis='y')

    # Rotate x-axis labels
    plt.xticks(pivot_issue.index, rotation=90)

    plt.tight_layout()
    # Save the plotting data as CSV (one-to-one with the PNG stem)
    if save_path and tables_dir is not None:
        _write_csv(grouped_issue, _default_csv_path(Path(save_path), tables_dir))

    plt.subplots_adjust(right=0.85) # For Legend
    plt.grid(True)
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
    else:
        plt.show()

def plot_chart_11_product_age_amount_split(df, save_path=None, tables_dir: Path | None = None):
    # =========================================================================
    # CELL 14: A/E BY PRODUCT (4-WAY SPLIT: AGE 48.5 x FACE AMOUNT 100K)

    # =========================================================================
    
    # Handle column names
    if 'Observation_Year' in df.columns:
        year_col = 'Observation_Year'
    else:
        year_col = 'Year'

    # 1. Define Split Logic
    def get_age_type(age):
        return '< 48.5' if age < 48.5 else '>= 48.5'

    def get_amt_type(band_str):
        # Band format example: '05: 100,000 - 249,999'
        # We can check the first 2 chars. 
        # 01-04 are < 100K. 05-08 are >= 100K.
        if not isinstance(band_str, str): return 'Unknown'
        code = band_str[:2]
        if code in ['01', '02', '03', '04']:
            return '< 100K'
        elif code in ['05', '06', '07', '08']:
            return '>= 100K'
        else:
            return 'Unknown'

    # 2. Aggregation First (Memory Optimization)
    # Group by Year, Plan, Attained_Age, Face_Amount_Band first to reduce size
    cols = [year_col, 'Insurance_Plan', 'Attained_Age', 'Face_Amount_Band', 'Death_Count', 'Expected_Deaths']
    df_minimal = df[cols].copy()

    # Apply Splits
    df_minimal['Age_Split'] = df_minimal['Attained_Age'].apply(get_age_type)
    df_minimal['Amt_Split'] = df_minimal['Face_Amount_Band'].apply(get_amt_type)

    # Filter for Core Products
    target_products = ['Perm', 'Term', 'UL', 'ULSG']
    df_prod = df_minimal[df_minimal['Insurance_Plan'].isin(target_products)]

    # Group By Split
    grouped = df_prod.groupby(['Age_Split', 'Amt_Split', year_col, 'Insurance_Plan']).agg({
        'Death_Count': 'sum',
        'Expected_Deaths': 'sum'
    }).reset_index()

    grouped['AE_Count'] = grouped['Death_Count'] / grouped['Expected_Deaths']

    # 3. Plot Layout (4x2 Grid of Subplots)
    # Rows: Age (< 48.5, >= 48.5) -> Spans 2 rows of subplots
    # Cols: Amount (< 100K, >= 100K) -> Spans 1 col of subplots (but we have 2 cols total)
    
    fig, axes = plt.subplots(
        4, 2, figsize=(18, 14), sharex=True,
        gridspec_kw={
            'height_ratios': [2, 1, 2, 1],
            'hspace': 0.32,   # ✅ 拉開上下間距
            'wspace': 0.12
        }
    )
    # ax_line.set_title(chart_title, fontsize=12, fontweight='bold', pad=10)
    splits_age = ['< 48.5', '>= 48.5']
    splits_amt = ['< 100K', '>= 100K']
    years = sorted(grouped[year_col].unique())

    color_map = {
        'Perm': '#003366',   # Dark Blue
        'Term': '#7d3f98',   # Purple
        'UL': '#5bc0de',     # Light Blue/Cyan
        'ULSG': '#d62728',   # Red
    }

    # Determine consistent Y-limits
    ae_ylim = (0.5, 1.5) 
    max_deaths = grouped['Death_Count'].max() * 1.1

    for r, age_cat in enumerate(splits_age):        # r=0 (Young), r=1 (Old)
        for c, amt_cat in enumerate(splits_amt):    # c=0 (Low), c=1 (High)
            
            # Determine Axes indices
            # Top Plot (A/E) is at row (r*2), col c
            # Bot Plot (Bar) is at row (r*2 + 1), col c
            ax_line = axes[r*2, c]
            ax_bar = axes[r*2 + 1, c]
            
            # Filter Data
            subset = grouped[(grouped['Age_Split'] == age_cat) & (grouped['Amt_Split'] == amt_cat)]
            
            pivot_ae = subset.pivot(index=year_col, columns='Insurance_Plan', values='AE_Count')
            pivot_death = subset.pivot(index=year_col, columns='Insurance_Plan', values='Death_Count').fillna(0)
            
            # --- Top Plot: A/E ---
            for prod in target_products:
                if prod in pivot_ae.columns:
                     ax_line.plot(pivot_ae.index, pivot_ae[prod], color=color_map.get(prod, 'gray'), linewidth=2.5, marker='o', markersize=3, label=prod)
            
            ax_line.axhline(1.0, color='gray', linestyle='-', linewidth=1, alpha=0.5)
            ax_line.set_ylim(ae_ylim)
            ax_line.grid(True, alpha=0.3)
            
            # Title
            chart_title = f"{age_cat} | Face Amount {amt_cat}"
            # ax_line.set_title(chart_title, fontsize=12, fontweight='bold', color='#444444')
            
            # Y-Label
            if c == 0:
                ax_line.set_ylabel('Actual vs Expected', fontsize=10)
                ax_line.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
            else:
                ax_line.set_yticklabels([])
                ax_line.set_ylabel('')

            # --- Bottom Plot: Deaths ---
            stack_order = ['ULSG', 'UL', 'Term', 'Perm']
            bottom = np.zeros(len(years))
            
            for prod in stack_order:
                if prod in pivot_death.columns:
                    values = pivot_death[prod].reindex(years, fill_value=0).values
                    ax_bar.bar(years, values, bottom=bottom, color=color_map.get(prod, 'gray'), alpha=0.9, label=prod)
                    bottom += values
            
            ax_bar.set_ylim(0, max_deaths)
            ax_bar.grid(True, alpha=0.3, axis='y')
            ax_bar.tick_params(axis='x', rotation=90)
            
            if c == 0:
                ax_bar.set_ylabel('Deaths', fontsize=10)
            else:
                ax_bar.set_yticklabels([])
                ax_bar.set_ylabel('')
                
            # Legend only on Top-Right Cell (r=0, c=1)
            if r == 0 and c == 1:
                ax_line.legend(loc='upper right', title='Plan', frameon=True, fontsize=8)

    # plt.suptitle('A/E BY PRODUCT (Split by Age 48.5 & Face Amount 100K)', fontsize=16, color='#003366', fontweight='bold', y=0.98)
    plt.grid(True)
    # Save the plotting data as CSV (one-to-one with the PNG stem)
    if save_path and tables_dir is not None:
        _write_csv(grouped, _default_csv_path(Path(save_path), tables_dir))

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
    else:
        plt.show()



def plot_chart_3_ae_by_product_and_year(df: pd.DataFrame, out_path: Path, tables_dir: Path | None = None):
    """
    A/E by Product and Observation Year (2-panel: lines for A/E, stacked bars for deaths).
    This mirrors the notebook/snippet versions but is packaged as a reusable function.
    """
    year_col = "Observation_Year" if "Observation_Year" in df.columns else "Year"

    target_products = ["Perm", "Term", "UL", "ULSG"]
    use_cols = [year_col, "Insurance_Plan", "Death_Count", "Expected_Deaths"]
    missing = [c for c in use_cols if c not in df.columns]
    if missing:
        raise KeyError(f"Missing required columns for plot_chart_3_ae_by_product_and_year: {missing}")

    g = (
        df.loc[df["Insurance_Plan"].isin(target_products), use_cols]
          .groupby([year_col, "Insurance_Plan"], observed=True, sort=True)
          .agg(Death_Count=("Death_Count", "sum"), Expected_Deaths=("Expected_Deaths", "sum"))
          .reset_index()
    )
    g["AE_Count"] = g["Death_Count"] / g["Expected_Deaths"]

    pivot_ae = g.pivot(index=year_col, columns="Insurance_Plan", values="AE_Count")
    pivot_death = g.pivot(index=year_col, columns="Insurance_Plan", values="Death_Count").fillna(0)

    color_map = {
        "Perm": "#003366",
        "Term": "#7d3f98",
        "UL": "#5bc0de",
        "ULSG": "#d62728",
    }

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(11, 8), sharex=True,
        gridspec_kw={"height_ratios": [1.0, 1.1], "hspace": 0.02}
    )

    # Top: A/E lines
    for i, prod in enumerate(target_products):
        if prod in pivot_ae.columns:
            ax1.plot(
                pivot_ae.index, pivot_ae[prod],
                linewidth=2.5, marker="o", markersize=4,
                color=color_map.get(prod, plt.cm.tab20(i)),
                label=prod
            )
    ax1.axhline(1.0, color="gray", linestyle="-", linewidth=1, alpha=0.5)
    ax1.set_ylabel("A/E Ratio", fontsize=11)
    ax1.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    ax1.grid(True, alpha=0.2)
    ax1.legend(loc="upper left", ncol=2, frameon=True, fontsize=9, title="Plan")

    # Bottom: stacked deaths
    bottom = np.zeros(len(pivot_death.index))
    for i, prod in enumerate(target_products):
        if prod in pivot_death.columns:
            vals = pivot_death[prod].values
            ax2.bar(
                pivot_death.index, vals, bottom=bottom,
                color=color_map.get(prod, plt.cm.tab20(i)),
                alpha=0.9, width=0.8, label=prod
            )
            bottom += vals

    ax2.set_ylabel("Number of Deaths", fontsize=11)
    ax2.set_xlabel("Observation Year", fontsize=11)
    ax2.grid(True, alpha=0.2, axis="y")
    plt.xticks(pivot_death.index, rotation=45)
    plt.grid(True)
    # fig.suptitle("A/E by Product and Observation Year", fontsize=14, fontweight="bold", y=0.98)
    # Save the plotting data as CSV (one-to-one with the PNG stem)
    if tables_dir is not None:
        _write_csv(g, _default_csv_path(out_path, tables_dir))

    _savefig_or_show(out_path)


def plot_chart_4_ae_by_product_age_split(df: pd.DataFrame, out_path: Path, age_threshold: float = 48.5, tables_dir: Path | None = None):
    """
    A/E by Product and Observation Year, split by Attained Age (< threshold vs >= threshold).
    Produces a 4-panel layout (2 rows: age splits; each row has line+bar).
    """
    year_col = "Observation_Year" if "Observation_Year" in df.columns else "Year"
    req = [year_col, "Insurance_Plan", "Attained_Age", "Death_Count", "Expected_Deaths"]
    missing = [c for c in req if c not in df.columns]
    if missing:
        raise KeyError(f"Missing required columns for plot_chart_4_ae_by_product_age_split: {missing}")

    target_products = ["Perm", "Term", "UL", "ULSG"]

    tmp = df.loc[df["Insurance_Plan"].isin(target_products), req].copy()
    tmp["Age_Split"] = np.where(tmp["Attained_Age"] < age_threshold, f"Attained Age < {age_threshold}", f"Attained Age >= {age_threshold}")

    g = (
        tmp.groupby([year_col, "Age_Split", "Insurance_Plan"], observed=True, sort=True)
           .agg(Death_Count=("Death_Count", "sum"), Expected_Deaths=("Expected_Deaths", "sum"))
           .reset_index()
    )
    g["AE_Count"] = g["Death_Count"] / g["Expected_Deaths"]

    color_map = {"Perm": "#003366", "Term": "#7d3f98", "UL": "#5bc0de", "ULSG": "#d62728"}
    age_splits = [f"Attained Age < {age_threshold}", f"Attained Age >= {age_threshold}"]

    fig, axes = plt.subplots(
        4, 1, figsize=(12, 12), sharex=True,
        gridspec_kw={"height_ratios": [1.0, 0.9, 1.0, 0.9], "hspace": 0.05}
    )

    for i, age_cat in enumerate(age_splits):
        ax_line = axes[i * 2]
        ax_bar = axes[i * 2 + 1]
        subset = g[g["Age_Split"] == age_cat]

        pivot_ae = subset.pivot(index=year_col, columns="Insurance_Plan", values="AE_Count")
        pivot_death = subset.pivot(index=year_col, columns="Insurance_Plan", values="Death_Count").fillna(0)

        for j, prod in enumerate(target_products):
            if prod in pivot_ae.columns:
                ax_line.plot(
                    pivot_ae.index, pivot_ae[prod],
                    color=color_map.get(prod, plt.cm.tab20(j)),
                    linewidth=2.3, marker="o", markersize=3.8, label=prod
                )

        ax_line.axhline(1.0, color="gray", linestyle="-", linewidth=1, alpha=0.5)
        ax_line.set_title(age_cat, fontsize=11, fontweight="bold", loc="left")
        ax_line.set_ylabel("A/E", fontsize=10)
        ax_line.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
        ax_line.grid(True, alpha=0.2)

        # useful: coarse scale for younger, fine scale for older (mirrors notebook intent)
        if age_cat.startswith("Attained Age <"):
            ax_line.set_ylim(0.6, 1.4)
        else:
            ax_line.set_ylim(0.85, 1.15)

        bottom = np.zeros(len(pivot_death.index))
        for j, prod in enumerate(target_products):
            if prod in pivot_death.columns:
                vals = pivot_death[prod].values
                ax_bar.bar(
                    pivot_death.index, vals, bottom=bottom,
                    color=color_map.get(prod, plt.cm.tab20(j)),
                    alpha=0.9, width=0.8
                )
                bottom += vals
        ax_bar.set_ylabel("Deaths", fontsize=10)
        ax_bar.grid(True, alpha=0.2, axis="y")

        if i == 0:
            ax_line.legend(loc="upper left", ncol=4, frameon=True, fontsize=9, title="Plan")

    axes[-1].set_xlabel("Observation Year", fontsize=11)
    plt.xticks(pivot_death.index, rotation=45)
    plt.grid(True)
    # fig.suptitle(f"A/E by Product and Observation Year (Age Split @ {age_threshold})", fontsize=14, fontweight="bold", y=0.98)
    # Save the plotting data as CSV (one-to-one with the PNG stem)
    if tables_dir is not None:
        _write_csv(g, _default_csv_path(out_path, tables_dir))

    _savefig_or_show(out_path)


def plot_chart_5_policies_exposed_by_product_issue_year(df: pd.DataFrame, out_path: Path, tables_dir: Path | None = None):
    """
    Policies Exposed by Product and Issue Year (1980–2019):
      - Top: 100% stacked bars (share)
      - Bottom: stacked bars (absolute)
    """
    year_col = "Observation_Year" if "Observation_Year" in df.columns else "Year"
    req = [year_col, "Duration", "Insurance_Plan", "Policies_Exposed"]
    missing = [c for c in req if c not in df.columns]
    if missing:
        raise KeyError(f"Missing required columns for plot_chart_5_policies_exposed_by_product_issue_year: {missing}")

    # Compute Issue_Year without mutating caller's df
    tmp = df[req].copy()
    tmp["Issue_Year"] = tmp[year_col] - tmp["Duration"] + 1
    tmp = tmp[(tmp["Issue_Year"] >= 1980) & (tmp["Issue_Year"] <= 2019)].copy()

    g = (
        tmp.groupby(["Issue_Year", "Insurance_Plan"], observed=True, sort=True)["Policies_Exposed"]
           .sum()
           .reset_index()
    )

    # Group rare products into "Other" for readability
    main = {"Perm", "Term", "UL", "ULSG", "VL", "VLSG"}
    g["Plan_Group"] = np.where(g["Insurance_Plan"].isin(list(main)), g["Insurance_Plan"], "Other")

    g2 = (
        g.groupby(["Issue_Year", "Plan_Group"], observed=True, sort=True)["Policies_Exposed"]
          .sum()
          .reset_index()
    )

    pivot = g2.pivot(index="Issue_Year", columns="Plan_Group", values="Policies_Exposed").fillna(0)

    # Percent share
    share = pivot.div(pivot.sum(axis=1).replace(0, np.nan), axis=0).fillna(0)

    color_map = {
        "Perm": "#003366",
        "Term": "#7d3f98",
        "UL": "#5bc0de",
        "ULSG": "#d62728",
        "VL": "#ffdd05",
        "VLSG": "#77ac30",
        "Other": "#001f3f",
    }

    order = [c for c in ["Perm","Term","UL","ULSG","VL","VLSG","Other"] if c in pivot.columns]

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(14, 10), sharex=True,
        gridspec_kw={"height_ratios": [1, 1], "hspace": 0.03}
    )

    bottom = np.zeros(len(share.index))
    for i, col in enumerate(order):
        vals = share[col].values
        ax1.bar(share.index, vals, bottom=bottom, color=color_map.get(col, plt.cm.tab20(i)), width=0.8, alpha=0.95, label=col)
        bottom += vals
    ax1.set_ylabel("Share of Policies Exposed", fontsize=11)
    ax1.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    ax1.grid(True, alpha=0.25, axis="y")
    ax1.legend(loc="upper center", ncol=min(7, len(order)), bbox_to_anchor=(0.5, 1.18), frameon=True, fontsize=9)

    bottom = np.zeros(len(pivot.index))
    for i, col in enumerate(order):
        vals = pivot[col].values
        ax2.bar(pivot.index, vals, bottom=bottom, color=color_map.get(col, plt.cm.tab20(i)), width=0.8, alpha=0.95, label=col)
        bottom += vals
    ax2.set_ylabel("Policies Exposed", fontsize=11)
    ax2.set_xlabel("Issue Year", fontsize=11)
    ax2.grid(True, alpha=0.25, axis="y")
    plt.xticks(pivot.index[::2], rotation=90)
    plt.grid(True)
    # fig.suptitle("Policies Exposed by Product and Issue Year (1980–2019)", fontsize=14, fontweight="bold", y=0.98)
    # Save the plotting data as CSV (one-to-one with the PNG stem)
    if tables_dir is not None:
        _write_csv(g2, _default_csv_path(out_path, tables_dir))

    _savefig_or_show(out_path)



if __name__ == '__main__':    
    df = load_data_optimized()

    # === Legacy charts (optional, still saved) ===
    plot_chart_1(df, ASSETS_DIR / "chart_1_overview.png", tables_dir=TABLES_DIR)
    plot_chart_2(df, ASSETS_DIR / "chart_2_overview.png", tables_dir=TABLES_DIR)
    plot_chart_10_deaths_product_issue_year(df, ASSETS_DIR / "deaths_product_issue_year.png", tables_dir=TABLES_DIR)
    plot_chart_11_product_age_amount_split(df, ASSETS_DIR / "ae_product_age_amount.png", tables_dir=TABLES_DIR)

    # === Additional charts from notebook/snippets ===
    plot_chart_3_ae_by_product_and_year(df, ASSETS_DIR / "ae_by_product_year.png", tables_dir=TABLES_DIR)
    plot_chart_4_ae_by_product_age_split(df, ASSETS_DIR / "ae_by_product_year_age_split.png", tables_dir=TABLES_DIR)
    plot_chart_5_policies_exposed_by_product_issue_year(df, ASSETS_DIR / "policies_product_issue_year.png", tables_dir=TABLES_DIR)

    # === Trend Pack core outputs ===
    trendpack_yearly_compare_year_sex(
        df,
        fig_path=ASSETS_DIR / "yearly_compare_year_sex.png",
        csv_path=TABLES_DIR / "yearly_compare_year_sex.csv"
    )

    trendpack_age_curve(
        df,
        fig_path=ASSETS_DIR / "age_curve.png",
        csv_path=TABLES_DIR / "age_curve.csv"
    )

    trendpack_product_mix_shift(
        df,
        fig_path=ASSETS_DIR / "product_mix_shift.png",
        csv_path=TABLES_DIR / "product_mix_shift.csv"
    )

    trendpack_drift_flags_from_year_sex(
        df,
        csv_path=TABLES_DIR / "drift_flags.csv"
    )
    manifest = {
        "trend_pack_dir": str(TREND_PACK_DIR),
        "assets_dir": str(ASSETS_DIR),
        "tables_dir": str(TABLES_DIR),
        "assets": sorted([p.name for p in ASSETS_DIR.glob("*.png")]),
        "tables": sorted([p.name for p in TABLES_DIR.glob("*.csv")]),
    }
    (TREND_PACK_DIR / "artifacts_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )
    print("✓ Wrote artifact manifest:", TREND_PACK_DIR / "artifacts_manifest.json")