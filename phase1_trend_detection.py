"""
Phase 1: Trend Detection
========================
Identify demographic segments with statistically significant mortality deterioration.

Methods:
1. A/E Ratio Calculation by segment and year
2. Mann-Kendall trend test for monotonic trends

Key result to verify:
- Male smokers aged 25-44: +24.5% increase in A/E ratio (from ~1.28 to ~1.63)
"""

import os
import pandas as pd
import numpy as np
import pymannkendall as mk
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

# Configuration — resolved relative to this file (soa_ai_v2/code/)
# OUTPUT_DIR honors the optional SOA_RESULTS_DIR env var so a reproduction run
# can write to a fresh folder without clobbering the shipped results/ reference.
DATA_DIR = Path(__file__).resolve().parents[1] / "data"
OUTPUT_DIR = Path(os.environ.get("SOA_RESULTS_DIR") or (Path(__file__).resolve().parents[1] / "results"))
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def load_data(file_path=None, sample_size=None):
    """Load the ILEC dataset."""
    if file_path is None:
        file_path = DATA_DIR / "ILEC_2012_19 - 20240429.txt"

    col_names = [
        'Observation_Year', 'Age_Ind', 'Sex', 'Smoker_Status', 'Insurance_Plan',
        'Issue_Age', 'Duration', 'Face_Amount_Band', 'Issue_Year', 'Attained_Age',
        'SOA_Antp_Lvl_TP', 'SOA_Guar_Lvl_TP', 'SOA_Post_Lvl_Ind', 'Slct_Ult_Ind',
        'Preferred_Indicator', 'Number_of_Pfd_Classes', 'Preferred_Class',
        'Amount_Exposed', 'Policies_Exposed', 'Death_Claim_Amount', 'Death_Count',
        'ExpDth_VBT2015_Cnt', 'ExpDth_VBT2015_Amt', 'ExpDth_VBT2015wMI_Cnt',
        'ExpDth_VBT2015wMI_Amt', 'Cen2MomP1wMI_Amt', 'Cen2MomP2wMI_Amt',
        'Cen3MomP1wMI_Amt', 'Cen3MomP2wMI_Amt', 'Cen3MomP3wMI_Amt'
    ]

    print(f"Loading data from {file_path}...")

    if sample_size:
        df = pd.read_csv(file_path, sep='\t', nrows=sample_size, names=col_names, skiprows=1)
    else:
        # Load in chunks for large file
        chunks = []
        chunk_size = 1_000_000
        for chunk in tqdm(pd.read_csv(file_path, sep='\t', chunksize=chunk_size,
                                       names=col_names, skiprows=1),
                          desc="Loading chunks"):
            chunks.append(chunk)
        df = pd.concat(chunks, ignore_index=True)

    print(f"Loaded {len(df):,} records")
    return df


def create_age_band(attained_age):
    """Create age band categories."""
    if attained_age < 25:
        return '0-24'
    elif attained_age <= 44:
        return '25-44'
    elif attained_age <= 64:
        return '45-64'
    else:
        return '65+'


def preprocess_data(df):
    """Preprocess data and create segment columns."""
    print("Preprocessing data...")

    # Create Age Band
    df['Age_Band'] = df['Attained_Age'].apply(create_age_band)

    # Create primary segment: Sex × Age Band
    df['Primary_Segment'] = df['Sex'] + '_' + df['Age_Band']

    # Create extended segment: Sex × Age Band × Smoker Status
    df['Extended_Segment'] = df['Sex'] + '_' + df['Age_Band'] + '_' + df['Smoker_Status']

    # Filter valid records
    df = df[df['Policies_Exposed'] > 0]
    df = df[df['ExpDth_VBT2015_Cnt'] > 0]

    print(f"After filtering: {len(df):,} records")
    return df


def calculate_ae_ratios(df, segment_col='Extended_Segment'):
    """
    Calculate A/E ratios by segment and year.

    A/E = Actual Deaths / Expected Deaths
    Expected Deaths = ExpDth_VBT2015_Cnt (from 2015 VBT mortality table)
    """
    print(f"\nCalculating A/E ratios by {segment_col}...")

    # Aggregate by segment and year
    agg = df.groupby([segment_col, 'Observation_Year']).agg({
        'Death_Count': 'sum',
        'ExpDth_VBT2015_Cnt': 'sum',
        'Policies_Exposed': 'sum',
        'Amount_Exposed': 'sum',
        'Death_Claim_Amount': 'sum'
    }).reset_index()

    # Calculate A/E ratio (count basis)
    agg['AE_Ratio'] = agg['Death_Count'] / agg['ExpDth_VBT2015_Cnt']

    # Also calculate amount-weighted A/E for comparison
    # (would need ExpDth_VBT2015_Amt for this)

    print(f"Calculated A/E for {agg[segment_col].nunique()} segments")
    return agg


def mann_kendall_trend_test(series):
    """
    Run Mann-Kendall trend test.

    Returns:
        dict: tau, p_value, trend direction, slope (Sen's slope)
    """
    if len(series) < 4:
        return {'tau': np.nan, 'p_value': np.nan, 'trend': 'insufficient data', 'slope': np.nan}

    try:
        result = mk.original_test(series)
        return {
            'tau': result.Tau,
            'p_value': result.p,
            'trend': result.trend,
            'slope': result.slope,
            's': result.s,
            'z': result.z
        }
    except Exception as e:
        print(f"MK test error: {e}")
        return {'tau': np.nan, 'p_value': np.nan, 'trend': 'error', 'slope': np.nan}


def analyze_trends(ae_data, segment_col='Extended_Segment'):
    """
    Analyze A/E ratio trends for all segments using Mann-Kendall test.
    """
    print("\n" + "="*60)
    print("TREND ANALYSIS RESULTS")
    print("="*60)

    # Pivot to get time series for each segment
    ae_pivot = ae_data.pivot(
        index='Observation_Year',
        columns=segment_col,
        values='AE_Ratio'
    )

    results = []
    for segment in tqdm(ae_pivot.columns, desc="Analyzing trends"):
        ts = ae_pivot[segment].dropna()

        if len(ts) < 2:
            continue

        # Mann-Kendall test (needs at least 4 points)
        if len(ts) >= 4:
            mk_result = mann_kendall_trend_test(ts)
        else:
            mk_result = {'tau': np.nan, 'p_value': np.nan, 'trend': 'insufficient data', 'slope': np.nan}

        # Calculate percent change from first to last value
        first_val = ts.iloc[0]
        last_val = ts.iloc[-1]
        pct_change = (last_val - first_val) / first_val * 100 if first_val != 0 else 0

        # Get counts for the segment
        seg_data = ae_data[ae_data[segment_col] == segment]
        total_deaths = seg_data['Death_Count'].sum()
        total_exposure = seg_data['Policies_Exposed'].sum()

        results.append({
            'Segment': segment,
            'First_Year': int(ts.index[0]),
            'Last_Year': int(ts.index[-1]),
            'First_AE': first_val,
            'Last_AE': last_val,
            'Pct_Change': pct_change,
            'MK_Tau': mk_result['tau'],
            'MK_P_Value': mk_result['p_value'],
            'MK_Trend': mk_result['trend'],
            'Sen_Slope': mk_result['slope'],
            'Total_Deaths': total_deaths,
            'Total_Exposure': total_exposure
        })

    results_df = pd.DataFrame(results)

    if len(results_df) == 0:
        print("Warning: No segments with sufficient data for trend analysis")
        return results_df, ae_pivot

    # Sort by percent change (descending for deterioration)
    results_df = results_df.sort_values('Pct_Change', ascending=False)

    return results_df, ae_pivot


def print_key_findings(results_df):
    """Print key findings from trend analysis."""
    print("\n" + "="*60)
    print("KEY FINDINGS")
    print("="*60)

    if len(results_df) == 0:
        print("No results to display")
        return

    # Statistically significant deteriorating trends (p < 0.05, positive change)
    deteriorating = results_df[
        (results_df['MK_P_Value'] < 0.05) &
        (results_df['Pct_Change'] > 0)
    ].copy()

    print(f"\n{len(deteriorating)} segments with significant deterioration (p < 0.05):")
    print("-" * 60)

    if len(deteriorating) > 0:
        for _, row in deteriorating.head(10).iterrows():
            print(f"\n{row['Segment']}:")
            print(f"  A/E: {row['First_AE']:.3f} → {row['Last_AE']:.3f} ({row['Pct_Change']:+.1f}%)")
            print(f"  MK tau: {row['MK_Tau']:.3f}, p-value: {row['MK_P_Value']:.4f}")
            print(f"  Total deaths: {row['Total_Deaths']:,.0f}")

    # Specific verification: Male smokers 25-44
    print("\n" + "="*60)
    print("VERIFICATION: Male Smokers 25-44")
    print("="*60)

    male_smoker_young = results_df[
        results_df['Segment'].str.contains('M_25-44_S', na=False)
    ]

    if not male_smoker_young.empty:
        row = male_smoker_young.iloc[0]
        print(f"\nActual Results:")
        print(f"  A/E 2012: {row['First_AE']:.4f}")
        print(f"  A/E 2019: {row['Last_AE']:.4f}")
        print(f"  Change: {row['Pct_Change']:+.1f}%")
        print(f"  MK tau: {row['MK_Tau']:.4f}")
        print(f"  p-value: {row['MK_P_Value']:.4f}")

        print(f"\nPaper Claims:")
        print(f"  A/E increase: +24.5% (from ~1.28 to ~1.63)")

        # Calculate match
        if abs(row['Pct_Change'] - 24.5) < 5:
            print("\n✓ RESULT MATCHES PAPER (within 5%)")
        else:
            print(f"\n⚠ RESULT DIFFERS FROM PAPER")
            print(f"  Difference: {row['Pct_Change'] - 24.5:+.1f} percentage points")
    else:
        print("Segment M_25-44_S not found in data")


def plot_ae_trends(ae_pivot, segments_to_plot=None, output_file=None):
    """Plot A/E ratio trends for selected segments."""
    if segments_to_plot is None:
        # Default: plot segments with largest changes
        segments_to_plot = ae_pivot.columns[:8]

    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.flatten()

    for idx, segment in enumerate(segments_to_plot[:8]):
        if segment not in ae_pivot.columns:
            continue

        ax = axes[idx]
        ts = ae_pivot[segment].dropna()

        ax.plot(ts.index, ts.values, 'b-o', linewidth=2, markersize=6)
        ax.axhline(y=1.0, color='r', linestyle='--', alpha=0.5, label='Expected (1.0)')

        ax.set_title(segment, fontsize=10)
        ax.set_xlabel('Year')
        ax.set_ylabel('A/E Ratio')
        ax.grid(True, alpha=0.3)

        # Add trend line
        if len(ts) >= 2:
            z = np.polyfit(ts.index, ts.values, 1)
            p = np.poly1d(z)
            ax.plot(ts.index, p(ts.index), 'g--', alpha=0.5)

    plt.tight_layout()

    if output_file:
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        print(f"Plot saved to {output_file}")

    plt.close()


def main(sample_size=None):
    """Main function for Phase 1 analysis."""
    print("\n" + "="*80)
    print("PHASE 1: TREND DETECTION")
    print("="*80)

    # Load data
    df = load_data(sample_size=sample_size)

    # Preprocess
    df = preprocess_data(df)

    # Calculate A/E ratios
    ae_data = calculate_ae_ratios(df, 'Extended_Segment')

    # Analyze trends
    results_df, ae_pivot = analyze_trends(ae_data, 'Extended_Segment')

    # Print findings
    print_key_findings(results_df)

    # Save results
    results_df.to_csv(OUTPUT_DIR / 'phase1_trend_results.csv', index=False)
    ae_data.to_csv(OUTPUT_DIR / 'phase1_ae_ratios.csv', index=False)
    print(f"\nResults saved to {OUTPUT_DIR}")

    # Plot key segments
    key_segments = [
        'M_25-44_S', 'M_25-44_NS', 'F_25-44_S', 'F_25-44_NS',
        'M_45-64_S', 'M_45-64_NS', 'F_45-64_S', 'F_45-64_NS'
    ]
    existing_segments = [s for s in key_segments if s in ae_pivot.columns]
    if existing_segments:
        plot_ae_trends(ae_pivot, existing_segments, OUTPUT_DIR / 'phase1_ae_trends.png')

    return results_df, ae_data, ae_pivot


if __name__ == "__main__":
    import sys

    # Use sample for testing, full data for final analysis
    sample = int(sys.argv[1]) if len(sys.argv) > 1 else None
    results_df, ae_data, ae_pivot = main(sample_size=sample)
