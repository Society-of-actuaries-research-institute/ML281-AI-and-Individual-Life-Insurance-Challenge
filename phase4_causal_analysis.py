"""
Phase 4: Conformal Prediction & Causal Attribution
==================================================

4A. Conformal Prediction: Uncertainty quantification for mortality predictions
4B. Synthetic Control Analysis: Causal attribution of mortality changes

Key results to verify:
- Conformal: 90% target → 91.6% actual coverage
- Causal effect: ~11.4 percentage points for males 25-44
- Financial impact: ~$540M excess claims
"""

import os
import pandas as pd
import numpy as np
from pathlib import Path
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# Configuration — resolved relative to this file (soa_ai_v2/code/)
# OUTPUT_DIR honors the optional SOA_RESULTS_DIR env var (default unchanged).
DATA_DIR = Path(__file__).resolve().parents[1] / "data"
OUTPUT_DIR = Path(os.environ.get("SOA_RESULTS_DIR") or (Path(__file__).resolve().parents[1] / "results"))
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


#=============================================================================
# PHASE 4A: CONFORMAL PREDICTION
#=============================================================================

def conformal_prediction_analysis(model, X_calib, y_calib, X_test, y_test, method='score'):
    """
    Apply conformal prediction for uncertainty quantification.

    Uses MAPIE library to wrap the base model and provide prediction sets
    with guaranteed coverage.

    Args:
        model: Trained sklearn-compatible classifier
        X_calib: Calibration features
        y_calib: Calibration labels
        X_test: Test features
        y_test: Test labels
        method: Conformal method ('score', 'cumulated_score', 'lac')

    Returns:
        dict: Coverage results at different confidence levels
    """
    print("\n" + "="*60)
    print("PHASE 4A: CONFORMAL PREDICTION")
    print("="*60)

    try:
        from mapie.classification import MapieClassifier

        # Fit MAPIE wrapper
        print("Fitting MAPIE classifier...")
        mapie = MapieClassifier(model, method=method, cv="prefit")
        mapie.fit(X_calib, y_calib)

        # Evaluate at multiple confidence levels
        results = []
        alphas = [0.1, 0.2, 0.3, 0.4]  # 90%, 80%, 70%, 60% confidence

        print("\nCoverage Analysis:")
        print("-" * 50)
        print(f"{'Target Coverage':>20} {'Actual Coverage':>20}")
        print("-" * 50)

        for alpha in alphas:
            target_coverage = 1 - alpha

            # Generate prediction sets
            y_pred, y_pred_set = mapie.predict(X_test, alpha=alpha)

            # Calculate actual coverage
            # For binary classification: check if true label is in prediction set
            coverage = calculate_coverage(y_test.values, y_pred_set)

            results.append({
                'Alpha': alpha,
                'Target_Coverage': target_coverage,
                'Actual_Coverage': coverage,
                'Coverage_Gap': coverage - target_coverage
            })

            print(f"{target_coverage:>18.0%} {coverage:>18.1%}")

        print("-" * 50)

        # Verification
        print("\n--- VERIFICATION ---")
        results_df = pd.DataFrame(results)

        # Check 90% coverage
        r90 = results_df[results_df['Alpha'] == 0.1]
        if not r90.empty:
            actual_90 = r90['Actual_Coverage'].iloc[0] * 100
            print(f"90% target coverage → {actual_90:.1f}% actual")
            print(f"Paper claims: 90% → 91.6%")

        # Check 80% coverage
        r80 = results_df[results_df['Alpha'] == 0.2]
        if not r80.empty:
            actual_80 = r80['Actual_Coverage'].iloc[0] * 100
            print(f"80% target coverage → {actual_80:.1f}% actual")
            print(f"Paper claims: 80% → 76.5%")

        return results_df

    except ImportError:
        print("Warning: MAPIE library not available. Skipping conformal prediction.")
        return None
    except Exception as e:
        print(f"Error in conformal prediction: {e}")
        return None


def calculate_coverage(y_true, y_pred_set):
    """
    Calculate the actual coverage of prediction sets.

    Args:
        y_true: True labels (array)
        y_pred_set: Prediction set array from MAPIE (n_samples, n_classes, n_alpha)

    Returns:
        float: Fraction of true labels contained in prediction sets
    """
    n_samples = len(y_true)
    covered = 0

    for i in range(n_samples):
        true_label = int(y_true[i])
        # Check if true label is in the prediction set
        if y_pred_set[i, true_label, 0]:  # First alpha level
            covered += 1

    return covered / n_samples


def plot_coverage_analysis(results_df, output_file=None):
    """Plot coverage analysis results."""
    fig, ax = plt.subplots(figsize=(10, 6))

    targets = results_df['Target_Coverage'] * 100
    actuals = results_df['Actual_Coverage'] * 100

    x = np.arange(len(targets))
    width = 0.35

    bars1 = ax.bar(x - width/2, targets, width, label='Target Coverage', color='steelblue')
    bars2 = ax.bar(x + width/2, actuals, width, label='Actual Coverage', color='coral')

    ax.axhline(y=100, color='gray', linestyle='--', alpha=0.3)
    ax.plot([x[0]-0.5, x[-1]+0.5], [targets.iloc[0], targets.iloc[-1]],
            'k--', alpha=0.3, label='Diagonal (perfect calibration)')

    ax.set_xlabel('Confidence Level')
    ax.set_ylabel('Coverage (%)')
    ax.set_title('Conformal Prediction Coverage Analysis')
    ax.set_xticks(x)
    ax.set_xticklabels([f"{t:.0f}%" for t in targets])
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()

    if output_file:
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        print(f"Coverage plot saved to {output_file}")

    plt.close()


#=============================================================================
# PHASE 4B: SYNTHETIC CONTROL ANALYSIS
#=============================================================================

class SyntheticControlMethod:
    """
    Synthetic Control Method for causal inference.

    Based on Abadie, Diamond, & Hainmueller (2010):
    "Synthetic Control Methods for Comparative Case Studies"
    """

    def __init__(self):
        self.weights = None
        self.treatment_unit = None
        self.control_units = None
        self.pre_period = None
        self.post_period = None

    def fit(self, Y, treatment_idx, pre_period_idx):
        """
        Fit synthetic control weights.

        Args:
            Y: (T x N) matrix of outcomes (time periods x units)
            treatment_idx: Index of treatment unit
            pre_period_idx: Indices of pre-treatment periods

        Returns:
            self
        """
        T, N = Y.shape

        # Separate treatment and control outcomes
        Y_treat = Y[pre_period_idx, treatment_idx]
        Y_control = Y[pre_period_idx, :]
        Y_control = np.delete(Y_control, treatment_idx, axis=1)

        N_control = Y_control.shape[1]

        # Optimize weights to minimize pre-period prediction error
        def objective(w):
            synthetic = Y_control @ w
            return np.sum((Y_treat - synthetic) ** 2)

        # Constraints: weights sum to 1
        constraints = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}

        # Bounds: weights are non-negative
        bounds = [(0, 1)] * N_control

        # Initial weights: equal
        w0 = np.ones(N_control) / N_control

        result = minimize(
            objective, w0, method='SLSQP',
            bounds=bounds, constraints=constraints,
            options={'maxiter': 1000}
        )

        self.weights = result.x
        self.treatment_idx = treatment_idx
        self.optimization_result = result

        return self

    def predict(self, Y):
        """
        Generate synthetic control outcome.

        Args:
            Y: (T x N) outcome matrix

        Returns:
            array: Synthetic control outcome for all time periods
        """
        Y_control = np.delete(Y, self.treatment_idx, axis=1)
        return Y_control @ self.weights

    def estimate_effect(self, Y, pre_period_idx, post_period_idx):
        """
        Estimate treatment effect.

        Returns:
            dict: Pre-period effect, post-period effect, and causal estimate
        """
        Y_treat = Y[:, self.treatment_idx]
        Y_synth = self.predict(Y)

        pre_effect = np.mean(Y_treat[pre_period_idx] - Y_synth[pre_period_idx])
        post_effect = np.mean(Y_treat[post_period_idx] - Y_synth[post_period_idx])
        causal_effect = post_effect - pre_effect

        return {
            'pre_effect': pre_effect,
            'post_effect': post_effect,
            'causal_effect': causal_effect,
            'treatment': Y_treat,
            'synthetic': Y_synth
        }


def synthetic_control_analysis(ae_data, treatment_segment, control_segments, pre_years, post_years):
    """
    Run synthetic control analysis for causal attribution.

    Args:
        ae_data: DataFrame with A/E ratios by segment and year
        treatment_segment: Segment pattern for treatment (e.g., 'M_25-44')
        control_segments: List of segment patterns for control pool
        pre_years: Pre-treatment years
        post_years: Post-treatment years

    Returns:
        dict: Analysis results
    """
    print("\n" + "="*60)
    print("PHASE 4B: SYNTHETIC CONTROL ANALYSIS")
    print("="*60)

    # Aggregate A/E by segment and year
    if 'Extended_Segment' in ae_data.columns:
        segment_col = 'Extended_Segment'
    else:
        segment_col = 'Primary_Segment'

    # Get treatment segment data
    treatment_data = ae_data[ae_data[segment_col].str.contains(treatment_segment, na=False)]
    treatment_ts = treatment_data.groupby('Observation_Year')['AE_Ratio'].mean()

    # Get control segments data
    control_data = {}
    for seg_pattern in control_segments:
        seg_data = ae_data[ae_data[segment_col].str.contains(seg_pattern, na=False)]
        if len(seg_data) > 0:
            control_data[seg_pattern] = seg_data.groupby('Observation_Year')['AE_Ratio'].mean()

    if not control_data:
        print("Warning: No valid control segments found")
        return None

    print(f"\nTreatment segment: {treatment_segment}")
    print(f"Control segments: {list(control_data.keys())}")
    print(f"Pre-treatment period: {pre_years}")
    print(f"Post-treatment period: {post_years}")

    # Create outcome matrix
    all_years = sorted(treatment_ts.index)
    units = [treatment_segment] + list(control_data.keys())

    Y = pd.DataFrame(index=all_years)
    Y[treatment_segment] = treatment_ts
    for seg, ts in control_data.items():
        Y[seg] = ts

    Y = Y.dropna()
    Y_matrix = Y.values

    # Define period indices
    years_list = list(Y.index)
    pre_idx = [years_list.index(y) for y in pre_years if y in years_list]
    post_idx = [years_list.index(y) for y in post_years if y in years_list]

    print(f"\nYears in analysis: {years_list}")
    print(f"Pre-period indices: {pre_idx}")
    print(f"Post-period indices: {post_idx}")

    # Fit synthetic control
    sc = SyntheticControlMethod()
    sc.fit(Y_matrix, treatment_idx=0, pre_period_idx=pre_idx)

    # Estimate effects
    effects = sc.estimate_effect(Y_matrix, pre_idx, post_idx)

    # Print results
    print("\n--- RESULTS ---")
    print(f"Pre-period average effect: {effects['pre_effect']:.4f}")
    print(f"Post-period average effect: {effects['post_effect']:.4f}")
    print(f"Estimated causal effect: {effects['causal_effect']:.4f}")
    print(f"  ({effects['causal_effect']*100:.1f} percentage points)")

    print("\n--- VERIFICATION ---")
    print(f"Paper claims: ~0.114 (11.4 percentage points)")
    print(f"for opioid crisis impact on males 25-44")

    if abs(effects['causal_effect'] - 0.114) < 0.05:
        print("✓ Result aligns with paper (within 0.05)")
    else:
        print(f"⚠ Result differs: {effects['causal_effect'] - 0.114:+.4f}")

    # Add weights info
    print("\n--- SYNTHETIC CONTROL WEIGHTS ---")
    for i, (seg, weight) in enumerate(zip(list(control_data.keys()), sc.weights)):
        if weight > 0.01:
            print(f"  {seg}: {weight:.3f}")

    # Prepare return dict
    results = {
        'treatment_segment': treatment_segment,
        'control_segments': list(control_data.keys()),
        'weights': dict(zip(control_data.keys(), sc.weights)),
        'pre_effect': effects['pre_effect'],
        'post_effect': effects['post_effect'],
        'causal_effect': effects['causal_effect'],
        'years': years_list,
        'treatment_ae': effects['treatment'],
        'synthetic_ae': effects['synthetic'],
        'Y_matrix': Y_matrix,
        'Y_df': Y
    }

    return results


def calculate_financial_impact(causal_effect, ae_data, treatment_segment, post_years, avg_face_amount=None):
    """
    Calculate financial impact (excess claims) due to mortality deterioration.

    Excess claims = (Causal A/E increase) × (Expected deaths) × (Average face amount)
    """
    print("\n--- FINANCIAL IMPACT CALCULATION ---")

    if 'Extended_Segment' in ae_data.columns:
        segment_col = 'Extended_Segment'
    else:
        segment_col = 'Primary_Segment'

    # Get treatment segment data for post period
    treatment_data = ae_data[
        (ae_data[segment_col].str.contains(treatment_segment, na=False)) &
        (ae_data['Observation_Year'].isin(post_years))
    ]

    if len(treatment_data) == 0:
        print("Warning: No data for financial impact calculation")
        return None

    # Total expected deaths
    total_expected_deaths = treatment_data['ExpDth_VBT2015_Cnt'].sum()

    # Average face amount (from death claims if available)
    if 'Death_Claim_Amount' in treatment_data.columns and 'Death_Count' in treatment_data.columns:
        total_claims = treatment_data['Death_Claim_Amount'].sum()
        total_deaths = treatment_data['Death_Count'].sum()
        if total_deaths > 0:
            avg_face_amount = total_claims / total_deaths

    if avg_face_amount is None:
        avg_face_amount = 150000  # Default assumption

    # Calculate excess claims
    excess_claims = causal_effect * total_expected_deaths * avg_face_amount

    print(f"Total expected deaths ({treatment_segment}, {min(post_years)}-{max(post_years)}): {total_expected_deaths:,.0f}")
    print(f"Average face amount: ${avg_face_amount:,.0f}")
    print(f"Causal A/E increase: {causal_effect:.4f}")
    print(f"\nEstimated excess claims: ${excess_claims/1e6:,.1f} million")

    print("\n--- VERIFICATION ---")
    print("Paper claims: ~$540 million for males 25-44 during 2015-2019")

    if abs(excess_claims/1e6 - 540) < 200:
        print("✓ Result in similar range (within $200M)")
    else:
        print(f"⚠ Result differs significantly: ${(excess_claims/1e6 - 540):+.1f}M difference")

    return {
        'causal_effect': causal_effect,
        'total_expected_deaths': total_expected_deaths,
        'avg_face_amount': avg_face_amount,
        'excess_claims': excess_claims,
        'treatment_segment': treatment_segment,
        'post_years': post_years
    }


def plot_synthetic_control(results, output_file=None):
    """Plot synthetic control analysis results."""
    fig, axes = plt.subplots(2, 1, figsize=(12, 10))

    years = results['years']
    treatment_ae = results['treatment_ae']
    synthetic_ae = results['synthetic_ae']

    # Top plot: Treatment vs Synthetic
    ax1 = axes[0]
    ax1.plot(years, treatment_ae, 'b-o', linewidth=2, markersize=8,
             label=f"Treatment ({results['treatment_segment']})")
    ax1.plot(years, synthetic_ae, 'r--s', linewidth=2, markersize=8,
             label='Synthetic Control')

    # Mark pre/post periods
    ax1.axvline(x=2014.5, color='gray', linestyle=':', linewidth=2)
    ax1.annotate('Pre-period', xy=(2013, ax1.get_ylim()[1]*0.95), fontsize=10)
    ax1.annotate('Post-period', xy=(2016, ax1.get_ylim()[1]*0.95), fontsize=10)

    ax1.set_xlabel('Year')
    ax1.set_ylabel('A/E Ratio')
    ax1.set_title('Synthetic Control Analysis: Treatment vs. Synthetic')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Bottom plot: Treatment Effect
    ax2 = axes[1]
    effect = treatment_ae - synthetic_ae
    ax2.bar(years, effect, color=['green' if e > 0 else 'red' for e in effect], alpha=0.7)
    ax2.axhline(y=0, color='black', linewidth=1)
    ax2.axvline(x=2014.5, color='gray', linestyle=':', linewidth=2)

    ax2.set_xlabel('Year')
    ax2.set_ylabel('Treatment Effect (A/E Difference)')
    ax2.set_title('Estimated Treatment Effect (Treatment - Synthetic)')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    if output_file:
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        print(f"Synthetic control plot saved to {output_file}")

    plt.close()


def main(ae_data=None, model=None, X_calib=None, y_calib=None, X_test=None, y_test=None):
    """Main function for Phase 4 analysis."""
    print("\n" + "="*80)
    print("PHASE 4: CONFORMAL PREDICTION & CAUSAL ATTRIBUTION")
    print("="*80)

    results = {}

    # Load A/E data if not provided
    if ae_data is None:
        ae_data_path = OUTPUT_DIR / 'phase1_ae_ratios.csv'
        if ae_data_path.exists():
            ae_data = pd.read_csv(ae_data_path)
        else:
            print("Warning: A/E data not found. Please run Phase 1 first.")

    # 4A: Conformal Prediction
    if model is not None and X_calib is not None:
        conformal_results = conformal_prediction_analysis(
            model, X_calib, y_calib, X_test, y_test
        )
        if conformal_results is not None:
            results['conformal'] = conformal_results
            conformal_results.to_csv(OUTPUT_DIR / 'phase4a_conformal_results.csv', index=False)
            plot_coverage_analysis(conformal_results, OUTPUT_DIR / 'phase4a_coverage_analysis.png')

    # 4B: Synthetic Control Analysis
    if ae_data is not None:
        # Treatment: Males 25-44
        treatment_segment = 'M_25-44'

        # Control pool: Older females and older males (less affected by opioid crisis)
        control_segments = ['F_45-64', 'F_65', 'M_65']

        # Time periods
        pre_years = [2012, 2013, 2014]
        post_years = [2015, 2016, 2017, 2018, 2019]

        sc_results = synthetic_control_analysis(
            ae_data, treatment_segment, control_segments, pre_years, post_years
        )

        if sc_results is not None:
            results['synthetic_control'] = sc_results

            # Plot
            plot_synthetic_control(sc_results, OUTPUT_DIR / 'phase4b_synthetic_control.png')

            # Financial impact
            financial_results = calculate_financial_impact(
                sc_results['causal_effect'], ae_data, treatment_segment, post_years
            )
            if financial_results:
                results['financial_impact'] = financial_results

            # Save synthetic control results
            sc_summary = pd.DataFrame([{
                'Treatment_Segment': sc_results['treatment_segment'],
                'Pre_Effect': sc_results['pre_effect'],
                'Post_Effect': sc_results['post_effect'],
                'Causal_Effect': sc_results['causal_effect']
            }])
            sc_summary.to_csv(OUTPUT_DIR / 'phase4b_synthetic_control_results.csv', index=False)

    print(f"\nResults saved to {OUTPUT_DIR}")

    return results


if __name__ == "__main__":
    # Run standalone (requires Phase 1 results)
    results = main()
