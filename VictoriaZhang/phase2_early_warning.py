"""
Phase 2: Change Point Detection & Early Warning System
======================================================
Compare AI-based vs traditional early warning systems for mortality surveillance.

Methods:
1. Bayesian Online Change Point Detection (BOCPD)
2. AI-based: z-score from rolling baseline
3. Traditional: rolling average threshold

Key results to verify:
- 60% of change points cluster in 2015-2016
- Average detection advantage: 14.3 months
- 47 segments where AI detected first
- 12 segments where only AI detected
"""

import os
import pandas as pd
import numpy as np
from pathlib import Path
from scipy.stats import norm
from scipy.special import gammaln
import matplotlib.pyplot as plt
from tqdm import tqdm

# Configuration — resolved relative to this file (soa_ai_v2/code/)
# OUTPUT_DIR honors the optional SOA_RESULTS_DIR env var (default unchanged).
OUTPUT_DIR = Path(os.environ.get("SOA_RESULTS_DIR") or (Path(__file__).resolve().parents[1] / "results"))
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


class BayesianOnlineChangePointDetection:
    """
    Bayesian Online Change Point Detection (BOCPD)
    Based on Adams & MacKay (2007): "Bayesian Online Changepoint Detection"

    This implementation uses a Gaussian observation model with unknown mean and variance.
    """

    def __init__(self, hazard_rate=1/50, prior_params=None):
        """
        Initialize BOCPD.

        Args:
            hazard_rate: Prior probability of changepoint at each step (1/expected_run_length)
            prior_params: Dict with 'mu0', 'kappa0', 'alpha0', 'beta0' for Normal-Inverse-Gamma prior
        """
        self.hazard_rate = hazard_rate

        # Normal-Inverse-Gamma prior hyperparameters
        if prior_params is None:
            self.mu0 = 0
            self.kappa0 = 1
            self.alpha0 = 1
            self.beta0 = 1
        else:
            self.mu0 = prior_params.get('mu0', 0)
            self.kappa0 = prior_params.get('kappa0', 1)
            self.alpha0 = prior_params.get('alpha0', 1)
            self.beta0 = prior_params.get('beta0', 1)

    def _student_t_pdf(self, x, mu, sigma, nu):
        """Evaluate Student-t PDF."""
        coef = np.exp(gammaln((nu + 1) / 2) - gammaln(nu / 2))
        coef *= 1 / (np.sqrt(nu * np.pi) * sigma)
        return coef * (1 + ((x - mu) ** 2) / (nu * sigma ** 2)) ** (-(nu + 1) / 2)

    def detect(self, data):
        """
        Run BOCPD on time series data.

        Args:
            data: 1D array of observations

        Returns:
            dict containing:
                - run_length_probs: (T, T) matrix of run length probabilities
                - change_point_probs: array of change point probabilities at each time
                - most_likely_cp: index of most likely change point
                - p_no_change: probability of no change
        """
        data = np.asarray(data)
        T = len(data)

        # Initialize sufficient statistics arrays
        # For each run length r at time t, we track the posterior parameters
        mu = np.zeros((T + 1, T + 1))
        kappa = np.zeros((T + 1, T + 1))
        alpha = np.zeros((T + 1, T + 1))
        beta = np.zeros((T + 1, T + 1))

        # Initialize with prior
        mu[0, :] = self.mu0
        kappa[0, :] = self.kappa0
        alpha[0, :] = self.alpha0
        beta[0, :] = self.beta0

        # Run length distribution
        R = np.zeros((T + 1, T + 1))
        R[0, 0] = 1.0

        # Predictive probabilities
        pred_probs = np.zeros(T + 1)

        # Change point probabilities
        cp_probs = np.zeros(T)

        for t in range(T):
            x = data[t]

            # Calculate predictive probability for each run length
            # Under Normal-Inverse-Gamma, the predictive is Student-t
            for r in range(t + 1):
                # Posterior parameters for run length r
                m = mu[t, r]
                k = kappa[t, r]
                a = alpha[t, r]
                b = beta[t, r]

                # Predictive Student-t parameters
                pred_mu = m
                pred_sigma = np.sqrt(b * (k + 1) / (a * k))
                pred_nu = 2 * a

                # Evaluate predictive probability
                pred_probs[r] = self._student_t_pdf(x, pred_mu, pred_sigma, pred_nu)

            # Growth probabilities (no change point)
            growth = R[t, :t + 1] * pred_probs[:t + 1] * (1 - self.hazard_rate)

            # Change point probability
            cp = np.sum(R[t, :t + 1] * pred_probs[:t + 1]) * self.hazard_rate
            cp_probs[t] = cp

            # Update run length distribution
            R[t + 1, 1:t + 2] = growth
            R[t + 1, 0] = cp

            # Normalize
            norm_const = np.sum(R[t + 1, :t + 2])
            if norm_const > 0:
                R[t + 1, :t + 2] /= norm_const

            # Update sufficient statistics for each run length
            for r in range(t + 2):
                if r == 0:
                    # Reset to prior
                    mu[t + 1, r] = self.mu0
                    kappa[t + 1, r] = self.kappa0
                    alpha[t + 1, r] = self.alpha0
                    beta[t + 1, r] = self.beta0
                else:
                    # Update posterior with new observation
                    k = kappa[t, r - 1]
                    m = mu[t, r - 1]
                    a = alpha[t, r - 1]
                    b = beta[t, r - 1]

                    kappa[t + 1, r] = k + 1
                    mu[t + 1, r] = (k * m + x) / (k + 1)
                    alpha[t + 1, r] = a + 0.5
                    beta[t + 1, r] = b + (k * (x - m) ** 2) / (2 * (k + 1))

        # Find most likely change point
        most_likely_cp = np.argmax(cp_probs)

        # Probability of no change (run length = T at end)
        p_no_change = R[T, T] if T < R.shape[1] else 0

        return {
            'run_length_probs': R,
            'change_point_probs': cp_probs,
            'most_likely_cp': most_likely_cp,
            'p_no_change': p_no_change
        }


def ai_early_warning(ae_series, threshold_sigma=1.5, window=3):
    """
    AI-based early warning system.

    Triggers when current A/E z-score exceeds threshold standard deviations
    from a rolling baseline.

    Args:
        ae_series: pandas Series with year index and A/E values
        threshold_sigma: z-score threshold for triggering
        window: rolling window size for baseline

    Returns:
        tuple: (trigger_year, z_score at trigger) or (None, None)
    """
    if len(ae_series) < window + 1:
        return None, None

    # Calculate rolling statistics
    rolling_mean = ae_series.rolling(window, min_periods=window).mean()
    rolling_std = ae_series.rolling(window, min_periods=window).std()

    # Calculate z-scores relative to previous baseline (avoid look-ahead)
    z_scores = pd.Series(index=ae_series.index, dtype=float)
    for i in range(window, len(ae_series)):
        if rolling_std.iloc[i - 1] > 0:
            z_scores.iloc[i] = (ae_series.iloc[i] - rolling_mean.iloc[i - 1]) / rolling_std.iloc[i - 1]
        else:
            z_scores.iloc[i] = 0

    # Find first trigger
    triggers = z_scores[z_scores > threshold_sigma]
    if len(triggers) > 0:
        trigger_year = triggers.index[0]
        return trigger_year, triggers.iloc[0]

    return None, None


def traditional_early_warning(ae_series, threshold_pct=0.10, window=3):
    """
    Traditional early warning system.

    Triggers when rolling average exceeds the baseline by threshold percentage.

    Args:
        ae_series: pandas Series with year index and A/E values
        threshold_pct: percentage threshold (e.g., 0.10 for 10%)
        window: rolling window size

    Returns:
        tuple: (trigger_year, rolling_avg at trigger) or (None, None)
    """
    if len(ae_series) < window + 1:
        return None, None

    # Baseline is the average of first 'window' observations
    baseline = ae_series.iloc[:window].mean()

    # Rolling average
    rolling_avg = ae_series.rolling(window, min_periods=window).mean()

    # Threshold
    threshold_value = baseline * (1 + threshold_pct)

    # Find first trigger (starting after baseline period)
    for year in ae_series.index[window:]:
        if rolling_avg[year] > threshold_value:
            return year, rolling_avg[year]

    return None, None


def analyze_early_warning(ae_pivot, segment_col='Extended_Segment'):
    """
    Compare AI vs traditional early warning across all segments.
    """
    print("\n" + "="*60)
    print("EARLY WARNING SYSTEM COMPARISON")
    print("="*60)

    results = []
    change_point_years = []

    # Initialize BOCPD
    bocpd = BayesianOnlineChangePointDetection(hazard_rate=1/4)

    for segment in tqdm(ae_pivot.columns, desc="Analyzing segments"):
        ts = ae_pivot[segment].dropna().sort_index()

        if len(ts) < 5:
            continue

        # BOCPD analysis
        bocpd_result = bocpd.detect(ts.values)
        cp_idx = bocpd_result['most_likely_cp']
        cp_year = ts.index[cp_idx] if cp_idx < len(ts) else None
        if cp_year:
            change_point_years.append(cp_year)

        # AI early warning (z-score based)
        ai_trigger, ai_zscore = ai_early_warning(ts, threshold_sigma=1.5, window=3)

        # Traditional early warning (rolling average based)
        trad_trigger, trad_value = traditional_early_warning(ts, threshold_pct=0.10, window=3)

        # Calculate detection advantage
        detection_advantage_months = None
        ai_detected_first = False
        ai_only_detected = False

        if ai_trigger is not None and trad_trigger is not None:
            # Convert years to approximate months (mid-year assumption)
            ai_months = ai_trigger * 12 + 6
            trad_months = trad_trigger * 12 + 6
            detection_advantage_months = trad_months - ai_months
            ai_detected_first = detection_advantage_months > 0
        elif ai_trigger is not None and trad_trigger is None:
            ai_only_detected = True
            ai_detected_first = True
        elif ai_trigger is None and trad_trigger is not None:
            ai_detected_first = False

        results.append({
            'Segment': segment,
            'CP_Year': cp_year,
            'CP_Probability': bocpd_result['change_point_probs'][cp_idx] if cp_idx < len(bocpd_result['change_point_probs']) else None,
            'P_No_Change': bocpd_result['p_no_change'],
            'AI_Trigger_Year': ai_trigger,
            'AI_Z_Score': ai_zscore,
            'Trad_Trigger_Year': trad_trigger,
            'Trad_Value': trad_value,
            'Detection_Advantage_Months': detection_advantage_months,
            'AI_Detected_First': ai_detected_first,
            'AI_Only_Detected': ai_only_detected
        })

    results_df = pd.DataFrame(results)
    return results_df, change_point_years


def print_phase2_findings(results_df, change_point_years):
    """Print key findings from Phase 2 analysis."""
    print("\n" + "="*60)
    print("PHASE 2 KEY FINDINGS")
    print("="*60)

    # Change point distribution
    print("\n--- Change Point Distribution ---")
    cp_series = pd.Series(change_point_years)
    if len(cp_series) > 0:
        print(f"Total segments analyzed: {len(results_df)}")
        print(f"Segments with change points: {len(cp_series)}")

        # Count by year
        cp_counts = cp_series.value_counts().sort_index()
        print("\nChange points by year:")
        for year, count in cp_counts.items():
            print(f"  {year}: {count}")

        # Calculate 2015-2016 clustering
        cp_2015_2016 = ((cp_series >= 2015) & (cp_series <= 2016)).sum()
        pct_2015_2016 = cp_2015_2016 / len(cp_series) * 100
        print(f"\nChange points in 2015-2016: {cp_2015_2016} ({pct_2015_2016:.1f}%)")
        print(f"Paper claims: 60% in 2015-2016")

        if abs(pct_2015_2016 - 60) < 10:
            print("✓ Result aligns with paper")
        else:
            print(f"⚠ Result differs: {pct_2015_2016 - 60:+.1f}% difference")

    # Early warning comparison
    print("\n--- Early Warning Comparison ---")

    ai_first_count = results_df['AI_Detected_First'].sum()
    ai_only_count = results_df['AI_Only_Detected'].sum()

    print(f"Segments where AI detected first: {ai_first_count}")
    print(f"Segments where only AI detected: {ai_only_count}")

    print(f"\nPaper claims: 47 AI first, 12 AI only")

    # Average detection advantage
    both_triggered = results_df[
        (results_df['AI_Trigger_Year'].notna()) &
        (results_df['Trad_Trigger_Year'].notna())
    ]

    if len(both_triggered) > 0:
        avg_advantage = both_triggered['Detection_Advantage_Months'].mean()
        median_advantage = both_triggered['Detection_Advantage_Months'].median()

        print(f"\nDetection advantage (when both triggered):")
        print(f"  Mean: {avg_advantage:.1f} months")
        print(f"  Median: {median_advantage:.1f} months")
        print(f"  Paper claims: 14.3 months average")

        if abs(avg_advantage - 14.3) < 5:
            print("✓ Result aligns with paper (within 5 months)")
        else:
            print(f"⚠ Result differs: {avg_advantage - 14.3:+.1f} months difference")

    # Segments where AI significantly outperformed
    print("\n--- Top Segments (AI Detection Advantage) ---")
    top_ai = results_df[results_df['Detection_Advantage_Months'].notna()].nlargest(
        5, 'Detection_Advantage_Months'
    )
    for _, row in top_ai.iterrows():
        print(f"  {row['Segment']}: {row['Detection_Advantage_Months']:.0f} months earlier")


def plot_change_point_analysis(ae_series, segment_name, bocpd_result, output_file=None):
    """Plot change point detection results for a segment."""
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))

    # Top: A/E time series
    ax1 = axes[0]
    ax1.plot(ae_series.index, ae_series.values, 'b-o', linewidth=2, markersize=8)
    ax1.axhline(y=1.0, color='r', linestyle='--', alpha=0.5, label='Expected (1.0)')

    cp_idx = bocpd_result['most_likely_cp']
    if cp_idx < len(ae_series):
        cp_year = ae_series.index[cp_idx]
        ax1.axvline(x=cp_year, color='g', linestyle='-', linewidth=2,
                   label=f'Change Point ({cp_year})')

    ax1.set_title(f'A/E Ratio Time Series: {segment_name}')
    ax1.set_xlabel('Year')
    ax1.set_ylabel('A/E Ratio')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Bottom: Change point probabilities
    ax2 = axes[1]
    years = ae_series.index
    cp_probs = bocpd_result['change_point_probs']

    ax2.bar(years[:len(cp_probs)], cp_probs, alpha=0.7, color='green')
    ax2.set_title('Change Point Probability')
    ax2.set_xlabel('Year')
    ax2.set_ylabel('Probability')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    if output_file:
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        print(f"Plot saved to {output_file}")

    plt.close()


def main(ae_pivot=None, ae_data=None):
    """Main function for Phase 2 analysis."""
    print("\n" + "="*80)
    print("PHASE 2: CHANGE POINT DETECTION & EARLY WARNING")
    print("="*80)

    if ae_pivot is None:
        # Load from Phase 1 results
        ae_data = pd.read_csv(OUTPUT_DIR / 'phase1_ae_ratios.csv')
        ae_pivot = ae_data.pivot(
            index='Observation_Year',
            columns='Extended_Segment',
            values='AE_Ratio'
        )

    # Run early warning analysis
    results_df, change_point_years = analyze_early_warning(ae_pivot)

    # Print findings
    print_phase2_findings(results_df, change_point_years)

    # Save results
    results_df.to_csv(OUTPUT_DIR / 'phase2_early_warning_results.csv', index=False)
    print(f"\nResults saved to {OUTPUT_DIR}")

    # Plot example for key segment
    if 'M_25-44_S' in ae_pivot.columns:
        ts = ae_pivot['M_25-44_S'].dropna()
        bocpd = BayesianOnlineChangePointDetection(hazard_rate=1/4)
        bocpd_result = bocpd.detect(ts.values)
        plot_change_point_analysis(
            ts, 'M_25-44_S (Male Smokers 25-44)',
            bocpd_result,
            OUTPUT_DIR / 'phase2_change_point_M_25-44_S.png'
        )

    return results_df, change_point_years


if __name__ == "__main__":
    results_df, cp_years = main()
