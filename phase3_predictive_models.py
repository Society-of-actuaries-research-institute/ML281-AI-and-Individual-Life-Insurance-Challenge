"""
Phase 3: Predictive Modeling - Poisson/Rate Regression (CORRECTED)
===================================================================
Builds and compares mortality prediction models with temporal validation.

CORRECTED VERSION: Uses Poisson/Tweedie regression for grouped actuarial data
instead of binary classification. Each row represents an aggregated cell with
Death_Count deaths out of Policies_Exposed exposure.

Key changes from original binary classification:
1. Target is Death_Count (count) or mortality_rate, not binary has_death
2. Models use Poisson/Tweedie objectives with log(exposure) offset
3. Evaluation metrics compare predicted vs actual mortality RATES
4. Calibration is exposure-weighted
"""
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import PoissonRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
import xgboost as xgb
import lightgbm as lgb
import os
import json
import warnings
import gc
from tqdm import tqdm
warnings.filterwarnings("ignore")

# Paths — resolved relative to this file (soa_ai_v2/code/)
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PARQUET_DIR = os.path.join(_ROOT, "data", "parquet")
# Results base honors the optional SOA_RESULTS_DIR env var (default unchanged).
_RESULTS = os.environ.get("SOA_RESULTS_DIR") or os.path.join(_ROOT, "results")
OUTPUT_DIR = os.path.join(_RESULTS, "models")
FIGURES_DIR = os.path.join(_RESULTS, "figures")
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)

plt.rcParams.update({
    "figure.figsize": (12, 7),
    "font.size": 12,
    "axes.titlesize": 14,
    "figure.dpi": 150,
})

# Features
CATEGORICAL_FEATURES = [
    "Sex", "Smoker_Status", "Insurance_Plan", "Age_Ind",
    "SOA_Post_Lvl_Ind", "Slct_Ult_Ind", "Preferred_Indicator",
    "Number_of_Pfd_Classes", "Preferred_Class", "Face_Amount_Band",
]
NUMERIC_FEATURES = [
    "Issue_Age", "Duration", "Attained_Age", "Issue_Year",
]

COL_NAMES = [
    'Observation_Year', 'Age_Ind', 'Sex', 'Smoker_Status', 'Insurance_Plan',
    'Issue_Age', 'Duration', 'Face_Amount_Band', 'Issue_Year', 'Attained_Age',
    'SOA_Antp_Lvl_TP', 'SOA_Guar_Lvl_TP', 'SOA_Post_Lvl_Ind', 'Slct_Ult_Ind',
    'Preferred_Indicator', 'Number_of_Pfd_Classes', 'Preferred_Class',
    'Amount_Exposed', 'Policies_Exposed', 'Death_Claim_Amount', 'Death_Count',
    'ExpDth_VBT2015_Cnt', 'ExpDth_VBT2015_Amt', 'ExpDth_VBT2015wMI_Cnt',
    'ExpDth_VBT2015wMI_Amt', 'Cen2MomP1wMI_Amt', 'Cen2MomP2wMI_Amt',
    'Cen3MomP1wMI_Amt', 'Cen3MomP2wMI_Amt', 'Cen3MomP3wMI_Amt'
]


def load_data():
    """Load the ILEC data from parquet files (memory efficient)."""
    print("Loading ILEC 2012-2019 data from parquet files...")

    parquet_files = sorted([
        os.path.join(PARQUET_DIR, f)
        for f in os.listdir(PARQUET_DIR)
        if f.endswith('.parquet')
    ])

    chunks = []
    for pf in tqdm(parquet_files, desc="Loading parquet files"):
        chunk = pd.read_parquet(pf)
        chunks.append(chunk)
        print(f"  Loaded {os.path.basename(pf)}: {len(chunk):,} rows")

    df = pd.concat(chunks, ignore_index=True)
    print(f"Loaded {len(df):,} rows (aggregated cells)")
    return df


def prepare_data(df):
    """Prepare data for modeling."""
    print("Preparing data...")

    # Filter to records with positive exposure
    df = df[df["Policies_Exposed"] > 0].copy()

    # CORRECTED: Create mortality rate target
    df["mortality_rate"] = df["Death_Count"] / df["Policies_Exposed"]

    # Log exposure for Poisson offset
    df["log_exposure"] = np.log(df["Policies_Exposed"])

    # Encode categoricals
    label_encoders = {}
    for col in CATEGORICAL_FEATURES:
        if col in df.columns:
            le = LabelEncoder()
            df[col] = df[col].astype(str)
            df[col + "_encoded"] = le.fit_transform(df[col])
            label_encoders[col] = le

    # Create additional features
    df["Age_Band"] = pd.cut(
        df["Attained_Age"],
        bins=[0, 24, 44, 64, 120],
        labels=[0, 1, 2, 3],
    ).astype(float).fillna(2.0)

    df["log_amount_exposed"] = np.log1p(df["Amount_Exposed"])
    df["duration_squared"] = df["Duration"] ** 2

    # Summary statistics
    total_exposure = df["Policies_Exposed"].sum()
    total_deaths = df["Death_Count"].sum()
    overall_rate = total_deaths / total_exposure

    print(f"After filtering: {len(df):,} rows (aggregated cells)")
    print(f"Total policy-years exposure: {total_exposure:,.0f}")
    print(f"Total deaths: {total_deaths:,.0f}")
    print(f"Overall mortality rate: {overall_rate:.6f} ({overall_rate*1000:.2f} per 1000)")
    print(f"Year range: {df['Observation_Year'].min()} - {df['Observation_Year'].max()}")

    return df, label_encoders


def get_feature_cols():
    """Get list of feature columns for modeling."""
    encoded_cats = [c + "_encoded" for c in CATEGORICAL_FEATURES]
    return NUMERIC_FEATURES + encoded_cats + ["Age_Band", "log_amount_exposed", "duration_squared"]


def temporal_train_test_split(df, train_end_year, test_year):
    """Split data by observation year for temporal validation."""
    train = df[df["Observation_Year"] <= train_end_year].copy()
    test = df[df["Observation_Year"] == test_year].copy()
    return train, test


def sample_for_training(df, max_rows=2_000_000):
    """
    Sample for training efficiency while preserving exposure distribution.
    CORRECTED: Sample by exposure weight, not by binary class balance.
    """
    if len(df) <= max_rows:
        return df

    # Stratified sample: ensure cells with deaths are represented
    cells_with_deaths = df[df["Death_Count"] > 0]
    cells_no_deaths = df[df["Death_Count"] == 0]

    # Keep all cells with deaths (they're rare and informative)
    n_death_cells = len(cells_with_deaths)
    n_no_death = min(len(cells_no_deaths), max_rows - n_death_cells)

    if n_no_death > 0 and n_no_death < len(cells_no_deaths):
        # Sample non-death cells proportional to exposure
        probs = cells_no_deaths["Policies_Exposed"] / cells_no_deaths["Policies_Exposed"].sum()
        cells_no_deaths = cells_no_deaths.sample(n=n_no_death, weights=probs, random_state=42)

    result = pd.concat([cells_with_deaths, cells_no_deaths]).sample(frac=1, random_state=42)
    return result


# =============================================================================
# Model Training Functions (Poisson/Rate Modeling)
# =============================================================================

def train_poisson_glm(X_train, y_count, exposure_train, X_test, exposure_test):
    """Poisson GLM - actuarial standard for mortality count modeling."""
    model = PoissonRegressor(alpha=0.1, max_iter=1000)

    # Fit on rates with exposure weights
    rates_train = y_count / exposure_train
    model.fit(X_train, rates_train, sample_weight=exposure_train)

    # Predict rates directly
    y_pred_rate = model.predict(X_test)
    y_pred_rate = np.clip(y_pred_rate, 0, 1)

    return model, y_pred_rate


def train_xgboost_poisson(X_train, y_count, exposure_train, X_test, exposure_test):
    """XGBoost with Poisson objective for count data."""
    # Create DMatrix with base_margin (offset)
    dtrain = xgb.DMatrix(X_train, label=y_count)
    dtrain.set_base_margin(np.log(exposure_train))

    dtest = xgb.DMatrix(X_test)
    dtest.set_base_margin(np.log(exposure_test))

    params = {
        "objective": "count:poisson",
        "max_depth": 6,
        "learning_rate": 0.1,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "min_child_weight": 10,
        "seed": 42,
        "verbosity": 0,
    }

    model = xgb.train(
        params,
        dtrain,
        num_boost_round=300,
        evals=[(dtest, "test")],
        verbose_eval=False,
    )

    # Predict expected counts, convert to rates
    y_pred_count = model.predict(dtest)
    y_pred_rate = y_pred_count / exposure_test
    y_pred_rate = np.clip(y_pred_rate, 0, 1)

    return model, y_pred_rate


def train_xgboost_tweedie(X_train, y_count, exposure_train, X_test, exposure_test):
    """XGBoost with Tweedie objective - flexible for count data with zeros."""
    rates_train = y_count / exposure_train

    params = {
        "objective": "reg:tweedie",
        "tweedie_variance_power": 1.5,
        "max_depth": 6,
        "learning_rate": 0.1,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "min_child_weight": 10,
        "seed": 42,
        "verbosity": 0,
    }

    model = xgb.XGBRegressor(**params, n_estimators=300)
    model.fit(X_train, rates_train, sample_weight=exposure_train, verbose=False)

    y_pred_rate = model.predict(X_test)
    y_pred_rate = np.clip(y_pred_rate, 0, 1)

    return model, y_pred_rate


def train_lightgbm_tweedie(X_train, y_count, exposure_train, X_test, exposure_test):
    """LightGBM with Tweedie objective - models rate directly."""
    rates_train = y_count / exposure_train

    model = lgb.LGBMRegressor(
        objective="tweedie",
        tweedie_variance_power=1.5,
        n_estimators=300,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_samples=20,
        random_state=42,
        verbose=-1,
    )

    model.fit(X_train, rates_train, sample_weight=exposure_train)

    y_pred_rate = model.predict(X_test)
    y_pred_rate = np.clip(y_pred_rate, 0, 1)

    return model, y_pred_rate


def train_catboost_poisson(X_train, y_count, exposure_train, X_test, exposure_test):
    """CatBoost with Poisson objective."""
    from catboost import CatBoostRegressor

    rates_train = y_count / exposure_train

    model = CatBoostRegressor(
        loss_function="Poisson",
        iterations=300,
        depth=6,
        learning_rate=0.1,
        random_seed=42,
        verbose=0,
    )

    model.fit(X_train, rates_train, sample_weight=exposure_train, verbose=0)

    y_pred_rate = model.predict(X_test)
    y_pred_rate = np.clip(y_pred_rate, 0, 1)

    return model, y_pred_rate


def train_ebm_regression(X_train, y_count, exposure_train, X_test, exposure_test):
    """Explainable Boosting Machine - regression on mortality rate."""
    from interpret.glassbox import ExplainableBoostingRegressor

    rates_train = y_count / exposure_train

    model = ExplainableBoostingRegressor(
        max_bins=256,
        max_rounds=300,
        learning_rate=0.01,
        random_state=42,
        n_jobs=-1,
    )

    # EBM doesn't support sample_weight in fit
    model.fit(X_train, rates_train)

    y_pred_rate = model.predict(X_test)
    y_pred_rate = np.clip(y_pred_rate, 0, 1)

    return model, y_pred_rate


# =============================================================================
# Evaluation Metrics (Rate-based)
# =============================================================================

def evaluate_model(y_rate_true, y_rate_pred, exposure, model_name):
    """Evaluate predictions against actual mortality rates."""
    metrics = {"model": model_name}

    # 1. Weighted RMSE
    weighted_mse = np.average((y_rate_true - y_rate_pred)**2, weights=exposure)
    metrics["weighted_rmse"] = np.sqrt(weighted_mse)

    # 2. Weighted MAE
    metrics["weighted_mae"] = np.average(np.abs(y_rate_true - y_rate_pred), weights=exposure)

    # 3. A/E Ratio (aggregate actual vs predicted deaths)
    total_actual_deaths = np.sum(y_rate_true * exposure)
    total_predicted_deaths = np.sum(y_rate_pred * exposure)
    metrics["ae_ratio"] = total_actual_deaths / total_predicted_deaths if total_predicted_deaths > 0 else np.nan

    # 4. Weighted R²
    ss_res = np.sum(exposure * (y_rate_true - y_rate_pred)**2)
    ss_tot = np.sum(exposure * (y_rate_true - np.average(y_rate_true, weights=exposure))**2)
    metrics["weighted_r2"] = 1 - (ss_res / ss_tot) if ss_tot > 0 else np.nan

    # 5. Calibration Error (by predicted rate decile)
    pred_df = pd.DataFrame({
        "pred": y_rate_pred,
        "actual": y_rate_true,
        "exposure": exposure,
    })

    try:
        pred_df["decile"] = pd.qcut(pred_df["pred"], 10, duplicates="drop", labels=False)
    except ValueError:
        pred_df["decile"] = pd.qcut(pred_df["pred"].rank(method="first"), 5, duplicates="drop", labels=False)

    cal = pred_df.groupby("decile").apply(
        lambda g: pd.Series({
            "pred_rate": np.average(g["pred"], weights=g["exposure"]),
            "actual_rate": np.average(g["actual"], weights=g["exposure"]),
            "total_exposure": g["exposure"].sum(),
        })
    )
    metrics["calibration_error"] = np.average(
        np.abs(cal["pred_rate"] - cal["actual_rate"]),
        weights=cal["total_exposure"]
    )

    return metrics


# =============================================================================
# Model Tournament
# =============================================================================

def run_model_tournament(df):
    """Run expanding window temporal validation with all models."""
    print("\n" + "="*70)
    print("MODEL TOURNAMENT (POISSON/RATE REGRESSION)")
    print("="*70)

    feature_cols = get_feature_cols()
    all_results = []

    # Expanding window: train up to year Y, test on Y+1
    train_test_configs = [
        (2014, 2015),
        (2015, 2016),
        (2016, 2017),
        (2017, 2018),
        (2018, 2019),
    ]

    models_to_train = {
        "Poisson GLM": train_poisson_glm,
        "XGBoost-Poisson": train_xgboost_poisson,
        "XGBoost-Tweedie": train_xgboost_tweedie,
        "LightGBM-Tweedie": train_lightgbm_tweedie,
        "CatBoost-Poisson": train_catboost_poisson,
        "EBM": train_ebm_regression,
    }

    best_models = {}

    for train_end, test_year in train_test_configs:
        print(f"\n--- Train: 2012-{train_end}, Test: {test_year} ---")
        train_df, test_df = temporal_train_test_split(df, train_end, test_year)

        # Sample for efficiency
        train_sampled = sample_for_training(train_df, max_rows=2_000_000)

        # Summary stats
        train_exposure = train_sampled["Policies_Exposed"].sum()
        train_deaths = train_sampled["Death_Count"].sum()
        test_exposure = test_df["Policies_Exposed"].sum()
        test_deaths = test_df["Death_Count"].sum()

        print(f"  Train: {len(train_sampled):,} cells, {train_exposure:,.0f} exposure, {train_deaths:,.0f} deaths")
        print(f"  Test:  {len(test_df):,} cells, {test_exposure:,.0f} exposure, {test_deaths:,.0f} deaths")
        print(f"  Train mortality rate: {train_deaths/train_exposure:.6f}")
        print(f"  Test mortality rate:  {test_deaths/test_exposure:.6f}")

        # Prepare features and targets
        X_train = train_sampled[feature_cols].values.astype(np.float32)
        y_count_train = train_sampled["Death_Count"].values.astype(np.float32)
        exposure_train = train_sampled["Policies_Exposed"].values.astype(np.float32)

        X_test = test_df[feature_cols].values.astype(np.float32)
        y_count_test = test_df["Death_Count"].values.astype(np.float32)
        exposure_test = test_df["Policies_Exposed"].values.astype(np.float32)
        y_rate_test = test_df["mortality_rate"].values.astype(np.float32)

        # Handle NaN
        X_train = np.nan_to_num(X_train, nan=0.0)
        X_test = np.nan_to_num(X_test, nan=0.0)

        for model_name, train_fn in models_to_train.items():
            print(f"  Training {model_name}...", end=" ", flush=True)
            try:
                model, y_pred_rate = train_fn(
                    X_train, y_count_train, exposure_train,
                    X_test, exposure_test
                )

                metrics = evaluate_model(y_rate_test, y_pred_rate, exposure_test, model_name)
                metrics["train_end"] = train_end
                metrics["test_year"] = test_year
                all_results.append(metrics)

                print(f"RMSE: {metrics['weighted_rmse']:.6f} | "
                      f"A/E: {metrics['ae_ratio']:.4f} | "
                      f"Cal Error: {metrics['calibration_error']:.6f}")

                # Store the model for later analysis (keep last window)
                if test_year == 2019:
                    best_models[model_name] = {
                        "model": model,
                        "X_test": X_test,
                        "y_rate_test": y_rate_test,
                        "y_pred_rate": y_pred_rate,
                        "exposure_test": exposure_test,
                        "test_df": test_df,
                        "feature_cols": feature_cols,
                    }

            except Exception as e:
                print(f"FAILED: {e}")
                import traceback
                traceback.print_exc()
                all_results.append({
                    "model": model_name,
                    "train_end": train_end,
                    "test_year": test_year,
                    "weighted_rmse": np.nan,
                    "weighted_mae": np.nan,
                    "ae_ratio": np.nan,
                    "calibration_error": np.nan,
                    "error": str(e),
                })

        gc.collect()

    results_df = pd.DataFrame(all_results)
    results_df.to_csv(os.path.join(OUTPUT_DIR, "model_comparison_poisson.csv"), index=False)
    print(f"\nSaved model_comparison_poisson.csv")

    return results_df, best_models


# =============================================================================
# Visualization
# =============================================================================

def create_model_comparison_plots(results_df):
    """Generate model comparison visualizations."""
    print("\n=== MODEL COMPARISON PLOTS ===")

    # 1. Metrics over time by model
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    metrics_to_plot = [
        ("weighted_rmse", axes[0], "Weighted RMSE (lower=better)"),
        ("ae_ratio", axes[1], "A/E Ratio (1.0=perfect)"),
        ("calibration_error", axes[2], "Calibration Error (lower=better)"),
    ]

    for metric, ax, title in metrics_to_plot:
        for model_name in results_df["model"].unique():
            model_data = results_df[results_df["model"] == model_name].sort_values("test_year")
            if not model_data[metric].isna().all():
                ax.plot(model_data["test_year"], model_data[metric], "o-", label=model_name, markersize=6)

        ax.set_title(title)
        ax.set_xlabel("Test Year")
        ax.set_ylabel(metric)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

        if metric == "ae_ratio":
            ax.axhline(y=1.0, color="k", linestyle="--", alpha=0.5, label="Perfect (1.0)")

    plt.suptitle("Model Performance Over Time (Poisson/Rate Regression)", fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "figure5_model_comparison_poisson.png"), bbox_inches="tight")
    plt.close()
    print("  Saved figure5_model_comparison_poisson.png")

    # 2. Summary bar chart
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    avg_metrics = results_df.groupby("model").agg({
        "weighted_rmse": "mean",
        "ae_ratio": lambda x: np.abs(x - 1.0).mean(),
        "calibration_error": "mean",
    }).reset_index()

    for i, (metric, title) in enumerate([
        ("weighted_rmse", "Avg Weighted RMSE"),
        ("ae_ratio", "Avg |A/E - 1.0|"),
        ("calibration_error", "Avg Calibration Error"),
    ]):
        ax = axes[i]
        sorted_df = avg_metrics.sort_values(metric)
        colors = plt.cm.RdYlGn_r(np.linspace(0.2, 0.8, len(sorted_df)))
        ax.barh(sorted_df["model"], sorted_df[metric], color=colors)
        ax.set_xlabel(metric)
        ax.set_title(title)
        ax.grid(True, alpha=0.3, axis="x")

    plt.suptitle("Model Comparison: Average Performance Across All Test Windows", fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "figure6_model_ranking_poisson.png"), bbox_inches="tight")
    plt.close()
    print("  Saved figure6_model_ranking_poisson.png")


def generate_calibration_plots(best_models):
    """Generate calibration plots (predicted vs actual rate by decile)."""
    print("\n=== CALIBRATION PLOTS ===")

    n_models = len(best_models)
    cols = min(3, n_models)
    rows = (n_models + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(6*cols, 5*rows))
    if n_models == 1:
        axes = [axes]
    else:
        axes = axes.flatten()

    for idx, (model_name, data) in enumerate(best_models.items()):
        if idx >= len(axes):
            break

        ax = axes[idx]

        y_rate_test = data["y_rate_test"]
        y_pred_rate = data["y_pred_rate"]
        exposure = data["exposure_test"]

        # Create calibration data by decile
        pred_df = pd.DataFrame({
            "pred": y_pred_rate,
            "actual": y_rate_test,
            "exposure": exposure,
        })

        try:
            pred_df["decile"] = pd.qcut(pred_df["pred"], 10, duplicates="drop", labels=False)
        except ValueError:
            pred_df["decile"] = pd.qcut(pred_df["pred"].rank(method="first"), 5, duplicates="drop", labels=False)

        cal = pred_df.groupby("decile").apply(
            lambda g: pd.Series({
                "pred_rate": np.average(g["pred"], weights=g["exposure"]),
                "actual_rate": np.average(g["actual"], weights=g["exposure"]),
                "exposure": g["exposure"].sum(),
            })
        ).reset_index()

        # Plot
        ax.scatter(cal["pred_rate"], cal["actual_rate"], s=cal["exposure"]/1e6, alpha=0.7)

        # Perfect calibration line
        max_rate = max(cal["pred_rate"].max(), cal["actual_rate"].max()) * 1.1
        ax.plot([0, max_rate], [0, max_rate], "k--", alpha=0.5, label="Perfect")

        ax.set_xlabel("Predicted Mortality Rate")
        ax.set_ylabel("Actual Mortality Rate")
        ax.set_title(f"{model_name}")
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, max_rate)
        ax.set_ylim(0, max_rate)

    # Hide unused subplots
    for idx in range(len(best_models), len(axes)):
        axes[idx].set_visible(False)

    plt.suptitle("Calibration: Predicted vs Actual Mortality Rate by Decile\n(Bubble size = exposure)", fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "figure11_calibration_poisson.png"), bbox_inches="tight")
    plt.close()
    print("  Saved figure11_calibration_poisson.png")


def print_summary(results_df):
    """Print summary of results."""
    print("\n" + "="*70)
    print("SUMMARY: Best Models by Metric (2019 Test Year)")
    print("="*70)

    final_results = results_df[results_df["test_year"] == 2019].copy()
    if len(final_results) > 0:
        print("\nBy Weighted RMSE (lower is better):")
        print(final_results.nsmallest(3, "weighted_rmse")[["model", "weighted_rmse", "ae_ratio", "calibration_error"]].to_string(index=False))

        print("\nBy A/E Ratio (closest to 1.0 is best):")
        final_results["ae_distance"] = np.abs(final_results["ae_ratio"] - 1.0)
        print(final_results.nsmallest(3, "ae_distance")[["model", "ae_ratio", "weighted_rmse", "calibration_error"]].to_string(index=False))

        print("\nBy Calibration Error (lower is better):")
        print(final_results.nsmallest(3, "calibration_error")[["model", "calibration_error", "weighted_rmse", "ae_ratio"]].to_string(index=False))

    # Save summary JSON
    summary = {
        "methodology": "Poisson/Tweedie rate regression with exposure weighting",
        "training_sample_size": 2_000_000,
        "temporal_windows": 5,
        "models_tested": list(results_df["model"].unique()),
        "best_by_rmse": final_results.nsmallest(1, "weighted_rmse")["model"].iloc[0] if len(final_results) > 0 else None,
        "best_by_ae": final_results.loc[final_results["ae_distance"].idxmin(), "model"] if len(final_results) > 0 else None,
        "best_by_calibration": final_results.nsmallest(1, "calibration_error")["model"].iloc[0] if len(final_results) > 0 else None,
    }

    with open(os.path.join(OUTPUT_DIR, "model_comparison_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)


# =============================================================================
# Main
# =============================================================================

def main():
    print("="*70)
    print("PHASE 3: PREDICTIVE MODELING - POISSON/RATE REGRESSION")
    print("(CORRECTED for grouped actuarial data)")
    print("="*70)

    df = load_data()
    df, label_encoders = prepare_data(df)
    results_df, best_models = run_model_tournament(df)
    create_model_comparison_plots(results_df)

    if best_models:
        generate_calibration_plots(best_models)

    print_summary(results_df)

    print("\n" + "="*70)
    print("PHASE 3 COMPLETE")
    print(f"Outputs saved to: {OUTPUT_DIR} and {FIGURES_DIR}")
    print("="*70)

    return results_df, best_models


if __name__ == "__main__":
    results_df, best_models = main()
