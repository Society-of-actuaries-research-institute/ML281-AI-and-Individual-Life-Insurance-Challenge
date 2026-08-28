#!/usr/bin/env python3
import pandas as pd
import numpy as np
import lightgbm as lgb
import matplotlib.pyplot as plt
import seaborn as sns
import os
import warnings
import gc

warnings.filterwarnings('ignore')

# Configuration
DATA_PATH = 'data/ilec_cleaned.parquet'
MODELS_DIR = 'models'
REPORTS_DIR = 'data/reports'
PLOTS_DIR = 'data/plots'
CUTOFF_YEARS = [2012, 2013, 2014, 2015, 2016, 2017, 2018]
TARGET = 'Death_Count'
EXPOSURE = 'Policies_Exposed'

NUMERICAL_FEATURES = ['Attained_Age', 'Issue_Age', 'Duration']
CATEGORICAL_FEATURES = [
    'Sex', 'Smoker_Status', 'Insurance_Plan', 'Face_Amount_Band',
    'Preferred_Class', 'SOA_Post_Lvl_Ind', 'SOA_Antp_Lvl_TP', 'SOA_Guar_Lvl_TP'
]

def prepare_dataset_for_inference(data):
    X = data[NUMERICAL_FEATURES + CATEGORICAL_FEATURES].copy()
    y = data[TARGET].values
    exposure = data[EXPOSURE].values
    # Offset logic: log(exposure)
    # We add epsilon to avoid log(0)
    offset = np.log(exposure + 1e-10) 
    
    for col in CATEGORICAL_FEATURES:
        X[col] = X[col].astype('category')
        
    return X, y, offset, exposure

def main():
    print("Loading data...")
    if not os.path.exists(DATA_PATH):
        print(f"Error: {DATA_PATH} not found.")
        return

    # Load full dataset - NO SAMPLING for test
    df_full = pd.read_parquet(DATA_PATH)
    
    print(f"Total Rows: {len(df_full):,}")
    
    metrics_list = []

    for cutoff in CUTOFF_YEARS:
        print(f"\n=== Evaluation Loop: Training Cutoff Year {cutoff} ===")
        
        model_path = f"{MODELS_DIR}/lgbm_cutoff_{cutoff}.txt"
        
        if not os.path.exists(model_path):
            print(f"  [SKIP] Model not found: {model_path}")
            continue
            
        print(f"  -> Loading model: {model_path}")
        try:
            model = lgb.Booster(model_file=model_path)
        except Exception as e:
            print(f"  [ERROR] Failed to load model: {e}")
            continue

        # Get test data (Future years)
        # USAGE OF FULL TEST DATA AS REQUESTED
        df_test = df_full[df_full['Observation_Year'] > cutoff].copy()
        
        if len(df_test) == 0:
            print(f"  [INFO] No future data for cutoff {cutoff}.")
            continue
            
        print(f"  -> Test Rows (Full): {len(df_test):,}")

        # Iterate through future years to calculate A/E per year
        future_years = sorted(df_test['Observation_Year'].unique())
        
        for f_year in future_years:
            df_year = df_test[df_test['Observation_Year'] == f_year]
            if len(df_year) == 0: continue

            X_test, y_test, offset_test, exposure_test = prepare_dataset_for_inference(df_year)
            
            # CRITICAL FIX: Correct prediction logic with offset
            # 1. Get raw scores (log-space output from trees)
            raw_scores = model.predict(X_test, raw_score=True)
            
            # 2. Add offset (log exposure)
            # 3. Exponentiate to get expected counts
            # E[Y] = exp(F(x) + log(E)) = exp(F(x)) * E
            pred_deaths = np.exp(raw_scores + offset_test)
            
            actual_deaths = y_test.sum()
            total_pred_deaths = pred_deaths.sum()
            
            if total_pred_deaths == 0:
                ae_ratio = 0
            else:
                ae_ratio = actual_deaths / total_pred_deaths
            
            # We don't have the original training row count here easily without metadata,
            # but we can infer or leave it generic.
            # Ideally we would save metadata alongside models. 
            # For now, we will label just by Cutoff Year.
            
            metrics_list.append({
                'Train_End_Year': cutoff,
                'Test_Year': f_year,
                'Actual_Deaths': actual_deaths,
                'Predicted_Deaths': total_pred_deaths,
                'AE_Ratio': ae_ratio
            })
            print(f"  -> Validated on {f_year}: A/E = {ae_ratio:.4f}")
            
        # Clean up to save memory
        del df_test
        gc.collect()

    # Save Metrics
    if not metrics_list:
        print("No metrics calculated.")
        return

    metrics_df = pd.DataFrame(metrics_list)
    csv_path = f"{REPORTS_DIR}/expanding_window_metrics.csv"
    metrics_df.to_csv(csv_path, index=False)
    print(f"\nMetrics saved to {csv_path}")
    
    # Plotting
    plot_results(metrics_df)

def plot_results(df):
    plt.figure(figsize=(12, 8))
    
    cutoffs = sorted(df['Train_End_Year'].unique())
    colors = sns.color_palette("viridis", n_colors=len(cutoffs))
    
    for i, cutoff in enumerate(cutoffs):
        subset = df[df['Train_End_Year'] == cutoff].sort_values('Test_Year')
        if len(subset) == 0: continue
            
        plt.plot(
            subset['Test_Year'], 
            subset['AE_Ratio'], 
            marker='o', 
            label=f'Model Trained Up To: {cutoff}',
            color=colors[i],
            linewidth=2,
            alpha=0.8
        )
        
    plt.axhline(1.0, color='red', linestyle='--', label='Target A/E = 1.0')
    plt.xlabel('Test Year (Observation Year)', fontsize=12)
    plt.ylabel('A/E Ratio', fontsize=12)
    plt.title('LGBM Model Stability: Expanding Window Validation\n(Performance on Future Unseen Data)', fontsize=14)
    plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', title='Training Cutoff')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    plot_path = f"{PLOTS_DIR}/expanding_window_ae.png"
    plt.savefig(plot_path)
    print(f"Plot saved to {plot_path}")

if __name__ == "__main__":
    main()
