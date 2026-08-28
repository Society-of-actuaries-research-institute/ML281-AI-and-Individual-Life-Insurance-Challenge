#!/usr/bin/env python3
import pandas as pd
import numpy as np
import lightgbm as lgb
import matplotlib.pyplot as plt
import seaborn as sns
import os
import warnings

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

def prepare_dataset(data):
    X = data[NUMERICAL_FEATURES + CATEGORICAL_FEATURES].copy()
    y = data[TARGET].values
    exposure = data[EXPOSURE].values
    offset = np.log(exposure + 1e-10) # Offset for Poisson
    
    for col in CATEGORICAL_FEATURES:
        X[col] = X[col].astype('category')
        
    return X, y, offset, exposure

def main():
    print("Loading data...")
    if not os.path.exists(DATA_PATH):
        print(f"Error: {DATA_PATH} not found.")
        return

    df = pd.read_parquet(DATA_PATH)
    df = df.sort_values('Observation_Year')
    
    print(f"Total Data Range: {df['Observation_Year'].min()} - {df['Observation_Year'].max()}")
    print(f"Total Rows: {len(df):,}")

    metrics_list = []

    for cutoff in CUTOFF_YEARS:
        print(f"\n=== Training/Validation Loop: Train <= {cutoff} ===")
        
        # 1. Split Data
        train_mask = df['Observation_Year'] <= cutoff
        test_mask = df['Observation_Year'] > cutoff
        
        df_train = df[train_mask]
        df_test = df[test_mask]

        # Downsample TRAIN only if needed (USER REQUEST: Cap training at 4.5M rows)
        # But KEEP TEST intact
        if len(df_train) > 4500000:
            print(f"  -> Downsampling train set from {len(df_train):,} to 4,500,000")
            df_train = df_train.sample(n=4500000, random_state=42)
        
        if len(df_test) == 0:
            print(f"Warning: No future data for cutoff {cutoff}. Skipping.")
            continue
            
        train_rows = len(df_train)
        print(f"Training Rows: {train_rows:,}")
        print(f"Test Rows: {len(df_test):,} (Full)")

        model_path = f"{MODELS_DIR}/lgbm_cutoff_{cutoff}.txt"
        
        # 2. Train or Load Model
        if os.path.exists(model_path):
            print(f"  -> Loading existing model from {model_path}")
            model = lgb.Booster(model_file=model_path)
        else:
            print("  -> Training new model...")
            X_train, y_train, offset_train, exposure_train = prepare_dataset(df_train)
            
            lgb_train = lgb.Dataset(
                X_train, 
                label=y_train,
                init_score=offset_train,
                categorical_feature=CATEGORICAL_FEATURES
            )
            
            params = {
                'objective': 'poisson',
                'metric': 'poisson',
                'boosting': 'gbdt',
                'num_leaves': 63,
                'learning_rate': 0.05,
                'feature_fraction': 0.8,
                'bagging_fraction': 0.8,
                'bagging_freq': 5,
                'min_data_in_leaf': 100,
                'verbose': -1,
                'seed': 42
            }
            
            model = lgb.train(
                params,
                lgb_train,
                num_boost_round=600
            )
            model.save_model(model_path)
            print(f"  -> Model saved to {model_path}")
        
        # 3. Predict on Future Years
        future_years = sorted(df_test['Observation_Year'].unique())
        
        for f_year in future_years:
            df_year = df_test[df_test['Observation_Year'] == f_year]
            if len(df_year) == 0: continue

            X_test_yr, y_test_yr, offset_test_yr, exposure_test_yr = prepare_dataset(df_year)
            
            # KEY FIX: Explicitly handle offset in prediction
            # raw_score=True gives the margin (log rate without offset if trained with offset?? No.)
            # If trained with init_score = log(E), expectation is E[y] = exp(margin + log(E))
            # So margin is log(rate).
            # To be safe and compliant with user request, we calculate:
            # pred_deaths = exp(raw_score + offset)
            
            raw_scores = model.predict(X_test_yr, raw_score=True)
            pred_deaths = np.exp(raw_scores + offset_test_yr)
            
            actual_deaths = y_test_yr.sum()
            total_pred_deaths = pred_deaths.sum()
            
            if total_pred_deaths == 0:
                ae_ratio = 0
            else:
                ae_ratio = actual_deaths / total_pred_deaths
                
            metrics_list.append({
                'Train_End_Year': cutoff,
                'Train_Rows': train_rows,
                'Test_Year': f_year,
                'Actual_Deaths': actual_deaths,
                'Predicted_Deaths': total_pred_deaths,
                'AE_Ratio': ae_ratio
            })
            print(f"  -> Validated on {f_year}: A/E = {ae_ratio:.4f}")

    # 4. Save Metrics
    metrics_df = pd.DataFrame(metrics_list)
    csv_path = f"{REPORTS_DIR}/expanding_window_metrics.csv"
    metrics_df.to_csv(csv_path, index=False)
    print(f"\nMetrics saved to {csv_path}")
    
    # 5. Plotting
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
            label=f'Train End: {cutoff} (N={subset["Train_Rows"].iloc[0]:,})',
            color=colors[i],
            linewidth=2
        )
        
    plt.axhline(1.0, color='red', linestyle='--', label='Target A/E = 1.0')
    plt.xlabel('Test Year (Observation Year)')
    plt.ylabel('A/E Ratio')
    plt.title('LGBM Model Stability: Expanding Window Validation\n(Performance on Future Unseen Data)')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    plot_path = f"{PLOTS_DIR}/expanding_window_ae.png"
    plt.savefig(plot_path)
    print(f"Plot saved to {plot_path}")

if __name__ == "__main__":
    main()
