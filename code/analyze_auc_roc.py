#!/usr/bin/env python3
import pandas as pd
import numpy as np
import lightgbm as lgb
import matplotlib.pyplot as plt
import seaborn as sns
import os
import warnings
import gc
from sklearn.metrics import roc_auc_score, roc_curve

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

def map_categorical(df, cat_cols):
    """
    Convert columns to category dtype in-place to save memory and processing time.
    """
    for col in cat_cols:
        if col in df.columns:
            df[col] = df[col].astype('category')
    return df

def plot_results(df):
    plt.figure(figsize=(12, 8))
    
    # Check if df is empty
    if df.empty:
        return

    cutoffs = sorted(df['Train_End_Year'].unique())
    colors = sns.color_palette("viridis", n_colors=len(cutoffs))
    
    for i, cutoff in enumerate(cutoffs):
        subset = df[df['Train_End_Year'] == cutoff].sort_values('Test_Year')
        if len(subset) == 0: continue
            
        plt.plot(
            subset['Test_Year'], 
            subset['AUC'], 
            marker='o', 
            label=f'Model Trained Up To: {cutoff}',
            color=colors[i],
            linewidth=2,
            alpha=0.8
        )
        
    plt.xlabel('Test Year')
    plt.ylabel('AUC Score')
    plt.title('AUC Stability Over Time (Discriminatory Power)')
    plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    auc_plot_path = f"{PLOTS_DIR}/auc_stability.png"
    plt.savefig(auc_plot_path)
    print(f"  -> AUC Stability Plot updated: {auc_plot_path}")
    plt.close() # Close plot to free memory

def main():
    if not os.path.exists(DATA_PATH):
        print(f"Error: {DATA_PATH} not found.")
        return

    print(f"Loading data from {DATA_PATH}...")
    df_full = pd.read_parquet(DATA_PATH)
    total_rows = len(df_full)
    print(f"Total Rows: {total_rows:,}")
    
    # OPTIMIZATION 1: Convert to category ONCE
    print("optimizing: converting categorical columns...")
    df_full = map_categorical(df_full, CATEGORICAL_FEATURES)
    
    # Pre-load models (they are small enough, ~few MB each usually)
    print("Pre-loading models...")
    models = {}
    for cutoff in CUTOFF_YEARS:
        model_path = f"{MODELS_DIR}/lgbm_cutoff_{cutoff}.txt"
        if os.path.exists(model_path):
            try:
                models[cutoff] = lgb.Booster(model_file=model_path)
                print(f"  -> Loaded model for cutoff {cutoff}")
            except Exception as e:
                print(f"  -> Failed to load {model_path}: {e}")
    
    metrics_list = []
    
    # Identify Future Years
    # We only care about years > min(CUTOFF_YEARS)
    min_cutoff = min(CUTOFF_YEARS)
    future_years = sorted(df_full[df_full['Observation_Year'] > min_cutoff]['Observation_Year'].unique())
    print(f"Future Years to Evaluate: {future_years}")

    # Initialize ROC Figure (OO Interface to avoid conflict)
    fig_roc, ax_roc = plt.subplots(figsize=(10, 8))
    ax_roc.plot([0, 1], [0, 1], 'k--', linestyle='--')
    ax_roc.set_xlabel('False Positive Rate')
    ax_roc.set_ylabel('True Positive Rate')
    ax_roc.set_title('ROC Curves (1-Year Forward Prediction) with Area')
    ax_roc.grid(True, alpha=0.3)

    # OPTIMIZATION 2: Iterate Years Outer Loop
    # Prepare X_test once per year
    for f_year in future_years:
        print(f"\nProcessing Year: {f_year}")
        
        # Slicing is fast on categorical
        df_year = df_full[df_full['Observation_Year'] == f_year]
        if len(df_year) == 0: continue
        
        # Prepare Dataset Part
        X_test = df_year[NUMERICAL_FEATURES + CATEGORICAL_FEATURES] 
        
        y_test = df_year[TARGET].values
        exposure = df_year[EXPOSURE].values
        offset_test = np.log(exposure + 1e-10)
        
        # Binary Target
        y_true_binary = (y_test > 0).astype(int)
        
        if y_true_binary.sum() == 0:
            print(f"  -> No deaths in {f_year}, skipping AUC.")
            continue

        # Evaluate against ALL applicable models
        # A model is applicable if cutoff < f_year
        for cutoff in CUTOFF_YEARS:
            if cutoff >= f_year:
                continue
                
            if cutoff not in models:
                continue
                
            model = models[cutoff]
            
            # Predict
            raw_scores = model.predict(X_test, raw_score=True)
            pred_deaths = np.exp(raw_scores + offset_test)
            
            y_score = pred_deaths
            auc = roc_auc_score(y_true_binary, y_score)
            
            metrics_list.append({
                'Train_End_Year': cutoff,
                'Test_Year': f_year,
                'AUC': auc,
                'Test_Rows': len(df_year),
                'Test_Deaths': y_true_binary.sum()
            })
            print(f"  -> Model {cutoff}: AUC = {auc:.4f}")
            
            # INCREMENTAL ROC PLOT (Area Chart style)
            # Only plot the "Immediate Next Year" prediction (Model 2012 -> Test 2013)
            # This gives us one clean curve per model.
            if f_year == cutoff + 1:
                fpr, tpr, _ = roc_curve(y_true_binary, y_score)
                # Plot Line
                ax_roc.plot(fpr, tpr, label=f'Model {cutoff} (Val {f_year}) AUC={auc:.4f}', linewidth=2)
                # Fill Area
                ax_roc.fill_between(fpr, tpr, alpha=0.1)
                
        # cleanup
        del df_year
        
        # INCREMENTAL SAVE (Metrics & Stability Plot)
        print(f"  -> Saving progress after year {f_year}...")
        metrics_df = pd.DataFrame(metrics_list)
        csv_path = f"{REPORTS_DIR}/auc_metrics.csv"
        metrics_df.to_csv(csv_path, index=False)
        
        # Generate Stability Plot
        plot_results(metrics_df)
        
        # Save ROC Plot Incrementally
        ax_roc.legend(loc='lower right')
        roc_path = f"{PLOTS_DIR}/roc_curves_area.png"
        fig_roc.savefig(roc_path)
        print(f"  -> ROC Area Plot updated: {roc_path}")
    
    print("\nanalysis Complete.")

if __name__ == "__main__":
    main()
