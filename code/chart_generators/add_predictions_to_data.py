# -*- coding: utf-8 -*-
"""
Add Model Predictions to Dataset (One-time setup)
"""

import pandas as pd
import numpy as np
import lightgbm as lgb
from pathlib import Path
from sklearn.preprocessing import LabelEncoder

# Paths
DATA_PATH = Path(__file__).parent.parent.parent / 'data' / 'ilec_cleaned.parquet'
MODEL_PATH = Path(__file__).parent.parent.parent / 'models' / 'lgbm_mortality_offset_poisson.txt'
OUTPUT_PATH = Path(__file__).parent.parent.parent / 'data' / 'ilec_with_predictions.parquet'

FEATURES = ['Attained_Age', 'Issue_Age', 'Duration', 'Sex', 'Smoker_Status',
            'Insurance_Plan', 'Face_Amount_Band', 'Preferred_Class',
            'SOA_Post_Lvl_Ind', 'SOA_Antp_Lvl_TP', 'SOA_Guar_Lvl_TP']
CAT_FEATURES = ['Sex', 'Smoker_Status', 'Insurance_Plan', 'Face_Amount_Band',
                'Preferred_Class', 'SOA_Post_Lvl_Ind', 'SOA_Antp_Lvl_TP', 'SOA_Guar_Lvl_TP']

def main():
    print('=' * 60)
    print('ADDING PREDICTIONS TO DATASET')
    print('=' * 60)
    
    print('\n[1] Loading data...')
    df = pd.read_parquet(DATA_PATH)
    print(f'Records: {len(df):,}')
    
    # Check if already exists
    if 'Expected_Deaths' in df.columns:
        print('Expected_Deaths already exists in dataset. Skipping.')
        return

    print('\n[2] Loading model...')
    print(f'\nMODEL_PATH:{MODEL_PATH}...')
    model = lgb.Booster(model_file=str(MODEL_PATH))
    
    print('\n[3] Encoding and Predicting...')
    df_encoded = df.copy()
    for col in CAT_FEATURES:
        le = LabelEncoder()
        df_encoded[col] = le.fit_transform(df_encoded[col].astype(str))
        
    # Pass as numpy array to avoid "categorical_feature do not match" error
    # This bypasses pandas column header/type checks against the model file
    df['Expected_Deaths'] = model.predict(df_encoded[FEATURES].values) * df['Policies_Exposed']
    
    print('\n[4] Saving to new file...')
    # Save to a new file to preserve original
    df.to_parquet(OUTPUT_PATH)
    print(f'✓ Saved: {OUTPUT_PATH}')
    print(f'  New Columns: {list(df.columns)}')

if __name__ == '__main__':
    main()
