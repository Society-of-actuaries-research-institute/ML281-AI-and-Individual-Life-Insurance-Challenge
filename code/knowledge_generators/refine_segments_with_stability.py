import pandas as pd
import numpy as np
import json
import os
import lightgbm as lgb
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import LabelEncoder

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, '../../data/ilec_cleaned.parquet')
MODEL_PATH = os.path.join(BASE_DIR, '../../models/lgbm_mortality_offset_poisson.txt')
OUTPUT_DIR = os.path.join(BASE_DIR, '../../knowledge_base/segments')
SPOTLIGHT_DIR = os.path.join(OUTPUT_DIR, 'spotlight')

os.makedirs(SPOTLIGHT_DIR, exist_ok=True)

# Mapping for Face_Amount_Band
FACE_BAND_MAP = {
    1: '1 - 9,999',
    2: '10,000 - 24,999',
    3: '25,000 - 49,999',
    4: '50,000 - 99,999',
    5: '100,000 - 249,999',
    6: '250,000 - 499,999',
    7: '500,000 - 999,999',
    8: '1,000,000 - 2,499,999',
    9: '2,500,000 - 4,999,999',
    10: '5,000,000 - 9,999,999',
    11: '10,000,000+'
}

def clean_preferred_class(val):
    v = str(val).strip().upper()
    if v in ['NA', 'U', 'NAN', 'NONE', 'UNKNOWN']: return 'U'
    v = v.replace('.0', '')
    return v

def translate_rule(feat, low, high):
    """Translate bounds into semantic text."""
    
    # helper for boolean/one-hot
    is_true = (low > -float('inf') and low >= 0.4)
    is_false = (high < float('inf') and high <= 0.6)

    # 1. Specialized Semantic Mappings
    if feat == 'Is_Male':
        if is_true: return "Sex is Male"
        if is_false: return "Sex is Female"
    
    if feat == 'Is_Smoker_Known':
        if is_true: return "Smoker Status is Known"
        if is_false: return "Smoker Status is Unknown"
        
    if feat == 'Is_Smoker':
        if is_true: return "Is Smoker"
        if is_false: return "Non-Smoker"
        
    if feat.startswith('Plan_'):
        plan = feat.replace('Plan_', '')
        if is_true: return f"Insurance Plan is {plan}"
        if is_false: return f"Insurance Plan is NOT {plan}"
        
    if feat.startswith('Pref_'):
        pref = feat.replace('Pref_', '')
        if is_true: return f"Preferred Class is {pref}"
        if is_false: return f"Preferred Class is NOT {pref}"
        
    if feat.startswith('SOA_'):
        label = feat.replace('_', ' ')
        if is_true: return f"{label} is Yes"
        if is_false: return f"{label} is No"
        
    if feat == 'Face_Amount_Band_Val':
        if low != -float('inf') and high != float('inf'):
            l_idx = int(np.ceil(low))
            h_idx = int(np.floor(high))
            if l_idx == h_idx: return f"Face Amount Band: {FACE_BAND_MAP.get(l_idx, l_idx)}"
            return f"Face Amount Band: {FACE_BAND_MAP.get(l_idx, l_idx)} to {FACE_BAND_MAP.get(h_idx, h_idx)}"
        elif low != -float('inf'):
            l_idx = int(np.ceil(low))
            return f"Face Amount Band > {FACE_BAND_MAP.get(l_idx, l_idx)}"
        else:
            h_idx = int(np.floor(high))
            return f"Face Amount Band <= {FACE_BAND_MAP.get(h_idx, h_idx)}"

    # 2. Default Numeric (Age, Duration)
    if low != -float('inf') and high != float('inf'):
        return f"{low:.1f} < {feat} <= {high:.1f}"
    elif low != -float('inf'):
        return f"{feat} > {low:.1f}"
    else:
        return f"{feat} <= {high:.1f}"

def get_leaf_path(tree, leaf_id, feature_names):
    """Extract, optimize and translate rules."""
    tree_ = tree.tree_
    
    def find_path(node, target, path=[]):
        if node == target: return path
        left, right = tree_.children_left[node], tree_.children_right[node]
        if left != -1:
            res = find_path(left, target, path + [(node, 'left')])
            if res: return res
        if right != -1:
            res = find_path(right, target, path + [(node, 'right')])
            if res: return res
        return None
    
    path = find_path(0, leaf_id)
    if not path: return []
    
    # Organize by feature
    bounds = {}
    for node, direction in path:
        feat = feature_names[tree_.feature[node]]
        thresh = tree_.threshold[node]
        if feat not in bounds: bounds[feat] = {'min': -float('inf'), 'max': float('inf')}
        
        if direction == 'right':
            bounds[feat]['min'] = max(bounds[feat]['min'], thresh)
        else:
            bounds[feat]['max'] = min(bounds[feat]['max'], thresh)
            
    # Translate
    final_rules = []
    for feat, b in bounds.items():
        rule = translate_rule(feat, b['min'], b['max'])
        if rule: final_rules.append(rule)
        
    return final_rules


def infer_year_column(df):
    # Infer observation/calendar year column name from dataframe columns.
    candidates = [
        'Observation_Year', 'Calendar_Year', 'Year', 'ObservationYear', 'CalendarYear',
        'Policy_Year', 'PolicyYear', 'Obsv_Year', 'OBS_YEAR', 'CAL_YEAR'
    ]
    cols = set(df.columns)
    for c in candidates:
        if c in cols:
            return c
    # Fallback: any column containing 'year' (case-insensitive) with reasonable unique values
    year_like = [c for c in df.columns if 'year' in c.lower()]
    for c in year_like:
        try:
            nunq = df[c].nunique(dropna=True)
            if 5 <= nunq <= 30:
                return c
        except Exception:
            continue
    return None

def compute_year_stability(df, leaf_col, year_col, ae_threshold=0.15):
    # Compute year-based stability metrics for each leaf.
    # Returns dict: leaf_id -> metrics dict.
    # Metrics are weighted by Expected_Deaths to reflect credibility.
    if year_col is None:
        return {}

    g = (df.groupby([leaf_col, year_col])
           .agg(deaths=('Death_Count', 'sum'),
                expected=('Expected_Deaths', 'sum'),
                exposure=('Policies_Exposed', 'sum'))
           .reset_index())

    g = g[g['expected'] > 0].copy()
    if g.empty:
        return {}

    g['ae'] = g['deaths'] / g['expected']
    g['ae_clip'] = g['ae'].clip(0.1, 10.0)
    g['log_ae'] = np.log(g['ae_clip'])
    g['sign'] = np.sign(g['ae'] - 1.0)
    g['is_anomaly'] = (abs(g['ae'] - 1.0) > ae_threshold)

    out = {}
    for leaf_id, gg in g.groupby(leaf_col):
        w = gg['expected'].values.astype(float)
        if w.sum() <= 0:
            continue

        log_ae = gg['log_ae'].values.astype(float)
        mu = float((log_ae * w).sum() / w.sum())
        var = float(((log_ae - mu) ** 2 * w).sum() / w.sum())
        sd = float(var ** 0.5)

        stability_score = float(np.exp(-sd))

        sign_mean = float((gg['sign'].values.astype(float) * w).sum() / w.sum())
        direction_consistency = float(abs(sign_mean))

        persistence = float((gg['is_anomaly'].values.astype(float) * w).sum() / w.sum())

        years_analyzed = int(gg[year_col].nunique(dropna=True))
        anomaly_years = int(gg.groupby(year_col)['is_anomaly'].max().sum())

        direction = "Over" if sign_mean > 0.05 else ("Under" if sign_mean < -0.05 else "Mixed")

        out[int(leaf_id)] = {
            "year_col": str(year_col),
            "years_analyzed": years_analyzed,
            "anomaly_years": anomaly_years,
            "direction": direction,
            "direction_consistency": round(direction_consistency, 3),
            "persistence": round(persistence, 3),
            "stability_score": round(stability_score, 3),
            "log_ae_sd": round(sd, 3)
        }
    return out


def main():
    print("Loading data...")
    df = pd.read_parquet(DATA_PATH)
    df = df.sample(frac=0.1, random_state=42).reset_index(drop=True)

    print("Loading model...")
    model = lgb.Booster(model_file=MODEL_PATH)

    print("Applying semantic encoding for Layer B (Full Model Alignment)...")
    
    # Preprocessing
    df['Preferred_Class_Clean'] = df['Preferred_Class'].apply(clean_preferred_class)
    df['Is_Male'] = (df['Sex'] == 'M').astype(int)
    df['Is_Smoker_Known'] = (df['Smoker_Status'] != 'U').astype(int)
    df['Is_Smoker'] = (df['Smoker_Status'] == 'S').astype(int)
    df['Face_Amount_Band_Val'] = df['Face_Amount_Band'].str.extract(r'(\d+)').astype(float)
    
    # Categoricals to Dummies
    plan_dummies = pd.get_dummies(df['Insurance_Plan'], prefix='Plan').astype(int)
    pref_dummies = pd.get_dummies(df['Preferred_Class_Clean'], prefix='Pref').astype(int)
    soa_post = pd.get_dummies(df['SOA_Post_Lvl_Ind'], prefix='SOA_Post').astype(int)
    soa_antp = pd.get_dummies(df['SOA_Antp_Lvl_TP'], prefix='SOA_Antp').astype(int)
    soa_guar = pd.get_dummies(df['SOA_Guar_Lvl_TP'], prefix='SOA_Guar').astype(int)
    
    df = pd.concat([df, plan_dummies, pref_dummies, soa_post, soa_antp, soa_guar], axis=1)

    # LGBM Predictions for A/E
    FEATURES_LGBM = ['Attained_Age', 'Issue_Age', 'Duration', 'Sex', 'Smoker_Status', 
                     'Insurance_Plan', 'Face_Amount_Band', 'Preferred_Class', 
                     'SOA_Post_Lvl_Ind', 'SOA_Antp_Lvl_TP', 'SOA_Guar_Lvl_TP']
    X_lgbm = df[FEATURES_LGBM].copy()
    for col in X_lgbm.select_dtypes(['object', 'category']).columns:
        X_lgbm[col] = LabelEncoder().fit_transform(X_lgbm[col].astype(str))
    
    df['Predicted_Rate'] = model.predict(X_lgbm.values)
    df['Expected_Deaths'] = df['Predicted_Rate'] * df['Policies_Exposed']
    df['Actual_Rate'] = df['Death_Count'] / df['Policies_Exposed'].replace(0, np.nan)
    overall_rate = df['Death_Count'].sum() / df['Policies_Exposed'].sum()

    # Feature Sets
    COV_FEAT = ['Attained_Age', 'Duration', 'Is_Male', 'Is_Smoker_Known', 'Is_Smoker'] + plan_dummies.columns.tolist()
    SPOT_FEAT = COV_FEAT + ['Issue_Age', 'Face_Amount_Band_Val'] + \
                pref_dummies.columns.tolist() + soa_post.columns.tolist() + \
                soa_antp.columns.tolist() + soa_guar.columns.tolist()

    # Layer A: Coverage
    print("Training Coverage Tree...")
    X_cov, y_cov = df[COV_FEAT], (df['Death_Count'] / df['Expected_Deaths'].replace(0, np.nan)).fillna(1.0).clip(0.1, 10)
    cov_tree = DecisionTreeRegressor(max_depth=5, min_samples_leaf=10000, random_state=42)
    cov_tree.fit(X_cov, y_cov, sample_weight=df['Policies_Exposed'])
    
    df['Coverage_Leaf'] = cov_tree.apply(X_cov)
    cov_agg = df.groupby('Coverage_Leaf').agg({'Policies_Exposed': 'sum', 'Death_Count': 'sum', 'Expected_Deaths': 'sum'}).reset_index()
    cov_registry = {"layer": "A", "overall_rate": round(overall_rate, 6), "segments": []}

    for _, row in cov_agg.iterrows():
        leaf_id = int(row['Coverage_Leaf'])
        rules = get_leaf_path(cov_tree, leaf_id, COV_FEAT)
        cov_registry["segments"].append({
            "segment_id": f"COV_{leaf_id:03d}",
            "rule_list": rules,
            "rule_text": " AND ".join(rules),
            "statistics": {
                "exposure": int(row['Policies_Exposed']),
                "actual_deaths": int(row['Death_Count']),
                "expected_deaths": round(float(row['Expected_Deaths']), 3),
                "ae_ratio": round(row['Death_Count'] / row['Expected_Deaths'], 3)
            }
        })
    with open(os.path.join(OUTPUT_DIR, 'coverage_registry.json'), 'w') as f: json.dump(cov_registry, f, indent=2)

    # Layer B: Spotlight (Independent A/E Anomaly Detection Tree)
    # Train a separate tree with lower thresholds to find fine-grained anomalies
    print("Training Spotlight Tree (A/E Anomalies, min_samples=1000)...")
    for f_name in os.listdir(SPOTLIGHT_DIR): os.remove(os.path.join(SPOTLIGHT_DIR, f_name))
    
    # Train independent Spotlight Tree with lower sample threshold
    # Target: A/E ratio (same as coverage, but with finer granularity)
    y_spot = (df['Death_Count'] / df['Expected_Deaths'].replace(0, np.nan)).fillna(1.0).clip(0.1, 10)
    spot_tree = DecisionTreeRegressor(max_depth=8, min_samples_leaf=1000, random_state=42)
    spot_tree.fit(df[SPOT_FEAT], y_spot, sample_weight=df['Policies_Exposed'])
    
    # Apply tree and aggregate
    df['Spotlight_Leaf'] = spot_tree.apply(df[SPOT_FEAT])
    spot_agg = df.groupby('Spotlight_Leaf').agg({
        'Policies_Exposed': 'sum', 
        'Death_Count': 'sum', 
        'Expected_Deaths': 'sum'
    }).reset_index()
    spot_agg['ae_ratio'] = spot_agg['Death_Count'] / spot_agg['Expected_Deaths']
    


    # Year-based stability (if a year column is available)
    year_col = infer_year_column(df)
    year_stability = compute_year_stability(df, 'Spotlight_Leaf', year_col, ae_threshold=0.15)

    # Filter: A/E deviation > 15% AND exposure >= 1000
    ae_anom = spot_agg[
        (abs(spot_agg['ae_ratio'] - 1) > 0.15) & 
        (spot_agg['Policies_Exposed'] >= 1000)
    ].copy()
    
    # Summary construction
    spotlight_summary = {
        "layer": "B",
        "overall_rate": round(overall_rate, 6),
        "statistics": {
            "total_exposure": int(df['Policies_Exposed'].sum()),
            "ae_anomaly_exposure": int(df[df['Spotlight_Leaf'].isin(ae_anom['Spotlight_Leaf'])]['Policies_Exposed'].sum()),
            "ae_anomaly_exposure_ratio":int(df[df['Spotlight_Leaf'].isin(ae_anom['Spotlight_Leaf'])]['Policies_Exposed'].sum())/int(df['Policies_Exposed'].sum()),
            "total_anomaly_count": len(ae_anom),
            "anomaly_count": len(ae_anom),
            "anomaly_count_ratio": len(ae_anom)/len(spot_agg)
        },
        "anomalies": []
    }
    
    ae_anom['credibility'] = np.minimum(np.sqrt(ae_anom['Policies_Exposed'] / 5000), 1.0)
    # Sort by credibility * abs(ae_ratio - 1) to prioritize meaningful anomalies
    ae_anom['priority'] = ae_anom['credibility'] * abs(ae_anom['ae_ratio'] - 1)
    ae_anom = ae_anom.sort_values('priority', ascending=False)
    
    for _, row in ae_anom.iterrows():
        leaf_id = int(row['Spotlight_Leaf'])
        rules = get_leaf_path(spot_tree, leaf_id, SPOT_FEAT)
        detail = {
            "segment_id": f"SPOT_{leaf_id:03d}",
            "rule_list": rules,
            "rule_text": " AND ".join(rules),
            "statistics": {
                "exposure": int(row['Policies_Exposed']),
                "actual_deaths": int(row['Death_Count']),
                "expected_deaths": round(float(row['Expected_Deaths']), 3),
                "ae_ratio": round(row['ae_ratio'], 3),
                "credibility_score": round(row['credibility'], 3),
                "actual_deaths": int(row['Death_Count']),
                "expected_deaths": round(float(row['Expected_Deaths']), 3),
                "stability": year_stability.get(leaf_id, None)
            }
        }
        with open(os.path.join(SPOTLIGHT_DIR, f"{detail['segment_id']}.json"), 'w') as f: json.dump(detail, f, indent=2)
        spotlight_summary["anomalies"].append({
            "segment_id": detail["segment_id"],
            "rule_list": rules,
            "statistics": {
                "exposure": detail["statistics"]["exposure"],
                "actual_deaths": detail["statistics"]["actual_deaths"],
                "expected_deaths": detail["statistics"]["expected_deaths"],
                "ae_ratio": detail["statistics"]["ae_ratio"],
                "credibility": detail["statistics"]["credibility_score"]
            }
        })

    with open(os.path.join(OUTPUT_DIR, 'spotlight_summary.json'), 'w') as f: json.dump(spotlight_summary, f, indent=2)

    print(f"Success: Found {len(ae_anom)} A/E anomalies with credibility scoring.")


if __name__ == "__main__":
    main()
