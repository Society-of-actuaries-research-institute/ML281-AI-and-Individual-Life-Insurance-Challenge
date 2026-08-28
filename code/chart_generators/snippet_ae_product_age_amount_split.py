# =========================================================================
# CELL: A/E BY PRODUCT (4-WAY SPLIT) - FIXED & OPTIMIZED
# =========================================================================
import matplotlib.ticker as mtick
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import gc

# 1. Force GC
gc.collect()

# Handle column names
if 'Observation_Year' in df.columns:
    year_col = 'Observation_Year'
else:
    year_col = 'Year'

# 2. Define Split Logic (Helpers)
def get_age_type(age):
    return '< 48.5' if age < 48.5 else '>= 48.5'

def get_amt_type(band_str):
    if not isinstance(band_str, str): return 'Unknown'
    code = band_str[:2]
    if code in ['01', '02', '03', '04']:
        return '< 100K'
    elif code in ['05', '06', '07', '08']:
        return '>= 100K'
    else:
        return 'Unknown'

print("Preparing data for 4-Way Split...")

# 3. Create Minimal Subset (Crucial for Memory)
cols = [year_col, 'Insurance_Plan', 'Attained_Age', 'Face_Amount_Band', 'Death_Count', 'Expected_Deaths']
df_minimal = df[cols].copy()

# Optimize data types
if df_minimal['Insurance_Plan'].dtype == 'object':
    df_minimal['Insurance_Plan'] = df_minimal['Insurance_Plan'].astype('category')

# Apply Splits
# Vectorized approach or apply is fine here since subset is smaller, 
# but for extra speed/memory, let's stick to apply as the logic is custom.
df_minimal['Age_Split'] = df_minimal['Attained_Age'].apply(get_age_type).astype('category')
df_minimal['Amt_Split'] = df_minimal['Face_Amount_Band'].apply(get_amt_type).astype('category')

# 4. Filter for Core Products -> THIS DEFINES df_prod
target_products = ['Perm', 'Term', 'UL', 'ULSG']
df_prod = df_minimal[df_minimal['Insurance_Plan'].isin(target_products)]

print(f"Aggregating {len(df_prod):,} records...")

# 5. Group By Split
grouped = df_prod.groupby(['Age_Split', 'Amt_Split', year_col, 'Insurance_Plan'], observed=True).agg({
    'Death_Count': 'sum',
    'Expected_Deaths': 'sum'
}).reset_index()

# Clean up memory
del df_minimal, df_prod
gc.collect()

# Calc Metrics
grouped['AE_Count'] = grouped['Death_Count'] / grouped['Expected_Deaths']
grouped['Death_Count_M'] = grouped['Death_Count'] / 1e6

# 6. Plot Layout (4x2 Grid of Subplots)
fig, axes = plt.subplots(4, 2, figsize=(18, 16), sharex=True, 
                         gridspec_kw={'height_ratios': [1, 1.2, 1, 1.2], 'hspace': 0.15, 'wspace': 0.1})

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
max_deaths_m = grouped.groupby(['Age_Split', 'Amt_Split', year_col])['Death_Count_M'].sum().max() * 1.2
ae_ylim = (0.5, 1.5) 

print("Plotting...")

for r, age_cat in enumerate(splits_age):        # r=0 (Young), r=1 (Old)
    for c, amt_cat in enumerate(splits_amt):    # c=0 (Low), c=1 (High)
        
        ax_line = axes[r*2, c]
        ax_bar = axes[r*2 + 1, c]
        
        subset = grouped[(grouped['Age_Split'] == age_cat) & (grouped['Amt_Split'] == amt_cat)]
        
        pivot_ae = subset.pivot(index=year_col, columns='Insurance_Plan', values='AE_Count')
        pivot_death_m = subset.pivot(index=year_col, columns='Insurance_Plan', values='Death_Count_M').fillna(0)
        
        # --- Top Plot: A/E ---
        for prod in target_products:
            if prod in pivot_ae.columns:
                 ax_line.plot(pivot_ae.index, pivot_ae[prod], color=color_map.get(prod, 'gray'), linewidth=2.5, marker='o', markersize=3, label=prod)
        
        ax_line.axhline(1.0, color='gray', linestyle='-', linewidth=1, alpha=0.5)
        ax_line.set_ylim(ae_ylim)
        ax_line.grid(True, alpha=0.3)
        ax_line.set_title(f"Age {age_cat} | Face {amt_cat}", fontsize=11, fontweight='bold', color='#333333')
        
        if c == 0:
            ax_line.set_ylabel('A/E', fontsize=10)
            ax_line.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
        else:
            ax_line.set_yticklabels([])

        # --- Bottom Plot: Deaths (Millions) ---
        stack_order = ['ULSG', 'UL', 'Term', 'Perm']
        bottom = np.zeros(len(years))
        
        for prod in stack_order:
            if prod in pivot_death_m.columns:
                values = pivot_death_m[prod].reindex(years, fill_value=0).values
                ax_bar.bar(years, values, bottom=bottom, color=color_map.get(prod, 'gray'), alpha=0.9, label=prod)
                bottom += values
        
        ax_bar.set_ylim(0, max_deaths_m)
        ax_bar.grid(True, alpha=0.3, axis='y')
        ax_bar.tick_params(axis='x', rotation=45)
        
        if c == 0:
            ax_bar.set_ylabel('Deaths (M)', fontsize=10)
        else:
            ax_bar.set_yticklabels([])
            
        # Legend only on top-right quadrant
        if r == 0 and c == 1:
            ax_line.legend(loc='upper right', title='Plan', frameon=True, fontsize=8)

plt.suptitle('MULTI-DIMENSIONAL A/E: AGE 48.5 x FACE 100K', fontsize=16, color='#003366', fontweight='bold', y=0.98)

# Use manual spacing instead of tight_layout to avoid warnings/errors
plt.subplots_adjust(left=0.08, right=0.95, top=0.92, bottom=0.05, hspace=0.25, wspace=0.1)
plt.show()
