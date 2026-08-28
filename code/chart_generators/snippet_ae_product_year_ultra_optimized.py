# =========================================================================
# CELL: A/E BY PRODUCT AND OBSERVATION YEAR (ULTRA MEMORY OPTIMIZED)
# =========================================================================
import matplotlib.ticker as mtick
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import gc

# 1. Force Garbage Collection to free up any unreferenced memory
gc.collect()

# Handle column names
if 'Observation_Year' in df.columns:
    year_col = 'Observation_Year'
else:
    year_col = 'Year'

print("Optimizing data for aggregation...")

# 2. Select ONLY the columns we strictly need. 
# This reduces the width of the DataFrame significantly.
cols_needed = [year_col, 'Insurance_Plan', 'Death_Count', 'Expected_Deaths']
df_subset = df[cols_needed].copy()

# 3. Optimize Data Types (Crucial for Memory)
# Converting string 'Insurance_Plan' to category can save ~90% memory for that column
if df_subset['Insurance_Plan'].dtype == 'object':
    df_subset['Insurance_Plan'] = df_subset['Insurance_Plan'].astype('category')

print("Aggregating...")
# 4. Groupby on the minimal subset
grouped_all = df_subset.groupby([year_col, 'Insurance_Plan'], observed=True)[['Death_Count', 'Expected_Deaths']].sum().reset_index()

# Free up the subset memory immediately
del df_subset
gc.collect()

# Filter for relevant products from the aggregated result (tiny size)
target_products = ['Perm', 'Term', 'UL', 'ULSG']
grouped = grouped_all[grouped_all['Insurance_Plan'].isin(target_products)].copy()

# Calculate A/E Ratio
grouped['AE_Count'] = grouped['Death_Count'] / grouped['Expected_Deaths']

# Prepare data for plotting
years = sorted(grouped[year_col].unique())
# Create pivoting for easier plotting
pivot_ae = grouped.pivot(index=year_col, columns='Insurance_Plan', values='AE_Count')
pivot_death = grouped.pivot(index=year_col, columns='Insurance_Plan', values='Death_Count').fillna(0)

# Colors
color_map = {
    'Perm': '#003366',   # Dark Blue
    'Term': '#7d3f98',   # Purple
    'UL': '#5bc0de',     # Light Blue/Cyan
    'ULSG': '#d62728',   # Red
}

print("Plotting...")
# Plot
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True, gridspec_kw={'height_ratios': [2, 1], 'hspace': 0.05})

# --- Top Plot: A/E Ratio (Lines) ---
for prod in target_products:
    if prod in pivot_ae.columns:
        ax1.plot(pivot_ae.index, pivot_ae[prod], color=color_map.get(prod, 'gray'), linewidth=3, marker='o', label=prod)

# Formatting Top
ax1.axhline(1.0, color='gray', linestyle='-', linewidth=1, alpha=0.5)
ax1.set_ylabel('Actual vs Expected', fontsize=12)
ax1.set_ylim(0.75, 1.25)
ax1.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
ax1.set_title('A/E BY PRODUCT AND OBSERVATION YEAR (CORE DATA)', fontsize=14, loc='left', pad=15, color='#003366', fontweight='bold')
ax1.legend(loc='upper right', title='Insurance Plan', frameon=True)
ax1.grid(True, alpha=0.3)

# --- Bottom Plot: Deaths (Stacked Bars) ---
stack_order = ['ULSG', 'UL', 'Term', 'Perm']
bottom = np.zeros(len(years))

for prod in stack_order:
    if prod in pivot_death.columns:
        values = pivot_death[prod].values
        ax2.bar(years, values, bottom=bottom, color=color_map.get(prod, 'gray'), alpha=0.9, label=prod)
        bottom += values

# Formatting Bottom
ax2.set_ylabel('Number of Deaths', fontsize=12)
ax2.set_xlabel('Observation Year', fontsize=12)
ax2.grid(True, alpha=0.3, axis='y')

plt.show()
