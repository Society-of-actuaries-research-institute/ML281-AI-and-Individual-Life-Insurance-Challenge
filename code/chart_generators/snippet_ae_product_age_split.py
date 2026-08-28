# =========================================================================
# CELL: A/E BY PRODUCT AND OBSERVATION YEAR (SPLIT BY ATTAINED AGE 48.5)
# Replaces Figure 4.3 Face Amount split with Age Split
# =========================================================================

# Handle column names
if 'Observation_Year' in df.columns:
    year_col = 'Observation_Year'
else:
    year_col = 'Year'

# Define Age Split
# Threshold 48.5 means <= 48 is 'Younger', >= 49 is 'Older'
def get_age_split(age):
    if age < 48.5:
        return 'Attained Age < 48.5'
    else:
        return 'Attained Age >= 48.5'

# Apply split if not exists
if 'Age_Split' not in df.columns:
    df['Age_Split'] = df['Attained_Age'].apply(get_age_split)

# Filter for Core Products
target_products = ['Perm', 'Term', 'UL', 'ULSG']
df_prod = df[df['Insurance_Plan'].isin(target_products)].copy()

# Aggregation: Group by Split, Year, Plan
grouped = df_prod.groupby(['Age_Split', year_col, 'Insurance_Plan']).agg({
    'Death_Count': 'sum',
    'Expected_Deaths': 'sum'
}).reset_index()

grouped['AE_Count'] = grouped['Death_Count'] / grouped['Expected_Deaths']

# Prepare Plot Data
splits = ['Attained Age < 48.5', 'Attained Age >= 48.5']
years = sorted(grouped[year_col].unique())

# Colors (Product Based)
color_map = {
    'Perm': '#003366',   # Dark Blue
    'Term': '#7d3f98',   # Purple
    'UL': '#5bc0de',     # Light Blue/Cyan
    'ULSG': '#d62728',   # Red
}

# Plot Layout: 2 Columns (Age Splits), 2 Rows (A/E Ratio, Deaths)
fig, axes = plt.subplots(2, 2, figsize=(16, 8), sharex=True, 
                         gridspec_kw={'height_ratios': [2, 1], 'hspace': 0.05, 'wspace': 0.1})

# Common Limits for consistency
ae_ylim = (0.75, 1.4) 
# Determine max deaths for unified Y-axis scaling on bars
max_deaths = grouped.groupby(['Age_Split', year_col])['Death_Count'].sum().max() * 1.1

for i, split in enumerate(splits):
    # Filter data for this column (Age Split)
    subset = grouped[grouped['Age_Split'] == split]
    
    # Pivot for easier plotting
    pivot_ae = subset.pivot(index=year_col, columns='Insurance_Plan', values='AE_Count')
    pivot_death = subset.pivot(index=year_col, columns='Insurance_Plan', values='Death_Count').fillna(0)
    
    # --- Top Row: A/E Ratio (Lines) ---
    ax_line = axes[0, i]
    for prod in target_products:
        if prod in pivot_ae.columns:
            ax_line.plot(pivot_ae.index, pivot_ae[prod], color=color_map.get(prod, 'gray'), linewidth=3, marker='o', markersize=4, label=prod)
            
    # Formatting Top
    ax_line.axhline(1.0, color='gray', linestyle='-', linewidth=1, alpha=0.5)
    ax_line.set_title(split, fontsize=12, fontweight='bold', color='#444444')
    ax_line.set_ylim(ae_ylim)
    ax_line.grid(True, alpha=0.3)
    
    # Y-Label only on left plot
    if i == 0:
        ax_line.set_ylabel('Actual vs Expected', fontsize=12)
        ax_line.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    else:
        ax_line.set_yticklabels([]) # Hide labels
        ax_line.set_ylabel('')

    # --- Bottom Row: Deaths (Stacked Bars) ---
    ax_bar = axes[1, i]
    # Stack Order: ULSG -> UL -> Term -> Perm
    stack_order = ['ULSG', 'UL', 'Term', 'Perm']
    bottom = np.zeros(len(years))
    
    for prod in stack_order:
        if prod in pivot_death.columns:
            # Reindex to ensure all years exist (fill 0 if missing)
            values = pivot_death[prod].reindex(years, fill_value=0).values
            ax_bar.bar(years, values, bottom=bottom, color=color_map.get(prod, 'gray'), alpha=0.9, label=prod)
            bottom += values
            
    # Formatting Bottom
    ax_bar.set_ylim(0, max_deaths)
    ax_bar.grid(True, alpha=0.3, axis='y')
    ax_bar.tick_params(axis='x', rotation=90) # Rotate Years
    
    # Y-Label only on left plot
    if i == 0:
        ax_bar.set_ylabel('Number of Deaths', fontsize=12)
    else:
        ax_bar.set_yticklabels([])
        ax_bar.set_ylabel('')

    # Legend only on the last column's top plot
    if i == 1:
        ax_line.legend(loc='upper right', title='Insurance Plan', frameon=True, fontsize=9)


plt.suptitle('A/E BY PRODUCT AND OBSERVATION YEAR (Split by Attained Age 48.5)', fontsize=16, color='#003366', fontweight='bold', y=0.98)
plt.show()
