# =========================================================================
# CELL: POLICIES EXPOSED BY PRODUCT AND ISSUE YEAR (1980-2019)
# =========================================================================
import matplotlib.ticker as mtick
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.cm as cm

# Ensure Issue_Year exists
if 'Issue_Year' not in df.columns:
    if 'Observation_Year' in df.columns and 'Duration' in df.columns:
        df['Issue_Year'] = df['Observation_Year'] - df['Duration'] + 1
        print("Computed Issue_Year from Observation_Year and Duration")
    elif 'Year' in df.columns and 'Duration' in df.columns:
         df['Issue_Year'] = df['Year'] - df['Duration'] + 1
         print("Computed Issue_Year from Year and Duration")

# Filter Data for Issue Years 1980-2019
df_issue = df[(df['Issue_Year'] >= 1980) & (df['Issue_Year'] <= 2019)].copy()

# Filter/Group Products
# Check available products:
available_products = df_issue['Insurance_Plan'].unique()

# Aggregation by Issue Year and Product
grouped_issue = df_issue.groupby(['Issue_Year', 'Insurance_Plan'])['Policies_Exposed'].sum().reset_index()

# Pivot for Stacked Bar
pivot_issue = grouped_issue.pivot(index='Issue_Year', columns='Insurance_Plan', values='Policies_Exposed').fillna(0)

# Calculate Percentages for 100% Stacked Bar
pivot_pct = pivot_issue.div(pivot_issue.sum(axis=1), axis=0)

# Colors (Approximated from image 4.1)
# Perm (Blue), Term (Purple), UL (Cyan), ULSG (Red), VL (Yellow), VLSG (Green), Other (Dark Blue/Black)
color_map_issue = {
    'Perm': '#003366',    # Dark Blue
    'Term': '#7d3f98',    # Purple
    'UL': '#5bc0de',      # Light Blue/Cyan
    'ULSG': '#d62728',    # Red
    'VL': '#ffdd05',      # Yellow
    'VLSG': '#77ac30',    # Green
    'Other': '#001f3f'    # Very Dark Blue
}

# Plot: 2 Subplots (Top: %, Bottom: Count)
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True, gridspec_kw={'height_ratios': [1, 1], 'hspace': 0.05})

# --- Top Plot: 100% Stacked Bar ---
bottom = np.zeros(len(pivot_pct))
for i, col in enumerate(pivot_pct.columns):
    values = pivot_pct[col].values
    color = color_map_issue.get(col, None) # Use defined or auto
    if color is None: color = plt.cm.tab20(i)
        
    ax1.bar(pivot_pct.index, values, bottom=bottom, label=col, color=color, alpha=0.9, width=0.8)
    bottom += values

ax1.set_ylabel('% of Policies Exposed', fontsize=12)
ax1.set_ylim(0, 1.0)
ax1.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
ax1.set_title('POLICIES EXPOSED BY PRODUCT AND ISSUE YEARS 1980-2019', fontsize=14, loc='left', pad=15, color='#003366', fontweight='bold')
ax1.legend(loc='upper left', bbox_to_anchor=(1, 1), title='Insurance Plan')
ax1.grid(True, alpha=0.3, axis='y')

# --- Bottom Plot: Stacked Bar (Count) ---
bottom = np.zeros(len(pivot_issue))
for i, col in enumerate(pivot_issue.columns):
    values = pivot_issue[col].values
    color = color_map_issue.get(col, None) 
    if color is None: color = plt.cm.tab20(i)
    
    ax2.bar(pivot_issue.index, values, bottom=bottom, label=col, color=color, alpha=0.9, width=0.8)
    bottom += values

ax2.set_ylabel('Policies Exposed', fontsize=12)
ax2.set_xlabel('Issue Year', fontsize=12)
ax2.grid(True, alpha=0.3, axis='y')

# Rotate x-axis labels
plt.xticks(pivot_issue.index, rotation=90)

plt.tight_layout()
plt.show()
