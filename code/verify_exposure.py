import pandas as pd
from pathlib import Path
import os

# Try to find the data file
possible_paths = [
    Path('data/ilec_with_predictions.parquet'),
    Path('../../data/ilec_with_predictions.parquet'),
    Path('/Users/BF/Development/SOA/data/ilec_with_predictions.parquet')
]

data_path = None
for p in possible_paths:
    if p.exists():
        data_path = p
        break

if data_path is None:
    print("Error: content file not found.")
    exit(1)

print(f"Loading data from: {data_path}")
df = pd.read_parquet(data_path)

# Check condition: Duration <= 12.5 AND Attained_Age <= 15.5 AND Smoker_Status == 'U'
print("Checking Segment Condition:")
print("  1. Duration <= 12.5")
print("  2. Attained_Age <= 15.5")
print("  3. Smoker_Status == 'U' (Unknown)")

# Note: Smoker_Status might be 'U' or 'Unknown', usually 'U' in ILEC
mask = (df['Duration'] <= 12.5) & \
       (df['Attained_Age'] <= 15.5) & \
       (df['Smoker_Status'] == 'U')

subset = df[mask]
row_count = len(subset)
exposure_sum = subset['Policies_Exposed'].sum()
death_count = subset['Death_Count'].sum()

print("-" * 30)
print(f"Row Count: {row_count:,}")
print(f"Sum of Policies_Exposed: {exposure_sum:,.2f}")
print(f"Sum of Death_Count: {death_count:,.2f}")
print("-" * 30)
