import os
import re
import pandas as pd
import numpy as np
from datetime import datetime

# 1. Load & normalize column names
excel_file = r"C:\Users\anish.nair\OneDrive - BAE Systems Inc\Desktop\Pipelines\PNOCs 2024-2025.xlsm"
sheet_name = "Baseline + Actual Dates"
df_raw = pd.read_excel(excel_file, sheet_name=sheet_name)
df_raw.columns = df_raw.columns.str.strip()  # trim whitespace

# 2. Auto-detect key columns
col_map = {}
for col in df_raw.columns:
    low = col.lower()
    if 'process' in low:
        col_map['process'] = col
    elif 'baseline' in low:
        col_map['baseline'] = col
    elif 'actual' in low:
        col_map['actual'] = col
    elif re.search(r'pnoc.*id', low):
        col_map['pnoc_id'] = col

# 3. Clean and standardize process names
df_raw[col_map['process']] = (
    df_raw[col_map['process']]
    .str.replace(r'\s*\(.*\)', '', regex=True)  # remove parentheses
    .str.strip()
)

# 4. Define the seven milestones (must match cleaned "Process" entries exactly)
milestones = [
    'PNOC/CI Issued',
    'R&C Due',
    'Enter TA',
    'Branch TA Due',
    'Staff TA/Au Due',
    'NOC Issued',
    'Revision Issued',
]

# 5. Pivot to one row per PNOC
base = df_raw.pivot(
    index=col_map['pnoc_id'],
    columns=col_map['process'],
    values=col_map['baseline']
)[milestones]

act  = df_raw.pivot(
    index=col_map['pnoc_id'],
    columns=col_map['process'],
    values=col_map['actual']
)[milestones]

# 6. Reset index (flatten)
base = base.reset_index(drop=True)
act  = act.reset_index(drop=True)

# 7. Handle missing milestone dates via spline interpolation
for df_date in (base, act):
    for m in milestones:
        if df_date[m].isnull().any():
            df_date[m] = df_date[m].interpolate(method='spline', order=3)

# 8. Compute business-day durations between successive milestones
holidays = np.array([
    datetime(2024,1,1),
    datetime(2024,1,15),
    datetime(2024,5,27),
    datetime(2024,7,4),
    datetime(2024,9,2),
    datetime(2024,10,14),
    datetime(2024,11,14),
    datetime(2024,11,28),
    datetime(2024,12,25),
], dtype='datetime64[D]')

for i in range(len(milestones)-1):
    start, end = milestones[i], milestones[i+1]
    base[f'{start}_to_{end}_BL'] = np.busday_count(
        base[start].values.astype('datetime64[D]'),
        base[end].values.astype('datetime64[D]'),
        holidays=holidays
    )
    act[f'{start}_to_{end}_ACT'] = np.busday_count(
        act[start].values.astype('datetime64[D]'),
        act[end].values.astype('datetime64[D]'),
        holidays=holidays
    )

# 9. Combine duration features and fill any remaining NaNs
df = pd.concat([
    base.filter(like='_BL'),
    act.filter(like='_ACT')
], axis=1).apply(lambda col: col.fillna(col.median()), axis=0)

# 10. Prepare X (baseline durations) and y (sum of actual durations)
feature_cols = [c for c in df.columns if c.endswith('_BL')]
X = df[feature_cols]
y = df[[c for c in df.columns if c.endswith('_ACT')]].sum(axis=1)

# 11. Inspect shapes
print("Features (X) shape:", X.shape)
print("Target (y) shape:", y.shape)

