import os
import re
import joblib
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_absolute_percentage_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# 1. Load data
excel_file = r"C:\Users\anish.nair\Downloads\PNOCs 2024-2025.xlsm"
sheet_name = "Baseline + Actual Dates"
df_raw = pd.read_excel(excel_file, sheet_name=sheet_name)
df_raw.columns = df_raw.columns.str.strip()

# 2. Detect key columns
col_map = {}
for col in df_raw.columns:
    l = col.lower()
    if 'process' in l:      col_map['process'] = col
    elif 'baseline' in l:   col_map['baseline'] = col
    elif 'actual' in l:     col_map['actual'] = col
    elif re.search(r'pnoc.*id', l): col_map['pnoc_id'] = col

# 3. Clean 'Process' & parse dates
df_raw[col_map['process']] = (
    df_raw[col_map['process']]
      .str.replace(r'\s*\(.*\)', '', regex=True)
      .str.strip()
)
df_raw[col_map['baseline']] = pd.to_datetime(df_raw[col_map['baseline']], errors='coerce')
df_raw[col_map['actual']]   = pd.to_datetime(df_raw[col_map['actual']]  , errors='coerce')

# 4. Milestones to pivot on
milestones = [
    'PNOC/CI Issued',
    'R&C Due',
    'Enter TA',
    'Branch TA Due',
    'Staff TA/Au Due',
    'NOC Issued',
    'Revision Issued',
]

# 5. Pivot wide: Baseline & Actual
base = df_raw.pivot(index=col_map['pnoc_id'], columns=col_map['process'], values=col_map['baseline'])[milestones].reset_index(drop=True)
act  = df_raw.pivot(index=col_map['pnoc_id'], columns=col_map['process'], values=col_map['actual'])[milestones].reset_index(drop=True)

# 6. Spline-interpolate missing dates (via int64)
for df_date in (base, act):
    for m in milestones:
        if df_date[m].isnull().any():
            ints      = df_date[m].astype('int64')
            interpol  = pd.Series(ints).interpolate(method='spline', order=3)
            df_date[m] = pd.to_datetime(interpol.astype('int64'))

# 7. Business-day holidays (2024)
holidays = np.array([
    datetime(2024,1,1), datetime(2024,1,15), datetime(2024,5,27),
    datetime(2024,7,4), datetime(2024,9,2),  datetime(2024,10,14),
    datetime(2024,11,14), datetime(2024,11,28), datetime(2024,12,25),
], dtype='datetime64[D]')

# 8. Compute durations between successive milestones
for i in range(len(milestones)-1):
    s, e = milestones[i], milestones[i+1]
    base[f'{s}_to_{e}_BL'] = np.busday_count(
        base[s].values.astype('datetime64[D]'),
        base[e].values.astype('datetime64[D]'), holidays=holidays
    )
    act[f'{s}_to_{e}_ACT'] = np.busday_count(
        act[s].values.astype('datetime64[D]'),
        act[e].values.astype('datetime64[D]'), holidays=holidays
    )

# 9. Build DataFrame of features & targets, fill any remaining NaNs
df = pd.concat([
    base.filter(like='_BL'),
    act.filter(like='_ACT')
], axis=1).apply(lambda c: c.fillna(c.median()), axis=0)

X = df[[c for c in df.columns if c.endswith('_BL')]]
y = df[[c for c in df.columns if c.endswith('_ACT')]]

# 10. Train / Test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 11. Multi-output regression with grid search
multi = MultiOutputRegressor(HistGradientBoostingRegressor(random_state=42))
param_grid = {
    'estimator__max_iter': [100, 200],
    'estimator__max_depth': [3, 5],
    'estimator__learning_rate': [0.01, 0.1],
}
grid = GridSearchCV(multi, param_grid, cv=3,
                    scoring='neg_mean_absolute_percentage_error',
                    n_jobs=-1)
grid.fit(X_train, y_train)

# 12. Evaluate per subprocess
best = grid.best_estimator_
y_pred = best.predict(X_test)

results = []
for idx, col in enumerate(y_test.columns):
    mape = mean_absolute_percentage_error(y_test[col], y_pred[:, idx])
    r2   = r2_score(y_test[col], y_pred[:, idx])
    results.append({'Phase': col, 'MAPE': f"{mape:.2%}", 'R2': f"{r2:.3f}"})

results_df = pd.DataFrame(results)
print("Per-phase predictions:\n", results_df)

# 13. Save model
joblib.dump(best, "pnoc_multiphase_predictor.pkl")
print("\nModel saved as pnoc_multiphase_predictor.pkl")


