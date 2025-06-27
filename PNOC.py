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
    low = col.lower()
    if 'process' in low:         col_map['process'] = col
    elif 'baseline' in low:      col_map['baseline'] = col
    elif 'actual' in low:        col_map['actual'] = col
    elif re.search(r'pnoc.*id', low): col_map['pnoc_id'] = col

# 3. Clean 'Process' & parse dates
df_raw[col_map['process']] = (
    df_raw[col_map['process']]
      .str.replace(r'\s*\(.*\)', '', regex=True)
      .str.strip()
)
df_raw[col_map['baseline']] = pd.to_datetime(df_raw[col_map['baseline']], errors='coerce')
df_raw[col_map['actual']]   = pd.to_datetime(df_raw[col_map['actual']],   errors='coerce')

# 4. Milestones
milestones = [
    'PNOC/CI Issued',
    'R&C Due',
    'Enter TA',
    'Branch TA Due',
    'Staff TA/Au Due',
    'NOC Issued',
    'Revision Issued',
]

# 5. Pivot
base = df_raw.pivot(
    index=col_map['pnoc_id'], columns=col_map['process'], values=col_map['baseline']
)[milestones].reset_index(drop=True)
act  = df_raw.pivot(
    index=col_map['pnoc_id'], columns=col_map['process'], values=col_map['actual']
)[milestones].reset_index(drop=True)

# 6. Interpolate missing dates via integer-spline
for df_date in (base, act):
    for m in milestones:
        if df_date[m].isnull().any():
            ints = df_date[m].astype('int64')
            interp = pd.Series(ints).interpolate(method='spline', order=3)
            df_date[m] = pd.to_datetime(interp.astype('int64'))

# 7. Define holidays
holidays = np.array([
    datetime(2024,1,1), datetime(2024,1,15), datetime(2024,5,27),
    datetime(2024,7,4), datetime(2024,9,2),  datetime(2024,10,14),
    datetime(2024,11,14), datetime(2024,11,28), datetime(2024,12,25),
], dtype='datetime64[D]')

# 8. Safe business-day count
def safe_busday_count(start_dates, end_dates, holidays):
    result = np.full(start_dates.shape, np.nan)
    mask = (~pd.isna(start_dates)) & (~pd.isna(end_dates))
    result[mask] = np.busday_count(
        start_dates[mask].astype('datetime64[D]'),
        end_dates[mask].astype('datetime64[D]'),
        holidays=holidays
    )
    return result

for i in range(len(milestones)-1):
    s, e = milestones[i], milestones[i+1]
    bl = f'{s}_to_{e}_BL'
    ac = f'{s}_to_{e}_ACT'
    base[bl] = safe_busday_count(base[s], base[e], holidays)
    act[ac] = safe_busday_count(act[s], act[e], holidays)

# 9. Combine & fill NaNs
df = pd.concat([
    base.filter(like='_BL'), act.filter(like='_ACT')
], axis=1).apply(lambda col: col.fillna(col.median()), axis=0)

# 10. Features & multi-target
feature_cols = [c for c in df.columns if c.endswith('_BL')]
target_cols  = [c for c in df.columns if c.endswith('_ACT')]
X = df[feature_cols]
y = df[target_cols]

# 11. Train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 12. Multi-output model
multi = MultiOutputRegressor(HistGradientBoostingRegressor(random_state=42))
param_grid = {
    'estimator__max_iter': [100, 200],
    'estimator__max_depth': [3, 5],
    'estimator__learning_rate': [0.01, 0.1],
}
grid = GridSearchCV(multi, param_grid, cv=3,
                    scoring='neg_mean_absolute_percentage_error', n_jobs=-1)
grid.fit(X_train, y_train)

# 13. Evaluate
best = grid.best_estimator_
y_pred = best.predict(X_test)
results = []
for idx, col in enumerate(target_cols):
    mape = mean_absolute_percentage_error(y_test[col], y_pred[:, idx])
    r2   = r2_score(y_test[col], y_pred[:, idx])
    results.append({'Phase': col, 'MAPE': f"{mape:.2%}", 'R2': f"{r2:.3f}"})
import ace_tools as tools; tools.display_dataframe_to_user("Regression Results", pd.DataFrame(results))

# 14. Save model
joblib.dump(best, "pnoc_multiphase_predictor.pkl")



