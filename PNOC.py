import re
import joblib
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_absolute_percentage_error, r2_score

# ────────────────────────────────────────────────────────────────
# 0. CONFIG ── update only if your sheet/path changes
# ────────────────────────────────────────────────────────────────
EXCEL_FILE = r"C:\Users\anish.nair\Downloads\PNOCs 2024-2025.xlsm"
SHEET_NAME = "Baseline + Actual Dates"

MILESTONES = [
    'PNOC/CI Issued',
    'R&C Due',
    'Enter TA',
    'Branch TA Due',
    'Staff TA/Au Due',
    'NOC Issued',
    'Revision Issued',
]

HOLIDAYS_2024 = np.array([
    datetime(2024, 1,  1), datetime(2024, 1, 15),
    datetime(2024, 5, 27), datetime(2024, 7,  4),
    datetime(2024, 9,  2), datetime(2024,10, 14),
    datetime(2024,11, 14), datetime(2024,11, 28),
    datetime(2024,12, 25),
], dtype='datetime64[D]')

# ────────────────────────────────────────────────────────────────
# 1. LOAD SHEET & AUTO-DETECT COLUMN HEADERS
# ────────────────────────────────────────────────────────────────
df_raw = pd.read_excel(EXCEL_FILE, sheet_name=SHEET_NAME)
df_raw.columns = df_raw.columns.str.strip()          # trim header whitespace

col_map = {}                                         # find key columns
for c in df_raw.columns:
    l = c.lower()
    if 'process' in l:      col_map['process']  = c
    elif 'baseline' in l:   col_map['baseline'] = c
    elif 'actual' in l:     col_map['actual']   = c
    elif re.search(r'pnoc.*id', l):
        col_map['pnoc_id'] = c

# ────────────────────────────────────────────────────────────────
# 2. CLEAN PROCESS LABELS  &  PARSE DATES
# ────────────────────────────────────────────────────────────────
df_raw[col_map['process']] = (
    df_raw[col_map['process']]
        .str.replace(r'\s*\(.*\)', '', regex=True)   # drop "(Need Date: …)"
        .str.strip()
)

df_raw[col_map['baseline']] = pd.to_datetime(df_raw[col_map['baseline']], errors='coerce')
df_raw[col_map['actual'  ]] = pd.to_datetime(df_raw[col_map['actual'  ]], errors='coerce')

# ────────────────────────────────────────────────────────────────
# 3. PIVOT  →  one row = one PNOC  (baseline & actual)
# ────────────────────────────────────────────────────────────────
base = (
    df_raw.pivot(index=col_map['pnoc_id'],
                 columns=col_map['process'],
                 values=col_map['baseline'])[MILESTONES]
      .reset_index(drop=True)
)
act  = (
    df_raw.pivot(index=col_map['pnoc_id'],
                 columns=col_map['process'],
                 values=col_map['actual'])[MILESTONES]
      .reset_index(drop=True)
)

# ────────────────────────────────────────────────────────────────
# 4. INTERPOLATE MISSING DATES  (int64-spline → back to datetime)
# ────────────────────────────────────────────────────────────────
for df_dates in (base, act):
    for m in MILESTONES:
        if df_dates[m].isna().any():
            ints = df_dates[m].astype('int64')
            ints_interp = (
                pd.Series(ints).interpolate(method='spline', order=3)
                               .astype('int64')
            )
            df_dates[m] = pd.to_datetime(ints_interp)

# ────────────────────────────────────────────────────────────────
# 5. SAFE BUSINESS-DAY COUNTS  (handles residual NaT)
# ────────────────────────────────────────────────────────────────
def busdays(start_series: pd.Series,
            end_series:   pd.Series,
            holi: np.ndarray) -> np.ndarray:
    """np.busday_count with NaT protection; returns float array (NaN where missing)."""
    mask = start_series.notna() & end_series.notna()
    out  = np.full(len(start_series), np.nan, dtype=float)
    if mask.any():
        s = start_series[mask].dt.normalize().to_numpy(dtype='datetime64[D]')
        e = end_series  [mask].dt.normalize().to_numpy(dtype='datetime64[D]')
        out[mask] = np.busday_count(s, e, holidays=holi)
    return out

for i in range(len(MILESTONES) - 1):
    a, b = MILESTONES[i], MILESTONES[i + 1]
    base[f'{a}_to_{b}_BL'] = busdays(base[a], base[b], HOLIDAYS_2024)
    act [f'{a}_to_{b}_ACT'] = busdays(act [a], act [b], HOLIDAYS_2024)

# ────────────────────────────────────────────────────────────────
# 6. FEATURE / TARGET MATRICES
# ────────────────────────────────────────────────────────────────
df = (
    pd.concat([base.filter(like='_BL'), act.filter(like='_ACT')], axis=1)
      .apply(lambda s: s.fillna(s.median()), axis=0)       # median-fill
)

X = df[[c for c in df.columns if c.endswith('_BL')]]
Y = df[[c for c in df.columns if c.endswith('_ACT')]]      # multi-target

# ────────────────────────────────────────────────────────────────
# 7. TRAIN / TEST  &  MULTI-OUTPUT REGRESSION
# ────────────────────────────────────────────────────────────────
X_tr, X_te, Y_tr, Y_te = train_test_split(X, Y, test_size=0.2, random_state=42)

grid = GridSearchCV(
    MultiOutputRegressor(HistGradientBoostingRegressor(random_state=42)),
    param_grid={
        'estimator__max_iter'      : [100, 200],
        'estimator__max_depth'     : [3, 5],
        'estimator__learning_rate' : [0.01, 0.1],
    },
    cv=3,
    scoring='neg_mean_absolute_percentage_error',
    n_jobs=-1,
)
grid.fit(X_tr, Y_tr)

best = grid.best_estimator_
Y_pred = best.predict(X_te)

# ────────────────────────────────────────────────────────────────
# 8. PHASE-BY-PHASE METRICS
# ────────────────────────────────────────────────────────────────
results = []
for idx, col in enumerate(Y_te.columns):
    results.append({
        'Phase': col,
        'MAPE' : f"{mean_absolute_percentage_error(Y_te.iloc[:, idx], Y_pred[:, idx]):.2%}",
        'R²'   : f"{r2_score(Y_te.iloc[:, idx],            Y_pred[:, idx]):.3f}"
    })

print("\nPHASE-LEVEL PERFORMANCE")
print(pd.DataFrame(results).to_string(index=False))

# ────────────────────────────────────────────────────────────────
# 9. SAVE MODEL
# ────────────────────────────────────────────────────────────────
joblib.dump(best, "pnoc_multiphase_predictor.pkl")
print("\n✅ Model saved to  pnoc_multiphase_predictor.pkl")




