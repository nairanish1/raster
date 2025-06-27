import re, joblib, numpy as np, pandas as pd
from datetime import datetime
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_absolute_percentage_error, r2_score

# ── CONFIG ──────────────────────────────────────────────────────────
EXCEL_FILE = r"C:\Users\anish.nair\Downloads\PNOCs 2024-2025.xlsm"
SHEET_NAME = "Baseline + Actual Dates"

MILESTONES = [
    "PNOC/CI Issued", "R&C Due", "Enter TA", "Branch TA Due",
    "Staff TA/Au Due", "NOC Issued", "Revision Issued",
]

HOLIDAYS_2024 = np.array(
    [datetime(2024,1,1), datetime(2024,1,15), datetime(2024,5,27),
     datetime(2024,7,4), datetime(2024,9,2), datetime(2024,10,14),
     datetime(2024,11,14), datetime(2024,11,28), datetime(2024,12,25)],
    dtype="datetime64[D]"
)

# ── 1. LOAD & IDENTIFY KEY COLUMNS ─────────────────────────────────
df = pd.read_excel(EXCEL_FILE, sheet_name=SHEET_NAME)
df.columns = df.columns.str.strip()

col = {}
for c in df.columns:
    l = c.lower()
    if "process"  in l:                col["proc"]  = c
    elif "baseline" in l:              col["base"]  = c
    elif "actual" in l:                col["act"]   = c
    elif re.search(r"pnoc.*id", l):    col["pnoc"]  = c

# ── 2. CLEAN PROCESS LABELS & PARSE DATES ──────────────────────────
df[col["proc"]] = (
    df[col["proc"]].str.replace(r"\s*\(.*\)", "", regex=True).str.strip()
)
df[col["base"]] = pd.to_datetime(df[col["base"]], errors="coerce")
df[col["act" ]] = pd.to_datetime(df[col["act" ]], errors="coerce")

# ── 3. PIVOT TO WIDE (ONE ROW = ONE PNOC) ──────────────────────────
base = (df.pivot(index=col["pnoc"], columns=col["proc"], values=col["base"])
          [MILESTONES].reset_index(drop=True))
act  = (df.pivot(index=col["pnoc"], columns=col["proc"], values=col["act"])
          [MILESTONES].reset_index(drop=True))

# ── 4. INT64-SPLINE INTERPOLATION FOR ANY MISSING DATES ────────────
for wide in (base, act):
    for m in MILESTONES:
        if wide[m].isna().any():
            ints = wide[m].astype("int64")
            ints = (pd.Series(ints)
                      .interpolate(method="spline", order=3)
                      .astype("int64"))
            wide[m] = pd.to_datetime(ints)

# ── 5. SAFE BUSINESS-DAY COUNTS (no NaT casts) ─────────────────────
def busdays(start: pd.Series, end: pd.Series,
            holidays: np.ndarray) -> np.ndarray:
    result = np.full(len(start), np.nan, dtype=float)
    ok = start.notna() & end.notna()
    if ok.any():
        s = start[ok].values.astype("datetime64[D]")
        e = end  [ok].values.astype("datetime64[D]")
        result[ok] = np.busday_count(s, e, holidays=holidays)
    return result

for i in range(len(MILESTONES)-1):
    a, b = MILESTONES[i], MILESTONES[i+1]
    base[f"{a}_to_{b}_BL"] = busdays(base[a], base[b], HOLIDAYS_2024)
    act [f"{a}_to_{b}_ACT"] = busdays(act [a], act [b], HOLIDAYS_2024)

# ── 6. BUILD FEATURE / TARGET MATRICES ─────────────────────────────
wide = (pd.concat([base.filter(like="_BL"), act.filter(like="_ACT")], axis=1)
          .apply(lambda s: s.fillna(s.median()), axis=0))

X = wide[[c for c in wide.columns if c.endswith("_BL")]]
Y = wide[[c for c in wide.columns if c.endswith("_ACT")]]

# ── 7. TRAIN / TEST & MULTI-OUTPUT REGRESSION ──────────────────────
X_tr, X_te, Y_tr, Y_te = train_test_split(X, Y, test_size=0.2, random_state=42)

grid = GridSearchCV(
    MultiOutputRegressor(HistGradientBoostingRegressor(random_state=42)),
    param_grid={
        "estimator__max_iter":      [100, 200],
        "estimator__max_depth":     [3, 5],
        "estimator__learning_rate": [0.01, 0.1],
    },
    cv=3, n_jobs=-1,
    scoring="neg_mean_absolute_percentage_error",
)
grid.fit(X_tr, Y_tr)

best = grid.best_estimator_
Y_pred = best.predict(X_te)

# ── 8. REPORT PER-PHASE METRICS ────────────────────────────────────
print("\nPHASE-LEVEL PERFORMANCE")
for i, colname in enumerate(Y_te.columns):
    mape = mean_absolute_percentage_error(Y_te.iloc[:, i], Y_pred[:, i])
    r2   = r2_score(Y_te.iloc[:, i],            Y_pred[:, i])
    print(f"{colname:<35}  MAPE: {mape:>6.2%}   R²: {r2:>6.3f}")

# ── 9. SAVE MODEL ─────────────────────────────────────────────────
joblib.dump(best, "pnoc_multiphase_predictor.pkl")
print("\n✅ Model saved to  pnoc_multiphase_predictor.pkl")



