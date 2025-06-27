import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_percentage_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# 1. Load & preprocess
excel_file = r"C:\Users\anish.nair\OneDrive - BAE Systems Inc\Desktop\Pipelines\PNOCs 2024-2025.xlsm"
sheet_name = "Baseline + Actual Dates"

# Read raw data
df_raw = pd.read_excel(excel_file, sheet_name=sheet_name)

# Clean 'Process' column: remove any "(...)" notes
df_raw['Process'] = df_raw['Process'].str.replace(r'\s*\(.*\)', '', regex=True).str.strip()

# Parse dates
df_raw['Baseline_Date'] = pd.to_datetime(df_raw['Baseline_Date'], errors='coerce')
df_raw['Actual_Date']   = pd.to_datetime(df_raw['Actual_Date'],   errors='coerce')

# Pivot so each PNOC_ID is one row, with one column per milestone/date type
milestones = ['PNOC CI/ Issued', 'R&C Due', 'Enter TA', 
              'Branch TA Due', 'Staff TA/Au Due', 
              'NOC Issued', 'Revision Due']

base = df_raw.pivot(index='PNOC_ID', columns='Process', values='Baseline_Date')[milestones]
act  = df_raw.pivot(index='PNOC_ID', columns='Process', values='Actual_Date')[milestones]

# 2. Derive phase durations (business days)
holidays = []  # fill in holidays array if needed
for i in range(len(milestones)-1):
    start, end = milestones[i], milestones[i+1]
    base[f'dur_{start}_to_{end}_BL'] = np.busday_count(
        base[start].values.astype('datetime64[D]'),
        base[end].values.astype('datetime64[D]'),
        holidays=holidays
    )
    act[f'dur_{start}_to_{end}_ACT'] = np.busday_count(
        act[start].values.astype('datetime64[D]'),
        act[end].values.astype('datetime64[D]'),
        holidays=holidays
    )

# Combine into one dataframe
df = pd.concat([base.filter(like='dur_'), act.filter(like='dur_')], axis=1)
df = df.dropna()  # drop PNOCs with missing dates

# 3. Prepare features & target
#    - Use all baseline phase durations as features
feature_cols = [c for c in df.columns if c.endswith('_BL')]
X = df[feature_cols]
y = df[[c for c in df.columns if c.endswith('_ACT')]].sum(axis=1)  # total actual days

# 4. Train/Test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 5. Build & tune model
pipe = Pipeline([
    ('scale', StandardScaler()),
    ('hgb', HistGradientBoostingRegressor(random_state=42))
])
param_grid = {
    'hgb__max_iter': [100, 200],
    'hgb__max_depth': [3, 5, None],
    'hgb__learning_rate': [0.01, 0.1]
}
grid = GridSearchCV(pipe, param_grid, cv=3, scoring='neg_mean_absolute_percentage_error')
grid.fit(X_train, y_train)

# 6. Evaluate
best = grid.best_estimator_
y_pred = best.predict(X_test)
mape = mean_absolute_percentage_error(y_test, y_pred)
r2   = r2_score(y_test, y_pred)

print(f"Best params: {grid.best_params_}")
print(f"Test MAPE: {mape:.2%}, R²: {r2:.3f}")

# 7. Save model for future baseline prediction
import joblib
joblib.dump(best, "pnoc_baseline_predictor.pkl")
