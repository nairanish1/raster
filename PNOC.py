import os
import re
import joblib
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_percentage_error, r2_score, mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# 1. File & sheet
excel_file = r"C:\Users\anish.nair\Downloads\PNOCs 2024-2025.xlsm"
sheet_name = "Baseline + Actual Dates"

# 2. Read & clean
if os.path.exists(excel_file):
    try:
        df_raw = pd.read_excel(excel_file, sheet_name=sheet_name)
        # strip any "(...)" from Process
        df_raw['Process'] = df_raw['Process'] \
            .str.replace(r'\s*\(.*\)', '', regex=True) \
            .str.strip()

        # parse dates
        df_raw['Baseline'] = pd.to_datetime(df_raw['Baseline'], errors='coerce')
        df_raw['Actual']   = pd.to_datetime(df_raw['Actual'],   errors='coerce')

        # 3. Milestones (match these exactly to your cleaned 'Process' values)
        milestones = [
            'PNOC/ CI Issued',
            'R&C Due',
            'Enter TA',
            'Branch TA Due',
            'Staff TA/Au Due',
            'NOC Issued',
            'Revision Issued',
        ]

        # normalize any variants of the Staff TA step
        def standardize_milestone_type(pt):
            if isinstance(pt, str) and 'Staff TA/Au Due' in pt:
                return 'Staff TA/Au Due'
            return pt

        df_raw['Process'] = df_raw['Process'].apply(standardize_milestone_type)

        # 4. Pivot to one row per PNOC
        base = df_raw.pivot(index='PNOC_ID', columns='Process', values='Baseline')[milestones]
        act  = df_raw.pivot(index='PNOC_ID', columns='Process', values='Actual')[milestones]

        # 5. Reset index so PNOC_ID isn’t in the index
        base = base.reset_index(drop=True)
        act  = act.reset_index(drop=True)

        # 6. Handle missing milestone dates via spline interpolation
        for df_date in (base, act):
            for col in milestones:
                if df_date[col].isnull().any():
                    df_date[col] = df_date[col].interpolate(method='spline', order=3)

        # 7. Define holidays
        holidays_2024 = [
            datetime(2024,1,1),
            datetime(2024,1,15),
            datetime(2024,5,27),
            datetime(2024,7,4),
            datetime(2024,9,2),
            datetime(2024,10,14),
            datetime(2024,11,14),
            datetime(2024,11,28),
            datetime(2024,12,25),
        ]
        holidays = np.array(holidays_2024, dtype='datetime64[D]')

        # 8. Compute business-day phase durations
        for i in range(len(milestones)-1):
            start, end = milestones[i], milestones[i+1]
            bl_col = f'dur_{start.replace("/", "_")}_to_{end.replace("/", "_")}_BL'
            ac_col = f'dur_{start.replace("/", "_")}_to_{end.replace("/", "_")}_ACT'

            base[bl_col] = np.busday_count(
                base[start].values.astype('datetime64[D]'),
                base[end].values.astype('datetime64[D]'),
                holidays=holidays
            )
            act[ac_col] = np.busday_count(
                act[start].values.astype('datetime64[D]'),
                act[end].values.astype('datetime64[D]'),
                holidays=holidays
            )

        # 9. Combine & fill any remaining NaNs with medians
        df = pd.concat([base.filter(like='_BL'), act.filter(like='_ACT')], axis=1)
        df = df.apply(lambda col: col.fillna(col.median()), axis=0)

        # 10. Prepare features & target
        feature_cols = [c for c in df.columns if c.endswith('_BL')]
        X = df[feature_cols]
        y = df[[c for c in df.columns if c.endswith('_ACT')]].sum(axis=1)

        # 11. Train/test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        # 12. Pipeline + grid search
        pipe = Pipeline([
            ('scale', StandardScaler()),
            ('hgb', HistGradientBoostingRegressor(random_state=42))
        ])
        param_grid = {
            'hgb__max_iter': [100, 200],
            'hgb__max_depth': [3, 5, None],
            'hgb__learning_rate': [0.01, 0.1]
        }
        grid = GridSearchCV(
            pipe, param_grid,
            cv=3,
            scoring='neg_mean_absolute_percentage_error',
            n_jobs=-1
        )
        grid.fit(X_train, y_train)

        # 13. Evaluate
        best = grid.best_estimator_
        y_pred = best.predict(X_test)
        mape = mean_absolute_percentage_error(y_test, y_pred)
        r2   = r2_score(y_test, y_pred)
        rmse = mean_squared_error(y_test, y_pred, squared=False)

        print(f"Best params: {grid.best_params_}")
        print(f"Test MAPE: {mape:.2%}")
        print(f"Test R²: {r2:.3f}")
        print(f"Test RMSE: {rmse:.2f}")

        # 14. Save model
        joblib.dump(best, "pnoc_baseline_predictor.pkl")
        print("Model saved as pnoc_baseline_predictor.pkl")

    except FileNotFoundError:
        print(f"Error: The file '{excel_file}' was not found.")
    except Exception as e:
        print(f"An error occurred: {e}")
else:
    print(f"Error: The file '{excel_file}' does not exist.")
