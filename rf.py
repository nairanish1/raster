import pandas as pd
import numpy as np
import re
import os
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer

# ---------- CONFIG (adjust these paths if needed) ----------
EXPECTED_ACTUAL_CSV = r"C:\Users\anish.nair\Downloads\Expected_and_Actual_Milestones.csv"
PNOC_FEATURES_CSV  = r"C:\Users\anish.nair\Downloads\Merged_PNOC_Data.csv"
OUTPUT_DIR = Path("feature_importances")  # will be created if missing

NUMERIC_IMPUTE_COLS = [
    'CI', 'RM', 'Total CI Closed', 'Avg Days', 'Total CI Late',
    'Total Critical CI Closed', 'Total Critical CI Late',
    'Total RM Closed', 'Total RM Late', 'Avg Days to close a RM'
]
CENTROID_COLS = ['Predicted 120 Centroid', 'Predicted Purpose Centroid']
CANONICAL_MILESTONES = [
    "PNOC/CI Issued", "R&C Due", "Enter TA", "Branch TA",
    "Staff TA/Au Due", "NOC Issued", "Revision Issued"
]

# ---------- UTILITY FUNCTIONS ----------

def safe_filename(s: str) -> str:
    # Remove or replace characters that are not allowed in filenames
    return re.sub(r'[\\/:"*?<>|]+', '_', s)

def load_and_preprocess_expected_actual(path: str):
    df = pd.read_csv(path, low_memory=False)
    required = {'PNOC ID', 'Process', 'Expected', 'Actual'}
    if not required.issubset(df.columns):
        missing = required - set(df.columns)
        raise ValueError(f"Expected/Actual file missing columns: {missing}")

    # Clean and normalize
    df['PNOC ID'] = df['PNOC ID'].astype(str).str.strip()
    df['Process'] = df['Process'].astype(str).str.replace(r"\s*/\s*", "/", regex=True).str.strip()
    df['Process'] = df['Process'].replace({r"PNOC/?\s*CI Issued": "PNOC/CI Issued"}, regex=True)
    df = df[df['Process'].isin(CANONICAL_MILESTONES)].copy()

    # Parse dates
    df['Actual'] = pd.to_datetime(df['Actual'], errors='coerce')
    df['Expected'] = pd.to_datetime(df['Expected'], errors='coerce')

    # Compute variance (Actual - Expected) in days
    df['variance'] = (df['Actual'] - df['Expected']).dt.total_seconds() / (60 * 60 * 24)

    # Order processes
    df['Process'] = pd.Categorical(df['Process'], categories=CANONICAL_MILESTONES, ordered=True)
    df = df.sort_values(['PNOC ID', 'Process'])

    # Pivot wide: variance per milestone per PNOC; use observed=True to avoid FutureWarning
    var_wide = (
        df.groupby(['PNOC ID', 'Process'], observed=True)['variance']
          .first()
          .unstack('Process')
          .reindex(columns=CANONICAL_MILESTONES)
    )

    # Per-milestone variance frames (if needed separately)
    milestone_variance_dfs = {
        milestone: var_wide[[milestone]].reset_index().rename(columns={milestone: 'variance'})
        for milestone in CANONICAL_MILESTONES
    }

    # Adjacent differences: EA_diff_<prev>_to_<curr>
    diffs = {}
    for i in range(1, len(CANONICAL_MILESTONES)):
        prev = CANONICAL_MILESTONES[i - 1]
        curr = CANONICAL_MILESTONES[i]
        col_name = f"EA_diff_{prev}_to_{curr}"
        if prev in var_wide.columns and curr in var_wide.columns:
            diffs[col_name] = var_wide[curr] - var_wide[prev]
        else:
            diffs[col_name] = pd.Series(np.nan, index=var_wide.index)
    ea_diff_df = pd.DataFrame(diffs)
    ea_diff_df.index.name = 'PNOC ID'
    ea_diff_df = ea_diff_df.reset_index()

    return ea_diff_df, milestone_variance_dfs

def load_and_preprocess_features(path: str):
    df = pd.read_csv(path, low_memory=False)
    if 'PNOC ID' not in df.columns:
        raise ValueError("Features file missing 'PNOC ID' column")
    df['PNOC ID'] = df['PNOC ID'].astype(str).str.strip()

    # Fill categorical defaults
    for col in ['critical', 'Group']:
        if col in df.columns:
            df[col] = df[col].fillna('Unknown').astype(str)
        else:
            df[col] = 'Unknown'

    # Numeric imputation
    for col in NUMERIC_IMPUTE_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        else:
            df[col] = np.nan
    num_imputer = SimpleImputer(strategy='median')
    df[NUMERIC_IMPUTE_COLS] = num_imputer.fit_transform(df[NUMERIC_IMPUTE_COLS])

    # One-hot encode centroid columns if present
    for cc in CENTROID_COLS:
        if cc in df.columns:
            dummies = pd.get_dummies(df[cc].astype(str), prefix=cc.replace(" ", "_"))
            df = pd.concat([df, dummies], axis=1)

    # One-hot encode 'critical' and 'Group'
    if 'critical' in df.columns:
        df = pd.concat([df, pd.get_dummies(df['critical'].astype(str), prefix='critical')], axis=1)
    if 'Group' in df.columns:
        df = pd.concat([df, pd.get_dummies(df['Group'].astype(str), prefix='Group')], axis=1)

    # Multi-label encode Requestor(s)
    if 'Requestor(s)' in df.columns:
        def split_requestors(x):
            if pd.isna(x):
                return []
            parts = re.split(r'[;,]', str(x))
            return [p.strip() for p in parts if p.strip()]
        req_lists = df['Requestor(s)'].apply(split_requestors)
        mlb = MultiLabelBinarizer(sparse_output=False)
        if req_lists.map(len).sum() > 0:
            req_encoded = pd.DataFrame(
                mlb.fit_transform(req_lists),
                columns=[f"Requestor_{c}" for c in mlb.classes_],
                index=df.index
            )
            df = pd.concat([df, req_encoded], axis=1)

    # Drop original encoded columns
    drop_cols = ['critical', 'Group'] + [c for c in CENTROID_COLS if c in df.columns] + ['Requestor(s)']
    df = df.drop(columns=[c for c in drop_cols if c in df.columns], errors='ignore')

    # Deduplicate: numeric mean, others first
    if df['PNOC ID'].duplicated().any():
        numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
        non_numeric = [c for c in df.columns if c not in numeric_cols and c != 'PNOC ID']
        agg = {c: 'mean' for c in numeric_cols}
        agg.update({c: 'first' for c in non_numeric})
        df = df.groupby('PNOC ID', as_index=False).agg(agg)

    return df

def merge_features_targets(features_df: pd.DataFrame, target_df: pd.DataFrame) -> pd.DataFrame:
    merged = target_df.merge(features_df, on='PNOC ID', how='inner')
    return merged

def train_and_get_importances(merged: pd.DataFrame, random_state=42):
    target_cols = [c for c in merged.columns if c.startswith("EA_diff_")]
    X_base = merged.drop(columns=['PNOC ID'] + target_cols)

    # One-hot any remaining object-type columns to avoid string->float errors
    obj_cols = X_base.select_dtypes(include=['object', 'category']).columns.tolist()
    if obj_cols:
        X_base = pd.get_dummies(X_base, columns=obj_cols, drop_first=True)

    results = {}
    for target in target_cols:
        y = merged[target].astype(float)
        X = X_base.copy()
        mask = ~y.isna()
        X = X.loc[mask]
        y = y.loc[mask]
        if y.empty:
            print(f"[WARN] No usable data for target {target}, skipping.")
            continue
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_state)
        model = RandomForestRegressor(n_estimators=200, random_state=random_state, n_jobs=-1)
        model.fit(X_train, y_train)
        imp_series = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)
        perm = permutation_importance(model, X_test, y_test, n_repeats=10, random_state=random_state, n_jobs=-1)
        perm_series = pd.Series(perm.importances_mean, index=X.columns).sort_values(ascending=False)
        df_imp = pd.DataFrame({
            'impurity_importance': imp_series,
            'permutation_importance': perm_series
        }).fillna(0).sort_values(by='permutation_importance', ascending=False)
        results[target] = {
            'model': model,
            'importances': df_imp,
            'r2_train': model.score(X_train, y_train),
            'r2_test': model.score(X_test, y_test),
            'X_train_shape': X_train.shape,
            'X_test_shape': X_test.shape,
        }
    return results

# ---------- EXECUTION ENTRYPOINT ----------

def main():
    # Ensure output directory exists
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load & preprocess
    ea_diff_df, milestone_variances = load_and_preprocess_expected_actual(EXPECTED_ACTUAL_CSV)
    df_features = load_and_preprocess_features(PNOC_FEATURES_CSV)

    # Merge
    merged = merge_features_targets(df_features, ea_diff_df)
    if merged.empty:
        raise RuntimeError("Merged dataset is empty. Check PNOC ID alignment between features and milestone diffs.")

    # Train models & get importances
    results = train_and_get_importances(merged)

    # Save / report
    for target, info in results.items():
        print(f"\n=== Target: {target} ===")
        print(f"Train R²: {info['r2_train']:.3f}, Test R²: {info['r2_test']:.3f}")
        print(f"Train shape: {info['X_train_shape']}, Test shape: {info['X_test_shape']}")
        print("Top 10 permutation importances:")
        print(info['importances'].head(10))
        # Sanitize filename
        safe_target = safe_filename(target)
        out_path = OUTPUT_DIR / f"feature_importances_{safe_target}.csv"
        info['importances'].to_csv(out_path, index=True)
        print(f"Saved importances to {out_path}")

if __name__ == "__main__":
    main()

