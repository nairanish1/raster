import pandas as pd
import numpy as np
import re
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import MultiLabelBinarizer

# ---------- CONFIG ----------
EXPECTED_ACTUAL_CSV = r"C:\Users\anish.nair\Downloads\Expected_and_Actual_Milestones.csv"
PNOC_FEATURES_CSV  = r"C:\Users\anish.nair\Downloads\Merged_PNOC_Data.csv"
OUTPUT_DIR = Path("feature_importances")  # output directory

# Allowed feature sets
NUMERIC_IMPUTE_COLS = [
    'CI', 'RM', 'Total CI Closed', 'Avg Days', 'Total CI Late',
    'Total Critical CI Closed', 'Total Critical CI Late',
    'Total RM Closed', 'Total RM Late', 'Avg Days to close a RM'
]
CATEGORICAL_SIMPLE = ['critical', 'Group']
MULTI_LABEL = ['Requestor(s)']
CENTROID_COLS = ['Predicted 120 Centroid', 'Predicted Purpose Centroid']

# Milestone order to derive six diff targets
CANONICAL_MILESTONES = [
    "PNOC/CI Issued", "R&C Due", "Enter TA", "Branch TA",
    "Staff TA/Au Due", "NOC Issued", "Revision Issued"
]

# ---------- UTILITIES ----------
def safe_filename(s: str) -> str:
    return re.sub(r'[\\/:"*?<>| ]+', '_', s)

# ---------- TARGET PREPROCESSING ----------
def load_expected_actual_diffs(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, low_memory=False)
    required = {'PNOC ID', 'Process', 'Expected', 'Actual'}
    if not required.issubset(df.columns):
        missing = required - set(df.columns)
        raise ValueError(f"Expected/Actual file missing columns: {missing}")

    df['PNOC ID'] = df['PNOC ID'].astype(str).str.strip()
    df['Process'] = df['Process'].astype(str).str.replace(r"\s*/\s*", "/", regex=True).str.strip()
    df['Process'] = df['Process'].replace({r"PNOC/?\s*CI Issued": "PNOC/CI Issued"}, regex=True)
    df = df[df['Process'].isin(CANONICAL_MILESTONES)].copy()

    df['Actual'] = pd.to_datetime(df['Actual'], errors='coerce')
    df['Expected'] = pd.to_datetime(df['Expected'], errors='coerce')
    df['variance'] = (df['Actual'] - df['Expected']).dt.total_seconds() / (60 * 60 * 24)

    df['Process'] = pd.Categorical(df['Process'], categories=CANONICAL_MILESTONES, ordered=True)
    df = df.sort_values(['PNOC ID', 'Process'])

    var_wide = (
        df.groupby(['PNOC ID', 'Process'], observed=True)['variance']
          .first()
          .unstack('Process')
          .reindex(columns=CANONICAL_MILESTONES)
    )

    # adjacent EA differences
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
    return ea_diff_df  # contains PNOC ID + six EA_diff_*

# ---------- FEATURE PREPROCESSING (restricted + centroid) ----------
def load_restricted_features(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, low_memory=False)
    if 'PNOC ID' not in df.columns:
        raise ValueError("Features file missing 'PNOC ID' column")
    df['PNOC ID'] = df['PNOC ID'].astype(str).str.strip()

    # Build minimal feature set (include centroids)
    required_columns = ['PNOC ID'] + NUMERIC_IMPUTE_COLS + CATEGORICAL_SIMPLE + MULTI_LABEL + CENTROID_COLS
    # Ensure presence
    for col in required_columns:
        if col not in df.columns:
            if col in NUMERIC_IMPUTE_COLS:
                df[col] = np.nan
            else:
                df[col] = 'Unknown'
    df = df[required_columns].copy()

    # Numeric imputation
    for col in NUMERIC_IMPUTE_COLS:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    num_imputer = SimpleImputer(strategy='median')
    df[NUMERIC_IMPUTE_COLS] = num_imputer.fit_transform(df[NUMERIC_IMPUTE_COLS])

    # Fill simple categoricals
    for col in CATEGORICAL_SIMPLE:
        df[col] = df[col].fillna('Unknown').astype(str)

    # One-hot encode simple categoricals
    if 'critical' in df.columns:
        crit_d = pd.get_dummies(df['critical'].astype(str), prefix='critical')
        df = pd.concat([df, crit_d], axis=1)
    if 'Group' in df.columns:
        grp_d = pd.get_dummies(df['Group'].astype(str), prefix='Group')
        df = pd.concat([df, grp_d], axis=1)

    # One-hot encode centroid columns
    for cc in CENTROID_COLS:
        if cc in df.columns:
            dummies = pd.get_dummies(df[cc].astype(str), prefix=cc.replace(" ", "_"))
            df = pd.concat([df, dummies], axis=1)

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

    # Drop original unencoded columns
    drop_cols = ['critical', 'Group', 'Requestor(s)'] + CENTROID_COLS
    df = df.drop(columns=[c for c in drop_cols if c in df.columns], errors='ignore')

    # Deduplicate if needed: numeric mean, others first
    if df['PNOC ID'].duplicated().any():
        numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
        non_numeric = [c for c in df.columns if c not in numeric_cols and c != 'PNOC ID']
        agg_map = {c: 'mean' for c in numeric_cols}
        agg_map.update({c: 'first' for c in non_numeric})
        df = df.groupby('PNOC ID', as_index=False).agg(agg_map)

    return df

# ---------- MERGE AND TRAIN ----------
def merge_features_targets(features_df: pd.DataFrame, target_df: pd.DataFrame) -> pd.DataFrame:
    return target_df.merge(features_df, on='PNOC ID', how='inner')

def train_and_rank(merged: pd.DataFrame, random_state=42):
    target_cols = [c for c in merged.columns if c.startswith("EA_diff_")]
    X_base = merged.drop(columns=['PNOC ID'] + target_cols)

    # One-hot any leftover object columns to avoid string-to-float errors
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
            print(f"[WARN] skipping {target}; no valid target values.")
            continue

        # 5-fold CV on training portion
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=random_state
        )
        rf_cv = RandomForestRegressor(n_estimators=200, random_state=random_state, n_jobs=-1)
        cv_scores = cross_val_score(rf_cv, X_train, y_train, cv=5, scoring='r2', n_jobs=-1)
        cv_mean, cv_std = cv_scores.mean(), cv_scores.std()

        # Train final model
        model = RandomForestRegressor(n_estimators=200, random_state=random_state, n_jobs=-1)
        model.fit(X_train, y_train)
        r2_train = model.score(X_train, y_train)
        r2_test = model.score(X_test, y_test)

        # Feature importances
        imp_series = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)
        perm = permutation_importance(model, X_test, y_test, n_repeats=10, random_state=random_state, n_jobs=-1)
        perm_series = pd.Series(perm.importances_mean, index=X.columns).sort_values(ascending=False)

        df_imp = pd.DataFrame({
            'feature': imp_series.index,
            'impurity_importance': imp_series.values,
            'permutation_importance': perm_series.reindex(imp_series.index).fillna(0).values
        })

        # Human-readable milestone pair label
        pair_raw = target.replace("EA_diff_", "")
        if "_to_" in pair_raw:
            from_m, to_m = pair_raw.split("_to_")
            pair_display = f"{from_m} → {to_m}"
        else:
            pair_display = target
        df_imp.insert(0, 'Milestone Pair', pair_display)

        # Normalize importance scores (optional)
        df_imp['impurity_importance_norm'] = df_imp['impurity_importance'] / df_imp['impurity_importance'].sum()
        df_imp['permutation_importance_norm'] = df_imp['permutation_importance'] / df_imp['permutation_importance'].sum()

        results[target] = {
            'model': model,
            'importances': df_imp.sort_values('permutation_importance', ascending=False),
            'r2_train': r2_train,
            'r2_test': r2_test,
            'cv_r2_mean': cv_mean,
            'cv_r2_std': cv_std,
            'X_train_shape': X_train.shape,
            'X_test_shape': X_test.shape
        }
    return results

# ---------- ENTRYPOINT ----------
def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load data
    ea_diff_df = load_expected_actual_diffs(EXPECTED_ACTUAL_CSV)
    df_features = load_restricted_features(PNOC_FEATURES_CSV)

    merged = merge_features_targets(df_features, ea_diff_df)
    if merged.empty:
        raise RuntimeError("Merged dataset empty: check PNOC ID alignment.")

    # Train and rank
    results = train_and_rank(merged)

    # Report + save
    for target, info in results.items():
        print(f"\n=== Target: {target} ===")
        print(f"Train R²: {info['r2_train']:.3f}, Test R²: {info['r2_test']:.3f}")
        print(f"CV R² mean±std: {info['cv_r2_mean']:.3f} ± {info['cv_r2_std']:.3f}")
        print(f"Shapes: train {info['X_train_shape']}, test {info['X_test_shape']}")
        print("Top features by permutation importance:")
        print(info['importances'][['feature', 'permutation_importance']].head(10))

        # Save with sanitized file name reflecting milestone pair
        pair_raw = target.replace("EA_diff_", "")
        if "_to_" in pair_raw:
            from_m, to_m = pair_raw.split("_to_")
            fname = f"feature_importances_{from_m}_to_{to_m}.csv"
        else:
            fname = f"feature_importances_{pair_raw}.csv"
        fname = safe_filename(fname)
        out_path = OUTPUT_DIR / fname
        info['importances'].to_csv(out_path, index=False)
        print(f"Saved CSV for {target} to {out_path}")

if __name__ == "__main__":
    main()

