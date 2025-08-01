import pandas as pd
import numpy as np
import re
from pathlib import Path
from collections import defaultdict, Counter

from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.model_selection import (
    RepeatedKFold,
    KFold,
    LeaveOneOut,
    RandomizedSearchCV,
    cross_val_score,
)
from sklearn.preprocessing import MultiLabelBinarizer

# ---------- CONFIG ----------
EXPECTED_ACTUAL_CSV = r"C:\Users\anish.nair\Downloads\Expected_and_Actual_Milestones.csv"
PNOC_FEATURES_CSV = r"C:\Users\anish.nair\Downloads\Merged_PNOC_Data.csv"
OUTPUT_DIR = Path("feature_importances_optimized")  # output directory

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

# expanded hyperparameter search space
PARAM_DIST = {
    'n_estimators': [100, 200, 500, 1000],
    'max_depth': [None, 5, 10, 20, 30],
    'min_samples_leaf': [1, 2, 4, 8],
    'min_samples_split': [2, 5, 10],
    'max_features': ['sqrt', 'log2', 0.3, 0.5, 0.8],
}

# ---------- UTILITIES ----------
def safe_filename(s: str) -> str:
    return re.sub(r'[\\/:"*?<>| ]+', '_', s)

def get_outer_cv(n_samples: int):
    if n_samples >= 150:
        return RepeatedKFold(n_splits=5, n_repeats=4, random_state=42)
    if n_samples >= 100:
        return RepeatedKFold(n_splits=5, n_repeats=3, random_state=42)
    if n_samples >= 50:
        return RepeatedKFold(n_splits=5, n_repeats=2, random_state=42)
    if n_samples >= 20:
        return RepeatedKFold(n_splits=4, n_repeats=2, random_state=42)
    return LeaveOneOut()

def get_inner_cv(n_samples: int):
    if n_samples >= 50:
        return KFold(n_splits=4, shuffle=True, random_state=1)
    if n_samples >= 20:
        return KFold(n_splits=3, shuffle=True, random_state=1)
    return LeaveOneOut()

def summarize_importances(impurity_dict: dict, permutation_dict: dict) -> pd.Series:
    """Combine impurity and permutation importances into normalized vector."""
    all_feats = sorted(set(impurity_dict) | set(permutation_dict))
    combined = {}
    for feat in all_feats:
        imp = impurity_dict.get(feat, 0.0)
        perm = permutation_dict.get(feat, 0.0)
        combined[feat] = (imp + perm) / 2  # simple average
    s = pd.Series(combined)
    if s.sum() > 0:
        s = s / s.sum()
    else:
        # fallback uniform if nothing
        s = pd.Series(1.0 / len(all_feats), index=all_feats)
    return s

# ---------- DATA LOADING / PREPROCESSING ----------
def load_and_preprocess_expected_actual(path: str):
    df = pd.read_csv(path, low_memory=False)
    required = {'PNOC ID', 'Process', 'Expected', 'Actual'}
    if not required.issubset(df.columns):
        raise ValueError(f"Expected/Actual file missing columns: {required - set(df.columns)}")

    df['PNOC ID'] = df['PNOC ID'].astype(str).str.strip()
    df['Process'] = (df['Process'].astype(str)
                     .str.replace(r"\s*/\s*", "/", regex=True)
                     .str.strip())
    df['Process'] = df['Process'].replace({r"PNOC/?\s*CI Issued": "PNOC/CI Issued"}, regex=True)
    df = df[df['Process'].isin(CANONICAL_MILESTONES)].copy()

    df['Actual'] = pd.to_datetime(df['Actual'], errors='coerce')
    df['Expected'] = pd.to_datetime(df['Expected'], errors='coerce')
    df['variance'] = (df['Actual'] - df['Expected']).dt.total_seconds() / (60 * 60 * 24)

    df['Process'] = pd.Categorical(df['Process'], categories=CANONICAL_MILESTONES, ordered=True)
    df = df.sort_values(['PNOC ID', 'Process'])

    var_wide = (df.groupby(['PNOC ID', 'Process'], observed=True)['variance']
                  .first()
                  .unstack('Process')
                  .reindex(columns=CANONICAL_MILESTONES))

    milestone_variance_dfs = {
        milestone: var_wide[[milestone]].reset_index().rename(columns={milestone: 'variance'})
        for milestone in CANONICAL_MILESTONES
    }

    # build EA_diff targets
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

    # drop leakage/date columns
    for leak in ['Baseline', 'Actual', 'Date', 'Need Date']:
        if leak in df.columns:
            df = df.drop(columns=[leak])

    # critical/group fill
    for col in ['critical', 'Group']:
        if col in df.columns:
            df[col] = df[col].fillna('Unknown').astype(str)
        else:
            df[col] = 'Unknown'

    # numeric impute columns
    for col in NUMERIC_IMPUTE_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        else:
            df[col] = np.nan
    num_imputer = SimpleImputer(strategy='median')
    df[NUMERIC_IMPUTE_COLS] = num_imputer.fit_transform(df[NUMERIC_IMPUTE_COLS])

    # centroid one-hot
    for cc in CENTROID_COLS:
        if cc in df.columns:
            dummies = pd.get_dummies(df[cc].astype(str), prefix=cc.replace(" ", "_"))
            df = pd.concat([df, dummies], axis=1)

    # one-hot critical/group
    if 'critical' in df.columns:
        df = pd.concat([df, pd.get_dummies(df['critical'].astype(str), prefix='critical')], axis=1)
    if 'Group' in df.columns:
        df = pd.concat([df, pd.get_dummies(df['Group'].astype(str), prefix='Group')], axis=1)

    # multi-label requestor(s)
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

    # drop originals
    drop_cols = ['critical', 'Group'] + [c for c in CENTROID_COLS if c in df.columns] + ['Requestor(s)']
    df = df.drop(columns=[c for c in drop_cols if c in df.columns], errors='ignore')

    # dedupe if duplicates
    if df['PNOC ID'].duplicated().any():
        numeric = df.select_dtypes(include=["number"]).columns.tolist()
        non_numeric = [c for c in df.columns if c not in numeric and c != 'PNOC ID']
        agg_map = {c: 'mean' for c in numeric}
        agg_map.update({c: 'first' for c in non_numeric})
        df = df.groupby('PNOC ID', as_index=False).agg(agg_map)

    return df

# ---------- MODELING / NESTED CV ----------
def choose_best_target_variant(y: pd.Series):
    lower = y.quantile(0.05)
    upper = y.quantile(0.95)
    clipped = y.clip(lower=lower, upper=upper)
    return {'raw': y, 'clipped': clipped}

def train_single_target_with_nested_cv(X: pd.DataFrame, y: pd.Series, random_state=42):
    mask = ~y.isna()
    X_valid = X.loc[mask].copy()
    y_valid = y.loc[mask].astype(float).copy()
    n = len(y_valid)
    if n == 0:
        return None

    outer_cv = get_outer_cv(n)
    inner_cv = get_inner_cv(n)

    candidate_variants = choose_best_target_variant(y_valid)
    best_overall = None
    for variant_name, y_variant in candidate_variants.items():
        outer_r2s = []
        impurity_accum = defaultdict(list)
        perm_accum = defaultdict(list)
        best_params_list = []

        for train_idx, test_idx in outer_cv.split(X_valid, y_variant):
            X_tr, X_te = X_valid.iloc[train_idx], X_valid.iloc[test_idx]
            y_tr, y_te = y_variant.iloc[train_idx], y_variant.iloc[test_idx]

            base = RandomForestRegressor(random_state=random_state)
            search = RandomizedSearchCV(
                base,
                param_distributions=PARAM_DIST,
                n_iter=50,
                scoring='r2',
                cv=inner_cv,
                n_jobs=-1,
                random_state=random_state,
                verbose=0
            )
            search.fit(X_tr, y_tr)
            best = search.best_estimator_
            best_params_list.append(tuple(sorted(search.best_params_.items())))

            r2_outer = best.score(X_te, y_te)
            outer_r2s.append(r2_outer)

            if hasattr(best, 'feature_importances_'):
                for feat, imp in zip(X_tr.columns, best.feature_importances_):
                    impurity_accum[feat].append(imp)

            perm = permutation_importance(best, X_te, y_te, n_repeats=10, random_state=random_state+1, n_jobs=-1)
            for feat, imp_mean in zip(X_te.columns, perm.importances_mean):
                perm_accum[feat].append(imp_mean)

        nested_cv_r2_mean = np.mean(outer_r2s)
        nested_cv_r2_std = np.std(outer_r2s)

        features = sorted(set(list(impurity_accum.keys()) + list(perm_accum.keys())))
        summary_rows = []
        for feat in features:
            imp_list = impurity_accum.get(feat, [])
            perm_list = perm_accum.get(feat, [])
            summary_rows.append({
                'feature': feat,
                'avg_impurity_importance': np.mean(imp_list) if imp_list else 0.0,
                'std_impurity_importance': np.std(imp_list) if imp_list else 0.0,
                'avg_permutation_importance': np.mean(perm_list) if perm_list else 0.0,
                'std_permutation_importance': np.std(perm_list) if perm_list else 0.0,
            })
        df_agg = pd.DataFrame(summary_rows).sort_values('avg_permutation_importance', ascending=False).reset_index(drop=True)

        final_params = {}
        if best_params_list:
            most_common = Counter(best_params_list).most_common(1)[0][0]
            final_params = dict(most_common)

        final_cv = RepeatedKFold(n_splits=5 if n >= 10 else max(2, n), n_repeats=2, random_state=7) if n >= 10 else LeaveOneOut()
        final_search = RandomizedSearchCV(
            RandomForestRegressor(random_state=random_state),
            param_distributions=PARAM_DIST,
            n_iter=50,
            scoring='r2',
            cv=final_cv,
            n_jobs=-1,
            random_state=random_state + 2,
            verbose=0
        )
        final_search.fit(X_valid, y_variant)
        final_model = final_search.best_estimator_

        final_cv_scores = cross_val_score(final_model, X_valid, y_variant, cv=final_cv, scoring='r2', n_jobs=-1)
        final_cv_r2_mean = np.mean(final_cv_scores)
        final_cv_r2_std = np.std(final_cv_scores)

        final_impurity = {}
        if hasattr(final_model, 'feature_importances_'):
            for feat, imp in zip(X_valid.columns, final_model.feature_importances_):
                final_impurity[feat] = imp
        final_perm_full = {}
        perm_full = permutation_importance(final_model, X_valid, y_variant, n_repeats=15, random_state=random_state + 3, n_jobs=-1)
        for feat, imp_mean in zip(X_valid.columns, perm_full.importances_mean):
            final_perm_full[feat] = imp_mean

        def safe_get(d, k):
            return d[k] if k in d else 0.0

        df_agg['final_impurity_importance'] = df_agg['feature'].apply(lambda f: safe_get(final_impurity, f))
        df_agg['final_permutation_importance'] = df_agg['feature'].apply(lambda f: safe_get(final_perm_full, f))

        for col in ['avg_impurity_importance', 'avg_permutation_importance',
                    'final_impurity_importance', 'final_permutation_importance']:
            total = df_agg[col].sum()
            norm_col = f'{col}_norm'
            df_agg[norm_col] = df_agg[col] / total if total > 0 else 0.0

        result = {
            'nested_cv_r2_mean': nested_cv_r2_mean,
            'nested_cv_r2_std': nested_cv_r2_std,
            'final_cv_r2_mean': final_cv_r2_mean,
            'final_cv_r2_std': final_cv_r2_std,
            'df_agg': df_agg,
            'final_model': final_model,
            'chosen_params': final_search.best_params_,
            'variant_used': variant_name,
        }

        if best_overall is None or result['nested_cv_r2_mean'] > best_overall['nested_cv_r2_mean']:
            best_overall = result

    return best_overall

def train_and_get_importances(merged: pd.DataFrame, random_state=42):
    target_cols = [c for c in merged.columns if c.startswith("EA_diff_")]
    X_base = merged.drop(columns=['PNOC ID'] + target_cols)

    # one-hot encode any remaining object-like
    obj_cols = X_base.select_dtypes(include=['object', 'category']).columns.tolist()
    if obj_cols:
        X_base = pd.get_dummies(X_base, columns=obj_cols, drop_first=True)

    results = {}
    for target in target_cols:
        y = merged[target]
        n_nonnull = y.notna().sum()
        print(f"\n--- Training target {target} ({n_nonnull} non-null examples) ---")
        best = train_single_target_with_nested_cv(X_base, y, random_state=random_state)
        if best is None:
            print(f"Skipping {target}: no valid data")
            continue

        pair_raw = target.replace("EA_diff_", "")
        if "_to_" in pair_raw:
            from_m, to_m = pair_raw.split("_to_")
            milestone_pair_display = f"{from_m} → {to_m}"
        else:
            milestone_pair_display = target

        df_final = best['df_agg'].copy()
        df_final.insert(0, 'Milestone Pair', milestone_pair_display)
        df_final['nested_cv_r2_mean'] = best['nested_cv_r2_mean']
        df_final['nested_cv_r2_std'] = best['nested_cv_r2_std']
        df_final['final_cv_r2_mean'] = best['final_cv_r2_mean']
        df_final['final_cv_r2_std'] = best['final_cv_r2_std']
        df_final['chosen_params'] = str(best['chosen_params'])
        df_final['variant_used'] = best['variant_used']

        results[target] = {
            'model': best['final_model'],
            'importances': df_final,
            'nested_cv_r2_mean': best['nested_cv_r2_mean'],
            'nested_cv_r2_std': best['nested_cv_r2_std'],
            'final_cv_r2_mean': best['final_cv_r2_mean'],
            'final_cv_r2_std': best['final_cv_r2_std'],
            'chosen_params': best['chosen_params'],
            'variant_used': best['variant_used'],
        }

    return results

# ---------- AUTOREGRESSIVE CASCADE ----------
def train_autoregressive_rf(merged: pd.DataFrame, random_state=42):
    target_cols = []
    milestone_pairs = []
    for i in range(1, len(CANONICAL_MILESTONES)):
        prev = CANONICAL_MILESTONES[i - 1]
        curr = CANONICAL_MILESTONES[i]
        col = f"EA_diff_{prev}_to_{curr}"
        target_cols.append(col)
        milestone_pairs.append((prev, curr))

    # base feature matrix (drop all EA_diff targets)
    X_base = merged.drop(columns=[c for c in merged.columns if c.startswith("EA_diff_")] + ['PNOC ID'])
    obj_cols = X_base.select_dtypes(include=['object', 'category']).columns.tolist()
    if obj_cols:
        X_base = pd.get_dummies(X_base, columns=obj_cols, drop_first=True)
    X_base_orig = X_base.copy()

    cascade_feature_vector = pd.Series(0.0, index=X_base.columns)  # initial empty memory
    previous_pred = pd.Series(0.0, index=X_base.index)  # placeholder for previous target prediction

    ar_results = {}
    for i, (prev, curr) in enumerate(milestone_pairs):
        target_col = f"EA_diff_{prev}_to_{curr}"
        y = merged[target_col]
        mask = ~y.isna()
        if mask.sum() == 0:
            print(f"Skipping autoregressive {target_col}: no data")
            continue

        # Build features for this step
        X = X_base_orig.loc[mask].copy()
        # add cascading importance features (same for all rows)
        for feat, val in cascade_feature_vector.items():
            X[f"AR_featimp_{feat}"] = val
        # add previous-step prediction
        if i > 0:
            X['AR_prev_target'] = previous_pred.reindex(X.index).fillna(0.0)

        # Train with nested CV
        best = train_single_target_with_nested_cv(X, y, random_state=random_state)
        if best is None:
            continue

        # Prepare summary frame
        pair_raw = target_col.replace("EA_diff_", "")
        if "_to_" in pair_raw:
            from_m, to_m = pair_raw.split("_to_")
            milestone_pair_display = f"{from_m} → {to_m}"
        else:
            milestone_pair_display = target_col

        df_final = best['df_agg'].copy()
        df_final.insert(0, 'Milestone Pair', milestone_pair_display)
        df_final['nested_cv_r2_mean'] = best['nested_cv_r2_mean']
        df_final['nested_cv_r2_std'] = best['nested_cv_r2_std']
        df_final['final_cv_r2_mean'] = best['final_cv_r2_mean']
        df_final['final_cv_r2_std'] = best['final_cv_r2_std']
        df_final['chosen_params'] = str(best['chosen_params'])
        df_final['variant_used'] = best['variant_used']

        # Store result
        ar_results[target_col] = {
            'model': best['final_model'],
            'importances': df_final,
            'nested_cv_r2_mean': best['nested_cv_r2_mean'],
            'nested_cv_r2_std': best['nested_cv_r2_std'],
            'final_cv_r2_mean': best['final_cv_r2_mean'],
            'final_cv_r2_std': best['final_cv_r2_std'],
            'chosen_params': best['chosen_params'],
            'variant_used': best['variant_used'],
        }

        # Update cascade memory from this step's final model
        X_valid = X.loc[~y.isna()]  # same mask
        y_valid = y.loc[~y.isna()].astype(float)
        # impurity
        final_model = best['final_model']
        impurity_full = {}
        if hasattr(final_model, 'feature_importances_'):
            for feat, imp in zip(X_valid.columns, final_model.feature_importances_):
                impurity_full[feat] = imp
        # permutation
        perm_full = permutation_importance(final_model, X_valid, y_valid, n_repeats=10,
                                          random_state=random_state + 10, n_jobs=-1)
        perm_full_dict = {feat: perm_full.importances_mean[idx] for idx, feat in enumerate(X_valid.columns)}

        cascade_feature_vector = summarize_importances(impurity_full, perm_full_dict)
        # previous prediction (in-sample for next step)
        previous_pred = pd.Series(final_model.predict(X), index=X.index)

    return ar_results

# ---------- ENTRYPOINT ----------
def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    ea_diff_df, milestone_variances = load_and_preprocess_expected_actual(EXPECTED_ACTUAL_CSV)
    df_features = load_and_preprocess_features(PNOC_FEATURES_CSV)
    merged = ea_diff_df.merge(df_features, on='PNOC ID', how='inner')
    if merged.empty:
        raise RuntimeError("Merged dataset is empty; check PNOC ID alignment.")

    print("===== Baseline independent RF models =====")
    baseline_results = train_and_get_importances(merged, random_state=42)

    print("\n===== Autoregressive cascade RF models =====")
    ar_results = train_autoregressive_rf(merged, random_state=42)

    # Compare and persist
    for target in sorted(set(list(baseline_results.keys()) + list(ar_results.keys()))):
        base = baseline_results.get(target)
        ar = ar_results.get(target)
        print(f"\n=== Target: {target} ===")
        if base:
            print(f"[Baseline] Nested CV R²: {base['nested_cv_r2_mean']:.3f} ± {base['nested_cv_r2_std']:.3f}; "
                  f"Final CV R²: {base['final_cv_r2_mean']:.3f} ± {base['final_cv_r2_std']:.3f}")
        else:
            print("[Baseline] skipped")
        if ar:
            print(f"[AR Cascade] Nested CV R²: {ar['nested_cv_r2_mean']:.3f} ± {ar['nested_cv_r2_std']:.3f}; "
                  f"Final CV R²: {ar['final_cv_r2_mean']:.3f} ± {ar['final_cv_r2_std']:.3f}")
        else:
            print("[AR Cascade] skipped")

        # Save both importances if present
        for prefix, info in (("baseline", base), ("ar", ar)):
            if info is None:
                continue
            df_out = info['importances']
            pair_raw = target.replace("EA_diff_", "")
            if "_to_" in pair_raw:
                from_m, to_m = pair_raw.split("_to_")
                fname = f"{prefix}_feature_importances_{from_m}_to_{to_m}.csv"
            else:
                fname = f"{prefix}_feature_importances_{pair_raw}.csv"
            fname = safe_filename(fname)
            out_path = OUTPUT_DIR / fname
            df_out.to_csv(out_path, index=False)
            print(f"Saved {prefix} importances for {target} to {out_path}")

if __name__ == "__main__":
    main()
