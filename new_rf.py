import pandas as pd
import numpy as np
import re
from pathlib import Path
from collections import defaultdict, Counter

from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.model_selection import (
    train_test_split,
    cross_val_score,
    RepeatedKFold,
    KFold,
    LeaveOneOut,
    RandomizedSearchCV,
)
from sklearn.preprocessing import MultiLabelBinarizer

# ---------- CONFIG ----------
EXPECTED_ACTUAL_CSV = r"C:\Users\anish.nair\Downloads\Expected_and_Actual_Milestones.csv"
PNOC_FEATURES_CSV = r"C:\Users\anish.nair\Downloads\Merged_PNOC_Data.csv"
OUTPUT_DIR = Path("feature_importances_nested_rf")  # directory to dump results

CANONICAL_MILESTONES = [
    "PNOC/CI Issued",
    "R&C Due",
    "Enter TA",
    "Branch TA",
    "Staff TA/Au Due",
    "NOC Issued",
    "Revision Issued",
]

# Hyperparameter search space
PARAM_DIST = {
    "n_estimators": [100, 200, 500],
    "max_depth": [None, 5, 10, 20],
    "min_samples_leaf": [1, 2, 4],
    "max_features": ["sqrt", "log2", 0.5],
}

# ---------- UTILITIES ----------
def safe_filename(s: str) -> str:
    return re.sub(r"[\\/:\*\?\"<>\| ]+", "_", s)

def choose_outer_cv(n: int):
    if n >= 100:
        return RepeatedKFold(n_splits=5, n_repeats=3, random_state=42)
    if n >= 50:
        return RepeatedKFold(n_splits=5, n_repeats=2, random_state=42)
    if n >= 20:
        return RepeatedKFold(n_splits=4, n_repeats=2, random_state=42)
    return LeaveOneOut()

def choose_inner_cv(n: int):
    if n >= 50:
        return KFold(n_splits=4, shuffle=True, random_state=1)
    if n >= 20:
        return KFold(n_splits=3, shuffle=True, random_state=1)
    return LeaveOneOut()

# ---------- DATA LOADING & PREPROCESSING ----------

def load_expected_actual_diffs(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, low_memory=False)
    required = {"PNOC ID", "Process", "Expected", "Actual"}
    if not required.issubset(df.columns):
        raise ValueError(f"Expected/Actual file missing columns: {required - set(df.columns)}")

    df["PNOC ID"] = df["PNOC ID"].astype(str).str.strip()
    df["Process"] = (
        df["Process"].astype(str)
        .str.replace(r"\s*/\s*", "/", regex=True)
        .str.strip()
    )
    df["Process"] = df["Process"].replace({r"PNOC/?\s*CI Issued": "PNOC/CI Issued"}, regex=True)
    df = df[df["Process"].isin(CANONICAL_MILESTONES)].copy()

    df["Actual"] = pd.to_datetime(df["Actual"], errors="coerce")
    df["Expected"] = pd.to_datetime(df["Expected"], errors="coerce")
    df["variance"] = (df["Actual"] - df["Expected"]).dt.total_seconds() / (60 * 60 * 24)

    df["Process"] = pd.Categorical(df["Process"], categories=CANONICAL_MILESTONES, ordered=True)
    df = df.sort_values(["PNOC ID", "Process"])

    var_wide = (
        df.groupby(["PNOC ID", "Process"], observed=True)["variance"]
        .first()
        .unstack("Process")
        .reindex(columns=CANONICAL_MILESTONES)
    )

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
    ea_diff_df.index.name = "PNOC ID"
    ea_diff_df = ea_diff_df.reset_index()
    return ea_diff_df  # PNOC ID + 6 targets

def load_all_features(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, low_memory=False)
    if "PNOC ID" not in df.columns:
        raise ValueError("Features file missing 'PNOC ID' column")
    df["PNOC ID"] = df["PNOC ID"].astype(str).str.strip()

    # Handle Requestor(s) multi-label
    if "Requestor(s)" in df.columns:
        def split_requestors(x):
            if pd.isna(x):
                return []
            parts = re.split(r"[;,]", str(x))
            return [p.strip() for p in parts if p.strip()]
        req_lists = df["Requestor(s)"].apply(split_requestors)
        mlb = MultiLabelBinarizer(sparse_output=False)
        if req_lists.map(len).sum() > 0:
            req_encoded = pd.DataFrame(
                mlb.fit_transform(req_lists),
                columns=[f"Requestor_{c}" for c in mlb.classes_],
                index=df.index,
            )
            df = pd.concat([df, req_encoded], axis=1)
        df = df.drop(columns=["Requestor(s)"], errors="ignore")

    # Coerce object columns that are mostly numeric
    numeric_candidates = []
    for col in df.columns:
        if col == "PNOC ID":
            continue
        if df[col].dtype == object:
            coerced = pd.to_numeric(df[col], errors="coerce")
            if coerced.notna().sum() / max(1, len(coerced)) >= 0.9:
                df[col] = coerced
                numeric_candidates.append(col)

    # Identify numeric columns
    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()

    # Impute numeric columns (median)
    if numeric_cols:
        num_imputer = SimpleImputer(strategy="median")
        df[numeric_cols] = num_imputer.fit_transform(df[numeric_cols])

    # Fill and one-hot encode remaining object/categorical (excluding PNOC ID)
    obj_cols = [c for c in df.select_dtypes(include=["object", "category"]).columns if c != "PNOC ID"]
    for c in obj_cols:
        df[c] = df[c].fillna("Unknown").astype(str)
    if obj_cols:
        df = pd.get_dummies(df, columns=obj_cols, drop_first=True)

    # Deduplicate if necessary
    if df["PNOC ID"].duplicated().any():
        numeric_now = df.select_dtypes(include=["number"]).columns.tolist()
        non_numeric = [c for c in df.columns if c not in numeric_now and c != "PNOC ID"]
        agg_map = {c: "mean" for c in numeric_now}
        agg_map.update({c: "first" for c in non_numeric})
        df = df.groupby("PNOC ID", as_index=False).agg(agg_map)

    return df  # fully preprocessed features

# ---------- MODEL / TRAINING ----------

def train_target_nested_rf(X: pd.DataFrame, y: pd.Series):
    """
    Nested CV + hyperparameter tuning + importance aggregation for a single target.
    Returns dict with aggregated metrics and a final model trained on full data.
    """
    # Drop any rows with NaN target
    mask = ~y.isna()
    X = X.loc[mask]
    y = y.loc[mask].astype(float)
    n = len(y)
    if n == 0:
        return None  # nothing to train

    outer_cv = choose_outer_cv(n)
    inner_cv = choose_inner_cv(n)

    outer_r2s = []
    impurity_accum = defaultdict(list)
    perm_accum = defaultdict(list)
    best_params_list = []

    for train_idx, test_idx in outer_cv.split(X, y):
        X_tr, X_te = X.iloc[train_idx], X.iloc[test_idx]
        y_tr, y_te = y.iloc[train_idx], y.iloc[test_idx]

        # Inner hyperparameter search
        base_rf = RandomForestRegressor(random_state=42)
        search = RandomizedSearchCV(
            base_rf,
            param_distributions=PARAM_DIST,
            n_iter=25,
            scoring="r2",
            cv=inner_cv,
            n_jobs=-1,
            random_state=42,
            verbose=0,
        )
        search.fit(X_tr, y_tr)
        best = search.best_estimator_
        best_params_list.append(tuple(sorted(search.best_params_.items())))

        # Evaluate outer fold
        r2 = best.score(X_te, y_te)
        outer_r2s.append(r2)

        # Impurity importance from this best estimator
        if hasattr(best, "feature_importances_"):
            for feat, imp in zip(X_tr.columns, best.feature_importances_):
                impurity_accum[feat].append(imp)

        # Permutation importance on outer test fold
        perm = permutation_importance(best, X_te, y_te, n_repeats=10, random_state=0, n_jobs=-1)
        for feat, imp_mean in zip(X_te.columns, perm.importances_mean):
            perm_accum[feat].append(imp_mean)

    # Aggregate metrics
    nested_cv_r2_mean = np.mean(outer_r2s)
    nested_cv_r2_std = np.std(outer_r2s)

    # Build feature importance summary from outer folds
    features = sorted(set(list(impurity_accum.keys()) + list(perm_accum.keys())))
    summary_rows = []
    for feat in features:
        impurity_list = impurity_accum.get(feat, [])
        perm_list = perm_accum.get(feat, [])
        summary_rows.append({
            "feature": feat,
            "avg_impurity_importance": np.mean(impurity_list) if impurity_list else 0.0,
            "std_impurity_importance": np.std(impurity_list) if impurity_list else 0.0,
            "avg_permutation_importance": np.mean(perm_list) if perm_list else 0.0,
            "std_permutation_importance": np.std(perm_list) if perm_list else 0.0,
        })
    summary_df = pd.DataFrame(summary_rows).sort_values("avg_permutation_importance", ascending=False)

    # Choose a final hyperparameter set: most common in outer best_params
    if best_params_list:
        most_common_params = Counter(best_params_list).most_common(1)[0][0]
        final_params = dict(most_common_params)
    else:
        final_params = {}

    # Final model fit on full X,y with hyperparameter tuning (refine on all data)
    final_cv = RepeatedKFold(n_splits=5 if n >= 10 else max(2, n), n_repeats=2, random_state=7) if n >= 10 else LeaveOneOut()
    rf_final_search = RandomizedSearchCV(
        RandomForestRegressor(random_state=42),
        param_distributions=PARAM_DIST,
        n_iter=30,
        scoring="r2",
        cv=final_cv,
        n_jobs=-1,
        random_state=1,
        verbose=0,
    )
    rf_final_search.fit(X, y)
    final_model = rf_final_search.best_estimator_
    final_r2_cv_scores = cross_val_score(final_model, X, y, cv=final_cv, scoring="r2", n_jobs=-1)
    final_cv_r2_mean = np.mean(final_r2_cv_scores)
    final_cv_r2_std = np.std(final_r2_cv_scores)
    # Impurity / permutation importance from final model (full-data)
    final_impurity = {}
    if hasattr(final_model, "feature_importances_"):
        for feat, imp in zip(X.columns, final_model.feature_importances_):
            final_impurity[feat] = imp
    final_perm = {}
    perm_full = permutation_importance(final_model, X, y, n_repeats=15, random_state=2, n_jobs=-1)
    for feat, imp in zip(X.columns, perm_full.importances_mean):
        final_perm[feat] = imp

    # Merge final importances into summary_df
    def get_or_zero(d, k):
        return d[k] if k in d else 0.0

    summary_df["final_impurity_importance"] = summary_df["feature"].apply(lambda f: get_or_zero(final_impurity, f))
    summary_df["final_permutation_importance"] = summary_df["feature"].apply(lambda f: get_or_zero(final_perm, f))

    # Normalize columns for readability
    for col in ["avg_impurity_importance", "avg_permutation_importance",
                "final_impurity_importance", "final_permutation_importance"]:
        total = summary_df[col].sum()
        summary_df[f"{col}_norm"] = summary_df[col] / total if total > 0 else 0.0

    result = {
        "nested_cv_r2_mean": nested_cv_r2_mean,
        "nested_cv_r2_std": nested_cv_r2_std,
        "final_cv_r2_mean": final_cv_r2_mean,
        "final_cv_r2_std": final_cv_r2_std,
        "summary_df": summary_df,
        "final_model": final_model,
        "chosen_params": rf_final_search.best_params_,
    }
    return result

# ---------- MAIN PIPELINE ----------
def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load targets and features
    ea_diff_df = load_expected_actual_diffs(EXPECTED_ACTUAL_CSV)
    df_features = load_all_features(PNOC_FEATURES_CSV)

    # Merge on PNOC ID
    merged = ea_diff_df.merge(df_features, on="PNOC ID", how="inner")
    if merged.empty:
        raise RuntimeError("Merged dataset is empty; check PNOC ID alignment.")

    target_cols = [c for c in merged.columns if c.startswith("EA_diff_")]
    X_base = merged.drop(columns=["PNOC ID"] + target_cols)

    for target in target_cols:
        print(f"\n--- Training target: {target} ---")
        y = merged[target]
        res = train_target_nested_rf(X_base, y)
        if res is None:
            print(f"Skipping {target}: no data.")
            continue

        # Milestone pair display
        raw = target.replace("EA_diff_", "")
        if "_to_" in raw:
            from_m, to_m = raw.split("_to_")
            milestone_pair = f"{from_m} → {to_m}"
        else:
            milestone_pair = target

        print(f"Nested CV R² mean±std: {res['nested_cv_r2_mean']:.3f} ± {res['nested_cv_r2_std']:.3f}")
        print(f"Final model CV R² mean±std: {res['final_cv_r2_mean']:.3f} ± {res['final_cv_r2_std']:.3f}")
        print(f"Chosen final params: {res['chosen_params']}")

        # Top features
        top = res["summary_df"].head(10)[
            [
                "feature",
                "avg_permutation_importance",
                "std_permutation_importance",
                "final_permutation_importance",
                "nested_cv_r2_mean",
            ]
        ] if "nested_cv_r2_mean" in res["summary_df"].columns else res["summary_df"].head(10)
        print("Top features (by aggregated permutation importance):")
        print(res["summary_df"][["feature", "avg_permutation_importance", "std_permutation_importance"]].head(10))

        # Save CSV
        df_out = res["summary_df"].copy()
        df_out.insert(0, "Milestone Pair", milestone_pair)
        # Add summary metrics as header rows or separate file
        metrics = {
            "nested_cv_r2_mean": res["nested_cv_r2_mean"],
            "nested_cv_r2_std": res["nested_cv_r2_std"],
            "final_cv_r2_mean": res["final_cv_r2_mean"],
            "final_cv_r2_std": res["final_cv_r2_std"],
            "chosen_params": res["chosen_params"],
        }
        fname = safe_filename(f"feature_importances_{milestone_pair}.csv")
        out_path = OUTPUT_DIR / fname
        # Prepend a small metadata block at top via writing two files: metrics + table
        df_out.to_csv(out_path, index=False)
        # Also save metrics separately
        meta_path = OUTPUT_DIR / safe_filename(f"metrics_{milestone_pair}.json")
        pd.Series(metrics).to_json(meta_path)
        print(f"Saved feature importance CSV to {out_path}")
        print(f"Saved metrics JSON to {meta_path}")


if __name__ == "__main__":
    main()
