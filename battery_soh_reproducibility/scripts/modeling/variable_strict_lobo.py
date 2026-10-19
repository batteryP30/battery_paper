"""Nested leave-one-battery-out HGB for the revised variable-protocol task."""
from __future__ import annotations

import argparse
import json
import math
import time
import warnings
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error

warnings.filterwarnings("ignore", category=RuntimeWarning)


TARGET_COL = "Target_SOH"

BATTERY_COL = "battery_id"

SOURCE_COL = "source_dataset"

EXPECTED_BATTERIES = ["CS2_3", "CS2_9", "RW3", "RW4", "RW5", "RW6"]

EXPECTED_N = 165

MIN_SOURCE_COVERAGE = 0.95

EXPLICIT_AGE_BASE_FEATURES = {
    "LEGACY__Cycle_Age",
    "LEGACY__Cycle_Index",
    "LEGACY__Log_Cycle_Age",
}

AGE_COLUMNS = ["Age_Coord", "Log_Age_Coord"]

CORR_THRESHOLDS = [0.90, 0.93, 0.95]

TOP_K_OPTIONS = [8, 12, 16]

HGB_CONFIGS = {
    "H0": {
        "learning_rate": 0.05,
        "max_iter": 300,
        "max_leaf_nodes": 15,
        "min_samples_leaf": 5,
        "l2_regularization": 1.0,
    },
    "H1": {
        "learning_rate": 0.04,
        "max_iter": 500,
        "max_leaf_nodes": 15,
        "min_samples_leaf": 10,
        "l2_regularization": 2.0,
    },
    "H2": {
        "learning_rate": 0.03,
        "max_iter": 700,
        "max_leaf_nodes": 31,
        "min_samples_leaf": 10,
        "l2_regularization": 3.0,
    },
}

FEATURE_TUNING_MODEL = "H1"

RANDOM_STATE = 42

PREDICTION_CLIP_LOW = 70.0

PREDICTION_CLIP_HIGH = 100.0

def rmse(y_true, y_pred):
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))

def mae(y_true, y_pred):
    return float(mean_absolute_error(y_true, y_pred))

def equal_battery_weights(frame):
    """
    Give each battery equal total training weight.
    This prevents CS2_9, with many more independent observations,
    from dominating the HGB fit.
    """
    counts = frame[BATTERY_COL].value_counts()
    w = frame[BATTERY_COL].map(lambda b: 1.0 / counts.loc[b]).astype(float).to_numpy()

    # Normalize mean weight to 1 for numerical stability.
    if np.mean(w) > 0:
        w = w / np.mean(w)

    return w

def numeric_frame(frame, columns):
    """
    Return numeric feature values with inf converted to NaN.
    """
    out = frame.loc[:, columns].apply(pd.to_numeric, errors="coerce")
    return out.replace([np.inf, -np.inf], np.nan)

def underlying_base_feature(expanded_name):
    """
    Convert:
        MED__LEGACY__Cycle_Age
    to:
        LEGACY__Cycle_Age
    and likewise for LAST__.
    """
    if expanded_name.startswith("MED__"):
        return expanded_name[len("MED__"):]
    if expanded_name.startswith("LAST__"):
        return expanded_name[len("LAST__"):]
    return expanded_name

def get_expanded_strict_columns(frame):
    """
    Step 3 already froze strict-safe cycle-level features.
    The independent-observation table contains MED__ and LAST__
    summaries of those cycle features.
    """
    return [
        c for c in frame.columns
        if c.startswith("MED__") or c.startswith("LAST__")
    ]

def get_electrical_candidate_columns(frame):
    """
    Primary strict model:
    remove explicit cycle-age/index features.

    This leaves electrical/dynamic/history-derived measurements that were
    classified STRICT_SAFE in Step 1 and carried into Step 3.
    """
    expanded = get_expanded_strict_columns(frame)

    keep = []
    removed_age = []

    for c in expanded:
        base = underlying_base_feature(c)

        if base in EXPLICIT_AGE_BASE_FEATURES:
            removed_age.append(c)
        else:
            keep.append(c)

    return keep, removed_age

def training_source_coverage_filter(train_frame, columns, minimum=0.95):
    """
    Training-only source-aware availability filter.

    A feature is eligible only when:
      1. It reaches minimum finite coverage in EACH source represented
         in the current training split.
      2. It has at least two distinct finite values in the training split.

    No held-out outer or inner test rows are used.
    """
    if len(columns) == 0:
        return [], []

    X = numeric_frame(train_frame, columns)

    keep = []
    audit_rows = []

    grouped_sources = list(train_frame[SOURCE_COL].unique())

    for c in columns:
        per_source = {}

        eligible = True

        for src in grouped_sources:
            idx = train_frame[SOURCE_COL].eq(src)
            values = X.loc[idx, c]
            coverage = float(values.notna().mean()) if idx.sum() else np.nan
            per_source[src] = coverage

            if not np.isfinite(coverage) or coverage < minimum:
                eligible = False

        nunique = int(X[c].nunique(dropna=True))

        if nunique <= 1:
            eligible = False

        audit_rows.append(
            {
                "Feature": c,
                "Eligible": bool(eligible),
                "Unique_Train_Values": nunique,
                **{
                    f"Coverage_{src}": per_source.get(src, np.nan)
                    for src in grouped_sources
                },
            }
        )

        if eligible:
            keep.append(c)

    return keep, audit_rows

def prepare_feature_selector(train_frame, candidate_columns):
    """
    Build all quantities needed for training-only feature ranking
    and correlation pruning.

    Steps:
      1. Training-only source coverage filter.
      2. Training-only median imputation.
      3. Absolute Spearman association with training target.
      4. Pairwise absolute Spearman feature correlation.

    The returned structure can be reused for multiple K/correlation
    choices on the SAME training split.
    """
    eligible, coverage_audit = training_source_coverage_filter(
        train_frame,
        candidate_columns,
        minimum=MIN_SOURCE_COVERAGE,
    )

    if len(eligible) == 0:
        raise RuntimeError("No eligible features remained after training-only coverage filtering.")

    X_raw = numeric_frame(train_frame, eligible)

    imputer = SimpleImputer(strategy="median")
    X_imp = imputer.fit_transform(X_raw)

    X = pd.DataFrame(
        X_imp,
        columns=eligible,
        index=train_frame.index,
    )

    # Spearman correlation is Pearson correlation of ranks.
    ranked_X = X.rank(method="average")
    ranked_y = pd.Series(
        train_frame[TARGET_COL].to_numpy(),
        index=train_frame.index,
    ).rank(method="average")

    target_scores = (
        ranked_X.corrwith(ranked_y)
        .abs()
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0.0)
    )

    ranked_order = (
        target_scores
        .sort_values(ascending=False, kind="mergesort")
        .index
        .tolist()
    )

    # Correlation matrix in target-score ranking order.
    feature_corr = ranked_X.loc[:, ranked_order].corr().abs()

    return {
        "eligible_features": eligible,
        "coverage_audit": coverage_audit,
        "ranked_order": ranked_order,
        "target_scores": target_scores,
        "feature_corr": feature_corr,
    }

def choose_features(prepared, top_k, corr_threshold):
    """
    Greedy correlation pruning in descending training-target association order.

    A feature is kept only if it is not too highly correlated with a
    previously retained feature.
    """
    order = prepared["ranked_order"]
    corr = prepared["feature_corr"]

    selected = []

    for feature in order:
        if len(selected) >= int(top_k):
            break

        if not selected:
            selected.append(feature)
            continue

        correlations = corr.loc[feature, selected]

        if bool((correlations >= float(corr_threshold)).any()):
            continue

        selected.append(feature)

    if len(selected) == 0:
        raise RuntimeError("Feature selection returned zero features.")

    return selected

def build_model_features(selected_electrical, include_explicit_age):
    features = list(selected_electrical)

    if include_explicit_age:
        for c in AGE_COLUMNS:
            if c not in features:
                features.append(c)

    return features

def fit_hgb(train_frame, selected_electrical, config_name, include_explicit_age):
    """
    Fit imputer and HGB using only outer/inner training rows.
    """
    model_features = build_model_features(
        selected_electrical,
        include_explicit_age,
    )

    for c in model_features:
        if c not in train_frame.columns:
            raise KeyError(f"Required model column missing: {c}")

    X_train_raw = numeric_frame(train_frame, model_features)

    imputer = SimpleImputer(strategy="median")
    X_train = imputer.fit_transform(X_train_raw)

    config = HGB_CONFIGS[config_name]

    model = HistGradientBoostingRegressor(
        loss="squared_error",
        early_stopping=False,
        random_state=RANDOM_STATE,
        **config,
    )

    model.fit(
        X_train,
        train_frame[TARGET_COL].to_numpy(),
        sample_weight=equal_battery_weights(train_frame),
    )

    return model, imputer, model_features

def predict_hgb(model, imputer, model_features, test_frame):
    X_test_raw = numeric_frame(test_frame, model_features)
    X_test = imputer.transform(X_test_raw)

    raw = model.predict(X_test)

    clipped = np.clip(
        raw,
        PREDICTION_CLIP_LOW,
        PREDICTION_CLIP_HIGH,
    )

    return raw, clipped

def prepare_inner_splits(outer_train, candidate_columns):
    """
    Inner LOBO preparation.

    The outer training set contains five batteries.
    Each inner split holds out one of those batteries.
    """
    batteries = sorted(outer_train[BATTERY_COL].unique())

    prepared = {}

    for heldout in batteries:
        inner_train = outer_train.loc[
            outer_train[BATTERY_COL] != heldout
        ].copy()

        inner_test = outer_train.loc[
            outer_train[BATTERY_COL] == heldout
        ].copy()

        selector = prepare_feature_selector(
            inner_train,
            candidate_columns,
        )

        prepared[heldout] = {
            "train": inner_train,
            "test": inner_test,
            "selector": selector,
        }

    return prepared

def tune_feature_selection(
    outer_train,
    candidate_columns,
    include_explicit_age,
):
    """
    Stage A of nested tuning.

    Tune correlation threshold and feature count using inner LOBO
    with one fixed HGB configuration.

    Selection and imputation are rebuilt inside every inner-training split.
    """
    inner = prepare_inner_splits(
        outer_train,
        candidate_columns,
    )

    rows = []

    for corr_threshold in CORR_THRESHOLDS:
        for top_k in TOP_K_OPTIONS:

            fold_rmses = []
            fold_maes = []

            for heldout, split in inner.items():
                selected = choose_features(
                    split["selector"],
                    top_k=top_k,
                    corr_threshold=corr_threshold,
                )

                model, imputer, model_features = fit_hgb(
                    split["train"],
                    selected,
                    FEATURE_TUNING_MODEL,
                    include_explicit_age,
                )

                pred_raw, _ = predict_hgb(
                    model,
                    imputer,
                    model_features,
                    split["test"],
                )

                fold_rmses.append(
                    rmse(
                        split["test"][TARGET_COL],
                        pred_raw,
                    )
                )
                fold_maes.append(
                    mae(
                        split["test"][TARGET_COL],
                        pred_raw,
                    )
                )

            rows.append(
                {
                    "Corr_Threshold": corr_threshold,
                    "Top_K": top_k,
                    "Inner_Macro_RMSE": float(np.mean(fold_rmses)),
                    "Inner_Macro_MAE": float(np.mean(fold_maes)),
                    "Feature_Tuning_Model": FEATURE_TUNING_MODEL,
                }
            )

    tuning = pd.DataFrame(rows)

    # Prefer better RMSE, then fewer features, then stricter pruning
    # if scores are tied.
    tuning = tuning.sort_values(
        ["Inner_Macro_RMSE", "Top_K", "Corr_Threshold"],
        ascending=[True, True, True],
        kind="mergesort",
    ).reset_index(drop=True)

    best = tuning.iloc[0]

    return (
        float(best["Corr_Threshold"]),
        int(best["Top_K"]),
        tuning,
        inner,
    )

def tune_hgb_config(
    inner_prepared,
    corr_threshold,
    top_k,
    include_explicit_age,
):
    """
    Stage B of nested tuning.

    With the feature-selection settings fixed from Stage A,
    choose the finite HGB candidate using inner LOBO.
    """
    rows = []

    for config_name in HGB_CONFIGS:

        fold_rmses = []
        fold_maes = []

        for heldout, split in inner_prepared.items():
            selected = choose_features(
                split["selector"],
                top_k=top_k,
                corr_threshold=corr_threshold,
            )

            model, imputer, model_features = fit_hgb(
                split["train"],
                selected,
                config_name,
                include_explicit_age,
            )

            pred_raw, _ = predict_hgb(
                model,
                imputer,
                model_features,
                split["test"],
            )

            fold_rmses.append(
                rmse(
                    split["test"][TARGET_COL],
                    pred_raw,
                )
            )
            fold_maes.append(
                mae(
                    split["test"][TARGET_COL],
                    pred_raw,
                )
            )

        rows.append(
            {
                "HGB_Config": config_name,
                "Inner_Macro_RMSE": float(np.mean(fold_rmses)),
                "Inner_Macro_MAE": float(np.mean(fold_maes)),
            }
        )

    tuning = pd.DataFrame(rows).sort_values(
        ["Inner_Macro_RMSE", "HGB_Config"],
        ascending=[True, True],
        kind="mergesort",
    ).reset_index(drop=True)

    return str(tuning.iloc[0]["HGB_Config"]), tuning

def run_formulation(
    frame,
    formulation_name,
    candidate_columns,
    include_explicit_age,
):
    """
    Full nested LOBO for one formulation.
    """
    prediction_rows = []
    fold_rows = []
    selected_rows = []
    coverage_rows = []
    feature_tuning_rows = []
    model_tuning_rows = []

    outer_batteries = sorted(frame[BATTERY_COL].unique())

    print("\n" + "=" * 80)
    print("FORMULATION:", formulation_name)
    print("=" * 80)

    for fold_number, heldout in enumerate(outer_batteries, start=1):

        fold_start = time.time()

        outer_train = frame.loc[
            frame[BATTERY_COL] != heldout
        ].copy()

        outer_test = frame.loc[
            frame[BATTERY_COL] == heldout
        ].copy()

        print(
            f"\nOuter fold {fold_number}/{len(outer_batteries)} "
            f"- holdout {heldout}"
        )

        # ----------------------------------------------------
        # Nested Stage A: feature-selection settings
        # ----------------------------------------------------

        best_corr, best_k, feature_tuning, inner_prepared = (
            tune_feature_selection(
                outer_train,
                candidate_columns,
                include_explicit_age,
            )
        )

        feature_tuning = feature_tuning.copy()
        feature_tuning.insert(0, "Formulation", formulation_name)
        feature_tuning.insert(1, "Outer_Holdout", heldout)
        feature_tuning_rows.append(feature_tuning)

        # ----------------------------------------------------
        # Nested Stage B: HGB configuration
        # ----------------------------------------------------

        best_config, model_tuning = tune_hgb_config(
            inner_prepared,
            best_corr,
            best_k,
            include_explicit_age,
        )

        model_tuning = model_tuning.copy()
        model_tuning.insert(0, "Formulation", formulation_name)
        model_tuning.insert(1, "Outer_Holdout", heldout)
        model_tuning["Chosen_Corr_Threshold"] = best_corr
        model_tuning["Chosen_Top_K"] = best_k
        model_tuning_rows.append(model_tuning)

        # ----------------------------------------------------
        # Rebuild feature selector on FULL outer training data
        # ----------------------------------------------------

        selector = prepare_feature_selector(
            outer_train,
            candidate_columns,
        )

        selected = choose_features(
            selector,
            top_k=best_k,
            corr_threshold=best_corr,
        )

        # Store source-coverage audit for this outer fold.
        for row in selector["coverage_audit"]:
            coverage_rows.append(
                {
                    "Formulation": formulation_name,
                    "Outer_Holdout": heldout,
                    **row,
                }
            )

        for rank, feature in enumerate(selected, start=1):
            selected_rows.append(
                {
                    "Formulation": formulation_name,
                    "Outer_Holdout": heldout,
                    "Selected_Rank": rank,
                    "Feature": feature,
                    "Training_Target_SpearmanAbs": float(
                        selector["target_scores"].loc[feature]
                    ),
                    "Corr_Threshold": best_corr,
                    "Top_K": best_k,
                    "HGB_Config": best_config,
                }
            )

        # ----------------------------------------------------
        # Final outer-fold fit
        # ----------------------------------------------------

        model, imputer, model_features = fit_hgb(
            outer_train,
            selected,
            best_config,
            include_explicit_age,
        )

        pred_raw, pred_clipped = predict_hgb(
            model,
            imputer,
            model_features,
            outer_test,
        )

        y_true = outer_test[TARGET_COL].to_numpy()

        fold_rmse_raw = rmse(y_true, pred_raw)
        fold_mae_raw = mae(y_true, pred_raw)

        fold_rmse_clipped = rmse(y_true, pred_clipped)
        fold_mae_clipped = mae(y_true, pred_clipped)

        elapsed = time.time() - fold_start

        fold_rows.append(
            {
                "Formulation": formulation_name,
                "Holdout_Battery": heldout,
                "Source": str(outer_test[SOURCE_COL].iloc[0]),
                "N_Test": len(outer_test),
                "Eligible_Training_Features": len(
                    selector["eligible_features"]
                ),
                "Selected_Electrical_Features": len(selected),
                "Explicit_Age_Features_Added": (
                    len(AGE_COLUMNS) if include_explicit_age else 0
                ),
                "Final_Model_Features": len(model_features),
                "Chosen_Corr_Threshold": best_corr,
                "Chosen_Top_K": best_k,
                "Chosen_HGB_Config": best_config,
                "RMSE_Raw": fold_rmse_raw,
                "MAE_Raw": fold_mae_raw,
                "RMSE_Clipped70_100": fold_rmse_clipped,
                "MAE_Clipped70_100": fold_mae_clipped,
                "Runtime_Seconds": elapsed,
            }
        )

        for (_, row), raw_pred, clipped_pred in zip(
            outer_test.iterrows(),
            pred_raw,
            pred_clipped,
        ):
            prediction_rows.append(
                {
                    "Formulation": formulation_name,
                    "Holdout_Battery": heldout,
                    "battery_id": row[BATTERY_COL],
                    "source_dataset": row[SOURCE_COL],
                    "R5E_Reference_Epoch": row.get(
                        "R5E_Reference_Epoch",
                        np.nan,
                    ),
                    "Age_Coord": row.get("Age_Coord", np.nan),
                    "Target_SOH": row[TARGET_COL],
                    "Prediction_Raw": float(raw_pred),
                    "Prediction_Clipped70_100": float(clipped_pred),
                    "Residual_Raw": float(
                        row[TARGET_COL] - raw_pred
                    ),
                    "Residual_Clipped70_100": float(
                        row[TARGET_COL] - clipped_pred
                    ),
                }
            )

        print(
            f"  eligible features: {len(selector['eligible_features'])}"
        )
        print(
            f"  selected: {len(selected)} "
            f"| corr={best_corr} | K={best_k} | HGB={best_config}"
        )
        print(
            f"  raw RMSE={fold_rmse_raw:.4f} "
            f"| clipped RMSE={fold_rmse_clipped:.4f}"
        )
        print(f"  runtime={elapsed:.1f} s")

    predictions = pd.DataFrame(prediction_rows)
    folds = pd.DataFrame(fold_rows)
    selected_df = pd.DataFrame(selected_rows)
    coverage_df = pd.DataFrame(coverage_rows)
    feature_tuning_df = pd.concat(
        feature_tuning_rows,
        ignore_index=True,
    )
    model_tuning_df = pd.concat(
        model_tuning_rows,
        ignore_index=True,
    )

    macro = pd.DataFrame(
        [
            {
                "Formulation": formulation_name,
                "Batteries": folds["Holdout_Battery"].nunique(),
                "Macro_RMSE_Raw": folds["RMSE_Raw"].mean(),
                "Macro_MAE_Raw": folds["MAE_Raw"].mean(),
                "Macro_RMSE_Clipped70_100": folds[
                    "RMSE_Clipped70_100"
                ].mean(),
                "Macro_MAE_Clipped70_100": folds[
                    "MAE_Clipped70_100"
                ].mean(),
            }
        ]
    )

    # Selection frequency across the six outer folds.
    selection_frequency = (
        selected_df.groupby("Feature")
        .agg(
            Outer_Folds_Selected=("Outer_Holdout", "nunique"),
            Mean_Training_Target_SpearmanAbs=(
                "Training_Target_SpearmanAbs",
                "mean",
            ),
        )
        .reset_index()
        .sort_values(
            [
                "Outer_Folds_Selected",
                "Mean_Training_Target_SpearmanAbs",
            ],
            ascending=[False, False],
        )
        .reset_index(drop=True)
    )

    selection_frequency.insert(
        0,
        "Formulation",
        formulation_name,
    )

    return {
        "predictions": predictions,
        "folds": folds,
        "macro": macro,
        "selected": selected_df,
        "selection_frequency": selection_frequency,
        "coverage_audit": coverage_df,
        "feature_tuning": feature_tuning_df,
        "model_tuning": model_tuning_df,
    }

def combine_results(results_by_name, key):
    frames = [
        result[key]
        for result in results_by_name.values()
        if key in result and result[key] is not None
    ]

    if not frames:
        return pd.DataFrame()

    return pd.concat(frames, ignore_index=True)


def run(data_path: Path, baseline_path: Path | None, output_dir: Path) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    frame = pd.read_csv(data_path)
    if len(frame) != EXPECTED_N:
        raise AssertionError(f"Expected {EXPECTED_N} observations, found {len(frame)}.")
    if sorted(frame[BATTERY_COL].astype(str).unique()) != sorted(EXPECTED_BATTERIES):
        raise AssertionError("Unexpected variable-protocol battery set.")

    baseline = pd.read_csv(baseline_path) if baseline_path and baseline_path.exists() else None
    expanded_strict = get_expanded_strict_columns(frame)
    electrical_candidates, removed_age = get_electrical_candidate_columns(frame)

    results = {
        "STRICT_ELECTRICAL_HGB": run_formulation(
            frame, "STRICT_ELECTRICAL_HGB", electrical_candidates, False
        ),
        "STRICT_ELECTRICAL_PLUS_AGE_HGB": run_formulation(
            frame, "STRICT_ELECTRICAL_PLUS_AGE_HGB", electrical_candidates, True
        ),
    }

    keys = ["predictions","folds","macro","selected","selection_frequency",
            "coverage_audit","feature_tuning","model_tuning"]
    combined = {key: combine_results(results, key) for key in keys}

    comparison_rows = []
    if baseline is not None:
        for _, row in baseline.iterrows():
            comparison_rows.append({
                "Model": row["Baseline"],
                "Type": "Baseline",
                "Macro_RMSE": row["Macro_RMSE"],
                "Macro_MAE": row["Macro_MAE"],
            })
    for _, row in combined["macro"].iterrows():
        comparison_rows.append({
            "Model": row["Formulation"],
            "Type": "HGB",
            "Macro_RMSE": row["Macro_RMSE_Raw"],
            "Macro_MAE": row["Macro_MAE_Raw"],
        })
    comparison = pd.DataFrame(comparison_rows)

    paths = {
        "macro": output_dir / "variable_hgb_macro.csv",
        "per_battery": output_dir / "variable_hgb_per_battery.csv",
        "predictions": output_dir / "variable_hgb_predictions.csv",
        "selected_features": output_dir / "variable_hgb_selected_features.csv",
        "selection_frequency": output_dir / "variable_hgb_feature_selection_frequency.csv",
        "coverage_audit": output_dir / "variable_hgb_coverage_audit.csv",
        "feature_tuning": output_dir / "variable_hgb_inner_feature_tuning.csv",
        "model_tuning": output_dir / "variable_hgb_inner_model_tuning.csv",
        "comparison": output_dir / "variable_hgb_vs_baselines.csv",
        "workbook": output_dir / "variable_strict_hgb_nested_lobo.xlsx",
    }
    mapping = {
        "macro":"macro","per_battery":"folds","predictions":"predictions",
        "selected_features":"selected","selection_frequency":"selection_frequency",
        "coverage_audit":"coverage_audit","feature_tuning":"feature_tuning",
        "model_tuning":"model_tuning",
    }
    for path_key, data_key in mapping.items():
        combined[data_key].to_csv(paths[path_key], index=False)
    comparison.to_csv(paths["comparison"], index=False)

    with pd.ExcelWriter(paths["workbook"], engine="openpyxl") as writer:
        for key, df in combined.items():
            df.to_excel(writer, sheet_name=key[:31], index=False)
        comparison.to_excel(writer, sheet_name="Baseline_Comparison", index=False)

    return paths


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("outputs/variable_dataset/variable_independent_observations.csv"))
    parser.add_argument("--baselines", type=Path, default=Path("outputs/variable_dataset/variable_baseline_macro.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/variable_hgb"))
    args = parser.parse_args()
    baseline = args.baselines if args.baselines.exists() else None
    for name, path in run(args.data, baseline, args.output_dir).items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
