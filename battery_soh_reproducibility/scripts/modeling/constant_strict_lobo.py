"""Nested leave-one-battery-out HGB for the revised constant-protocol task."""
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


TARGET_COL = "Calculated_SOH(%)"

BATTERY_COL = "battery_id"

SOURCE_COL = "source_dataset"

EXPECTED_BATTERIES = [
    "CS2_33", "CS2_34", "CS2_35",
    "CS2_36", "CS2_37", "CS2_38",
    "B0005", "B0006", "B0007", "B0018",
]

EXPECTED_STRICT_FEATURE_COUNT = 335

EXPECTED_N = 4590

MIN_SOURCE_COVERAGE = 0.95

EXPLICIT_AGE_STRICT_FEATURES = {
    "LEGACY__Cycle_Age",
    "LEGACY__Cycle_Index",
    "LEGACY__Log_Cycle_Age",
}

AGE_COLUMNS = [
    "Cycle_Age",
    "Log_Cycle_Age",
]

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

def rmse(y_true, y_pred):
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))

def mae(y_true, y_pred):
    return float(mean_absolute_error(y_true, y_pred))

def equal_battery_weights(frame):
    """
    Each battery contributes equal total training weight, regardless of its
    number of retained cycles.
    """
    counts = frame[BATTERY_COL].value_counts()

    weights = frame[BATTERY_COL].map(
        lambda b: 1.0 / counts.loc[b]
    ).astype(float).to_numpy()

    if np.mean(weights) > 0:
        weights = weights / np.mean(weights)

    return weights

def numeric_frame(frame, columns):
    out = frame.loc[:, columns].apply(
        pd.to_numeric,
        errors="coerce",
    )

    return out.replace(
        [np.inf, -np.inf],
        np.nan,
    )

def load_frozen_feature_list(path: Path) -> list[str]:
    """Load the frozen strict constant-protocol feature-name list."""
    table = pd.read_csv(path)
    if "Feature" not in table.columns:
        raise KeyError("Strict feature file must contain a 'Feature' column.")
    return table["Feature"].dropna().astype(str).tolist()

def training_source_coverage_filter(
    train_frame,
    columns,
    minimum=MIN_SOURCE_COVERAGE,
):
    """
    TRAINING-ONLY availability filter.

    A feature remains eligible only when:
      1. finite coverage is >= minimum within EACH source represented in the
         current training data; and
      2. it has more than one distinct finite training value.

    No held-out outer or inner test rows are inspected.
    """
    if len(columns) == 0:
        return [], []

    X = numeric_frame(
        train_frame,
        columns,
    )

    sources = sorted(
        train_frame[SOURCE_COL].dropna().unique()
    )

    keep = []
    audit = []

    for feature in columns:
        eligible = True
        coverages = {}

        for source in sources:
            mask = train_frame[SOURCE_COL].eq(source)
            coverage = float(
                X.loc[mask, feature].notna().mean()
            )

            coverages[source] = coverage

            if not np.isfinite(coverage) or coverage < minimum:
                eligible = False

        unique_values = int(
            X[feature].nunique(dropna=True)
        )

        if unique_values <= 1:
            eligible = False

        audit.append(
            {
                "Feature": feature,
                "Eligible": bool(eligible),
                "Unique_Training_Values": unique_values,
                **{
                    f"Coverage_{source}": coverages[source]
                    for source in sources
                },
            }
        )

        if eligible:
            keep.append(feature)

    return keep, audit

def prepare_feature_selector(
    train_frame,
    candidate_columns,
):
    """
    Prepare target ranking and correlation information from the current
    training split only.
    """
    eligible, coverage_audit = (
        training_source_coverage_filter(
            train_frame,
            candidate_columns,
        )
    )

    if len(eligible) == 0:
        raise RuntimeError(
            "No features remained after training-only source coverage filtering."
        )

    X_raw = numeric_frame(
        train_frame,
        eligible,
    )

    imputer = SimpleImputer(
        strategy="median",
    )

    X_imp = imputer.fit_transform(
        X_raw,
    )

    X = pd.DataFrame(
        X_imp,
        columns=eligible,
        index=train_frame.index,
    )

    # Absolute Spearman association:
    # Pearson correlation of ranked values.
    ranked_X = X.rank(
        method="average",
    )

    ranked_y = pd.Series(
        pd.to_numeric(
            train_frame[TARGET_COL],
            errors="coerce",
        ).to_numpy(),
        index=train_frame.index,
    ).rank(
        method="average",
    )

    target_scores = (
        ranked_X.corrwith(ranked_y)
        .abs()
        .replace(
            [np.inf, -np.inf],
            np.nan,
        )
        .fillna(0.0)
    )

    ranked_order = (
        target_scores
        .sort_values(
            ascending=False,
            kind="mergesort",
        )
        .index
        .tolist()
    )

    # Spearman feature-feature correlation.
    feature_corr = (
        ranked_X.loc[:, ranked_order]
        .corr()
        .abs()
    )

    return {
        "eligible_features": eligible,
        "coverage_audit": coverage_audit,
        "target_scores": target_scores,
        "ranked_order": ranked_order,
        "feature_corr": feature_corr,
    }

def choose_features(
    prepared,
    top_k,
    corr_threshold,
):
    """
    Greedy correlation pruning, in descending training-target association order.
    """
    selected = []

    for feature in prepared["ranked_order"]:

        if len(selected) >= int(top_k):
            break

        if not selected:
            selected.append(feature)
            continue

        correlations = prepared[
            "feature_corr"
        ].loc[
            feature,
            selected,
        ]

        if bool(
            (correlations >= float(corr_threshold)).any()
        ):
            continue

        selected.append(feature)

    if not selected:
        raise RuntimeError(
            "Feature selection returned zero features."
        )

    return selected

def model_features(
    selected_electrical,
    include_age,
):
    cols = list(selected_electrical)

    if include_age:
        for age_col in AGE_COLUMNS:
            if age_col not in cols:
                cols.append(age_col)

    return cols

def fit_hgb(
    train_frame,
    selected_electrical,
    config_name,
    include_age,
):
    cols = model_features(
        selected_electrical,
        include_age,
    )

    X_raw = numeric_frame(
        train_frame,
        cols,
    )

    imputer = SimpleImputer(
        strategy="median",
    )

    X = imputer.fit_transform(
        X_raw,
    )

    model = HistGradientBoostingRegressor(
        loss="squared_error",
        early_stopping=False,
        random_state=RANDOM_STATE,
        **HGB_CONFIGS[config_name],
    )

    model.fit(
        X,
        pd.to_numeric(
            train_frame[TARGET_COL],
            errors="coerce",
        ).to_numpy(),
        sample_weight=equal_battery_weights(
            train_frame
        ),
    )

    return model, imputer, cols

def predict_hgb(
    model,
    imputer,
    columns,
    test_frame,
):
    X_raw = numeric_frame(
        test_frame,
        columns,
    )

    X = imputer.transform(
        X_raw,
    )

    raw = model.predict(X)

    clipped = np.clip(
        raw,
        CLIP_LOW,
        CLIP_HIGH,
    )

    return raw, clipped

def prepare_inner_splits(
    outer_train,
    candidate_columns,
):
    """
    Prepare each of the nine inner LOBO splits for one outer fold.
    """
    prepared = {}

    batteries = sorted(
        outer_train[BATTERY_COL].unique()
    )

    for inner_holdout in batteries:
        inner_train = outer_train.loc[
            outer_train[BATTERY_COL] != inner_holdout
        ].copy()

        inner_test = outer_train.loc[
            outer_train[BATTERY_COL] == inner_holdout
        ].copy()

        selector = prepare_feature_selector(
            inner_train,
            candidate_columns,
        )

        prepared[inner_holdout] = {
            "train": inner_train,
            "test": inner_test,
            "selector": selector,
        }

    return prepared

def tune_feature_selection(
    outer_train,
    candidate_columns,
    include_age,
):
    """
    Stage A:
    select correlation threshold and top-K using inner LOBO with one fixed
    HGB configuration.
    """
    inner = prepare_inner_splits(
        outer_train,
        candidate_columns,
    )

    rows = []

    for corr_threshold in CORR_THRESHOLDS:
        for top_k in TOP_K_OPTIONS:

            fold_rmse = []
            fold_mae = []

            for inner_holdout, split in inner.items():

                selected = choose_features(
                    split["selector"],
                    top_k=top_k,
                    corr_threshold=corr_threshold,
                )

                model, imputer, columns = fit_hgb(
                    split["train"],
                    selected,
                    FEATURE_TUNING_MODEL,
                    include_age,
                )

                pred, _ = predict_hgb(
                    model,
                    imputer,
                    columns,
                    split["test"],
                )

                y = pd.to_numeric(
                    split["test"][TARGET_COL],
                    errors="coerce",
                ).to_numpy()

                fold_rmse.append(
                    rmse(y, pred)
                )

                fold_mae.append(
                    mae(y, pred)
                )

            rows.append(
                {
                    "Corr_Threshold": corr_threshold,
                    "Top_K": top_k,
                    "Inner_Macro_RMSE": float(
                        np.mean(fold_rmse)
                    ),
                    "Inner_Macro_MAE": float(
                        np.mean(fold_mae)
                    ),
                    "Feature_Tuning_Model": FEATURE_TUNING_MODEL,
                }
            )

    table = pd.DataFrame(rows).sort_values(
        [
            "Inner_Macro_RMSE",
            "Top_K",
            "Corr_Threshold",
        ],
        ascending=[
            True,
            True,
            True,
        ],
        kind="mergesort",
    ).reset_index(
        drop=True
    )

    best = table.iloc[0]

    return (
        float(best["Corr_Threshold"]),
        int(best["Top_K"]),
        table,
        inner,
    )

def tune_hgb_config(
    inner_prepared,
    corr_threshold,
    top_k,
    include_age,
):
    """
    Stage B:
    select one predefined HGB configuration using the same inner LOBO splits.
    """
    rows = []

    for config_name in HGB_CONFIGS:

        fold_rmse = []
        fold_mae = []

        for inner_holdout, split in inner_prepared.items():

            selected = choose_features(
                split["selector"],
                top_k=top_k,
                corr_threshold=corr_threshold,
            )

            model, imputer, columns = fit_hgb(
                split["train"],
                selected,
                config_name,
                include_age,
            )

            pred, _ = predict_hgb(
                model,
                imputer,
                columns,
                split["test"],
            )

            y = pd.to_numeric(
                split["test"][TARGET_COL],
                errors="coerce",
            ).to_numpy()

            fold_rmse.append(
                rmse(y, pred)
            )

            fold_mae.append(
                mae(y, pred)
            )

        rows.append(
            {
                "HGB_Config": config_name,
                "Inner_Macro_RMSE": float(
                    np.mean(fold_rmse)
                ),
                "Inner_Macro_MAE": float(
                    np.mean(fold_mae)
                ),
            }
        )

    table = pd.DataFrame(rows).sort_values(
        [
            "Inner_Macro_RMSE",
            "HGB_Config",
        ],
        ascending=[
            True,
            True,
        ],
        kind="mergesort",
    ).reset_index(
        drop=True
    )

    return (
        str(table.iloc[0]["HGB_Config"]),
        table,
    )

def run_formulation(
    frame,
    formulation_name,
    candidate_columns,
    include_age,
):
    prediction_rows = []
    fold_rows = []
    selected_rows = []
    coverage_rows = []
    feature_tuning_rows = []
    model_tuning_rows = []

    batteries = sorted(
        frame[BATTERY_COL].unique()
    )

    print("\n" + "=" * 90)
    print("FORMULATION:", formulation_name)
    print("=" * 90)

    for fold_number, outer_holdout in enumerate(
        batteries,
        start=1,
    ):
        fold_start = time.time()

        outer_train = frame.loc[
            frame[BATTERY_COL] != outer_holdout
        ].copy()

        outer_test = frame.loc[
            frame[BATTERY_COL] == outer_holdout
        ].copy()

        print(
            f"\nOuter fold {fold_number}/{len(batteries)} "
            f"- holdout {outer_holdout}"
        )

        # ----------------------------------------------------
        # Stage A: feature-selection settings
        # ----------------------------------------------------
        (
            best_corr,
            best_k,
            feature_tuning,
            inner_prepared,
        ) = tune_feature_selection(
            outer_train,
            candidate_columns,
            include_age,
        )

        feature_tuning = feature_tuning.copy()

        feature_tuning.insert(
            0,
            "Formulation",
            formulation_name,
        )

        feature_tuning.insert(
            1,
            "Outer_Holdout",
            outer_holdout,
        )

        feature_tuning_rows.append(
            feature_tuning
        )

        # ----------------------------------------------------
        # Stage B: HGB configuration
        # ----------------------------------------------------
        best_config, model_tuning = (
            tune_hgb_config(
                inner_prepared,
                best_corr,
                best_k,
                include_age,
            )
        )

        model_tuning = model_tuning.copy()

        model_tuning.insert(
            0,
            "Formulation",
            formulation_name,
        )

        model_tuning.insert(
            1,
            "Outer_Holdout",
            outer_holdout,
        )

        model_tuning[
            "Chosen_Corr_Threshold"
        ] = best_corr

        model_tuning[
            "Chosen_Top_K"
        ] = best_k

        model_tuning_rows.append(
            model_tuning
        )

        # ----------------------------------------------------
        # Rebuild feature selection using only full outer train
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

        # Coverage audit
        for row in selector["coverage_audit"]:
            coverage_rows.append(
                {
                    "Formulation": formulation_name,
                    "Outer_Holdout": outer_holdout,
                    **row,
                }
            )

        # Selected-feature audit
        for rank, feature in enumerate(
            selected,
            start=1,
        ):
            selected_rows.append(
                {
                    "Formulation": formulation_name,
                    "Outer_Holdout": outer_holdout,
                    "Selected_Rank": rank,
                    "Feature": feature,
                    "Training_Target_SpearmanAbs": float(
                        selector[
                            "target_scores"
                        ].loc[feature]
                    ),
                    "Corr_Threshold": best_corr,
                    "Top_K": best_k,
                    "HGB_Config": best_config,
                }
            )

        # ----------------------------------------------------
        # Final outer fit
        # ----------------------------------------------------
        model, imputer, columns = fit_hgb(
            outer_train,
            selected,
            best_config,
            include_age,
        )

        pred_raw, pred_clipped = (
            predict_hgb(
                model,
                imputer,
                columns,
                outer_test,
            )
        )

        y = pd.to_numeric(
            outer_test[TARGET_COL],
            errors="coerce",
        ).to_numpy()

        raw_rmse = rmse(
            y,
            pred_raw,
        )

        raw_mae = mae(
            y,
            pred_raw,
        )

        clipped_rmse = rmse(
            y,
            pred_clipped,
        )

        clipped_mae = mae(
            y,
            pred_clipped,
        )

        elapsed = time.time() - fold_start

        fold_rows.append(
            {
                "Formulation": formulation_name,
                "Holdout_Battery": outer_holdout,
                "Source": str(
                    outer_test[SOURCE_COL].iloc[0]
                ),
                "N_Test_Cycles": int(
                    len(outer_test)
                ),
                "Eligible_Training_Features": int(
                    len(
                        selector[
                            "eligible_features"
                        ]
                    )
                ),
                "Selected_Electrical_Features": int(
                    len(selected)
                ),
                "Explicit_Age_Features_Added": (
                    len(AGE_COLUMNS)
                    if include_age
                    else 0
                ),
                "Final_Model_Features": int(
                    len(columns)
                ),
                "Chosen_Corr_Threshold": best_corr,
                "Chosen_Top_K": best_k,
                "Chosen_HGB_Config": best_config,
                "RMSE_Raw": raw_rmse,
                "MAE_Raw": raw_mae,
                "RMSE_Clipped70_100": clipped_rmse,
                "MAE_Clipped70_100": clipped_mae,
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
                    "Holdout_Battery": outer_holdout,
                    BATTERY_COL: row[BATTERY_COL],
                    SOURCE_COL: row[SOURCE_COL],
                    "Cycle_Index": row["Cycle_Index"],
                    "Cycle_Age": row["Cycle_Age"],
                    TARGET_COL: row[TARGET_COL],
                    "Prediction_Raw": float(
                        raw_pred
                    ),
                    "Prediction_Clipped70_100": float(
                        clipped_pred
                    ),
                    "Residual_Raw": float(
                        row[TARGET_COL] - raw_pred
                    ),
                    "Residual_Clipped70_100": float(
                        row[TARGET_COL] - clipped_pred
                    ),
                }
            )

        print(
            f"  eligible training features: "
            f"{len(selector['eligible_features'])}"
        )

        print(
            f"  selected={len(selected)} "
            f"| corr={best_corr} "
            f"| K={best_k} "
            f"| HGB={best_config}"
        )

        print(
            f"  raw RMSE={raw_rmse:.4f} "
            f"| clipped RMSE={clipped_rmse:.4f}"
        )

        print(
            f"  runtime={elapsed:.1f} s"
        )

    predictions = pd.DataFrame(
        prediction_rows
    )

    folds = pd.DataFrame(
        fold_rows
    )

    selected_df = pd.DataFrame(
        selected_rows
    )

    coverage_df = pd.DataFrame(
        coverage_rows
    )

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
                "Batteries": int(
                    folds[
                        "Holdout_Battery"
                    ].nunique()
                ),
                "Macro_RMSE_Raw": float(
                    folds["RMSE_Raw"].mean()
                ),
                "Macro_MAE_Raw": float(
                    folds["MAE_Raw"].mean()
                ),
                "Macro_RMSE_Clipped70_100": float(
                    folds[
                        "RMSE_Clipped70_100"
                    ].mean()
                ),
                "Macro_MAE_Clipped70_100": float(
                    folds[
                        "MAE_Clipped70_100"
                    ].mean()
                ),
            }
        ]
    )

    frequency = (
        selected_df.groupby(
            "Feature"
        )
        .agg(
            Outer_Folds_Selected=(
                "Outer_Holdout",
                "nunique",
            ),
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
            ascending=[
                False,
                False,
            ],
        )
        .reset_index(
            drop=True
        )
    )

    frequency.insert(
        0,
        "Formulation",
        formulation_name,
    )

    return {
        "predictions": predictions,
        "folds": folds,
        "macro": macro,
        "selected": selected_df,
        "selection_frequency": frequency,
        "coverage_audit": coverage_df,
        "feature_tuning": feature_tuning_df,
        "model_tuning": model_tuning_df,
    }

def combine_results(
    results,
    key,
):
    frames = [
        result[key]
        for result in results.values()
        if key in result
        and result[key] is not None
    ]

    if not frames:
        return pd.DataFrame()

    return pd.concat(
        frames,
        ignore_index=True,
    )

def run(data_path: Path, feature_list_path: Path, baseline_path: Path | None, output_dir: Path) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    frame = pd.read_csv(data_path)
    strict_features = load_frozen_feature_list(feature_list_path)
    if len(frame) != EXPECTED_N:
        raise AssertionError(f"Expected {EXPECTED_N} observations, found {len(frame)}.")
    if len(strict_features) != EXPECTED_STRICT_FEATURE_COUNT:
        raise AssertionError(
            f"Expected {EXPECTED_STRICT_FEATURE_COUNT} strict features, found {len(strict_features)}."
        )
    if sorted(frame[BATTERY_COL].astype(str).unique()) != sorted(EXPECTED_BATTERIES):
        raise AssertionError("Unexpected constant-protocol battery set.")

    baseline = pd.read_csv(baseline_path) if baseline_path and baseline_path.exists() else None

    # The primary model removes explicit age/index from the frozen strict set.
    electrical_candidates = [
        f for f in strict_features if f not in EXPLICIT_AGE_STRICT_FEATURES
    ]

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
        "macro": output_dir / "constant_hgb_macro.csv",
        "per_battery": output_dir / "constant_hgb_per_battery.csv",
        "predictions": output_dir / "constant_hgb_predictions.csv",
        "selected_features": output_dir / "constant_hgb_selected_features.csv",
        "selection_frequency": output_dir / "constant_hgb_feature_selection_frequency.csv",
        "coverage_audit": output_dir / "constant_hgb_coverage_audit.csv",
        "feature_tuning": output_dir / "constant_hgb_inner_feature_tuning.csv",
        "model_tuning": output_dir / "constant_hgb_inner_model_tuning.csv",
        "comparison": output_dir / "constant_hgb_vs_baselines.csv",
        "workbook": output_dir / "constant_strict_hgb_nested_lobo.xlsx",
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
    parser.add_argument("--data", type=Path, default=Path("outputs/constant_dataset/constant_observations.csv"))
    parser.add_argument("--features", type=Path, default=Path("outputs/constant_dataset/constant_strict_feature_names.csv"))
    parser.add_argument("--baselines", type=Path, default=Path("outputs/constant_dataset/constant_baseline_macro.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/constant_hgb"))
    args = parser.parse_args()
    baseline = args.baselines if args.baselines.exists() else None
    for name, path in run(args.data, args.features, baseline, args.output_dir).items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
