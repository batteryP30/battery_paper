"""Constant-protocol cross-dataset transfer (NASA <-> CALCE).

Model/feature tuning is performed using source-dataset labels only. The target
source is used only for the final transfer evaluation.
"""
from __future__ import annotations

import argparse
import time
import warnings
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

EXPECTED_STRICT_FEATURES = 335

EXPLICIT_AGE_STRICT_FEATURES = {
    "LEGACY__Cycle_Age",
    "LEGACY__Cycle_Index",
    "LEGACY__Log_Cycle_Age",
}

AGE_COLUMNS = [
    "Cycle_Age",
    "Log_Cycle_Age",
]

COMMON_SOURCE_COVERAGE = 0.95

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

CLIP_LOW = 70.0

CLIP_HIGH = 100.0

def rmse(y_true, y_pred):
    return float(
        np.sqrt(
            mean_squared_error(
                y_true,
                y_pred,
            )
        )
    )

def mae(y_true, y_pred):
    return float(
        mean_absolute_error(
            y_true,
            y_pred,
        )
    )

def numeric_frame(frame, columns):
    return (
        frame.loc[:, columns]
        .apply(
            pd.to_numeric,
            errors="coerce",
        )
        .replace(
            [np.inf, -np.inf],
            np.nan,
        )
    )

def equal_battery_weights(frame):
    counts = frame[BATTERY_COL].value_counts()

    weights = frame[BATTERY_COL].map(
        lambda b: 1.0 / counts.loc[b]
    ).astype(float).to_numpy()

    if np.mean(weights) > 0:
        weights = weights / np.mean(weights)

    return weights

def define_interoperable_schema(
    frame,
    strict_features,
):
    """
    Define the feature schema using FEATURE AVAILABILITY ONLY.

    No target values are consulted.

    A feature must have >=95% finite coverage in both CALCE and NASA.

    This is intentionally done once before transfer as an interoperability
    constraint. It is not a target-dependent selection procedure.
    """
    rows = []
    keep = []

    for feature in strict_features:
        values = pd.to_numeric(
            frame[feature],
            errors="coerce",
        )

        coverage = {}

        for source in ["CALCE", "NASA"]:
            mask = frame[SOURCE_COL].eq(source)

            coverage[source] = float(
                values.loc[mask]
                .notna()
                .mean()
            )

        eligible = (
            coverage["CALCE"] >= COMMON_SOURCE_COVERAGE
            and coverage["NASA"] >= COMMON_SOURCE_COVERAGE
        )

        rows.append(
            {
                "Feature": feature,
                "CALCE_Coverage": coverage["CALCE"],
                "NASA_Coverage": coverage["NASA"],
                "Interoperable": bool(eligible),
            }
        )

        if eligible:
            keep.append(feature)

    return keep, pd.DataFrame(rows)

def prepare_selector(
    train_frame,
    candidate_columns,
):
    """
    Source-training-only selector preparation.

    The target-source test dataset is never used.
    """
    X_raw = numeric_frame(
        train_frame,
        candidate_columns,
    )

    # Drop source-training constants or entirely missing columns.
    usable = [
        c
        for c in candidate_columns
        if X_raw[c].notna().any()
        and X_raw[c].nunique(dropna=True) > 1
    ]

    if not usable:
        raise RuntimeError(
            "No usable features remained in the source training dataset."
        )

    X_raw = X_raw[usable]

    imputer = SimpleImputer(
        strategy="median",
    )

    X_imp = imputer.fit_transform(
        X_raw
    )

    X = pd.DataFrame(
        X_imp,
        columns=usable,
        index=train_frame.index,
    )

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

    feature_corr = (
        ranked_X[ranked_order]
        .corr()
        .abs()
    )

    return {
        "usable": usable,
        "target_scores": target_scores,
        "ranked_order": ranked_order,
        "feature_corr": feature_corr,
    }

def choose_features(
    selector,
    top_k,
    corr_threshold,
):
    selected = []

    for feature in selector["ranked_order"]:

        if len(selected) >= int(top_k):
            break

        if not selected:
            selected.append(feature)
            continue

        corr = selector[
            "feature_corr"
        ].loc[
            feature,
            selected,
        ]

        if bool(
            (corr >= float(corr_threshold)).any()
        ):
            continue

        selected.append(feature)

    if not selected:
        raise RuntimeError(
            "Feature selection returned no features."
        )

    return selected

def build_model_columns(
    selected_electrical,
    include_age,
):
    columns = list(
        selected_electrical
    )

    if include_age:
        for col in AGE_COLUMNS:
            if col not in columns:
                columns.append(col)

    return columns

def fit_model(
    train_frame,
    selected_electrical,
    config_name,
    include_age,
):
    columns = build_model_columns(
        selected_electrical,
        include_age,
    )

    X_raw = numeric_frame(
        train_frame,
        columns,
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

    return model, imputer, columns

def predict_model(
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

    raw = model.predict(
        X
    )

    clipped = np.clip(
        raw,
        CLIP_LOW,
        CLIP_HIGH,
    )

    return raw, clipped

def prepare_source_inner_splits(
    source_train,
    candidate_columns,
):
    """
    Leave-one-battery-out splits entirely within the source dataset.
    """
    prepared = {}

    batteries = sorted(
        source_train[BATTERY_COL]
        .unique()
    )

    if len(batteries) < 2:
        raise RuntimeError(
            "Cross-dataset source tuning requires at least two source batteries."
        )

    for heldout in batteries:
        inner_train = source_train.loc[
            source_train[BATTERY_COL] != heldout
        ].copy()

        inner_valid = source_train.loc[
            source_train[BATTERY_COL] == heldout
        ].copy()

        selector = prepare_selector(
            inner_train,
            candidate_columns,
        )

        prepared[heldout] = {
            "train": inner_train,
            "valid": inner_valid,
            "selector": selector,
        }

    return prepared

def tune_feature_settings(
    source_train,
    candidate_columns,
    include_age,
):
    inner = prepare_source_inner_splits(
        source_train,
        candidate_columns,
    )

    rows = []

    for corr_threshold in CORR_THRESHOLDS:
        for top_k in TOP_K_OPTIONS:

            rmses = []
            maes = []

            for heldout, split in inner.items():
                selected = choose_features(
                    split["selector"],
                    top_k=top_k,
                    corr_threshold=corr_threshold,
                )

                model, imputer, columns = fit_model(
                    split["train"],
                    selected,
                    FEATURE_TUNING_MODEL,
                    include_age,
                )

                pred, _ = predict_model(
                    model,
                    imputer,
                    columns,
                    split["valid"],
                )

                y = pd.to_numeric(
                    split["valid"][TARGET_COL],
                    errors="coerce",
                ).to_numpy()

                rmses.append(
                    rmse(
                        y,
                        pred,
                    )
                )

                maes.append(
                    mae(
                        y,
                        pred,
                    )
                )

            rows.append(
                {
                    "Corr_Threshold": corr_threshold,
                    "Top_K": top_k,
                    "Source_LOBO_Macro_RMSE": float(
                        np.mean(rmses)
                    ),
                    "Source_LOBO_Macro_MAE": float(
                        np.mean(maes)
                    ),
                    "Feature_Tuning_Model": FEATURE_TUNING_MODEL,
                }
            )

    table = pd.DataFrame(
        rows
    ).sort_values(
        [
            "Source_LOBO_Macro_RMSE",
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
        float(
            best["Corr_Threshold"]
        ),
        int(
            best["Top_K"]
        ),
        table,
        inner,
    )

def tune_hgb(
    inner,
    corr_threshold,
    top_k,
    include_age,
):
    rows = []

    for config_name in HGB_CONFIGS:

        rmses = []
        maes = []

        for heldout, split in inner.items():
            selected = choose_features(
                split["selector"],
                top_k=top_k,
                corr_threshold=corr_threshold,
            )

            model, imputer, columns = fit_model(
                split["train"],
                selected,
                config_name,
                include_age,
            )

            pred, _ = predict_model(
                model,
                imputer,
                columns,
                split["valid"],
            )

            y = pd.to_numeric(
                split["valid"][TARGET_COL],
                errors="coerce",
            ).to_numpy()

            rmses.append(
                rmse(
                    y,
                    pred,
                )
            )

            maes.append(
                mae(
                    y,
                    pred,
                )
            )

        rows.append(
            {
                "HGB_Config": config_name,
                "Source_LOBO_Macro_RMSE": float(
                    np.mean(rmses)
                ),
                "Source_LOBO_Macro_MAE": float(
                    np.mean(maes)
                ),
            }
        )

    table = pd.DataFrame(
        rows
    ).sort_values(
        [
            "Source_LOBO_Macro_RMSE",
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
        str(
            table.iloc[0]["HGB_Config"]
        ),
        table,
    )

def run_transfer_direction(
    frame,
    candidate_columns,
    train_source,
    test_source,
    formulation,
    include_age,
):
    """
    Source-only tuning followed by untouched target-source evaluation.
    """
    start = time.time()

    source_train = frame.loc[
        frame[SOURCE_COL].eq(
            train_source
        )
    ].copy()

    target_test = frame.loc[
        frame[SOURCE_COL].eq(
            test_source
        )
    ].copy()

    print("\n" + "=" * 90)
    print(
        f"{formulation}: {train_source} -> {test_source}"
    )
    print("=" * 90)

    print(
        "Source training batteries:",
        sorted(
            source_train[BATTERY_COL]
            .unique()
        ),
    )

    print(
        "Target test batteries:",
        sorted(
            target_test[BATTERY_COL]
            .unique()
        ),
    )

    # --------------------------------------------------------
    # Source-only nested tuning
    # --------------------------------------------------------
    (
        best_corr,
        best_k,
        feature_tuning,
        inner,
    ) = tune_feature_settings(
        source_train,
        candidate_columns,
        include_age,
    )

    best_config, model_tuning = (
        tune_hgb(
            inner,
            best_corr,
            best_k,
            include_age,
        )
    )

    # --------------------------------------------------------
    # Final feature ranking on all source training batteries
    # --------------------------------------------------------
    selector = prepare_selector(
        source_train,
        candidate_columns,
    )

    selected = choose_features(
        selector,
        top_k=best_k,
        corr_threshold=best_corr,
    )

    # --------------------------------------------------------
    # Final source-only model
    # --------------------------------------------------------
    model, imputer, columns = fit_model(
        source_train,
        selected,
        best_config,
        include_age,
    )

    pred_raw, pred_clip = predict_model(
        model,
        imputer,
        columns,
        target_test,
    )

    target_test = (
        target_test.copy()
        .reset_index(
            drop=True
        )
    )

    target_test[
        "Prediction_Raw"
    ] = pred_raw

    target_test[
        "Prediction_Clipped70_100"
    ] = pred_clip

    target_test[
        "Train_Source"
    ] = train_source

    target_test[
        "Test_Source"
    ] = test_source

    target_test[
        "Formulation"
    ] = formulation

    # --------------------------------------------------------
    # Per-target-battery performance
    # --------------------------------------------------------
    battery_rows = []

    for battery, g in target_test.groupby(
        BATTERY_COL,
        sort=True,
    ):
        y = pd.to_numeric(
            g[TARGET_COL],
            errors="coerce",
        ).to_numpy()

        raw = g[
            "Prediction_Raw"
        ].to_numpy()

        clipped = g[
            "Prediction_Clipped70_100"
        ].to_numpy()

        battery_rows.append(
            {
                "Formulation": formulation,
                "Train_Source": train_source,
                "Test_Source": test_source,
                "Target_Battery": battery,
                "N_Test": int(
                    len(g)
                ),
                "RMSE_Raw": rmse(
                    y,
                    raw,
                ),
                "MAE_Raw": mae(
                    y,
                    raw,
                ),
                "RMSE_Clipped70_100": rmse(
                    y,
                    clipped,
                ),
                "MAE_Clipped70_100": mae(
                    y,
                    clipped,
                ),
            }
        )

    per_battery = pd.DataFrame(
        battery_rows
    )

    summary = pd.DataFrame(
        [
            {
                "Formulation": formulation,
                "Train_Source": train_source,
                "Test_Source": test_source,
                "Source_Training_Batteries": int(
                    source_train[
                        BATTERY_COL
                    ].nunique()
                ),
                "Target_Test_Batteries": int(
                    target_test[
                        BATTERY_COL
                    ].nunique()
                ),
                "N_Source_Train": int(
                    len(
                        source_train
                    )
                ),
                "N_Target_Test": int(
                    len(
                        target_test
                    )
                ),
                "Interoperable_Candidate_Features": int(
                    len(
                        candidate_columns
                    )
                ),
                "Selected_Electrical_Features": int(
                    len(
                        selected
                    )
                ),
                "Explicit_Age_Features_Added": (
                    len(
                        AGE_COLUMNS
                    )
                    if include_age
                    else 0
                ),
                "Chosen_Corr_Threshold": best_corr,
                "Chosen_Top_K": best_k,
                "Chosen_HGB_Config": best_config,
                "Macro_RMSE_Raw": float(
                    per_battery[
                        "RMSE_Raw"
                    ].mean()
                ),
                "Macro_MAE_Raw": float(
                    per_battery[
                        "MAE_Raw"
                    ].mean()
                ),
                "Macro_RMSE_Clipped70_100": float(
                    per_battery[
                        "RMSE_Clipped70_100"
                    ].mean()
                ),
                "Macro_MAE_Clipped70_100": float(
                    per_battery[
                        "MAE_Clipped70_100"
                    ].mean()
                ),
                "Runtime_Seconds": float(
                    time.time() - start
                ),
            }
        ]
    )

    selected_table = pd.DataFrame(
        [
            {
                "Formulation": formulation,
                "Train_Source": train_source,
                "Test_Source": test_source,
                "Selected_Rank": rank,
                "Feature": feature,
                "Source_Training_Target_SpearmanAbs": float(
                    selector[
                        "target_scores"
                    ].loc[
                        feature
                    ]
                ),
                "Corr_Threshold": best_corr,
                "Top_K": best_k,
                "HGB_Config": best_config,
            }
            for rank, feature in enumerate(
                selected,
                start=1,
            )
        ]
    )

    # Narrow prediction export.
    prediction_cols = [
        "Formulation",
        "Train_Source",
        "Test_Source",
        BATTERY_COL,
        "Cycle_Index",
        "Cycle_Age",
        TARGET_COL,
        "Prediction_Raw",
        "Prediction_Clipped70_100",
    ]

    predictions = target_test[
        prediction_cols
    ].copy()

    feature_tuning = (
        feature_tuning.copy()
    )

    feature_tuning.insert(
        0,
        "Formulation",
        formulation,
    )

    feature_tuning.insert(
        1,
        "Train_Source",
        train_source,
    )

    feature_tuning.insert(
        2,
        "Test_Source",
        test_source,
    )

    model_tuning = (
        model_tuning.copy()
    )

    model_tuning.insert(
        0,
        "Formulation",
        formulation,
    )

    model_tuning.insert(
        1,
        "Train_Source",
        train_source,
    )

    model_tuning.insert(
        2,
        "Test_Source",
        test_source,
    )

    print(
        summary.to_string(
            index=False
        )
    )

    return {
        "summary": summary,
        "per_battery": per_battery,
        "predictions": predictions,
        "selected": selected_table,
        "feature_tuning": feature_tuning,
        "model_tuning": model_tuning,
    }

def combine(results, key):
    return pd.concat(
        [
            r[key]
            for r in results
        ],
        ignore_index=True,
    )

def run(data_path: Path, feature_list_path: Path, output_dir: Path) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    frame = pd.read_csv(data_path)
    feature_table = pd.read_csv(feature_list_path)
    if "Feature" not in feature_table.columns:
        raise KeyError("Strict feature file must contain a 'Feature' column.")
    strict_features = feature_table["Feature"].dropna().astype(str).tolist()
    if len(strict_features) != EXPECTED_STRICT_FEATURES:
        raise AssertionError(
            f"Expected {EXPECTED_STRICT_FEATURES} strict features, found {len(strict_features)}."
        )

    interoperable, interoperability_audit = define_interoperable_schema(
        frame, strict_features
    )
    electrical_candidates = [
        feature for feature in interoperable
        if feature not in EXPLICIT_AGE_STRICT_FEATURES
    ]

    results = []
    for train_source, test_source in [("NASA", "CALCE"), ("CALCE", "NASA")]:
        results.append(
            run_transfer_direction(
                frame, electrical_candidates, train_source, test_source,
                "STRICT_ELECTRICAL_HGB", False
            )
        )
        results.append(
            run_transfer_direction(
                frame, electrical_candidates, train_source, test_source,
                "STRICT_ELECTRICAL_PLUS_AGE_HGB", True
            )
        )

    keys = ["summary","per_battery","predictions","selected","feature_tuning","model_tuning"]
    combined = {key: combine(results, key) for key in keys}

    paths = {
        "summary": output_dir / "constant_transfer_summary.csv",
        "per_battery": output_dir / "constant_transfer_per_battery.csv",
        "predictions": output_dir / "constant_transfer_predictions.csv",
        "selected_features": output_dir / "constant_transfer_selected_features.csv",
        "feature_tuning": output_dir / "constant_transfer_inner_feature_tuning.csv",
        "model_tuning": output_dir / "constant_transfer_inner_model_tuning.csv",
        "interoperability_audit": output_dir / "constant_transfer_interoperability_audit.csv",
        "workbook": output_dir / "constant_cross_dataset_transfer.xlsx",
    }

    for key in ["summary","per_battery","predictions","feature_tuning","model_tuning"]:
        combined[key].to_csv(paths[key], index=False)
    combined["selected"].to_csv(paths["selected_features"], index=False)
    interoperability_audit.to_csv(paths["interoperability_audit"], index=False)

    with pd.ExcelWriter(paths["workbook"], engine="openpyxl") as writer:
        interoperability_audit.to_excel(writer, sheet_name="Interoperability", index=False)
        for key, data in combined.items():
            data.to_excel(writer, sheet_name=key[:31], index=False)

    return paths


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("outputs/constant_dataset/constant_observations.csv"))
    parser.add_argument("--features", type=Path, default=Path("outputs/constant_dataset/constant_strict_feature_names.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/constant_transfer"))
    args = parser.parse_args()
    for name, path in run(args.data, args.features, args.output_dir).items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
