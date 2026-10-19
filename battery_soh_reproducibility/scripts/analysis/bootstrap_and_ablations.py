"""Battery-level bootstrap confidence intervals and strict feature-family ablations.

The bootstrap resamples whole batteries, not cycles. Ablations are run on the
corrected constant-protocol strict electrical-plus-age HGB and retrain the
nested feature-selection/model-selection pipeline after each feature family is
removed. Feature-family definitions overlap and are interpreted as sensitivity
analyses rather than additive variance decomposition.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import time
import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error

warnings.filterwarnings("ignore", category=RuntimeWarning)

TARGET_COL = "Calculated_SOH(%)"
BATTERY_COL = "battery_id"
SOURCE_COL = "source_dataset"
EXPECTED_N = 4590
EXPECTED_BATTERIES = 10
EXPECTED_STRICT_FEATURES = 335

EXPLICIT_AGE_STRICT_FEATURES = {
    "LEGACY__Cycle_Age", "LEGACY__Cycle_Index", "LEGACY__Log_Cycle_Age"
}
AGE_COLUMNS = ["Cycle_Age", "Log_Cycle_Age"]
MIN_SOURCE_COVERAGE = 0.95
CORR_THRESHOLDS = [0.90, 0.93, 0.95]
TOP_K_OPTIONS = [8, 12, 16]
HGB_CONFIGS = {
    "H0": {"learning_rate": 0.05, "max_iter": 300, "max_leaf_nodes": 15,
           "min_samples_leaf": 5, "l2_regularization": 1.0},
    "H1": {"learning_rate": 0.04, "max_iter": 500, "max_leaf_nodes": 15,
           "min_samples_leaf": 10, "l2_regularization": 2.0},
    "H2": {"learning_rate": 0.03, "max_iter": 700, "max_leaf_nodes": 31,
           "min_samples_leaf": 10, "l2_regularization": 3.0},
}
FEATURE_TUNING_MODEL = "H1"
RANDOM_STATE = 42
BOOTSTRAP_REPLICATES = 20000
BOOTSTRAP_SEED = 20261006


def bootstrap_macro_metric(
    values,
    n_boot=BOOTSTRAP_REPLICATES,
    seed=BOOTSTRAP_SEED,
):
    """
    Battery-level percentile bootstrap.

    Input is already one scalar metric per battery.
    """
    values = np.asarray(
        values,
        dtype=float,
    )

    values = values[
        np.isfinite(
            values
        )
    ]

    if len(
        values
    ) < 2:
        return {
            "N_Batteries": int(
                len(
                    values
                )
            ),
            "Estimate": float(
                np.mean(
                    values
                )
            )
            if len(
                values
            )
            else np.nan,
            "CI95_Lower": np.nan,
            "CI95_Upper": np.nan,
        }

    rng = np.random.default_rng(
        seed
    )

    indices = rng.integers(
        0,
        len(
            values
        ),
        size=(
            n_boot,
            len(
                values
            ),
        ),
    )

    boot = values[
        indices
    ].mean(
        axis=1
    )

    return {
        "N_Batteries": int(
            len(
                values
            )
        ),
        "Estimate": float(
            np.mean(
                values
            )
        ),
        "CI95_Lower": float(
            np.percentile(
                boot,
                2.5,
            )
        ),
        "CI95_Upper": float(
            np.percentile(
                boot,
                97.5,
            )
        ),
    }

def add_bootstrap_rows(
    rows,
    frame,
    study,
    group_columns,
    label_columns,
    rmse_col="RMSE_Raw",
    mae_col="MAE_Raw",
):
    """
    Add one RMSE and one MAE bootstrap row per requested model grouping.
    """
    grouped = frame.groupby(
        group_columns,
        dropna=False,
        sort=True,
    )

    for key, group in grouped:

        if not isinstance(
            key,
            tuple,
        ):
            key = (
                key,
            )

        labels = dict(
            zip(
                label_columns,
                key,
            )
        )

        for metric_name, metric_col in [
            (
                "Macro_RMSE",
                rmse_col,
            ),
            (
                "Macro_MAE",
                mae_col,
            ),
        ]:
            result = bootstrap_macro_metric(
                pd.to_numeric(
                    group[
                        metric_col
                    ],
                    errors="coerce",
                ).to_numpy()
            )

            rows.append(
                {
                    "Study": study,
                    **labels,
                    "Metric": metric_name,
                    "Bootstrap_Replicates": BOOTSTRAP_REPLICATES,
                    **result,
                }
            )

def rmse(
    y,
    pred,
):
    return float(
        np.sqrt(
            mean_squared_error(
                y,
                pred,
            )
        )
    )

def mae(
    y,
    pred,
):
    return float(
        mean_absolute_error(
            y,
            pred,
        )
    )

def numeric_frame(
    frame,
    columns,
):
    return (
        frame.loc[
            :,
            columns,
        ]
        .apply(
            pd.to_numeric,
            errors="coerce",
        )
        .replace(
            [
                np.inf,
                -np.inf,
            ],
            np.nan,
        )
    )

def equal_battery_weights(
    frame,
):
    counts = frame[
        BATTERY_COL
    ].value_counts()

    weights = frame[
        BATTERY_COL
    ].map(
        lambda b: 1.0
        / counts.loc[
            b
        ]
    ).astype(
        float
    ).to_numpy()

    if np.mean(
        weights
    ) > 0:
        weights = (
            weights
            / np.mean(
                weights
            )
        )

    return weights

def is_voltage_shape_feature(
    feature,
):
    """
    Discharge voltage level/shape/statistics.

    Dynamic-response slope features are excluded from this family so they
    can be tested separately.
    """
    feature = str(
        feature
    )

    voltage_signal = (
        "Discharge_Voltage"
        in feature
        or "DeltaV_discharge"
        in feature
        or "Discharge_startVoltage"
        in feature
        or "Discharge_endVoltage"
        in feature
    )

    dynamic_signal = (
        "dVdt"
        in feature
        or "CC_slope"
        in feature
    )

    return (
        voltage_signal
        and not dynamic_signal
    )

def is_dynamic_response_feature(
    feature,
):
    """
    Resistance / voltage-rate / CC discharge slope family.
    """
    feature = str(
        feature
    )

    return (
        "dVdt"
        in feature
        or "Internal_Resistance"
        in feature
        or "Resistance"
        in feature
        or "Discharge_Step7_CC_slope"
        in feature
    )

def is_temporal_trend_feature(
    feature,
):
    """
    Target-free trailing-history / temporal-trend transforms.

    This family intentionally includes rolling, exponentially weighted,
    lag-difference, and short-window slope transforms.

    Early-baseline current-cycle delta/ratio/pct transforms are retained;
    the ablation targets explicit temporal trend/history operators.
    """
    feature = str(
        feature
    )

    tokens = [
        "LEGACYH__",
        "__pct_lag1_delta",
        "__pct_slope5",
        "__pct_ewm5",
        "__rolling_mean_3",
        "__rolling_mean_5",
        "__rolling_std_5",
        "__rolling_slope_5",
        "__ewm_5",
        "__lag1_delta",
        "__per_cycle_delta",
    ]

    return any(
        token
        in feature
        for token
        in tokens
    )

def training_source_filter(
    train_frame,
    columns,
):
    X = numeric_frame(
        train_frame,
        columns,
    )

    sources = sorted(
        train_frame[
            SOURCE_COL
        ].dropna().unique()
    )

    keep = []

    for feature in columns:
        eligible = True

        for source in sources:
            mask = train_frame[
                SOURCE_COL
            ].eq(
                source
            )

            coverage = float(
                X.loc[
                    mask,
                    feature,
                ]
                .notna()
                .mean()
            )

            if (
                not np.isfinite(
                    coverage
                )
                or coverage
                < MIN_SOURCE_COVERAGE
            ):
                eligible = False

        if (
            X[
                feature
            ].nunique(
                dropna=True
            )
            <= 1
        ):
            eligible = False

        if eligible:
            keep.append(
                feature
            )

    return keep

def prepare_selector(
    train_frame,
    candidate_columns,
):
    eligible = training_source_filter(
        train_frame,
        candidate_columns,
    )

    if not eligible:
        raise RuntimeError(
            "No features remained after training-only coverage filtering."
        )

    X_raw = numeric_frame(
        train_frame,
        eligible,
    )

    imputer = SimpleImputer(
        strategy="median",
    )

    X = pd.DataFrame(
        imputer.fit_transform(
            X_raw
        ),
        columns=eligible,
        index=train_frame.index,
    )

    ranked_X = X.rank(
        method="average"
    )

    ranked_y = pd.Series(
        pd.to_numeric(
            train_frame[
                TARGET_COL
            ],
            errors="coerce",
        ).to_numpy(),
        index=train_frame.index,
    ).rank(
        method="average"
    )

    scores = (
        ranked_X.corrwith(
            ranked_y
        )
        .abs()
        .replace(
            [
                np.inf,
                -np.inf,
            ],
            np.nan,
        )
        .fillna(
            0.0
        )
    )

    order = (
        scores.sort_values(
            ascending=False,
            kind="mergesort",
        )
        .index
        .tolist()
    )

    correlations = (
        ranked_X[
            order
        ]
        .corr()
        .abs()
    )

    return {
        "eligible": eligible,
        "scores": scores,
        "order": order,
        "correlations": correlations,
    }

def choose_features(
    selector,
    top_k,
    corr_threshold,
):
    selected = []

    for feature in selector[
        "order"
    ]:
        if len(
            selected
        ) >= int(
            top_k
        ):
            break

        if not selected:
            selected.append(
                feature
            )
            continue

        corr = selector[
            "correlations"
        ].loc[
            feature,
            selected,
        ]

        if bool(
            (
                corr
                >= float(
                    corr_threshold
                )
            ).any()
        ):
            continue

        selected.append(
            feature
        )

    if not selected:
        raise RuntimeError(
            "Feature selection returned zero features."
        )

    return selected

def model_columns(
    selected_electrical,
):
    columns = list(
        selected_electrical
    )

    # Step 9 ablations use the strongest strict Step 6 formulation:
    # strict electrical + explicit age.
    for age_col in AGE_COLUMNS:
        if age_col not in columns:
            columns.append(
                age_col
            )

    return columns

def fit_model(
    train_frame,
    selected_electrical,
    config_name,
):
    columns = model_columns(
        selected_electrical
    )

    X_raw = numeric_frame(
        train_frame,
        columns,
    )

    imputer = SimpleImputer(
        strategy="median",
    )

    X = imputer.fit_transform(
        X_raw
    )

    model = HistGradientBoostingRegressor(
        loss="squared_error",
        early_stopping=False,
        random_state=RANDOM_STATE,
        **HGB_CONFIGS[
            config_name
        ],
    )

    model.fit(
        X,
        pd.to_numeric(
            train_frame[
                TARGET_COL
            ],
            errors="coerce",
        ).to_numpy(),
        sample_weight=equal_battery_weights(
            train_frame
        ),
    )

    return (
        model,
        imputer,
        columns,
    )

def predict_model(
    model,
    imputer,
    columns,
    test_frame,
):
    X = imputer.transform(
        numeric_frame(
            test_frame,
            columns,
        )
    )

    return model.predict(
        X
    )

def prepare_inner(
    outer_train,
    candidate_columns,
):
    prepared = {}

    for heldout in sorted(
        outer_train[
            BATTERY_COL
        ].unique()
    ):
        inner_train = outer_train.loc[
            outer_train[
                BATTERY_COL
            ]
            != heldout
        ].copy()

        inner_valid = outer_train.loc[
            outer_train[
                BATTERY_COL
            ]
            == heldout
        ].copy()

        prepared[
            heldout
        ] = {
            "train": inner_train,
            "valid": inner_valid,
            "selector": prepare_selector(
                inner_train,
                candidate_columns,
            ),
        }

    return prepared

def tune_feature_settings(
    outer_train,
    candidate_columns,
):
    inner = prepare_inner(
        outer_train,
        candidate_columns,
    )

    rows = []

    for corr_threshold in CORR_THRESHOLDS:
        for top_k in TOP_K_OPTIONS:
            rmses = []

            for heldout, split in inner.items():

                selected = choose_features(
                    split[
                        "selector"
                    ],
                    top_k,
                    corr_threshold,
                )

                model, imputer, columns = fit_model(
                    split[
                        "train"
                    ],
                    selected,
                    FEATURE_TUNING_MODEL,
                )

                pred = predict_model(
                    model,
                    imputer,
                    columns,
                    split[
                        "valid"
                    ],
                )

                y = pd.to_numeric(
                    split[
                        "valid"
                    ][
                        TARGET_COL
                    ],
                    errors="coerce",
                ).to_numpy()

                rmses.append(
                    rmse(
                        y,
                        pred,
                    )
                )

            rows.append(
                {
                    "Corr_Threshold": corr_threshold,
                    "Top_K": top_k,
                    "Inner_Macro_RMSE": float(
                        np.mean(
                            rmses
                        )
                    ),
                }
            )

    table = (
        pd.DataFrame(
            rows
        )
        .sort_values(
            [
                "Inner_Macro_RMSE",
                "Top_K",
                "Corr_Threshold",
            ],
            kind="mergesort",
        )
        .reset_index(
            drop=True
        )
    )

    best = table.iloc[
        0
    ]

    return (
        float(
            best[
                "Corr_Threshold"
            ]
        ),
        int(
            best[
                "Top_K"
            ]
        ),
        table,
        inner,
    )

def tune_hgb(
    inner,
    corr_threshold,
    top_k,
):
    rows = []

    for config_name in HGB_CONFIGS:
        rmses = []

        for heldout, split in inner.items():

            selected = choose_features(
                split[
                    "selector"
                ],
                top_k,
                corr_threshold,
            )

            model, imputer, columns = fit_model(
                split[
                    "train"
                ],
                selected,
                config_name,
            )

            pred = predict_model(
                model,
                imputer,
                columns,
                split[
                    "valid"
                ],
            )

            y = pd.to_numeric(
                split[
                    "valid"
                ][
                    TARGET_COL
                ],
                errors="coerce",
            ).to_numpy()

            rmses.append(
                rmse(
                    y,
                    pred,
                )
            )

        rows.append(
            {
                "HGB_Config": config_name,
                "Inner_Macro_RMSE": float(
                    np.mean(
                        rmses
                    )
                ),
            }
        )

    table = (
        pd.DataFrame(
            rows
        )
        .sort_values(
            [
                "Inner_Macro_RMSE",
                "HGB_Config",
            ],
            kind="mergesort",
        )
        .reset_index(
            drop=True
        )
    )

    return (
        str(
            table.iloc[
                0
            ][
                "HGB_Config"
            ]
        ),
        table,
    )

def run_one_ablation(
    frame,
    full_electrical_candidates,
    ablation_name,
    removal_rule,
):
    removed = [
        feature
        for feature in full_electrical_candidates
        if removal_rule(
            feature
        )
    ]

    candidates = [
        feature
        for feature in full_electrical_candidates
        if feature not in set(
            removed
        )
    ]

    print(
        "\n"
        + "=" * 100
    )

    print(
        ablation_name
    )

    print(
        "=" * 100
    )

    print(
        f"Removed features: {len(removed)}"
    )

    print(
        f"Remaining strict electrical candidates: {len(candidates)}"
    )

    fold_rows = []
    selected_rows = []
    feature_tuning_rows = []
    model_tuning_rows = []

    batteries = sorted(
        frame[
            BATTERY_COL
        ].unique()
    )

    for fold_number, heldout in enumerate(
        batteries,
        start=1,
    ):
        start = time.time()

        outer_train = frame.loc[
            frame[
                BATTERY_COL
            ]
            != heldout
        ].copy()

        outer_test = frame.loc[
            frame[
                BATTERY_COL
            ]
            == heldout
        ].copy()

        (
            best_corr,
            best_k,
            feature_tuning,
            inner,
        ) = tune_feature_settings(
            outer_train,
            candidates,
        )

        best_config, model_tuning = tune_hgb(
            inner,
            best_corr,
            best_k,
        )

        selector = prepare_selector(
            outer_train,
            candidates,
        )

        selected = choose_features(
            selector,
            best_k,
            best_corr,
        )

        model, imputer, columns = fit_model(
            outer_train,
            selected,
            best_config,
        )

        pred = predict_model(
            model,
            imputer,
            columns,
            outer_test,
        )

        y = pd.to_numeric(
            outer_test[
                TARGET_COL
            ],
            errors="coerce",
        ).to_numpy()

        fold_rows.append(
            {
                "Ablation": ablation_name,
                "Holdout_Battery": heldout,
                "Source": str(
                    outer_test[
                        SOURCE_COL
                    ].iloc[
                        0
                    ]
                ),
                "N_Test": int(
                    len(
                        outer_test
                    )
                ),
                "Removed_Family_Features": int(
                    len(
                        removed
                    )
                ),
                "Remaining_Candidate_Features": int(
                    len(
                        candidates
                    )
                ),
                "Selected_Electrical_Features": int(
                    len(
                        selected
                    )
                ),
                "Chosen_Corr_Threshold": best_corr,
                "Chosen_Top_K": best_k,
                "Chosen_HGB_Config": best_config,
                "RMSE": rmse(
                    y,
                    pred,
                ),
                "MAE": mae(
                    y,
                    pred,
                ),
                "Runtime_Seconds": float(
                    time.time()
                    - start
                ),
            }
        )

        for rank, feature in enumerate(
            selected,
            start=1,
        ):
            selected_rows.append(
                {
                    "Ablation": ablation_name,
                    "Outer_Holdout": heldout,
                    "Selected_Rank": rank,
                    "Feature": feature,
                    "Training_Target_SpearmanAbs": float(
                        selector[
                            "scores"
                        ].loc[
                            feature
                        ]
                    ),
                }
            )

        feature_tuning = feature_tuning.copy()

        feature_tuning.insert(
            0,
            "Ablation",
            ablation_name,
        )

        feature_tuning.insert(
            1,
            "Outer_Holdout",
            heldout,
        )

        feature_tuning_rows.append(
            feature_tuning
        )

        model_tuning = model_tuning.copy()

        model_tuning.insert(
            0,
            "Ablation",
            ablation_name,
        )

        model_tuning.insert(
            1,
            "Outer_Holdout",
            heldout,
        )

        model_tuning_rows.append(
            model_tuning
        )

        print(
            f"  {fold_number}/{len(batteries)} "
            f"{heldout}: RMSE={fold_rows[-1]['RMSE']:.4f}"
        )

    folds = pd.DataFrame(
        fold_rows
    )

    macro_rmse = float(
        folds[
            "RMSE"
        ].mean()
    )

    macro_mae = float(
        folds[
            "MAE"
        ].mean()
    )

    return {
        "ablation_name": ablation_name,
        "removed_features": removed,
        "remaining_features": candidates,
        "folds": folds,
        "macro_rmse": macro_rmse,
        "macro_mae": macro_mae,
        "selected": pd.DataFrame(
            selected_rows
        ),
        "feature_tuning": pd.concat(
            feature_tuning_rows,
            ignore_index=True,
        ),
        "model_tuning": pd.concat(
            model_tuning_rows,
            ignore_index=True,
        ),
    }


ABLATION_RULES = {
    "NO_VOLTAGE_SHAPE": is_voltage_shape_feature,
    "NO_DYNAMIC_RESPONSE": is_dynamic_response_feature,
    "NO_TEMPORAL_TREND_TRANSFORMS": is_temporal_trend_feature,
}


def build_bootstrap_table(
    variable_lobo_path: Path,
    constant_lobo_path: Path,
    transfer_path: Path,
    partial_path: Path,
) -> pd.DataFrame:
    """Build manuscript battery-bootstrap intervals from per-battery metrics."""
    rows: list[dict] = []

    step4 = pd.read_csv(variable_lobo_path)
    add_bootstrap_rows(
        rows, step4, "VARIABLE_LOBO", ["Formulation"], ["Model"]
    )

    step6 = pd.read_csv(constant_lobo_path)
    add_bootstrap_rows(
        rows, step6, "CONSTANT_LOBO", ["Formulation"], ["Model"]
    )

    step7 = pd.read_csv(transfer_path)
    add_bootstrap_rows(
        rows, step7, "CONSTANT_TRANSFER",
        ["Formulation", "Train_Source", "Test_Source"],
        ["Model", "Train_Source", "Test_Source"],
    )

    step8 = pd.read_csv(partial_path)
    add_bootstrap_rows(
        rows, step8, "CONSTANT_3P6", ["Formulation"], ["Model"]
    )
    return pd.DataFrame(rows)


def run(
    constant_data_path: Path,
    strict_feature_path: Path,
    constant_macro_path: Path,
    variable_per_battery_path: Path,
    constant_per_battery_path: Path,
    transfer_per_battery_path: Path,
    partial_per_battery_path: Path,
    output_dir: Path,
) -> dict[str, Path]:
    """Run 20,000-battery bootstrap intervals and all strict ablations."""
    output_dir.mkdir(parents=True, exist_ok=True)
    bootstrap = build_bootstrap_table(
        variable_per_battery_path, constant_per_battery_path,
        transfer_per_battery_path, partial_per_battery_path,
    )

    constant = pd.read_csv(constant_data_path)
    if len(constant) != EXPECTED_N or constant[BATTERY_COL].nunique() != EXPECTED_BATTERIES:
        raise AssertionError("Unexpected corrected constant-protocol dataset.")

    strict_features = (
        pd.read_csv(strict_feature_path)["Feature"].dropna().astype(str).tolist()
    )
    if len(strict_features) != EXPECTED_STRICT_FEATURES:
        raise AssertionError(
            f"Expected {EXPECTED_STRICT_FEATURES} strict features, found {len(strict_features)}."
        )
    electrical_candidates = [
        f for f in strict_features if f not in EXPLICIT_AGE_STRICT_FEATURES
    ]

    family_audit = pd.DataFrame([
        {
            "Feature": f,
            "Voltage_Shape": is_voltage_shape_feature(f),
            "Dynamic_Response": is_dynamic_response_feature(f),
            "Temporal_Trend_Transform": is_temporal_trend_feature(f),
        }
        for f in electrical_candidates
    ])

    results = [
        run_one_ablation(constant, electrical_candidates, name, rule)
        for name, rule in ABLATION_RULES.items()
    ]

    step6_macro = pd.read_csv(constant_macro_path)
    full_row = step6_macro.loc[
        step6_macro["Formulation"].eq("STRICT_ELECTRICAL_PLUS_AGE_HGB")
    ].iloc[0]
    no_age_row = step6_macro.loc[
        step6_macro["Formulation"].eq("STRICT_ELECTRICAL_HGB")
    ].iloc[0]

    full_rmse = float(full_row["Macro_RMSE_Raw"])
    full_mae = float(full_row["Macro_MAE_Raw"])
    summary_rows = [
        {
            "Analysis": "FULL_STRICT_ELECTRICAL_PLUS_AGE",
            "Type": "Reference", "Removed_Features": 0,
            "Macro_RMSE": full_rmse, "Macro_MAE": full_mae,
            "RMSE_Change_vs_Full_Pct": 0.0, "MAE_Change_vs_Full_Pct": 0.0,
        },
        {
            "Analysis": "NO_EXPLICIT_AGE",
            "Type": "Frozen_Constant_Age_Ablation",
            "Removed_Features": len(AGE_COLUMNS),
            "Macro_RMSE": float(no_age_row["Macro_RMSE_Raw"]),
            "Macro_MAE": float(no_age_row["Macro_MAE_Raw"]),
            "RMSE_Change_vs_Full_Pct":
                (float(no_age_row["Macro_RMSE_Raw"]) - full_rmse) / full_rmse * 100.0,
            "MAE_Change_vs_Full_Pct":
                (float(no_age_row["Macro_MAE_Raw"]) - full_mae) / full_mae * 100.0,
        },
    ]
    for result in results:
        summary_rows.append({
            "Analysis": result["ablation_name"],
            "Type": "Nested_LOBO_Drop_Family",
            "Removed_Features": len(result["removed_features"]),
            "Macro_RMSE": result["macro_rmse"],
            "Macro_MAE": result["macro_mae"],
            "RMSE_Change_vs_Full_Pct":
                (result["macro_rmse"] - full_rmse) / full_rmse * 100.0,
            "MAE_Change_vs_Full_Pct":
                (result["macro_mae"] - full_mae) / full_mae * 100.0,
        })

    summary = pd.DataFrame(summary_rows)
    folds = pd.concat([r["folds"] for r in results], ignore_index=True)
    selected = pd.concat([r["selected"] for r in results], ignore_index=True)
    feature_tuning = pd.concat([r["feature_tuning"] for r in results], ignore_index=True)
    model_tuning = pd.concat([r["model_tuning"] for r in results], ignore_index=True)
    removed = pd.DataFrame([
        {"Ablation": r["ablation_name"], "Feature": f}
        for r in results for f in r["removed_features"]
    ])

    paths = {
        "bootstrap": output_dir / "battery_bootstrap_confidence_intervals.csv",
        "ablation_summary": output_dir / "strict_ablation_summary.csv",
        "ablation_per_battery": output_dir / "strict_ablation_per_battery.csv",
        "family_audit": output_dir / "strict_ablation_family_audit.csv",
        "removed_features": output_dir / "strict_ablation_removed_features.csv",
        "selected_features": output_dir / "strict_ablation_selected_features.csv",
        "feature_tuning": output_dir / "strict_ablation_inner_feature_tuning.csv",
        "model_tuning": output_dir / "strict_ablation_inner_model_tuning.csv",
    }
    bootstrap.to_csv(paths["bootstrap"], index=False)
    summary.to_csv(paths["ablation_summary"], index=False)
    folds.to_csv(paths["ablation_per_battery"], index=False)
    family_audit.to_csv(paths["family_audit"], index=False)
    removed.to_csv(paths["removed_features"], index=False)
    selected.to_csv(paths["selected_features"], index=False)
    feature_tuning.to_csv(paths["feature_tuning"], index=False)
    model_tuning.to_csv(paths["model_tuning"], index=False)
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--constant-data", type=Path,
                        default=Path("outputs/constant_dataset/constant_observations.csv"))
    parser.add_argument("--strict-features", type=Path,
                        default=Path("outputs/constant_dataset/constant_strict_feature_names.csv"))
    parser.add_argument("--constant-macro", type=Path,
                        default=Path("outputs/constant_hgb/constant_hgb_macro.csv"))
    parser.add_argument("--variable-per-battery", type=Path,
                        default=Path("outputs/variable_hgb/variable_hgb_per_battery.csv"))
    parser.add_argument("--constant-per-battery", type=Path,
                        default=Path("outputs/constant_hgb/constant_hgb_per_battery.csv"))
    parser.add_argument("--transfer-per-battery", type=Path,
                        default=Path("outputs/constant_transfer/constant_transfer_per_battery.csv"))
    parser.add_argument("--partial-per-battery", type=Path,
                        default=Path("outputs/constant_3p6/constant_3p6_per_battery.csv"))
    parser.add_argument("--output-dir", type=Path,
                        default=Path("outputs/statistical_analysis"))
    args = parser.parse_args()
    for name, path in run(
        args.constant_data, args.strict_features, args.constant_macro,
        args.variable_per_battery, args.constant_per_battery,
        args.transfer_per_battery, args.partial_per_battery,
        args.output_dir,
    ).items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
