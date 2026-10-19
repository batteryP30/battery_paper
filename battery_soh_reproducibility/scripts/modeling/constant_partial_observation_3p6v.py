"""Constant-protocol 3.6 V partial-observation sensitivity experiment.

This is the validated secondary information setting from the manuscript.
It augments the frozen strict constant-protocol table with three observables
measured only up to the first 3.6 V crossing: capacity, energy, and elapsed
time. The 3.6 V variables are not part of the strict model and are reported
separately as partial-observation-assisted information.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import tempfile
import time
import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error

from battery_soh_repro.data_pipeline import (
    CALCE_CONSTANT_BATTERIES,
    NASA_CONSTANT_BATTERIES,
    process_files,
    load_nasa_constant_measurements_targetfree,
    prepare_raw_archive,
)

warnings.filterwarnings("ignore", category=RuntimeWarning)

CALCE_CONSTANT = dict(CALCE_CONSTANT_BATTERIES)
NASA_CONSTANT = list(NASA_CONSTANT_BATTERIES)
PARTIAL_VOLTAGE_V = 3.6
PARTIAL_FEATURES = ["P36_Capacity_Ah", "P36_Energy_Wh", "P36_Time_h"]

TARGET_COL = "Calculated_SOH(%)"
BATTERY_COL = "battery_id"
SOURCE_COL = "source_dataset"
MIN_SOURCE_COVERAGE = 0.95

EXPLICIT_AGE_STRICT_FEATURES = {
    "LEGACY__Cycle_Age", "LEGACY__Cycle_Index", "LEGACY__Log_Cycle_Age"
}
AGE_COLUMNS = ["Cycle_Age", "Log_Cycle_Age"]
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
CLIP_LOW, CLIP_HIGH = 70.0, 100.0


def calce_standard_columns(
    raw,
):
    cycle_col = find_column(
        raw,
        [
            "Cycle_Index",
        ],
    )

    time_col = find_column(
        raw,
        [
            "Test_Time(s)",
            "Test Time(s)",
            "Test_Time",
        ],
    )

    voltage_col = find_column(
        raw,
        [
            "Voltage(V)",
            "Voltage",
        ],
    )

    current_col = find_column(
        raw,
        [
            "Current(A)",
            "Current",
        ],
    )

    try:
        step_col = find_column(
            raw,
            [
                "Step_Index",
            ],
        )
    except Exception:
        step_col = None

    out = pd.DataFrame(
        {
            "Cycle_Index": pd.to_numeric(
                raw[
                    cycle_col
                ],
                errors="coerce",
            ),
            "Time_s": pd.to_numeric(
                raw[
                    time_col
                ],
                errors="coerce",
            ),
            "Voltage_V": pd.to_numeric(
                raw[
                    voltage_col
                ],
                errors="coerce",
            ),
            "Signed_Current_A": pd.to_numeric(
                raw[
                    current_col
                ],
                errors="coerce",
            ),
        }
    )

    if step_col is not None:
        out[
            "Step_Index"
        ] = pd.to_numeric(
            raw[
                step_col
            ],
            errors="coerce",
        )

    else:
        out[
            "Step_Index"
        ] = np.nan

    return out.dropna(
        subset=[
            "Cycle_Index",
            "Time_s",
            "Voltage_V",
            "Signed_Current_A",
        ]
    )

def integrate_series(
    time_s,
    voltage_v,
    current_a,
):
    t = np.asarray(
        time_s,
        dtype=float,
    )

    v = np.asarray(
        voltage_v,
        dtype=float,
    )

    i = np.abs(
        np.asarray(
            current_a,
            dtype=float,
        )
    )

    valid = (
        np.isfinite(
            t
        )
        & np.isfinite(
            v
        )
        & np.isfinite(
            i
        )
    )

    t = t[
        valid
    ]

    v = v[
        valid
    ]

    i = i[
        valid
    ]

    if len(
        t
    ) < 2:
        return (
            np.nan,
            np.nan,
            np.nan,
        )

    order = np.argsort(
        t
    )

    t = t[
        order
    ]

    v = v[
        order
    ]

    i = i[
        order
    ]

    unique_t, unique_idx = np.unique(
        t,
        return_index=True,
    )

    t = t[
        unique_idx
    ]

    v = v[
        unique_idx
    ]

    i = i[
        unique_idx
    ]

    if len(
        t
    ) < 2:
        return (
            np.nan,
            np.nan,
            np.nan,
        )

    q = float(
        np.trapezoid(
            i,
            t,
        )
        / 3600.0
    )

    e = float(
        np.trapezoid(
            v * i,
            t,
        )
        / 3600.0
    )

    h = float(
        (
            np.max(
                t
            )
            - np.min(
                t
            )
        )
        / 3600.0
    )

    return (
        q,
        e,
        h,
    )

def extract_partial_and_full(
    time_s,
    voltage_v,
    current_a,
    threshold=PARTIAL_VOLTAGE_V,
):
    """
    Return:
        partial Q/E/time to first downward threshold crossing
        full Q/E/time for the selected discharge segment

    The full quantities are used only for the reviewer-facing audit.
    They are NOT predictors.
    """
    t = np.asarray(
        time_s,
        dtype=float,
    )

    v = np.asarray(
        voltage_v,
        dtype=float,
    )

    i = np.abs(
        np.asarray(
            current_a,
            dtype=float,
        )
    )

    valid = (
        np.isfinite(
            t
        )
        & np.isfinite(
            v
        )
        & np.isfinite(
            i
        )
    )

    t = t[
        valid
    ]

    v = v[
        valid
    ]

    i = i[
        valid
    ]

    if len(
        t
    ) < 2:
        return (
            np.nan,
            np.nan,
            np.nan,
            np.nan,
            np.nan,
            np.nan,
        )

    order = np.argsort(
        t
    )

    t = t[
        order
    ]

    v = v[
        order
    ]

    i = i[
        order
    ]

    unique_t, unique_idx = np.unique(
        t,
        return_index=True,
    )

    t = t[
        unique_idx
    ]

    v = v[
        unique_idx
    ]

    i = i[
        unique_idx
    ]

    full_q, full_e, full_h = integrate_series(
        t,
        v,
        i,
    )

    if (
        len(
            t
        ) < 2
        or v[
            0
        ]
        <= threshold
    ):
        return (
            np.nan,
            np.nan,
            np.nan,
            full_q,
            full_e,
            full_h,
        )

    crossings = np.flatnonzero(
        v <= threshold
    )

    if len(
        crossings
    ) == 0:
        return (
            np.nan,
            np.nan,
            np.nan,
            full_q,
            full_e,
            full_h,
        )

    j = int(
        crossings[
            0
        ]
    )

    if j == 0:
        return (
            np.nan,
            np.nan,
            np.nan,
            full_q,
            full_e,
            full_h,
        )

    t_prev = float(
        t[
            j - 1
        ]
    )

    t_next = float(
        t[
            j
        ]
    )

    v_prev = float(
        v[
            j - 1
        ]
    )

    v_next = float(
        v[
            j
        ]
    )

    i_prev = float(
        i[
            j - 1
        ]
    )

    i_next = float(
        i[
            j
        ]
    )

    denom = (
        v_prev
        - v_next
    )

    if (
        np.isfinite(
            denom
        )
        and abs(
            denom
        ) > 1e-12
    ):
        fraction = (
            v_prev
            - threshold
        ) / denom

        fraction = float(
            np.clip(
                fraction,
                0.0,
                1.0,
            )
        )

    else:
        fraction = 1.0

    t_cross = (
        t_prev
        + fraction
        * (
            t_next
            - t_prev
        )
    )

    i_cross = (
        i_prev
        + fraction
        * (
            i_next
            - i_prev
        )
    )

    t_prefix = np.concatenate(
        [
            t[
                :j
            ],
            [
                t_cross
            ],
        ]
    )

    v_prefix = np.concatenate(
        [
            v[
                :j
            ],
            [
                threshold
            ],
        ]
    )

    i_prefix = np.concatenate(
        [
            i[
                :j
            ],
            [
                i_cross
            ],
        ]
    )

    partial_q, partial_e, partial_h = integrate_series(
        t_prefix,
        v_prefix,
        i_prefix,
    )

    return (
        partial_q,
        partial_e,
        partial_h,
        full_q,
        full_e,
        full_h,
    )

def compute_calce_constant_partial(
    raw,
):
    d = calce_standard_columns(
        raw
    )

    rows = []

    for cycle, group in d.groupby(
        "Cycle_Index",
        sort=True,
    ):

        step7 = group.loc[
            group[
                "Step_Index"
            ].eq(
                7
            )
        ].copy()

        if len(
            step7
        ) >= 2:
            discharge = step7

        else:
            discharge = group.loc[
                group[
                    "Signed_Current_A"
                ]
                < -0.02
            ].copy()

        discharge = discharge.sort_values(
            "Time_s"
        )

        (
            p_q,
            p_e,
            p_h,
            f_q,
            f_e,
            f_h,
        ) = extract_partial_and_full(
            discharge[
                "Time_s"
            ],
            discharge[
                "Voltage_V"
            ],
            discharge[
                "Signed_Current_A"
            ].abs(),
        )

        rows.append(
            {
                "Cycle_Index": float(
                    cycle
                ),
                "P36_Capacity_Ah": p_q,
                "P36_Energy_Wh": p_e,
                "P36_Time_h": p_h,
                "SelectedSegment_Full_Capacity_Ah": f_q,
                "SelectedSegment_Full_Energy_Wh": f_e,
                "SelectedSegment_Full_Time_h": f_h,
            }
        )

    return pd.DataFrame(
        rows
    )

def compute_nasa_constant_partial(
    frame,
):
    rows = []

    for cycle, group in frame.groupby(
        "Cycle_Index",
        sort=True,
    ):
        (
            p_q,
            p_e,
            p_h,
            f_q,
            f_e,
            f_h,
        ) = extract_partial_and_full(
            group[
                "Time_s"
            ],
            group[
                "Voltage_V"
            ],
            group[
                "Current_A"
            ],
        )

        rows.append(
            {
                "Cycle_Index": float(
                    cycle
                ),
                "P36_Capacity_Ah": p_q,
                "P36_Energy_Wh": p_e,
                "P36_Time_h": p_h,
                "SelectedSegment_Full_Capacity_Ah": f_q,
                "SelectedSegment_Full_Energy_Wh": f_e,
                "SelectedSegment_Full_Time_h": f_h,
            }
        )

    return pd.DataFrame(
        rows
    )

def build_observation_depth_audit(
    augmented,
):
    fraction_cols = [
        "P36_CapacityFraction_of_TargetCapacity",
        "P36_CapacityFraction_of_SelectedSegment",
        "P36_EnergyFraction_of_SelectedSegment",
        "P36_TimeFraction_of_SelectedSegment",
    ]

    rows = []

    for (
        source,
        battery,
    ), group in augmented.groupby(
        [
            SOURCE_COL,
            BATTERY_COL,
        ],
        sort=True,
    ):

        for metric in fraction_cols:
            values = pd.to_numeric(
                group[
                    metric
                ],
                errors="coerce",
            )

            values = values[
                np.isfinite(
                    values
                )
            ]

            rows.append(
                {
                    "Source": source,
                    "Battery": battery,
                    "Metric": metric,
                    "N_Total": int(
                        len(
                            group
                        )
                    ),
                    "N_Valid": int(
                        len(
                            values
                        )
                    ),
                    "Coverage": float(
                        len(
                            values
                        )
                        /
                        len(
                            group
                        )
                    ),
                    "Median": (
                        float(
                            values.median()
                        )
                        if len(
                            values
                        )
                        else np.nan
                    ),
                    "Q10": (
                        float(
                            values.quantile(
                                0.10
                            )
                        )
                        if len(
                            values
                        )
                        else np.nan
                    ),
                    "Q90": (
                        float(
                            values.quantile(
                                0.90
                            )
                        )
                        if len(
                            values
                        )
                        else np.nan
                    ),
                    "Min": (
                        float(
                            values.min()
                        )
                        if len(
                            values
                        )
                        else np.nan
                    ),
                    "Max": (
                        float(
                            values.max()
                        )
                        if len(
                            values
                        )
                        else np.nan
                    ),
                }
            )

    return pd.DataFrame(
        rows
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
        ].unique()
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
            "No strict electrical features survived training-only filtering."
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
        ) >= top_k:
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
                >= corr_threshold
            ).any()
        ):
            continue

        selected.append(
            feature
        )

    return selected

def build_columns(
    selected_strict,
    include_age,
):
    columns = list(
        selected_strict
    )

    for feature in PARTIAL_FEATURES:
        if feature not in columns:
            columns.append(
                feature
            )

    if include_age:
        for feature in AGE_COLUMNS:
            if feature not in columns:
                columns.append(
                    feature
                )

    return columns

def fit_model(
    train_frame,
    selected_strict,
    config_name,
    include_age,
):
    columns = build_columns(
        selected_strict,
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

    raw = model.predict(
        X
    )

    clipped = np.clip(
        raw,
        CLIP_LOW,
        CLIP_HIGH,
    )

    return (
        raw,
        clipped,
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
        train = outer_train.loc[
            outer_train[
                BATTERY_COL
            ]
            != heldout
        ].copy()

        valid = outer_train.loc[
            outer_train[
                BATTERY_COL
            ]
            == heldout
        ].copy()

        prepared[
            heldout
        ] = {
            "train": train,
            "valid": valid,
            "selector": prepare_selector(
                train,
                candidate_columns,
            ),
        }

    return prepared

def tune_feature_settings(
    outer_train,
    candidate_columns,
    include_age,
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
                    include_age,
                )

                pred, _ = predict_model(
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
    include_age,
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
                include_age,
            )

            pred, _ = predict_model(
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

def run_partial_lobo(
    frame,
    candidate_columns,
    formulation,
    include_age,
):
    folds = []
    predictions = []
    selected_rows = []
    feature_tuning_rows = []
    model_tuning_rows = []

    batteries = sorted(
        frame[
            BATTERY_COL
        ].unique()
    )

    print(
        "\n"
        + "=" * 90
    )

    print(
        formulation
    )

    print(
        "=" * 90
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
            candidate_columns,
            include_age,
        )

        best_config, model_tuning = tune_hgb(
            inner,
            best_corr,
            best_k,
            include_age,
        )

        selector = prepare_selector(
            outer_train,
            candidate_columns,
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
            include_age,
        )

        pred_raw, pred_clip = predict_model(
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

        folds.append(
            {
                "Formulation": formulation,
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
                "Selected_Strict_Electrical": int(
                    len(
                        selected
                    )
                ),
                "Forced_Partial_Features": len(
                    PARTIAL_FEATURES
                ),
                "Explicit_Age_Features": (
                    len(
                        AGE_COLUMNS
                    )
                    if include_age
                    else 0
                ),
                "Chosen_Corr_Threshold": best_corr,
                "Chosen_Top_K": best_k,
                "Chosen_HGB_Config": best_config,
                "RMSE_Raw": rmse(
                    y,
                    pred_raw,
                ),
                "MAE_Raw": mae(
                    y,
                    pred_raw,
                ),
                "RMSE_Clipped70_100": rmse(
                    y,
                    pred_clip,
                ),
                "MAE_Clipped70_100": mae(
                    y,
                    pred_clip,
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
                    "Formulation": formulation,
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
            "Formulation",
            formulation,
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
            "Formulation",
            formulation,
        )

        model_tuning.insert(
            1,
            "Outer_Holdout",
            heldout,
        )

        model_tuning_rows.append(
            model_tuning
        )

        for (
            _,
            row,
        ), raw_pred, clip_pred in zip(
            outer_test.iterrows(),
            pred_raw,
            pred_clip,
        ):
            predictions.append(
                {
                    "Formulation": formulation,
                    BATTERY_COL: row[
                        BATTERY_COL
                    ],
                    SOURCE_COL: row[
                        SOURCE_COL
                    ],
                    "Cycle_Index": row[
                        "Cycle_Index"
                    ],
                    TARGET_COL: row[
                        TARGET_COL
                    ],
                    "Prediction_Raw": float(
                        raw_pred
                    ),
                    "Prediction_Clipped70_100": float(
                        clip_pred
                    ),
                }
            )

        print(
            f"  {fold_number}/{len(batteries)} "
            f"{heldout}: "
            f"RMSE={folds[-1]['RMSE_Raw']:.4f}"
        )

    folds = pd.DataFrame(
        folds
    )

    macro = pd.DataFrame(
        [
            {
                "Formulation": formulation,
                "Batteries": int(
                    folds[
                        "Holdout_Battery"
                    ].nunique()
                ),
                "Macro_RMSE_Raw": float(
                    folds[
                        "RMSE_Raw"
                    ].mean()
                ),
                "Macro_MAE_Raw": float(
                    folds[
                        "MAE_Raw"
                    ].mean()
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

    return {
        "folds": folds,
        "macro": macro,
        "predictions": pd.DataFrame(
            predictions
        ),
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


def _standardize_nasa_partial(raw: pd.DataFrame) -> pd.DataFrame:
    """Convert the target-free NASA constant loader to Step-8 column names."""
    return pd.DataFrame({
        "Cycle_Index": pd.to_numeric(raw["Cycle_Index"], errors="coerce"),
        "Time_s": pd.to_numeric(raw["Test_Time(s)"], errors="coerce"),
        "Voltage_V": pd.to_numeric(raw["Voltage(V)"], errors="coerce"),
        "Current_A": pd.to_numeric(raw["Current(A)"], errors="coerce").abs(),
    }).dropna().sort_values(["Cycle_Index", "Time_s"]).reset_index(drop=True)


def build_constant_augmented(
    manifest: dict[str, dict],
    frozen_data_path: Path,
) -> pd.DataFrame:
    """Rebuild 3.6 V observables from raw data and merge to frozen Step-5 rows."""
    frozen = pd.read_csv(frozen_data_path)
    partial_parts: list[pd.DataFrame] = []

    for battery, sheets in CALCE_CONSTANT.items():
        raw = process_files(manifest[battery]["folder"], sheets)
        partial = compute_calce_constant_partial(raw)
        partial[BATTERY_COL] = battery
        partial_parts.append(partial)

    for battery in NASA_CONSTANT:
        raw = load_nasa_constant_measurements_targetfree(
            manifest[battery]["csv_path"], battery
        )
        partial = compute_nasa_constant_partial(_standardize_nasa_partial(raw))
        partial[BATTERY_COL] = battery
        partial_parts.append(partial)

    partial_all = pd.concat(partial_parts, ignore_index=True)
    frozen["_cycle_key"] = pd.to_numeric(
        frozen["Cycle_Index"], errors="coerce"
    ).round(6)
    partial_all["_cycle_key"] = pd.to_numeric(
        partial_all["Cycle_Index"], errors="coerce"
    ).round(6)

    merge_cols = [
        BATTERY_COL, "_cycle_key", *PARTIAL_FEATURES,
        "SelectedSegment_Full_Capacity_Ah",
        "SelectedSegment_Full_Energy_Wh",
        "SelectedSegment_Full_Time_h",
    ]
    augmented = frozen.merge(
        partial_all[merge_cols],
        on=[BATTERY_COL, "_cycle_key"],
        how="left",
        validate="one_to_one",
    ).drop(columns="_cycle_key")

    augmented["P36_CapacityFraction_of_TargetCapacity"] = (
        pd.to_numeric(augmented["P36_Capacity_Ah"], errors="coerce")
        / pd.to_numeric(augmented["Measured_Discharge_Capacity_Ah"], errors="coerce")
    )
    augmented["P36_CapacityFraction_of_SelectedSegment"] = (
        pd.to_numeric(augmented["P36_Capacity_Ah"], errors="coerce")
        / pd.to_numeric(augmented["SelectedSegment_Full_Capacity_Ah"], errors="coerce")
    )
    augmented["P36_EnergyFraction_of_SelectedSegment"] = (
        pd.to_numeric(augmented["P36_Energy_Wh"], errors="coerce")
        / pd.to_numeric(augmented["SelectedSegment_Full_Energy_Wh"], errors="coerce")
    )
    augmented["P36_TimeFraction_of_SelectedSegment"] = (
        pd.to_numeric(augmented["P36_Time_h"], errors="coerce")
        / pd.to_numeric(augmented["SelectedSegment_Full_Time_h"], errors="coerce")
    )
    return augmented


def run(
    raw_zip: Path,
    constant_data_path: Path,
    strict_feature_path: Path,
    strict_macro_path: Path | None,
    output_dir: Path,
    work_dir: Path | None = None,
) -> dict[str, Path]:
    """Run the complete constant-only 3.6 V sensitivity analysis."""
    output_dir.mkdir(parents=True, exist_ok=True)
    scratch = work_dir or (output_dir / "_raw_extract")
    manifest, raw_validation = prepare_raw_archive(raw_zip, scratch)

    augmented = build_constant_augmented(manifest, constant_data_path)
    if len(augmented) != 4590 or augmented[BATTERY_COL].nunique() != 10:
        raise AssertionError("Unexpected constant dataset shape in 3.6 V analysis.")

    strict_features = (
        pd.read_csv(strict_feature_path)["Feature"].dropna().astype(str).tolist()
    )
    if len(strict_features) != 335:
        raise AssertionError(f"Expected 335 strict features, found {len(strict_features)}.")

    coverage_rows = []
    for battery, group in augmented.groupby(BATTERY_COL, sort=True):
        source = str(group[SOURCE_COL].iloc[0])
        complete = group[PARTIAL_FEATURES].notna().all(axis=1)
        coverage_rows.append({
            "Battery": battery, "Source": source, "N": len(group),
            "N_With_All_P36": int(complete.sum()),
            "Coverage": float(complete.mean()),
        })
    coverage = pd.DataFrame(coverage_rows)
    overall_coverage = float(augmented[PARTIAL_FEATURES].notna().all(axis=1).mean())
    if overall_coverage < 0.99:
        raise RuntimeError(
            f"Constant 3.6 V coverage is {overall_coverage:.3%}; expected approximately 99%."
        )

    depth = build_observation_depth_audit(augmented)

    electrical_candidates = [
        f for f in strict_features
        if f not in EXPLICIT_AGE_STRICT_FEATURES and f in augmented.columns
    ]

    results = [
        run_partial_lobo(
            augmented, electrical_candidates,
            "STRICT_ELECTRICAL_PLUS_3P6_HGB", include_age=False,
        ),
        run_partial_lobo(
            augmented, electrical_candidates,
            "STRICT_ELECTRICAL_PLUS_AGE_PLUS_3P6_HGB", include_age=True,
        ),
    ]

    folds = pd.concat([r["folds"] for r in results], ignore_index=True)
    predictions = pd.concat([r["predictions"] for r in results], ignore_index=True)
    selected = pd.concat([r["selected"] for r in results], ignore_index=True)
    feature_tuning = pd.concat([r["feature_tuning"] for r in results], ignore_index=True)
    model_tuning = pd.concat([r["model_tuning"] for r in results], ignore_index=True)

    macro = pd.concat([r["macro"] for r in results], ignore_index=True)

    comparison = macro.copy()
    if strict_macro_path is not None and strict_macro_path.exists():
        strict_macro = pd.read_csv(strict_macro_path)
        keep = strict_macro[
            ["Formulation", "Batteries", "Macro_RMSE_Raw", "Macro_MAE_Raw",
             "Macro_RMSE_Clipped70_100", "Macro_MAE_Clipped70_100"]
        ]
        comparison = pd.concat([keep, macro], ignore_index=True, sort=False)

    paths = {
        "augmented": output_dir / "constant_3p6_augmented.csv",
        "coverage": output_dir / "constant_3p6_coverage.csv",
        "observation_depth": output_dir / "constant_3p6_observation_depth.csv",
        "macro": output_dir / "constant_3p6_macro.csv",
        "per_battery": output_dir / "constant_3p6_per_battery.csv",
        "predictions": output_dir / "constant_3p6_predictions.csv",
        "selected_features": output_dir / "constant_3p6_selected_features.csv",
        "feature_tuning": output_dir / "constant_3p6_inner_feature_tuning.csv",
        "model_tuning": output_dir / "constant_3p6_inner_model_tuning.csv",
        "comparison": output_dir / "constant_3p6_strict_comparison.csv",
        "raw_validation": output_dir / "raw_data_validation.csv",
    }
    augmented.to_csv(paths["augmented"], index=False)
    coverage.to_csv(paths["coverage"], index=False)
    depth.to_csv(paths["observation_depth"], index=False)
    macro.to_csv(paths["macro"], index=False)
    folds.to_csv(paths["per_battery"], index=False)
    predictions.to_csv(paths["predictions"], index=False)
    selected.to_csv(paths["selected_features"], index=False)
    feature_tuning.to_csv(paths["feature_tuning"], index=False)
    model_tuning.to_csv(paths["model_tuning"], index=False)
    comparison.to_csv(paths["comparison"], index=False)
    raw_validation.to_csv(paths["raw_validation"], index=False)
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-zip", type=Path, required=True)
    parser.add_argument("--constant-data", type=Path,
                        default=Path("outputs/constant_dataset/constant_observations.csv"))
    parser.add_argument("--strict-features", type=Path,
                        default=Path("outputs/constant_dataset/constant_strict_feature_names.csv"))
    parser.add_argument("--strict-macro", type=Path,
                        default=Path("outputs/constant_hgb/constant_hgb_macro.csv"))
    parser.add_argument("--output-dir", type=Path,
                        default=Path("outputs/constant_3p6"))
    parser.add_argument("--work-dir", type=Path)
    args = parser.parse_args()
    macro = args.strict_macro if args.strict_macro.exists() else None
    for name, path in run(
        args.raw_zip, args.constant_data, args.strict_features,
        macro, args.output_dir, args.work_dir
    ).items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
