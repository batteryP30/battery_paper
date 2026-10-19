"""Raw-measurement robustness analysis for the frozen constant HGB model.

Raw voltage/current measurements are perturbed before feature extraction.
The Step-6 selected features and HGB configuration are frozen for each held-out
battery, the clean SoH target is never recomputed from perturbed current, and
raw clean reconstruction must reproduce the frozen Step-6 fold before a
perturbation result is accepted.
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

from battery_soh_repro.data_pipeline import (
    CALCE_CONSTANT_BATTERIES,
    NASA_CONSTANT_BATTERIES,
    process_files,
    load_nasa_constant_measurements_targetfree,
    build_cycle_features_run2,
    add_dataset_identity,
    add_legacy_history_features,
    add_run2_invariant_features,
    add_past_only_features,
    add_joint_legacy_aliases,
    prepare_raw_archive,
)

warnings.filterwarnings("ignore", category=RuntimeWarning)

TARGET_COL = "Calculated_SOH(%)"
GROUP_COL = "battery_id"
CYCLE_COL = "Cycle_Index"
CONSTANT_PROTOCOL = 1
AGE_COLUMNS = ["Cycle_Age", "Log_Cycle_Age"]
RANDOM_STATE = 42
REBUILT_CLEAN_RMSE_TOLERANCE = 1e-3

EXPECTED_BATTERIES = [
    "B0005", "B0006", "B0007", "B0018",
    "CS2_33", "CS2_34", "CS2_35", "CS2_36", "CS2_37", "CS2_38",
]

HGB_CONFIGS = {
    "H0": {"learning_rate": 0.05, "max_iter": 300, "max_leaf_nodes": 15,
           "min_samples_leaf": 5, "l2_regularization": 1.0},
    "H1": {"learning_rate": 0.04, "max_iter": 500, "max_leaf_nodes": 15,
           "min_samples_leaf": 10, "l2_regularization": 2.0},
    "H2": {"learning_rate": 0.03, "max_iter": 700, "max_leaf_nodes": 31,
           "min_samples_leaf": 10, "l2_regularization": 3.0},
}

# Step 10 + Step 10B final systematic scenarios.
SYSTEMATIC_SCENARIOS = {
    "V_OFFSET_NEG_1mV": {"voltage_offset_V": -0.001, "current_gain": 1.0},
    "V_OFFSET_POS_1mV": {"voltage_offset_V":  0.001, "current_gain": 1.0},
    "V_OFFSET_NEG_5mV": {"voltage_offset_V": -0.005, "current_gain": 1.0},
    "V_OFFSET_POS_5mV": {"voltage_offset_V":  0.005, "current_gain": 1.0},
    "V_OFFSET_NEG_10mV": {"voltage_offset_V": -0.010, "current_gain": 1.0},
    "V_OFFSET_POS_10mV": {"voltage_offset_V":  0.010, "current_gain": 1.0},
    "I_GAIN_NEG_0p5pct": {"voltage_offset_V": 0.0, "current_gain": 0.995},
    "I_GAIN_POS_0p5pct": {"voltage_offset_V": 0.0, "current_gain": 1.005},
    "I_GAIN_NEG_1p0pct": {"voltage_offset_V": 0.0, "current_gain": 0.990},
    "I_GAIN_POS_1p0pct": {"voltage_offset_V": 0.0, "current_gain": 1.010},
    "COMBINED_NEG_10mV_NEG_1pct": {"voltage_offset_V": -0.010, "current_gain": 0.990},
    "COMBINED_POS_10mV_POS_1pct": {"voltage_offset_V": 0.010, "current_gain": 1.010},
}

RANDOM_SCENARIOS = {
    "RANDOM_1mV_0p1pct": {
        "voltage_sigma_V": 0.001, "current_sigma_fraction": 0.001, "replicates": 20
    },
    "RANDOM_5mV_0p5pct": {
        "voltage_sigma_V": 0.005, "current_sigma_fraction": 0.005, "replicates": 20
    },
}


def load_clean_raw_battery(
    manifest,
    battery_id,
):
    calce_map = dict(
        CALCE_CONSTANT_BATTERIES
    )

    if battery_id in calce_map:
        raw = process_files(
            str(
                manifest[
                    battery_id
                ][
                    "folder"
                ]
            ),
            calce_map[
                battery_id
            ],
        )

        raw[
            GROUP_COL
        ] = battery_id

        return (
            raw,
            "CALCE",
        )

    if battery_id in NASA_CONSTANT_BATTERIES:
        raw = load_nasa_constant_measurements_targetfree(
            manifest[
                battery_id
            ][
                "csv_path"
            ],
            battery_id,
        )

        raw[
            GROUP_COL
        ] = battery_id

        return (
            raw,
            "NASA",
        )

    raise KeyError(
        f"Unknown constant battery: {battery_id}"
    )

def characteristic_nonzero_current(
    raw,
):
    current = pd.to_numeric(
        raw[
            "Current(A)"
        ],
        errors="coerce",
    ).to_numpy(
        dtype=float
    )

    values = np.abs(
        current[
            np.isfinite(
                current
            )
            & (
                np.abs(
                    current
                )
                > 1e-9
            )
        ]
    )

    if len(
        values
    ) == 0:
        raise RuntimeError(
            "Could not determine characteristic non-zero current."
        )

    return float(
        np.median(
            values
        )
    )

def apply_systematic_perturbation(
    raw,
    voltage_offset_V,
    current_gain,
):
    """
    Apply systematic sensor calibration errors to raw measurements.

    Exact current zeros remain zero and current polarity is preserved.
    """
    out = raw.copy()

    voltage = pd.to_numeric(
        out[
            "Voltage(V)"
        ],
        errors="coerce",
    ).to_numpy(
        dtype=float
    )

    current = pd.to_numeric(
        out[
            "Current(A)"
        ],
        errors="coerce",
    ).to_numpy(
        dtype=float
    )

    finite_v = np.isfinite(
        voltage
    )

    finite_i = np.isfinite(
        current
    )

    voltage_new = voltage.copy()
    current_new = current.copy()

    voltage_new[
        finite_v
    ] = (
        voltage[
            finite_v
        ]
        + float(
            voltage_offset_V
        )
    )

    nonzero = (
        finite_i
        & (
            np.abs(
                current
            )
            > 1e-12
        )
    )

    zero = (
        finite_i
        & ~nonzero
    )

    current_new[
        nonzero
    ] = (
        current[
            nonzero
        ]
        * float(
            current_gain
        )
    )

    current_new[
        zero
    ] = 0.0

    out[
        "Voltage(V)"
    ] = voltage_new

    out[
        "Current(A)"
    ] = current_new

    return out

def rebuild_full_battery_features_from_raw(
    raw,
    battery_id,
    source,
):
    """
    Rebuild target-independent Step 5 feature representation from raw signals.

    IMPORTANT:
    histories are constructed over all available raw cycles BEFORE the
    frozen 70-100% evaluation-cycle membership is applied.
    """
    features = build_cycle_features_run2(
        raw,
        cycle_col=CYCLE_COL,
        time_col="Test_Time(s)",
        volt_col="Voltage(V)",
        curr_col="Current(A)",
        step_col="Step_Index",
        segment_col=None,
    )

    features = add_dataset_identity(
        features,
        battery_id,
        source,
        CONSTANT_PROTOCOL,
    )

    features[
        CYCLE_COL
    ] = pd.to_numeric(
        features[
            CYCLE_COL
        ],
        errors="coerce",
    )

    features = (
        features.sort_values(
            CYCLE_COL
        )
        .reset_index(
            drop=True
        )
    )

    first_cycle = pd.to_numeric(
        features[
            CYCLE_COL
        ],
        errors="coerce",
    ).min()

    features[
        "Cycle_Age"
    ] = (
        pd.to_numeric(
            features[
                CYCLE_COL
            ],
            errors="coerce",
        )
        - first_cycle
    ).clip(
        lower=0
    )

    features[
        "Log_Cycle_Age"
    ] = np.log1p(
        features[
            "Cycle_Age"
        ]
    )

    features[
        "source_is_nasa"
    ] = int(
        str(
            source
        ).upper()
        == "NASA"
    )

    features = add_legacy_history_features(
        features
    )

    features = add_run2_invariant_features(
        features
    )

    features = add_past_only_features(
        features
    )

    features = add_joint_legacy_aliases(
        features
    )

    return (
        features.sort_values(
            CYCLE_COL
        )
        .reset_index(
            drop=True
        )
    )

def extract_frozen_test_membership(
    frozen,
    battery_id,
):
    test = frozen.loc[
        frozen[
            GROUP_COL
        ].eq(
            battery_id
        )
    ].copy()

    if test.empty:
        raise RuntimeError(
            f"No frozen Step 5 rows for {battery_id}."
        )

    test[
        CYCLE_COL
    ] = pd.to_numeric(
        test[
            CYCLE_COL
        ],
        errors="coerce",
    )

    return (
        test.sort_values(
            CYCLE_COL
        )
        .reset_index(
            drop=True
        )
    )

def align_rebuilt_to_frozen_test(
    rebuilt,
    clean_frozen_test,
    selected_features,
):
    """
    Preserve exact Step 5 evaluation-cycle membership and selected schema.
    """
    expected = clean_frozen_test[
        [
            CYCLE_COL,
            TARGET_COL,
        ]
    ].copy()

    expected[
        CYCLE_COL
    ] = pd.to_numeric(
        expected[
            CYCLE_COL
        ],
        errors="coerce",
    )

    rebuilt = rebuilt.copy()

    rebuilt[
        CYCLE_COL
    ] = pd.to_numeric(
        rebuilt[
            CYCLE_COL
        ],
        errors="coerce",
    )

    merged = expected.merge(
        rebuilt,
        on=CYCLE_COL,
        how="left",
        validate="one_to_one",
        suffixes=(
            "_TARGET",
            "",
        ),
    )

    # The target from expected is the clean frozen target.
    if f"{TARGET_COL}_TARGET" in merged.columns:
        merged[
            TARGET_COL
        ] = merged[
            f"{TARGET_COL}_TARGET"
        ]

    for feature in (
        list(
            selected_features
        )
        + AGE_COLUMNS
    ):
        if feature in merged.columns:
            continue

        if feature not in clean_frozen_test.columns:
            raise RuntimeError(
                f"Selected feature absent from rebuilt and frozen schema: {feature}"
            )

        clean_values = pd.to_numeric(
            clean_frozen_test[
                feature
            ],
            errors="coerce",
        )

        if clean_values.notna().any():
            raise RuntimeError(
                f"Selected feature {feature} has clean finite values but "
                "was not reconstructed from raw data."
            )

        merged[
            feature
        ] = np.nan

    return merged

def equal_battery_weights(
    frame,
):
    counts = frame[
        GROUP_COL
    ].value_counts()

    weights = frame[
        GROUP_COL
    ].map(
        lambda battery:
        1.0
        / counts.loc[
            battery
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

def fit_frozen_step6_fold_model(
    clean_train,
    selected_electrical,
    config_name,
):
    columns = (
        list(
            selected_electrical
        )
        + AGE_COLUMNS
    )

    X_raw = numeric_frame(
        clean_train,
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
            clean_train[
                TARGET_COL
            ],
            errors="coerce",
        ).to_numpy(),
        sample_weight=equal_battery_weights(
            clean_train
        ),
    )

    return (
        model,
        imputer,
        columns,
    )

def predict_fold_model(
    model,
    imputer,
    columns,
    frame,
):
    X = imputer.transform(
        numeric_frame(
            frame,
            columns,
        )
    )

    return model.predict(
        X
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

def macro_by_replicate(
    results,
):
    frame = pd.DataFrame(
        results
    )

    if frame.empty:
        return pd.DataFrame()

    ok = frame.loc[
        frame[
            "Status"
        ].eq(
            "OK"
        )
    ].copy()

    if ok.empty:
        return pd.DataFrame()

    return (
        ok.groupby(
            [
                "Scenario",
                "Replicate",
            ],
            as_index=False,
        )
        .agg(
            Macro_RMSE=(
                "RMSE",
                "mean",
            ),
            Macro_MAE=(
                "MAE",
                "mean",
            ),
            N_Batteries=(
                "Battery",
                "nunique",
            ),
        )
        .sort_values(
            [
                "Scenario",
                "Replicate",
            ]
        )
        .reset_index(
            drop=True
        )
    )


def apply_random_perturbation(
    raw: pd.DataFrame,
    seed: int,
    voltage_sigma_V: float,
    current_sigma_fraction: float,
) -> pd.DataFrame:
    """Apply Gaussian sample-level noise while preserving current polarity."""
    rng = np.random.default_rng(int(seed))
    out = raw.copy()
    voltage = pd.to_numeric(out["Voltage(V)"], errors="coerce").to_numpy(float)
    current = pd.to_numeric(out["Current(A)"], errors="coerce").to_numpy(float)
    current_sigma = float(current_sigma_fraction) * characteristic_nonzero_current(raw)

    voltage_new = voltage.copy()
    current_new = current.copy()
    finite_v = np.isfinite(voltage)
    finite_i = np.isfinite(current)

    voltage_noise = rng.normal(0.0, float(voltage_sigma_V), size=len(voltage))
    current_noise = rng.normal(0.0, current_sigma, size=len(current))
    voltage_new[finite_v] = voltage[finite_v] + voltage_noise[finite_v]

    positive = finite_i & (current > 1e-12)
    negative = finite_i & (current < -1e-12)
    zero = finite_i & ~positive & ~negative
    current_new[positive] = np.maximum(current[positive] + current_noise[positive], 1e-12)
    current_new[negative] = np.minimum(current[negative] + current_noise[negative], -1e-12)
    current_new[zero] = 0.0

    out["Voltage(V)"] = voltage_new
    out["Current(A)"] = current_new
    return out


def _scenario_summary(results: pd.DataFrame, clean_rmse: float, clean_mae: float) -> pd.DataFrame:
    """Summarize equal-battery macro metrics and replicate intervals."""
    rows = [{
        "Scenario": "CLEAN_STEP6_REFERENCE",
        "Scenario_Type": "CLEAN",
        "Replicates": 1,
        "Macro_RMSE": clean_rmse,
        "Macro_MAE": clean_mae,
        "RMSE_Change_vs_Clean_Pct": 0.0,
        "MAE_Change_vs_Clean_Pct": 0.0,
        "RMSE_Replicate_CI95_Lower": np.nan,
        "RMSE_Replicate_CI95_Upper": np.nan,
    }]
    ok = results.loc[results["Status"].eq("OK")].copy()
    for scenario, group in ok.groupby("Scenario", sort=False):
        scenario_type = str(group["Scenario_Type"].iloc[0])
        by_rep = macro_by_replicate(group)
        rmse_vals = pd.to_numeric(by_rep["Macro_RMSE"], errors="coerce").dropna().to_numpy()
        mae_vals = pd.to_numeric(by_rep["Macro_MAE"], errors="coerce").dropna().to_numpy()
        macro_rmse = float(np.mean(rmse_vals))
        macro_mae = float(np.mean(mae_vals))
        rows.append({
            "Scenario": scenario,
            "Scenario_Type": scenario_type,
            "Replicates": int(len(rmse_vals)),
            "Macro_RMSE": macro_rmse,
            "Macro_MAE": macro_mae,
            "RMSE_Change_vs_Clean_Pct": (macro_rmse-clean_rmse)/clean_rmse*100.0,
            "MAE_Change_vs_Clean_Pct": (macro_mae-clean_mae)/clean_mae*100.0,
            "RMSE_Replicate_CI95_Lower":
                float(np.percentile(rmse_vals, 2.5)) if len(rmse_vals)>1 else np.nan,
            "RMSE_Replicate_CI95_Upper":
                float(np.percentile(rmse_vals, 97.5)) if len(rmse_vals)>1 else np.nan,
        })
    return pd.DataFrame(rows)


def run(
    raw_zip: Path,
    constant_data_path: Path,
    selected_features_path: Path,
    fold_metrics_path: Path,
    output_dir: Path,
    work_dir: Path | None = None,
) -> dict[str, Path]:
    """Run the final Step-10/10B robustness suite."""
    output_dir.mkdir(parents=True, exist_ok=True)
    scratch = work_dir or (output_dir / "_raw_extract")
    manifest, raw_validation = prepare_raw_archive(raw_zip, scratch)

    frozen = pd.read_csv(constant_data_path)
    selected_table = pd.read_csv(selected_features_path)
    fold_table = pd.read_csv(fold_metrics_path)
    formulation = "STRICT_ELECTRICAL_PLUS_AGE_HGB"
    selected_table = selected_table.loc[selected_table["Formulation"].eq(formulation)].copy()
    fold_table = fold_table.loc[fold_table["Formulation"].eq(formulation)].copy()

    if sorted(frozen[GROUP_COL].astype(str).unique()) != sorted(EXPECTED_BATTERIES):
        raise AssertionError("Unexpected constant-protocol battery set.")

    clean_macro_rmse = float(fold_table["RMSE_Raw"].mean())
    clean_macro_mae = float(fold_table["MAE_Raw"].mean())
    result_rows: list[dict] = []
    validation_rows: list[dict] = []

    for battery_index, battery in enumerate(sorted(EXPECTED_BATTERIES), start=1):
        print(f"[{battery_index}/{len(EXPECTED_BATTERIES)}] {battery}")
        clean_train = frozen.loc[~frozen[GROUP_COL].eq(battery)].copy()
        clean_test = extract_frozen_test_membership(frozen, battery)

        fold_selected = selected_table.loc[
            selected_table["Outer_Holdout"].eq(battery)
        ].sort_values("Selected_Rank")
        selected_features = fold_selected["Feature"].astype(str).tolist()
        fold_rows = fold_table.loc[fold_table["Holdout_Battery"].eq(battery)]
        if len(fold_rows) != 1:
            raise RuntimeError(f"{battery}: expected one frozen fold row.")
        fold_row = fold_rows.iloc[0]
        config_name = str(fold_row["Chosen_HGB_Config"])

        model, imputer, model_columns = fit_frozen_step6_fold_model(
            clean_train, selected_features, config_name
        )
        y = pd.to_numeric(clean_test[TARGET_COL], errors="coerce").to_numpy()
        frozen_pred = predict_fold_model(model, imputer, model_columns, clean_test)
        frozen_reproduced_rmse = rmse(y, frozen_pred)
        frozen_reference_rmse = float(fold_row["RMSE_Raw"])
        if abs(frozen_reproduced_rmse - frozen_reference_rmse) > 1e-9:
            raise RuntimeError(f"{battery}: frozen Step-6 fold reproduction failed.")

        raw_clean, source = load_clean_raw_battery(manifest, battery)
        rebuilt_clean_all = rebuild_full_battery_features_from_raw(raw_clean, battery, source)
        rebuilt_clean_test = align_rebuilt_to_frozen_test(
            rebuilt_clean_all, clean_test, selected_features
        )
        rebuilt_clean_pred = predict_fold_model(
            model, imputer, model_columns, rebuilt_clean_test
        )
        rebuilt_clean_rmse = rmse(y, rebuilt_clean_pred)
        rebuild_diff = abs(rebuilt_clean_rmse - frozen_reference_rmse)
        validation_rows.append({
            "Battery": battery, "Source": source, "N_Test": len(clean_test),
            "Step6_RMSE": frozen_reference_rmse,
            "Frozen_Table_Reproduced_RMSE": frozen_reproduced_rmse,
            "Raw_Rebuilt_Clean_RMSE": rebuilt_clean_rmse,
            "Absolute_RMSE_Difference_Raw_Rebuild": rebuild_diff,
            "Characteristic_Nonzero_Current_A": characteristic_nonzero_current(raw_clean),
            "Selected_Electrical_Features": len(selected_features),
            "HGB_Config": config_name,
        })
        if rebuild_diff > REBUILT_CLEAN_RMSE_TOLERANCE:
            raise RuntimeError(
                f"{battery}: clean raw reconstruction differs by {rebuild_diff:.6g}, "
                f"above tolerance {REBUILT_CLEAN_RMSE_TOLERANCE}."
            )

        for scenario, params in SYSTEMATIC_SCENARIOS.items():
            started = time.time()
            perturbed = apply_systematic_perturbation(
                raw_clean, params["voltage_offset_V"], params["current_gain"]
            )
            rebuilt = rebuild_full_battery_features_from_raw(perturbed, battery, source)
            test = align_rebuilt_to_frozen_test(rebuilt, clean_test, selected_features)
            pred = predict_fold_model(model, imputer, model_columns, test)
            result_rows.append({
                "Battery": battery, "Source": source, "Scenario": scenario,
                "Scenario_Type": "SYSTEMATIC", "Replicate": 0, "Status": "OK",
                "RMSE": rmse(y, pred), "MAE": mae(y, pred),
                "Clean_RMSE": frozen_reference_rmse, "Clean_MAE": float(fold_row["MAE_Raw"]),
                "Voltage_Offset_mV": params["voltage_offset_V"]*1000.0,
                "Current_Gain_Error_Pct": (params["current_gain"]-1.0)*100.0,
                "Voltage_Noise_Sigma_mV": 0.0, "Current_Noise_Sigma_A": 0.0,
                "Runtime_Seconds": time.time()-started,
            })

        for scenario_index, (scenario, params) in enumerate(RANDOM_SCENARIOS.items()):
            for replicate in range(int(params["replicates"])):
                started = time.time()
                # Step 10 and Step 10B were run separately with the same
                # deterministic per-battery seed schedule; only noise scale changes.
                seed = RANDOM_STATE + 100000*battery_index + replicate
                perturbed = apply_random_perturbation(
                    raw_clean, seed, params["voltage_sigma_V"],
                    params["current_sigma_fraction"],
                )
                rebuilt = rebuild_full_battery_features_from_raw(perturbed, battery, source)
                test = align_rebuilt_to_frozen_test(rebuilt, clean_test, selected_features)
                pred = predict_fold_model(model, imputer, model_columns, test)
                result_rows.append({
                    "Battery": battery, "Source": source, "Scenario": scenario,
                    "Scenario_Type": "STOCHASTIC", "Replicate": replicate,
                    "Status": "OK", "RMSE": rmse(y, pred), "MAE": mae(y, pred),
                    "Clean_RMSE": frozen_reference_rmse, "Clean_MAE": float(fold_row["MAE_Raw"]),
                    "Voltage_Offset_mV": 0.0, "Current_Gain_Error_Pct": 0.0,
                    "Voltage_Noise_Sigma_mV": params["voltage_sigma_V"]*1000.0,
                    "Current_Noise_Sigma_A":
                        params["current_sigma_fraction"]*characteristic_nonzero_current(raw_clean),
                    "Runtime_Seconds": time.time()-started,
                })

    results = pd.DataFrame(result_rows)
    validation = pd.DataFrame(validation_rows)
    summary = _scenario_summary(results, clean_macro_rmse, clean_macro_mae)
    replicates = macro_by_replicate(results.loc[results["Status"].eq("OK")])

    paths = {
        "summary": output_dir / "measurement_robustness_summary.csv",
        "per_battery": output_dir / "measurement_robustness_per_battery.csv",
        "macro_replicates": output_dir / "measurement_robustness_macro_replicates.csv",
        "clean_validation": output_dir / "clean_raw_reconstruction_validation.csv",
        "raw_validation": output_dir / "raw_data_validation.csv",
    }
    summary.to_csv(paths["summary"], index=False)
    results.to_csv(paths["per_battery"], index=False)
    replicates.to_csv(paths["macro_replicates"], index=False)
    validation.to_csv(paths["clean_validation"], index=False)
    raw_validation.to_csv(paths["raw_validation"], index=False)
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-zip", type=Path, required=True)
    parser.add_argument("--constant-data", type=Path,
                        default=Path("outputs/constant_dataset/constant_observations.csv"))
    parser.add_argument("--selected-features", type=Path,
                        default=Path("outputs/constant_hgb/constant_hgb_selected_features.csv"))
    parser.add_argument("--fold-metrics", type=Path,
                        default=Path("outputs/constant_hgb/constant_hgb_per_battery.csv"))
    parser.add_argument("--output-dir", type=Path,
                        default=Path("outputs/measurement_robustness"))
    parser.add_argument("--work-dir", type=Path)
    args = parser.parse_args()
    for name, path in run(
        args.raw_zip, args.constant_data, args.selected_features,
        args.fold_metrics, args.output_dir, args.work_dir
    ).items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
