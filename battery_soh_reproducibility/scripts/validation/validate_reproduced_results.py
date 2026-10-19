"""Validate reproduced outputs against the frozen manuscript results.

This is a regression/integrity check, not an experiment. It should be run after
the full workflow. Tolerances are deliberately small because the validated
pipeline uses fixed seeds, deterministic folds, and fixed candidate sets.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import math

import pandas as pd

ABS_TOL = 1e-4

EXPECTED_MACRO = {
    ("constant_hgb", "STRICT_ELECTRICAL_HGB"): (1.500262, 1.212214),
    ("constant_hgb", "STRICT_ELECTRICAL_PLUS_AGE_HGB"): (1.332906, 1.046004),
    ("variable_hgb", "STRICT_ELECTRICAL_HGB"): (4.223979, 3.555678),
    ("variable_hgb", "STRICT_ELECTRICAL_PLUS_AGE_HGB"): (2.696856, 2.119289),
}

EXPECTED_TRANSFER = {
    ("STRICT_ELECTRICAL_HGB", "NASA", "CALCE"): (4.424228, 3.488138),
    ("STRICT_ELECTRICAL_PLUS_AGE_HGB", "NASA", "CALCE"): (4.200279, 3.381922),
    ("STRICT_ELECTRICAL_HGB", "CALCE", "NASA"): (3.055213, 2.248472),
    ("STRICT_ELECTRICAL_PLUS_AGE_HGB", "CALCE", "NASA"): (4.215245, 3.440050),
}

EXPECTED_PARTIAL = {
    "STRICT_ELECTRICAL_PLUS_3P6_HGB": (1.427909, 1.104017),
    "STRICT_ELECTRICAL_PLUS_AGE_PLUS_3P6_HGB": (1.431725, 1.121442),
}

EXPECTED_ABLATION_RMSE = {
    "FULL_STRICT_ELECTRICAL_PLUS_AGE": 1.332906,
    "NO_EXPLICIT_AGE": 1.500262,
    "NO_VOLTAGE_SHAPE": 1.422263,
    "NO_DYNAMIC_RESPONSE": 2.314930,
    "NO_TEMPORAL_TREND_TRANSFORMS": 1.588642,
}

EXPECTED_ROBUSTNESS_RMSE = {
    "CLEAN_STEP6_REFERENCE": 1.332906,
    "V_OFFSET_NEG_1mV": 1.334644,
    "V_OFFSET_POS_1mV": 1.329005,
    "V_OFFSET_NEG_5mV": 1.335260,
    "V_OFFSET_POS_5mV": 1.317873,
    "V_OFFSET_NEG_10mV": 1.331929,
    "V_OFFSET_POS_10mV": 1.307438,
    "I_GAIN_NEG_0p5pct": 1.345923,
    "I_GAIN_POS_0p5pct": 1.338403,
    "I_GAIN_NEG_1p0pct": 1.348399,
    "I_GAIN_POS_1p0pct": 1.356443,
    "COMBINED_NEG_10mV_NEG_1pct": 1.346616,
    "COMBINED_POS_10mV_POS_1pct": 1.330127,
    "RANDOM_1mV_0p1pct": 1.837595,  # mean is checked at 1e-3 below
    "RANDOM_5mV_0p5pct": 2.500906,
}


def _assert_close(actual: float, expected: float, label: str, tol: float = ABS_TOL) -> None:
    if not math.isfinite(actual) or abs(actual - expected) > tol:
        raise AssertionError(
            f"{label}: expected {expected:.6f} +/- {tol:g}, got {actual:.6f}"
        )


def run(root: Path) -> None:
    constant_data = pd.read_csv(root/"constant_dataset/constant_observations.csv")
    variable_data = pd.read_csv(root/"variable_dataset/variable_independent_observations.csv")
    if len(constant_data) != 4590 or constant_data["battery_id"].nunique() != 10:
        raise AssertionError("Constant dataset contract failed.")
    expected_variable_counts = {"CS2_3": 26, "CS2_9": 92, "RW3": 13, "RW4": 12, "RW5": 13, "RW6": 9}
    if variable_data.groupby("battery_id").size().to_dict() != expected_variable_counts:
        raise AssertionError("Variable independent-observation contract failed.")

    for folder, formulation in EXPECTED_MACRO:
        table = pd.read_csv(root/folder/f"{folder}_macro.csv")
        row = table.loc[table["Formulation"].eq(formulation)].iloc[0]
        exp_rmse, exp_mae = EXPECTED_MACRO[(folder, formulation)]
        _assert_close(float(row["Macro_RMSE_Raw"]), exp_rmse, f"{folder}/{formulation} RMSE")
        _assert_close(float(row["Macro_MAE_Raw"]), exp_mae, f"{folder}/{formulation} MAE")

    transfer = pd.read_csv(root/"constant_transfer/constant_transfer_summary.csv")
    for key, expected in EXPECTED_TRANSFER.items():
        model, train, test = key
        row = transfer.loc[
            transfer["Formulation"].eq(model)
            & transfer["Train_Source"].eq(train)
            & transfer["Test_Source"].eq(test)
        ].iloc[0]
        _assert_close(float(row["Macro_RMSE_Raw"]), expected[0], f"transfer {key} RMSE")
        _assert_close(float(row["Macro_MAE_Raw"]), expected[1], f"transfer {key} MAE")

    partial = pd.read_csv(root/"constant_3p6/constant_3p6_macro.csv")
    for formulation, expected in EXPECTED_PARTIAL.items():
        row = partial.loc[partial["Formulation"].eq(formulation)].iloc[0]
        _assert_close(float(row["Macro_RMSE_Raw"]), expected[0], f"3.6V {formulation} RMSE")
        _assert_close(float(row["Macro_MAE_Raw"]), expected[1], f"3.6V {formulation} MAE")

    ablation = pd.read_csv(root/"statistical_analysis/strict_ablation_summary.csv")
    for name, expected in EXPECTED_ABLATION_RMSE.items():
        row = ablation.loc[ablation["Analysis"].eq(name)].iloc[0]
        _assert_close(float(row["Macro_RMSE"]), expected, f"ablation {name}")

    robustness = pd.read_csv(root/"measurement_robustness/measurement_robustness_summary.csv")
    for scenario, expected in EXPECTED_ROBUSTNESS_RMSE.items():
        row = robustness.loc[robustness["Scenario"].eq(scenario)].iloc[0]
        tol = 1e-3 if scenario.startswith("RANDOM") else ABS_TOL
        _assert_close(float(row["Macro_RMSE"]), expected, f"robustness {scenario}", tol)

    clean = pd.read_csv(root/"measurement_robustness/clean_raw_reconstruction_validation.csv")
    max_diff = pd.to_numeric(
        clean["Absolute_RMSE_Difference_Raw_Rebuild"], errors="coerce"
    ).max()
    if not math.isfinite(float(max_diff)) or float(max_diff) > 1e-3:
        raise AssertionError(
            f"Clean raw reconstruction exceeded 1e-3 tolerance: {max_diff}"
        )

    print("All frozen manuscript regression checks passed.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outputs-root", type=Path, default=Path("outputs"))
    args = parser.parse_args()
    run(args.outputs_root)


if __name__ == "__main__":
    main()
