#!/usr/bin/env python
"""Reconstruct variable-protocol reference targets and trivial baselines.

CALCE Type-3 cells contribute one independent 0.22 A reference observation per
macro-cycle. NASA RW3--RW6 contribute one independent observation per measured
reference discharge, using target-free summaries of the preceding random-walk
interval. Source SOH and SOH_interpolated fields are never used as targets.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from battery_soh_repro.data_pipeline import (
    GROUP_COL,
    SOURCE_COL,
    R5E_INTERVAL_ID_COL,
    STRICT_VARIABLE_FEATURES,
    build_independent_observation_table,
    build_revised_variable_cycle_features,
    dataset_summary,
    feature_coverage_table,
    prepare_raw_archive,
    run_variable_lobo_trivial_baselines,
)


def run(raw_zip: Path, output_dir: Path, work_dir: Path) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest, validation = prepare_raw_archive(raw_zip, work_dir)

    cycle_features, calce_audit, nasa_events, nasa_map = (
        build_revised_variable_cycle_features(manifest)
    )
    all_observations, primary = build_independent_observation_table(
        cycle_features, calce_audit, nasa_events, nasa_map
    )

    expected = {"CS2_3": 26, "CS2_9": 92, "RW3": 13, "RW4": 12, "RW5": 13, "RW6": 9}
    actual = primary.groupby(GROUP_COL).size().to_dict()
    if actual != expected:
        raise AssertionError(f"Unexpected variable observation counts: {actual}")
    if len(primary) != 165:
        raise AssertionError(f"Expected 165 variable observations, found {len(primary)}.")
    if len(STRICT_VARIABLE_FEATURES) != 314:
        raise AssertionError("Frozen strict variable feature count is not 314.")

    per_battery, macro, predictions = run_variable_lobo_trivial_baselines(primary)
    summary = dataset_summary(all_observations, primary)
    coverage = feature_coverage_table(primary)

    paths = {
        "observations": output_dir / "variable_independent_observations.csv",
        "strict_features": output_dir / "variable_strict_feature_names.csv",
        "calce_target_audit": output_dir / "calce_variable_target_audit.csv",
        "nasa_reference_targets": output_dir / "nasa_rw_reference_targets.csv",
        "nasa_cycle_map": output_dir / "nasa_rw_cycle_to_reference_map.csv",
        "dataset_summary": output_dir / "variable_dataset_summary.csv",
        "coverage": output_dir / "variable_strict_feature_coverage.csv",
        "baseline_per_battery": output_dir / "variable_baseline_per_battery.csv",
        "baseline_macro": output_dir / "variable_baseline_macro.csv",
        "baseline_predictions": output_dir / "variable_baseline_predictions.csv",
        "validation": output_dir / "raw_dataset_validation.csv",
        "workbook": output_dir / "variable_dataset_and_baselines.xlsx",
    }

    primary.to_csv(paths["observations"], index=False)
    pd.DataFrame({"Feature": STRICT_VARIABLE_FEATURES}).to_csv(
        paths["strict_features"], index=False
    )
    calce_audit.to_csv(paths["calce_target_audit"], index=False)
    nasa_events.to_csv(paths["nasa_reference_targets"], index=False)
    nasa_map.to_csv(paths["nasa_cycle_map"], index=False)
    summary.to_csv(paths["dataset_summary"], index=False)
    coverage.to_csv(paths["coverage"], index=False)
    per_battery.to_csv(paths["baseline_per_battery"], index=False)
    macro.to_csv(paths["baseline_macro"], index=False)
    predictions.to_csv(paths["baseline_predictions"], index=False)
    validation.to_csv(paths["validation"], index=False)

    with pd.ExcelWriter(paths["workbook"], engine="openpyxl") as writer:
        summary.to_excel(writer, sheet_name="Dataset_Summary", index=False)
        primary.to_excel(writer, sheet_name="Primary_Observations", index=False)
        calce_audit.to_excel(writer, sheet_name="CALCE_Target_Audit", index=False)
        nasa_events.to_excel(writer, sheet_name="NASA_Reference_Audit", index=False)
        nasa_map.to_excel(writer, sheet_name="NASA_Cycle_to_Reference", index=False)
        coverage.to_excel(writer, sheet_name="Strict_Feature_Coverage", index=False)
        per_battery.to_excel(writer, sheet_name="Baseline_Per_Battery", index=False)
        macro.to_excel(writer, sheet_name="Baseline_Macro", index=False)
        predictions.to_excel(writer, sheet_name="Baseline_Predictions", index=False)
        validation.to_excel(writer, sheet_name="Dataset_Validation", index=False)

    return paths


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-zip", required=True, type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/variable_dataset"))
    parser.add_argument("--work-dir", type=Path, default=Path("work/variable_raw"))
    args = parser.parse_args()
    paths = run(args.raw_zip, args.output_dir, args.work_dir)
    for name, path in paths.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
