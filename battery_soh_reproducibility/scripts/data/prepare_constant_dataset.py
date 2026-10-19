#!/usr/bin/env python
"""Reconstruct the constant-protocol dataset and trivial baselines.

This is the publication version of the validated Step 5 workflow. It rebuilds
capacity-based SoH from raw current/time measurements, does not use source SoH
fields, does not perform target-inspection cleaning, applies the 70--100% range
only as a study scope, and freezes the 335 strict-safe candidate features.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import shutil

import pandas as pd

from battery_soh_repro.data_pipeline import (
    FROZEN_STRICT_CONSTANT_FEATURES,
    GROUP_COL,
    SOURCE_COL,
    TARGET_COL,
    build_constant_dataset_summary,
    build_revised_constant_cycle_table,
    constant_feature_coverage,
    prepare_raw_archive,
    run_constant_trivial_baselines,
)


def run(raw_zip: Path, output_dir: Path, work_dir: Path) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest, validation = prepare_raw_archive(raw_zip, work_dir)

    combined, primary, target_audit = build_revised_constant_cycle_table(manifest)

    expected_batteries = {
        "CS2_33", "CS2_34", "CS2_35", "CS2_36", "CS2_37", "CS2_38",
        "B0005", "B0006", "B0007", "B0018",
    }
    if set(primary[GROUP_COL].astype(str).unique()) != expected_batteries:
        raise AssertionError("Unexpected constant-protocol battery set.")
    if len(primary) != 4590:
        raise AssertionError(f"Expected 4590 constant observations, found {len(primary)}.")
    if len(FROZEN_STRICT_CONSTANT_FEATURES) != 335:
        raise AssertionError("Frozen strict constant feature count is not 335.")
    if not primary[TARGET_COL].between(70.0, 100.0, inclusive="both").all():
        raise AssertionError("Constant target table contains values outside 70--100%.")

    per_battery, macro, predictions = run_constant_trivial_baselines(primary)
    summary = build_constant_dataset_summary(combined, primary)
    coverage = constant_feature_coverage(primary)

    paths = {
        "observations": output_dir / "constant_observations.csv",
        "strict_features": output_dir / "constant_strict_feature_names.csv",
        "target_audit": output_dir / "constant_target_reconstruction_audit.csv",
        "dataset_summary": output_dir / "constant_dataset_summary.csv",
        "coverage": output_dir / "constant_strict_feature_coverage.csv",
        "baseline_per_battery": output_dir / "constant_baseline_per_battery.csv",
        "baseline_macro": output_dir / "constant_baseline_macro.csv",
        "baseline_predictions": output_dir / "constant_baseline_predictions.csv",
        "validation": output_dir / "raw_dataset_validation.csv",
        "workbook": output_dir / "constant_dataset_and_baselines.xlsx",
    }

    primary.to_csv(paths["observations"], index=False)
    pd.DataFrame({"Feature": FROZEN_STRICT_CONSTANT_FEATURES}).to_csv(
        paths["strict_features"], index=False
    )
    target_audit.to_csv(paths["target_audit"], index=False)
    summary.to_csv(paths["dataset_summary"], index=False)
    coverage.to_csv(paths["coverage"], index=False)
    per_battery.to_csv(paths["baseline_per_battery"], index=False)
    macro.to_csv(paths["baseline_macro"], index=False)
    predictions.to_csv(paths["baseline_predictions"], index=False)
    validation.to_csv(paths["validation"], index=False)

    with pd.ExcelWriter(paths["workbook"], engine="openpyxl") as writer:
        summary.to_excel(writer, sheet_name="Dataset_Summary", index=False)
        target_audit.to_excel(writer, sheet_name="Target_Reconstruction_Audit", index=False)
        primary.to_excel(writer, sheet_name="Primary_Observations", index=False)
        pd.DataFrame({"Feature": FROZEN_STRICT_CONSTANT_FEATURES}).to_excel(
            writer, sheet_name="Frozen_Strict_Features", index=False
        )
        coverage.to_excel(writer, sheet_name="Strict_Feature_Coverage", index=False)
        per_battery.to_excel(writer, sheet_name="Baseline_Per_Battery", index=False)
        macro.to_excel(writer, sheet_name="Baseline_Macro", index=False)
        predictions.to_excel(writer, sheet_name="Baseline_Predictions", index=False)
        validation.to_excel(writer, sheet_name="Dataset_Validation", index=False)

    return paths


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-zip", required=True, type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/constant_dataset"))
    parser.add_argument("--work-dir", type=Path, default=Path("work/constant_raw"))
    args = parser.parse_args()
    paths = run(args.raw_zip, args.output_dir, args.work_dir)
    for name, path in paths.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
