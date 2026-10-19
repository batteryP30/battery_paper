#!/usr/bin/env python
"""Run the frozen feature-admissibility audit used by the final study."""
from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd

from battery_soh_repro.data_pipeline import (
    GROUP_COL,
    SOURCE_COL,
    build_revised_constant_cycle_table,
    build_revised_variable_cycle_features,
    prepare_raw_archive,
)
from battery_soh_repro.feature_policy import (
    audit_frozen_candidates,
    assert_manuscript_audit_counts,
)


def run(raw_zip: Path, output_dir: Path, work_dir: Path) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest, validation = prepare_raw_archive(raw_zip, work_dir)

    constant_features, _, _ = build_revised_constant_cycle_table(manifest)
    variable_cycle_features, _, _, _ = build_revised_variable_cycle_features(manifest)

    constant_audit = audit_frozen_candidates(
        constant_features, "CONSTANT", GROUP_COL, SOURCE_COL
    )
    variable_audit = audit_frozen_candidates(
        variable_cycle_features, "VARIABLE", GROUP_COL, SOURCE_COL
    )

    assert_manuscript_audit_counts(constant_audit, "CONSTANT")
    assert_manuscript_audit_counts(variable_audit, "VARIABLE")

    audit = pd.concat([constant_audit, variable_audit], ignore_index=True)
    counts = (
        audit.groupby(["Protocol", "Classification"], as_index=False)
        .agg(N_Features=("Feature", "nunique"))
        .sort_values(["Protocol", "Classification"])
    )

    paths = {
        "audit": output_dir / "feature_admissibility_audit.csv",
        "summary": output_dir / "feature_admissibility_summary.csv",
        "validation": output_dir / "raw_dataset_validation.csv",
        "workbook": output_dir / "feature_admissibility_audit.xlsx",
    }
    audit.to_csv(paths["audit"], index=False)
    counts.to_csv(paths["summary"], index=False)
    validation.to_csv(paths["validation"], index=False)

    with pd.ExcelWriter(paths["workbook"], engine="openpyxl") as writer:
        counts.to_excel(writer, sheet_name="Classification_Summary", index=False)
        constant_audit.to_excel(writer, sheet_name="Constant", index=False)
        variable_audit.to_excel(writer, sheet_name="Variable", index=False)
        validation.to_excel(writer, sheet_name="Dataset_Validation", index=False)

    return paths


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-zip", required=True, type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/feature_audit"))
    parser.add_argument("--work-dir", type=Path, default=Path("work/feature_audit"))
    args = parser.parse_args()
    for name, path in run(args.raw_zip, args.output_dir, args.work_dir).items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
