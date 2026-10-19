#!/usr/bin/env python
"""Reconstruct NASA RW3--RW6 reference-discharge SoH provenance from raw data."""
from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd

from battery_soh_repro.data_pipeline import (
    NASA_VARIABLE_BATTERIES,
    build_nasa_reconstructed_reference_targets,
    prepare_raw_archive,
)


def run(raw_zip: Path, output_dir: Path, work_dir: Path) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest, validation = prepare_raw_archive(raw_zip, work_dir)

    event_tables = []
    map_tables = []
    for battery in NASA_VARIABLE_BATTERIES:
        entry = manifest[battery]
        events, cycle_map = build_nasa_reconstructed_reference_targets(
            Path(entry["csv_path"]), battery
        )
        event_tables.append(events)
        map_tables.append(cycle_map)

    events = pd.concat(event_tables, ignore_index=True)
    cycle_map = pd.concat(map_tables, ignore_index=True)

    if len(events) != 73:
        raise AssertionError(f"Expected 73 reconstructed NASA reference events, found {len(events)}.")

    paths = {
        "reference_targets": output_dir / "nasa_rw_reference_targets.csv",
        "cycle_map": output_dir / "nasa_rw_cycle_to_reference_map.csv",
        "validation": output_dir / "raw_dataset_validation.csv",
        "workbook": output_dir / "nasa_rw_target_provenance.xlsx",
    }
    events.to_csv(paths["reference_targets"], index=False)
    cycle_map.to_csv(paths["cycle_map"], index=False)
    validation.to_csv(paths["validation"], index=False)

    with pd.ExcelWriter(paths["workbook"], engine="openpyxl") as writer:
        events.to_excel(writer, sheet_name="Reference_Targets", index=False)
        cycle_map.to_excel(writer, sheet_name="Cycle_to_Reference", index=False)
        validation.to_excel(writer, sheet_name="Dataset_Validation", index=False)

    return paths


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-zip", required=True, type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/nasa_rw_provenance"))
    parser.add_argument("--work-dir", type=Path, default=Path("work/nasa_rw_provenance"))
    args = parser.parse_args()
    for name, path in run(args.raw_zip, args.output_dir, args.work_dir).items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
