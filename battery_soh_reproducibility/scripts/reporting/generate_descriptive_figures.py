"""Generate the four descriptive figures referenced by the manuscript.

The charge/discharge profile figures are reconstructed from one configurable
CALCE cell using representative cycle quantiles. The resistance and dV/dt
figures are derived from the corrected constant-protocol cycle table.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import tempfile

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from battery_soh_repro.data_pipeline import (
    CALCE_CONSTANT_BATTERIES,
    find_column,
    prepare_raw_archive,
    process_files,
)


def _representative_cycles(values: pd.Series, n: int = 5) -> list[float]:
    cycles = np.array(sorted(pd.to_numeric(values, errors="coerce").dropna().unique()))
    if len(cycles) <= n:
        return cycles.tolist()
    positions = np.linspace(0, len(cycles) - 1, n).round().astype(int)
    return cycles[positions].tolist()


def _plot_profiles(raw: pd.DataFrame, output_dir: Path) -> tuple[Path, Path]:
    cycle_col = find_column(raw, ["Cycle_Index", "Cycle_Index_Combined", "cycle"])
    time_col = find_column(raw, ["Test_Time(s)", "Test_Time", "Time", "time"])
    voltage_col = find_column(raw, ["Voltage(V)", "Voltage", "Voltage_measured"])
    current_col = find_column(raw, ["Current(A)", "Current", "Current_measured"])

    data = raw[[cycle_col, time_col, voltage_col, current_col]].copy()
    for column in data.columns:
        data[column] = pd.to_numeric(data[column], errors="coerce")
    data = data.dropna()
    cycles = _representative_cycles(data[cycle_col])

    charge_path = output_dir / "charge.png"
    fig, ax = plt.subplots(figsize=(8, 5))
    for cycle in cycles:
        part = data.loc[(data[cycle_col] == cycle) & (data[current_col] > 0.02)].copy()
        if len(part) < 2:
            continue
        elapsed = (part[time_col] - part[time_col].min()) / 60.0
        ax.plot(elapsed, part[voltage_col], label=f"Cycle {int(cycle)}")
    ax.set_xlabel("Charge duration (min)")
    ax.set_ylabel("Voltage (V)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(charge_path, dpi=300)
    plt.close(fig)

    discharge_path = output_dir / "Discharge.png"
    fig, ax = plt.subplots(figsize=(8, 5))
    for cycle in cycles:
        part = data.loc[(data[cycle_col] == cycle) & (data[current_col] < -0.02)].copy()
        if len(part) < 2:
            continue
        elapsed = (part[time_col] - part[time_col].min()) / 60.0
        ax.plot(elapsed, part[voltage_col], label=f"Cycle {int(cycle)}")
    ax.set_xlabel("Discharge duration (min)")
    ax.set_ylabel("Voltage (V)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(discharge_path, dpi=300)
    plt.close(fig)
    return charge_path, discharge_path


def _plot_cycle_indicators(constant_data: pd.DataFrame, output_dir: Path) -> tuple[Path, Path]:
    data = constant_data.copy()
    resistance_candidates = [
        "Internal_Resistance_median", "LEGACY__Internal_Resistance_median"
    ]
    resistance_col = next((c for c in resistance_candidates if c in data.columns), None)
    dvdt_candidates = ["avg_abs_dvdt", "LEGACY__avg_abs_dvdt"]
    dvdt_col = next((c for c in dvdt_candidates if c in data.columns), None)

    resistance_path = output_dir / "resistance.png"
    fig, ax = plt.subplots(figsize=(7, 5))
    if resistance_col is not None:
        for battery, group in data.groupby("battery_id", sort=True):
            x = pd.to_numeric(group["Cycle_Index"], errors="coerce")
            y = pd.to_numeric(group[resistance_col], errors="coerce")
            mask = x.notna() & y.notna()
            if mask.any():
                ax.plot(x[mask], y[mask], linewidth=1, label=str(battery))
        ax.set_ylabel("Median internal resistance (Ohm)")
        ax.legend(fontsize=7, ncol=2)
    else:
        ax.text(0.5, 0.5, "Internal resistance not available in this dataset",
                ha="center", va="center", transform=ax.transAxes)
        ax.set_ylabel("Internal resistance")
    ax.set_xlabel("Cycle index")
    fig.tight_layout()
    fig.savefig(resistance_path, dpi=300)
    plt.close(fig)

    dvdt_path = output_dir / "dv_dt.png"
    fig, ax = plt.subplots(figsize=(7, 5))
    if dvdt_col is None:
        raise KeyError("No average absolute dV/dt feature found in constant cycle table.")
    x = pd.to_numeric(data["Calculated_SOH(%)"], errors="coerce")
    y = pd.to_numeric(data[dvdt_col], errors="coerce")
    mask = x.notna() & y.notna()
    ax.scatter(x[mask], y[mask], s=8, alpha=0.5)
    ax.set_xlabel("SoH (%)")
    ax.set_ylabel("Average |dV/dt| (V/s)")
    fig.tight_layout()
    fig.savefig(dvdt_path, dpi=300)
    plt.close(fig)
    return resistance_path, dvdt_path


def run(
    raw_zip: Path,
    constant_data_path: Path,
    output_dir: Path,
    calce_battery: str = "CS2_35",
    work_dir: Path | None = None,
) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    scratch = work_dir or (output_dir / "_raw_extract")
    manifest, _ = prepare_raw_archive(raw_zip, scratch)
    calce_map = dict(CALCE_CONSTANT_BATTERIES)
    if calce_battery not in calce_map:
        raise KeyError(f"{calce_battery} is not a configured constant CALCE battery.")
    raw = process_files(manifest[calce_battery]["folder"], calce_map[calce_battery])
    constant_data = pd.read_csv(constant_data_path)
    return [
        *_plot_profiles(raw, output_dir),
        *_plot_cycle_indicators(constant_data, output_dir),
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-zip", type=Path, required=True)
    parser.add_argument("--constant-data", type=Path,
                        default=Path("outputs/constant_dataset/constant_observations.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/manuscript_figures"))
    parser.add_argument("--calce-battery", default="CS2_35")
    parser.add_argument("--work-dir", type=Path)
    args = parser.parse_args()
    for path in run(args.raw_zip, args.constant_data, args.output_dir,
                    args.calce_battery, args.work_dir):
        print(path)


if __name__ == "__main__":
    main()
