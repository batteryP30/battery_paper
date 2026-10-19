import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd

from battery_soh_repro.data_pipeline import add_dataset_identity, build_cycle_features


ROOT = Path(__file__).resolve().parents[1]


def _synthetic_cycle_rows() -> pd.DataFrame:
    rows = []
    for cycle in (1, 2):
        time_s = 0.0
        for step, current, v_start, v_end, samples in (
            (2, 1.0, 3.5, 4.0, 20),
            (4, 0.4, 4.2, 4.2, 20),
            (7, -1.0, 4.1, 3.0, 40),
        ):
            for sample in range(samples):
                fraction = sample / (samples - 1)
                rows.append(
                    {
                        "Cycle_Index": cycle,
                        "Test_Time(s)": time_s,
                        "Voltage(V)": v_start + (v_end - v_start) * fraction,
                        "Current(A)": current,
                        "Step_Index": step,
                        "Internal_Resistance(Ohm)": 0.05 + 0.001 * cycle,
                    }
                )
                time_s += 10.0
    return pd.DataFrame(rows)


def test_historical_base_table_schema_has_109_columns():
    features = build_cycle_features(_synthetic_cycle_rows(), step_col="Step_Index")
    assert len(features.columns) == 106
    assert len(features.columns.drop("Cycle_Index")) == 105

    table = add_dataset_identity(features, "TEST", "CALCE", 1)
    assert len(table.columns) == 109
    assert {"Cycle_Index", "battery_id", "source_dataset", "protocol"}.issubset(table.columns)


def test_run_all_resolves_project_root():
    script = ROOT / "scripts" / "run_all.py"
    spec = importlib.util.spec_from_file_location("run_all", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert module.repository_root() == ROOT
