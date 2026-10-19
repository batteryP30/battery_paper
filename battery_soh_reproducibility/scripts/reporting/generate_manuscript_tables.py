"""Generate manuscript-ready tables from reproduced pipeline outputs.

The transformations in this module only format and combine outputs produced by
the scientific pipeline. No manuscript result values are hard-coded here,
apart from the finite HGB candidate configurations defined by the method.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


CONSTANT_BATTERY_ORDER = [
    "B0005", "B0006", "B0007", "B0018",
    "CS2_33", "CS2_34", "CS2_35", "CS2_36", "CS2_37", "CS2_38",
]
VARIABLE_BATTERY_ORDER = ["CS2_3", "CS2_9", "RW3", "RW4", "RW5", "RW6"]

MODEL_LABELS = {
    "STRICT_ELECTRICAL_HGB": "Strict electrical",
    "STRICT_ELECTRICAL_PLUS_AGE_HGB": "Strict electrical + age",
    "STRICT_ELECTRICAL_PLUS_3P6_HGB":
        "Strict electrical + 3.6 V partial observation",
    "STRICT_ELECTRICAL_PLUS_AGE_PLUS_3P6_HGB":
        "Strict electrical + age + 3.6 V partial observation",
}

ABLATION_LABELS = {
    "FULL_STRICT_ELECTRICAL_PLUS_AGE": "Full strict electrical + age",
    "NO_EXPLICIT_AGE": "No explicit age",
    "NO_VOLTAGE_SHAPE": "No voltage-shape family",
    "NO_DYNAMIC_RESPONSE": "No dynamic-response family",
    "NO_TEMPORAL_TREND_TRANSFORMS": "No temporal/trend transforms",
}

SCENARIO_LABELS = {
    "CLEAN_STEP6_REFERENCE": "Clean reference",
    "V_OFFSET_NEG_1mV": "Voltage offset -1 mV",
    "V_OFFSET_POS_1mV": "Voltage offset +1 mV",
    "V_OFFSET_NEG_5mV": "Voltage offset -5 mV",
    "V_OFFSET_POS_5mV": "Voltage offset +5 mV",
    "V_OFFSET_NEG_10mV": "Voltage offset -10 mV",
    "V_OFFSET_POS_10mV": "Voltage offset +10 mV",
    "I_GAIN_NEG_0p5pct": "Current gain -0.5%",
    "I_GAIN_POS_0p5pct": "Current gain +0.5%",
    "I_GAIN_NEG_1p0pct": "Current gain -1.0%",
    "I_GAIN_POS_1p0pct": "Current gain +1.0%",
    "COMBINED_NEG_10mV_NEG_1pct": "-10 mV and -1% gain",
    "COMBINED_POS_10mV_POS_1pct": "+10 mV and +1% gain",
    "RANDOM_1mV_0p1pct": "Random: 1 mV + 0.1% current",
    "RANDOM_5mV_0p5pct": "Random: 5 mV + 0.5% current",
}

SCENARIO_ORDER = list(SCENARIO_LABELS)


def _write(
    frame: pd.DataFrame,
    name: str,
    output_dir: Path,
) -> tuple[Path, Path]:
    """Write one table as CSV plus a LaTeX fragment."""
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / f"{name}.csv"
    tex_path = output_dir / f"{name}.tex"

    frame.to_csv(csv_path, index=False)
    tex_path.write_text(
        frame.to_latex(
            index=False,
            escape=False,
            float_format=lambda value: f"{value:.3f}",
            na_rep="--",
        ),
        encoding="utf-8",
    )
    return csv_path, tex_path


def _baseline_lookup(
    path: Path,
    baseline: str,
) -> tuple[float, float]:
    table = pd.read_csv(path)
    row = table.loc[table["Baseline"].eq(baseline)]
    if len(row) != 1:
        raise KeyError(f"Expected one {baseline!r} row in {path}, found {len(row)}.")
    return (
        float(row.iloc[0]["Macro_RMSE"]),
        float(row.iloc[0]["Macro_MAE"]),
    )


def _ci_string(
    lower: float,
    upper: float,
) -> str:
    if not np.isfinite(lower) or not np.isfinite(upper):
        return "--"
    return f"[{lower:.3f}, {upper:.3f}]"


def _ordered(
    frame: pd.DataFrame,
    column: str,
    order: list[str],
) -> pd.DataFrame:
    rank = {name: i for i, name in enumerate(order)}
    out = frame.copy()
    out["_sort"] = out[column].map(rank).fillna(len(order))
    return out.sort_values(["_sort", column]).drop(columns="_sort").reset_index(drop=True)


def build_dataset_summary(
    constant: pd.DataFrame,
    variable: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict] = []
    for protocol, frame in [("Constant", constant), ("Variable", variable)]:
        for source in ["CALCE", "NASA"]:
            subset = frame.loc[frame["source_dataset"].eq(source)]
            rows.append({
                "Protocol group": protocol,
                "Source": source,
                "Batteries": int(subset["battery_id"].nunique()),
                "Independent observations": int(len(subset)),
            })
        rows.append({
            "Protocol group": f"{protocol} total",
            "Source": "",
            "Batteries": int(frame["battery_id"].nunique()),
            "Independent observations": int(len(frame)),
        })
    return pd.DataFrame(rows)


def build_feature_audit(
    path: Path,
) -> pd.DataFrame:
    audit = pd.read_csv(path)
    pivot = audit.pivot_table(
        index="Protocol",
        columns="Classification",
        values="N_Features",
        aggfunc="first",
    ).reset_index()

    expected_columns = {
        "STRICT_SAFE": "Strict-safe",
        "PARTIAL_OBSERVATION": "Partial",
        "HISTORY_ONLY": "History-only",
        "INADMISSIBLE": "Inadmissible",
        "REVIEW_REQUIRED": "Review/excluded",
    }
    for raw in expected_columns:
        if raw not in pivot.columns:
            raise KeyError(f"Feature-audit summary is missing classification {raw!r}.")
    pivot = pivot.rename(columns=expected_columns)
    return pivot[
        [
            "Protocol", "Strict-safe", "Partial", "History-only",
            "Inadmissible", "Review/excluded",
        ]
    ]


def build_hgb_configs() -> pd.DataFrame:
    return pd.DataFrame(
        [
            ["H0", 0.05, 300, 15, 5, 1.0],
            ["H1", 0.04, 500, 15, 10, 2.0],
            ["H2", 0.03, 700, 31, 10, 3.0],
        ],
        columns=[
            "Config.", "Learning rate", "Max. iter.", "Max. leaves",
            "Min. leaf samples", "L2",
        ],
    )


def build_lobo_macro(
    constant_baselines: Path,
    variable_baselines: Path,
    constant_macro: Path,
    variable_macro: Path,
) -> pd.DataFrame:
    rows: list[list[object]] = []
    for protocol, baseline_path, macro_path in [
        ("Constant", constant_baselines, constant_macro),
        ("Variable", variable_baselines, variable_macro),
    ]:
        macro = pd.read_csv(macro_path)
        for label, key in [
            ("Equal-battery training mean", "EqualBatteryTrainMean"),
            ("Age-only linear", "AgeOnlyLinear"),
            (
                "Previous-target persistence (history-assisted)",
                "PreviousTargetPersistence_HISTORY_ASSISTED",
            ),
        ]:
            rmse, mae = _baseline_lookup(baseline_path, key)
            rows.append([protocol, label, rmse, mae])

        for formulation in [
            "STRICT_ELECTRICAL_HGB",
            "STRICT_ELECTRICAL_PLUS_AGE_HGB",
        ]:
            row = macro.loc[macro["Formulation"].eq(formulation)]
            if len(row) != 1:
                raise KeyError(f"Missing unique {formulation!r} in {macro_path}.")
            rows.append([
                protocol,
                f"{MODEL_LABELS[formulation]} HGB",
                float(row.iloc[0]["Macro_RMSE_Raw"]),
                float(row.iloc[0]["Macro_MAE_Raw"]),
            ])

    return pd.DataFrame(
        rows,
        columns=["Protocol", "Model / baseline", "RMSE", "MAE"],
    )


def build_per_battery(
    path: Path,
    battery_order: list[str],
) -> pd.DataFrame:
    table = pd.read_csv(path)
    pivot = table.pivot_table(
        index=["Holdout_Battery", "Source"],
        columns="Formulation",
        values="RMSE_Raw",
        aggfunc="first",
    ).reset_index()
    pivot = pivot.rename(columns={
        "Holdout_Battery": "Battery",
        "STRICT_ELECTRICAL_HGB": "Strict electrical",
        "STRICT_ELECTRICAL_PLUS_AGE_HGB": "Strict electrical + age",
    })
    required = ["Battery", "Source", "Strict electrical", "Strict electrical + age"]
    missing = [column for column in required if column not in pivot.columns]
    if missing:
        raise KeyError(f"{path} is missing per-battery fields after pivot: {missing}")
    return _ordered(pivot[required], "Battery", battery_order)


def _bootstrap_row(
    bootstrap: pd.DataFrame,
    study: str,
    model: str,
    metric: str,
    train_source: str | None = None,
    test_source: str | None = None,
) -> pd.Series:
    mask = (
        bootstrap["Study"].eq(study)
        & bootstrap["Model"].eq(model)
        & bootstrap["Metric"].eq(metric)
    )
    if train_source is not None:
        mask &= bootstrap["Train_Source"].eq(train_source)
    if test_source is not None:
        mask &= bootstrap["Test_Source"].eq(test_source)

    rows = bootstrap.loc[mask]
    if len(rows) != 1:
        raise KeyError(
            "Expected one bootstrap row for "
            f"{study}/{model}/{metric}/{train_source}/{test_source}; found {len(rows)}."
        )
    return rows.iloc[0]


def build_bootstrap_primary(
    path: Path,
) -> pd.DataFrame:
    bootstrap = pd.read_csv(path)
    specifications = [
        ("CONSTANT_LOBO", "Constant LOBO", "STRICT_ELECTRICAL_HGB",
         "Strict electrical", "Macro_RMSE", "RMSE"),
        ("CONSTANT_LOBO", "Constant LOBO", "STRICT_ELECTRICAL_PLUS_AGE_HGB",
         "Strict electrical + age", "Macro_RMSE", "RMSE"),
        ("CONSTANT_LOBO", "Constant LOBO", "STRICT_ELECTRICAL_PLUS_AGE_HGB",
         "Strict electrical + age", "Macro_MAE", "MAE"),
        ("VARIABLE_LOBO", "Variable LOBO", "STRICT_ELECTRICAL_HGB",
         "Strict electrical", "Macro_RMSE", "RMSE"),
        ("VARIABLE_LOBO", "Variable LOBO", "STRICT_ELECTRICAL_PLUS_AGE_HGB",
         "Strict electrical + age", "Macro_RMSE", "RMSE"),
        ("VARIABLE_LOBO", "Variable LOBO", "STRICT_ELECTRICAL_PLUS_AGE_HGB",
         "Strict electrical + age", "Macro_MAE", "MAE"),
    ]

    rows: list[list[object]] = []
    for study, study_label, model, model_label, metric, metric_label in specifications:
        item = _bootstrap_row(bootstrap, study, model, metric)
        rows.append([
            study_label,
            model_label,
            metric_label,
            float(item["Estimate"]),
            _ci_string(float(item["CI95_Lower"]), float(item["CI95_Upper"])),
        ])

    return pd.DataFrame(
        rows,
        columns=["Study", "Model", "Metric", "Estimate", "95% CI"],
    )


def build_transfer_results(
    summary_path: Path,
    bootstrap_path: Path,
) -> pd.DataFrame:
    summary = pd.read_csv(summary_path)
    bootstrap = pd.read_csv(bootstrap_path)
    rows: list[list[object]] = []

    for train, test in [("NASA", "CALCE"), ("CALCE", "NASA")]:
        for formulation in [
            "STRICT_ELECTRICAL_HGB",
            "STRICT_ELECTRICAL_PLUS_AGE_HGB",
        ]:
            result = summary.loc[
                summary["Train_Source"].eq(train)
                & summary["Test_Source"].eq(test)
                & summary["Formulation"].eq(formulation)
            ]
            if len(result) != 1:
                raise KeyError(
                    f"Expected one transfer row for {formulation} {train}->{test}."
                )
            result = result.iloc[0]
            boot = _bootstrap_row(
                bootstrap,
                "CONSTANT_TRANSFER",
                formulation,
                "Macro_RMSE",
                train_source=train,
                test_source=test,
            )
            rows.append([
                f"{train} -> {test}",
                MODEL_LABELS[formulation],
                float(result["Macro_RMSE_Raw"]),
                float(result["Macro_MAE_Raw"]),
                _ci_string(float(boot["CI95_Lower"]), float(boot["CI95_Upper"])),
            ])

    return pd.DataFrame(
        rows,
        columns=["Train -> test", "Model", "RMSE", "MAE", "RMSE 95% battery CI"],
    )


def build_partial_depth(
    path: Path,
) -> pd.DataFrame:
    depth = pd.read_csv(path)
    depth = depth.loc[
        depth["Metric"].eq("P36_CapacityFraction_of_TargetCapacity"),
        ["Battery", "Source", "Median"],
    ].copy()
    depth["Median observed capacity fraction (%)"] = (
        pd.to_numeric(depth["Median"], errors="coerce") * 100.0
    )
    depth = depth.drop(columns="Median")
    return _ordered(
        depth,
        "Battery",
        CONSTANT_BATTERY_ORDER,
    )


def build_partial_comparison(
    path: Path,
) -> pd.DataFrame:
    table = pd.read_csv(path)
    label_map = {
        "STRICT_ELECTRICAL_HGB": "Strict electrical",
        "STRICT_ELECTRICAL_PLUS_AGE_HGB": "Strict electrical + age",
        "STRICT_ELECTRICAL_PLUS_3P6_HGB":
            "Strict electrical + 3.6 V partial observation",
        "STRICT_ELECTRICAL_PLUS_AGE_PLUS_3P6_HGB":
            "Strict electrical + age + 3.6 V partial observation",
    }
    order = list(label_map)
    rows = []
    for model in order:
        result = table.loc[table["Model"].eq(model)]
        if len(result) != 1:
            raise KeyError(f"Expected one partial-observation row for {model!r}.")
        result = result.iloc[0]
        rows.append([
            label_map[model],
            float(result["Macro_RMSE"]),
            float(result["Macro_MAE"]),
        ])
    return pd.DataFrame(
        rows,
        columns=["Information setting / model", "Macro RMSE", "Macro MAE"],
    )


def build_ablation_table(
    path: Path,
) -> pd.DataFrame:
    table = pd.read_csv(path)
    order = list(ABLATION_LABELS)
    rows = []
    for analysis in order:
        result = table.loc[table["Analysis"].eq(analysis)]
        if len(result) != 1:
            raise KeyError(f"Expected one ablation row for {analysis!r}.")
        result = result.iloc[0]
        change = float(result["RMSE_Change_vs_Full_Pct"])
        rows.append([
            ABLATION_LABELS[analysis],
            int(result["Removed_Features"]),
            float(result["Macro_RMSE"]),
            np.nan if analysis == "FULL_STRICT_ELECTRICAL_PLUS_AGE" else change,
        ])
    return pd.DataFrame(
        rows,
        columns=["Configuration", "Removed features", "Macro RMSE", "RMSE change (%)"],
    )


def build_measurement_table(
    path: Path,
) -> pd.DataFrame:
    table = pd.read_csv(path)
    rows = []
    for scenario in SCENARIO_ORDER:
        result = table.loc[table["Scenario"].eq(scenario)]
        if len(result) != 1:
            raise KeyError(f"Expected one robustness row for {scenario!r}.")
        result = result.iloc[0]

        scenario_type = str(
            result.get("Scenario_Type", result.get("Type", ""))
        ).upper()
        if scenario_type == "RANDOM":
            scenario_type = "STOCHASTIC"
        if scenario_type == "CLEAN":
            type_label = "Clean"
        elif scenario_type == "SYSTEMATIC":
            type_label = "Systematic"
        elif scenario_type == "STOCHASTIC":
            type_label = "Stochastic"
        else:
            type_label = scenario_type.title()

        rmse_column = (
            "Macro_RMSE"
            if "Macro_RMSE" in table.columns
            else "Macro_RMSE_Mean"
        )
        lower_column = (
            "RMSE_Replicate_CI95_Lower"
            if "RMSE_Replicate_CI95_Lower" in table.columns
            else "Macro_RMSE_CI95_Lower"
        )
        upper_column = (
            "RMSE_Replicate_CI95_Upper"
            if "RMSE_Replicate_CI95_Upper" in table.columns
            else "Macro_RMSE_CI95_Upper"
        )

        if scenario.startswith("RANDOM_"):
            interval = _ci_string(
                float(result[lower_column]),
                float(result[upper_column]),
            )
        else:
            interval = "--"

        change = float(result["RMSE_Change_vs_Clean_Pct"])
        rows.append([
            SCENARIO_LABELS[scenario],
            type_label,
            float(result[rmse_column]),
            np.nan if scenario == "CLEAN_STEP6_REFERENCE" else change,
            interval,
        ])

    return pd.DataFrame(
        rows,
        columns=[
            "Scenario", "Type", "Macro RMSE", "RMSE change (%)",
            "95% replicate interval",
        ],
    )


def run(
    root: Path,
    output_dir: Path,
) -> list[Path]:
    """Create all twelve current-manuscript table artifacts."""
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs: list[Path] = []

    constant = pd.read_csv(root / "constant_dataset/constant_observations.csv")
    variable = pd.read_csv(root / "variable_dataset/variable_independent_observations.csv")

    tables: list[tuple[str, pd.DataFrame]] = [
        ("dataset_summary", build_dataset_summary(constant, variable)),
        (
            "feature_audit",
            build_feature_audit(
                root / "feature_audit/feature_admissibility_summary.csv"
            ),
        ),
        ("hgb_configs", build_hgb_configs()),
        (
            "lobo_macro",
            build_lobo_macro(
                root / "constant_dataset/constant_baseline_macro.csv",
                root / "variable_dataset/variable_baseline_macro.csv",
                root / "constant_hgb/constant_hgb_macro.csv",
                root / "variable_hgb/variable_hgb_macro.csv",
            ),
        ),
        (
            "constant_perbattery",
            build_per_battery(
                root / "constant_hgb/constant_hgb_per_battery.csv",
                CONSTANT_BATTERY_ORDER,
            ),
        ),
        (
            "variable_perbattery",
            build_per_battery(
                root / "variable_hgb/variable_hgb_per_battery.csv",
                VARIABLE_BATTERY_ORDER,
            ),
        ),
        (
            "bootstrap_primary",
            build_bootstrap_primary(
                root / "statistical_analysis/battery_bootstrap_confidence_intervals.csv"
            ),
        ),
        (
            "transfer_results",
            build_transfer_results(
                root / "constant_transfer/constant_transfer_summary.csv",
                root / "statistical_analysis/battery_bootstrap_confidence_intervals.csv",
            ),
        ),
        (
            "partial_depth",
            build_partial_depth(
                root / "constant_3p6/constant_3p6_observation_depth.csv"
            ),
        ),
        (
            "partial_3p6",
            build_partial_comparison(
                root / "constant_3p6/constant_3p6_strict_comparison.csv"
            ),
        ),
        (
            "feature_ablation",
            build_ablation_table(
                root / "statistical_analysis/strict_ablation_summary.csv"
            ),
        ),
        (
            "measurement_perturbation",
            build_measurement_table(
                root / "measurement_robustness/measurement_robustness_summary.csv"
            ),
        ),
    ]

    for name, table in tables:
        outputs.extend(_write(table, name, output_dir))
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outputs-root", type=Path, default=Path("outputs"))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/manuscript_tables"),
    )
    args = parser.parse_args()
    for path in run(args.outputs_root, args.output_dir):
        print(path)


if __name__ == "__main__":
    main()
