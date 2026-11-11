# STEP 11L — FIXED ELAPSED-DISCHARGE-TIME SENSITIVITY
# Run locally with explicit raw and frozen-input paths.
# Adapts the unchanged validated Step 11G raw loaders and Step 6 nested-LOBO helpers.
# Install prerequisites in Colab before running: pip install openpyxl pandas numpy scipy scikit-learn matplotlib

from __future__ import annotations

from pathlib import Path
from typing import Any
import re
import shutil
import time
import warnings
import zipfile

import numpy as np
import pandas as pd

from scipy.signal import find_peaks, peak_widths, peak_prominences

from sklearn.impute import SimpleImputer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error


warnings.filterwarnings("ignore", category=RuntimeWarning)

_HAS_SCIPY = True


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path('.')

DRIVE_ZIP = (
    PROJECT_ROOT
    / "raw_data"
    / "collab.zip"
)

STEP5_DIR = (
    PROJECT_ROOT
    / "frozen_data"
    / "step5_constant"
)

STEP6_DIR = (
    PROJECT_ROOT
    / "results"
    / "step6_constant_hgb"
)

STEP8_DIR = (
    PROJECT_ROOT
    / "results"
    / "step8_constant_3p6"
)

STEP5_DATA_PATH = (
    STEP5_DIR
    / "Step5_Revised_Constant_Cycle_Observations.csv"
)

STEP5_FEATURE_LIST_PATH = (
    STEP5_DIR
    / "Step5_Frozen_Strict_Constant_Feature_Names.csv"
)

# Alias required by the original validated Step 6 helper:
#     load_frozen_feature_list()
FEATURE_LIST_PATH = STEP5_FEATURE_LIST_PATH

STEP6_PREDICTIONS_PATH = (
    STEP6_DIR
    / "Step6_Constant_HGB_Predictions.csv"
)

STEP6_MACRO_PATH = (
    STEP6_DIR
    / "Step6_Constant_HGB_Macro_Summary.csv"
)

STEP8_MACRO_PATH = (
    STEP8_DIR
    / "Step8_Constant_3p6V_Macro_Summary.csv"
)

STEP8_VS_STRICT_PATH = (
    STEP8_DIR
    / "Step8_Constant_3p6V_vs_Strict.csv"
)

STEP8_AUGMENTED_PATH = (
    STEP8_DIR
    / "Step8_Constant_3p6V_Augmented_Dataset.csv"
)

LOCAL_ZIP = Path(
    "work/step11g_collab.zip"
)

EXTRACT_ROOT = Path(
    "work/step11g_constant_raw"
)

LOCAL_OUTPUT_DIR = Path(
    "work/step11g_local"
)

DRIVE_OUTPUT_DIR = (
    PROJECT_ROOT
    / "reviewer_response"
    / "step11g_true_3p6_censoring"
)

LOCAL_OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

DRIVE_OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

OUTPUT_WORKBOOK = (
    LOCAL_OUTPUT_DIR
    / "Reviewer_Step11G_True_3p6V_Censored_Prediction.xlsx"
)

DRIVE_WORKBOOK = (
    DRIVE_OUTPUT_DIR
    / OUTPUT_WORKBOOK.name
)

AUTO_DOWNLOAD_RESULTS = False


# ============================================================
# FROZEN FEATURE-ENGINEERING GLOBALS FROM STEP 10
# ============================================================

TARGET_COL = "Calculated_SOH(%)"
GROUP_COL = "battery_id"
BATTERY_COL = "battery_id"
CYCLE_COL = "Cycle_Index"
SOURCE_COL = "source_dataset"
PROTOCOL_COL = "protocol"

CONSTANT_PROTOCOL = 1
VARIABLE_PROTOCOL = 0

EARLY_BASELINE_FINITE_OBSERVATIONS = 5

CALCE_CONSTANT_BATTERIES = [
    ("CS2_33", ["Channel_1-006"]),
    ("CS2_34", ["Channel_1-007"]),
    ("CS2_35", ["Channel_1-008"]),
    ("CS2_36", ["Channel_1-009"]),
    ("CS2_37", ["Channel_1-010"]),
    ("CS2_38", ["Channel_1-011"]),
]

NASA_CONSTANT_BATTERIES = [
    "B0005",
    "B0006",
    "B0007",
    "B0018",
]

DISCHARGE_PROGRESS_LANDMARKS = tuple(
    i / 10
    for i in range(1, 10)
)

DISCHARGE_VOLTAGE_LANDMARKS = (
    4.0,
    3.8,
    3.6,
    3.4,
    3.2,
)

INVARIANT_EXACT_BASE_FEATURES = {
    "Internal_Resistance_median",
    "dVdt_mean",
    "avg_abs_dvdt",
    "mean_abs_dVdt_over_I",
    "median_abs_dVdt_over_I",
    "Discharge_cumulativeCapacity",
    "Discharge_cumulativeEnergy",
    "Discharge_duration",
    "Discharge_startVoltage",
    "Discharge_endVoltage",
    "DeltaV_discharge",
    "Discharge_Voltage_mean",
    "Discharge_Voltage_std",
    "Discharge_Voltage_q10",
    "Discharge_Voltage_q25",
    "Discharge_Voltage_q50",
    "Discharge_Voltage_q75",
    "Discharge_Voltage_q90",
    "Discharge_Voltage_IQR",
    "Discharge_Voltage_MAD",
    "Discharge_Current_mean",
    "Discharge_Current_std",
    "Discharge_Current_IQR",
    "Discharge_Current_MAD",
    "Discharge_AverageVoltage",
    "Discharge_Voltage_EarlyMean",
    "Discharge_Voltage_MiddleMean",
    "Discharge_Voltage_LateMean",
    "Discharge_LateMinusEarlyVoltage",
    "Discharge_VoltageDropPerHour",
    "Discharge_VoltageDropPerAh",
    "Discharge_EnergyPerHour",
    "Discharge_Step7_IC_peakLocation",
    "Discharge_Step7_IC_peakProminence",
    "Discharge_Step7_IC_peakWidth",
    "Discharge_Step7_IC_peakCount",
    "Discharge_Step7_CC_slope",
}

LEGACY_HISTORY_BASE_FEATURES = [
    "Internal_Resistance_median",
    "dVdt_mean",
    "avg_abs_dvdt",
    "mean_abs_dVdt_over_I",
    "median_abs_dVdt_over_I",
    "Discharge_cumulativeCapacity",
    "Discharge_cumulativeEnergy",
    "Discharge_duration",
    "Discharge_startVoltage",
    "Discharge_endVoltage",
    "DeltaV_discharge",
    "Discharge_Voltage_mean",
    "Discharge_Voltage_std",
    "Discharge_Voltage_IQR",
    "Discharge_Voltage_MAD",
    "Discharge_AverageVoltage",
    "Discharge_VoltageAt_Q50",
    "Discharge_Voltage_LateMean",
    "Discharge_Step7_IC_peakLocation",
    "Discharge_Step7_IC_peakProminence",
    "Discharge_Step7_CC_slope",
]

PAST_ONLY_BASE_FEATURES = [
    "Discharge_cumulativeCapacity",
    "Discharge_cumulativeEnergy",
    "Discharge_duration",
    "Discharge_AverageVoltage",
    "Discharge_Voltage_mean",
    "Discharge_Voltage_std",
]

JOINT_NON_FEATURE_COLUMNS = {
    TARGET_COL,
    GROUP_COL,
    SOURCE_COL,
    PROTOCOL_COL,
}


# ============================================================
# STEP 6 MODEL GLOBALS
# ============================================================

EXPECTED_STRICT_FEATURE_COUNT = 335

MIN_SOURCE_COVERAGE = 0.95

EXPLICIT_AGE_STRICT_FEATURES = {
    "LEGACY__Cycle_Age",
    "LEGACY__Cycle_Index",
    "LEGACY__Log_Cycle_Age",
}

AGE_COLUMNS = [
    "Cycle_Age",
    "Log_Cycle_Age",
]

CORR_THRESHOLDS = [
    0.90,
    0.93,
    0.95,
]

TOP_K_OPTIONS = [
    8,
    12,
    16,
]

HGB_CONFIGS = {
    "H0": {
        "learning_rate": 0.05,
        "max_iter": 300,
        "max_leaf_nodes": 15,
        "min_samples_leaf": 5,
        "l2_regularization": 1.0,
    },
    "H1": {
        "learning_rate": 0.04,
        "max_iter": 500,
        "max_leaf_nodes": 15,
        "min_samples_leaf": 10,
        "l2_regularization": 2.0,
    },
    "H2": {
        "learning_rate": 0.03,
        "max_iter": 700,
        "max_leaf_nodes": 31,
        "min_samples_leaf": 10,
        "l2_regularization": 3.0,
    },
}

FEATURE_TUNING_MODEL = "H1"

RANDOM_STATE = 42

CLIP_LOW = 70.0
CLIP_HIGH = 100.0


# ============================================================
# STEP 11G OBSERVATION BOUNDARY
# ============================================================

CENSOR_VOLTAGE_V = 3.6

EXPECTED_FULL_PRIMARY = 4590

EXPECTED_CROSSING_COUNTS = {
    "CS2_33": 609,
    "CS2_34": 595,
    "CS2_35": 653,
    "CS2_36": 636,
    "CS2_37": 724,
    "CS2_38": 778,
    "B0005": 163,
    "B0006": 104,
    "B0007": 168,
    "B0018": 132,
}

EXPECTED_CROSSING_TOTAL = sum(
    EXPECTED_CROSSING_COUNTS.values()
)

EXPECTED_SOURCE_COUNTS = {
    "CALCE": 3995,
    "NASA": 567,
}


# ============================================================
# VALIDATED STEP 10 RAW/FEATURE HELPERS
# ============================================================

def parse_calce_workbook_date(path: Path):
    """
    Parse the acquisition date embedded in CALCE workbook filenames.

    Expected examples
    -----------------
    CS2_33_8_17_10.xlsx  -> 2010-08-17
    CS2_33_1_10_11.xlsx  -> 2011-01-10

    The battery id itself contains an underscore and digits, so we scan numeric
    triplets from right to left and accept the first valid month/day/year
    triplet. Two-digit years are interpreted as 2000-2099.
    """
    path = Path(path)
    parts = path.stem.split("_")
    for start in range(len(parts) - 3, -1, -1):
        month_s, day_s, year_s = parts[start:start + 3]
        if not (month_s.isdigit() and day_s.isdigit() and year_s.isdigit()):
            continue
        if len(year_s) not in (2, 4):
            continue
        month = int(month_s)
        day = int(day_s)
        year = int(year_s)
        if len(year_s) == 2:
            year += 2000
        try:
            stamp = pd.Timestamp(year=year, month=month, day=day)
        except ValueError:
            continue
        return stamp
    return None

def _calce_workbook_sort_key(path: Path, folder: Path):
    stamp = parse_calce_workbook_date(path)
    rel = str(Path(path).relative_to(folder)).lower()
    if stamp is None:
        return (1, 9999, 12, 31, rel)
    return (0, int(stamp.year), int(stamp.month), int(stamp.day), rel)

def extract_files(folder_path):
    """
    Return CALCE xlsx files in acquisition-date order.

    IMPORTANT: ZIP extraction on Colab destroys the original Windows creation
    times used by Run 1. Therefore Run 2.2 parses month/day/year from each CALCE
    workbook filename and sorts chronologically. For a multi-workbook battery,
    training is stopped if any workbook date cannot be parsed rather than
    silently using an incorrect lexical order.
    """
    folder = Path(folder_path)
    if not folder.is_dir():
        raise FileNotFoundError(f"Battery folder does not exist: {folder}")
    excel_files = [
        path for path in folder.rglob("*.xlsx")
        if not path.name.startswith("~$")
    ]
    if not excel_files:
        raise FileNotFoundError(f"No .xlsx files were found in: {folder}")

    if len(excel_files) > 1:
        unparsed = [p for p in excel_files if parse_calce_workbook_date(p) is None]
        if unparsed:
            names = [str(p.relative_to(folder)) for p in unparsed]
            raise ValueError(
                "Could not parse acquisition dates from CALCE workbook filename(s): "
                f"{names}. Expected a month_day_year pattern such as _8_17_10.xlsx. "
                "Training is stopped to prevent corrupt cycle chronology."
            )

    excel_files.sort(key=lambda p: _calce_workbook_sort_key(p, folder))
    return [str(path) for path in excel_files]

def process_files(folder_path, sheet_names_to_import):
    """
    Import the requested worksheets and continue Cycle_Index across
    consecutive files and sheets.
    """
    file_paths = extract_files(folder_path)

    dfs = []
    last_cycle_index = 0

    for file_path in file_paths:
        with pd.ExcelFile(file_path) as excel_file:
            available_sheets = excel_file.sheet_names

            for target_sheet in sheet_names_to_import:
                if target_sheet not in available_sheets:
                    continue

                df = pd.read_excel(
                    excel_file,
                    sheet_name=target_sheet,
                )

                if "Cycle_Index" not in df.columns:
                    raise KeyError(
                        f"'Cycle_Index' was not found in "
                        f"{file_path}, sheet {target_sheet}."
                    )

                df["Cycle_Index"] = pd.to_numeric(
                    df["Cycle_Index"],
                    errors="coerce",
                )

                df = df.loc[df["Cycle_Index"].notna()].copy()

                if df.empty:
                    continue

                df["Cycle_Index"] = (
                    df["Cycle_Index"] + last_cycle_index
                )

                last_cycle_index = float(
                    df["Cycle_Index"].max()
                )

                dfs.append(df)

    if not dfs:
        raise ValueError(
            "None of the requested worksheets produced usable data "
            f"for folder: {folder_path}"
        )

    return pd.concat(dfs, ignore_index=True)

def _normalise_column_name(name):
    """Return a case-insensitive alphanumeric column key."""
    return "".join(character.lower() for character in str(name) if character.isalnum())

def find_column(df, candidates, required=True):
    """Find the first matching column using case-insensitive keys."""
    lookup = {
        _normalise_column_name(column): column
        for column in df.columns
    }

    for candidate in candidates:
        match = lookup.get(_normalise_column_name(candidate))
        if match is not None:
            return match

    if required:
        raise KeyError(
            "None of the required columns were found. Tried: "
            + ", ".join(candidates)
        )

    return None

def read_nasa_csv(csv_path, battery_id):
    """Read one NASA preprocessing CSV and provide a clear error."""
    csv_path = Path(csv_path)

    if not csv_path.exists():
        raise FileNotFoundError(
            f"NASA CSV for {battery_id} was not found:\n{csv_path}\n"
            "Run the corresponding NASA download/preprocessing script first."
        )

    df = pd.read_csv(csv_path, low_memory=False)

    if df.empty:
        raise ValueError(f"NASA CSV for {battery_id} is empty: {csv_path}")

    return df

def _safe_stats(s: pd.Series, prefix: str):
    s = pd.to_numeric(s, errors="coerce").dropna()

    if len(s) == 0:
        return {
            f"{prefix}_max": np.nan,
            f"{prefix}_min": np.nan,
            f"{prefix}_mean": np.nan,
            f"{prefix}_std": np.nan,
            f"{prefix}_skewness": np.nan,
            f"{prefix}_kurtosis": np.nan,
        }

    return {
        f"{prefix}_max": float(s.max()),
        f"{prefix}_min": float(s.min()),
        f"{prefix}_mean": float(s.mean()),
        f"{prefix}_std": float(s.std(ddof=1)) if len(s) > 1 else 0.0,
        f"{prefix}_skewness": float(s.skew()) if len(s) > 2 else 0.0,
        f"{prefix}_kurtosis": float(s.kurt()) if len(s) > 3 else 0.0,
    }

def _linear_slope(x: np.ndarray, y: np.ndarray):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    m = np.isfinite(x) & np.isfinite(y)
    x = x[m]
    y = y[m]

    if len(x) < 2:
        return np.nan

    x0 = x - x.mean()
    denom = np.dot(x0, x0)
    if denom == 0:
        return np.nan

    return float(np.dot(x0, y - y.mean()) / denom)

def _ic_features_from_segment(
    seg: pd.DataFrame,
    time_col: str,
    volt_col: str,
    curr_col: str,
    dt_h_col: str,
    peak_sign: str,
    prefix: str,
):
    """IC features with derivative/voltage alignment preserved by one mask."""
    out = {
        f"{prefix}_IC_area": np.nan,
        f"{prefix}_IC_areaVoltage": np.nan,
        f"{prefix}_IC_max": np.nan,
        f"{prefix}_IC_min": np.nan,
        f"{prefix}_IC_mean": np.nan,
        f"{prefix}_IC_std": np.nan,
        f"{prefix}_IC_skewness": np.nan,
        f"{prefix}_IC_kurtosis": np.nan,
        f"{prefix}_IC_peak": np.nan,
        f"{prefix}_IC_peakWidth": np.nan,
        f"{prefix}_IC_peakLocation": np.nan,
        f"{prefix}_IC_peakProminence": np.nan,
        f"{prefix}_IC_peaksArea": np.nan,
        f"{prefix}_IC_peakLeftSlope": np.nan,
        f"{prefix}_IC_peakRightSlope": np.nan,
        f"{prefix}_IC_peakCount": np.nan,
        f"{prefix}_IC_peakToPeakDistance": np.nan,
    }
    if seg is None or len(seg) < 5:
        return out
    seg = seg.copy().sort_values(time_col)
    I = pd.to_numeric(seg[curr_col], errors="coerce").to_numpy(dtype=float)
    V = pd.to_numeric(seg[volt_col], errors="coerce").to_numpy(dtype=float)
    dt_h = pd.to_numeric(seg[dt_h_col], errors="coerce").to_numpy(dtype=float)
    valid_base = np.isfinite(I) & np.isfinite(V) & np.isfinite(dt_h)
    I, V, dt_h = I[valid_base], V[valid_base], dt_h[valid_base]
    if len(I) < 5:
        return out
    dQ = np.abs(I) * dt_h
    Q = np.cumsum(dQ)
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        dQ_dV = np.gradient(Q, V)
    valid_ic = np.isfinite(dQ_dV) & np.isfinite(V)
    y = np.asarray(dQ_dV[valid_ic], dtype=float)
    xV = np.asarray(V[valid_ic], dtype=float)
    if len(y) < 1:
        return out
    out[f"{prefix}_IC_area"] = float(np.trapezoid(y, dx=1.0))
    out[f"{prefix}_IC_max"] = float(np.nanmax(y))
    out[f"{prefix}_IC_min"] = float(np.nanmin(y))
    out[f"{prefix}_IC_mean"] = float(np.nanmean(y))
    out[f"{prefix}_IC_std"] = float(np.nanstd(y, ddof=1)) if len(y) > 1 else 0.0
    out[f"{prefix}_IC_skewness"] = float(pd.Series(y).skew()) if len(y) > 2 else 0.0
    out[f"{prefix}_IC_kurtosis"] = float(pd.Series(y).kurt()) if len(y) > 3 else 0.0
    if len(y) >= 2:
        order = np.argsort(xV)
        out[f"{prefix}_IC_areaVoltage"] = float(np.trapezoid(y[order], x=xV[order]))
    if not _HAS_SCIPY or len(y) < 5:
        return out
    y_for_peaks = -y if peak_sign == "negative" else y
    peaks, _ = find_peaks(y_for_peaks)
    out[f"{prefix}_IC_peakCount"] = float(len(peaks))
    if len(peaks) == 0:
        return out
    prominences = peak_prominences(y_for_peaks, peaks)[0]
    k = int(np.argmax(prominences))
    p = int(peaks[k])
    out[f"{prefix}_IC_peak"] = float(y[p])
    out[f"{prefix}_IC_peakLocation"] = float(xV[p])
    out[f"{prefix}_IC_peakProminence"] = float(prominences[k])
    widths_res = peak_widths(y_for_peaks, [p], rel_height=0.5)
    out[f"{prefix}_IC_peakWidth"] = float(widths_res[0][0])
    left = max(int(np.floor(widths_res[2][0])), 0)
    right = min(int(np.ceil(widths_res[3][0])), len(y) - 1)
    if right > left:
        out[f"{prefix}_IC_peaksArea"] = float(np.trapezoid(y[left:right + 1], dx=1.0))
    if p - 1 >= 0:
        out[f"{prefix}_IC_peakLeftSlope"] = float(y[p] - y[p - 1])
    if p + 1 < len(y):
        out[f"{prefix}_IC_peakRightSlope"] = float(y[p + 1] - y[p])
    if len(peaks) >= 2:
        peak_order = np.argsort(prominences)[::-1]
        p1, p2 = int(peaks[peak_order[0]]), int(peaks[peak_order[1]])
        out[f"{prefix}_IC_peakToPeakDistance"] = float(abs(xV[p1] - xV[p2]))
    return out

def build_cycle_features(
    df: pd.DataFrame,
    cycle_col="Cycle_Index",
    time_col="Test_Time(s)",
    volt_col="Voltage(V)",
    curr_col="Current(A)",
    step_col=None,
    segment_col=None,
    charge_mask=None,
    discharge_mask=None,
    step2_value=2,
    step4_value=4,
    step7_value=7,
    cc_tol_frac=0.05,
    cv_vtol_frac=0.005,
    eps=1e-12,
):
    d = df.copy()

    d[time_col] = pd.to_numeric(d[time_col], errors="coerce")
    d[volt_col] = pd.to_numeric(d[volt_col], errors="coerce")
    d[curr_col] = pd.to_numeric(d[curr_col], errors="coerce")
    d["_original_row_order"] = np.arange(len(d))

    grouping_columns = [cycle_col]
    sorting_columns = [cycle_col]

    if segment_col is not None:
        if segment_col not in d.columns:
            raise KeyError(
                f"segment_col '{segment_col}' was requested but is absent."
            )

        d[segment_col] = pd.to_numeric(d[segment_col], errors="coerce")
        grouping_columns.append(segment_col)
        sorting_columns.append(segment_col)

    sorting_columns.extend([time_col, "_original_row_order"])
    d = d.sort_values(sorting_columns)

    # For NASA Random Walk data, time restarts in every MATLAB segment.
    # Grouping by both cycle and segment prevents capacity, energy, and dV/dt
    # from being integrated across a segment boundary.
    dt_s = d.groupby(grouping_columns, dropna=False)[time_col].diff()
    d["dt_h"] = dt_s.fillna(0) / 3600.0
    d["dt_h"] = d["dt_h"].clip(lower=0)

    dV = d.groupby(grouping_columns, dropna=False)[volt_col].diff()
    dT = dt_s
    d["dVdt_V_per_s"] = dV / dT
    d.loc[(dT <= 0) | (~np.isfinite(d["dVdt_V_per_s"])), "dVdt_V_per_s"] = np.nan

    absI = np.abs(d[curr_col].to_numpy(dtype=float))
    d["abs_dVdt_over_absI"] = np.abs(d["dVdt_V_per_s"].to_numpy(dtype=float)) / np.maximum(absI, eps)
    d.loc[~np.isfinite(d["abs_dVdt_over_absI"]), "abs_dVdt_over_absI"] = np.nan

    if charge_mask is None:
        charge_mask = lambda x: x[curr_col] > 0
    if discharge_mask is None:
        discharge_mask = lambda x: x[curr_col] < 0

    d["P_W"] = d[volt_col] * d[curr_col]
    d["dE_Wh"] = d["P_W"] * d["dt_h"]

    d["I_ch_A"] = d[curr_col].clip(lower=0)
    d["I_dis_A"] = (-d[curr_col]).clip(lower=0)
    d["dQ_ch_Ah"] = d["I_ch_A"] * d["dt_h"]
    d["dQ_dis_Ah"] = d["I_dis_A"] * d["dt_h"]

    rows = []

    for cyc, g in d.groupby(cycle_col, sort=True):
        out = {"Cycle_Index": float(cyc)}
        g = g.dropna(subset=[time_col, volt_col, curr_col, "dt_h"])

        if len(g) == 0:
            rows.append(out)
            continue

        if "Internal_Resistance(Ohm)" in g.columns:
            r = pd.to_numeric(g["Internal_Resistance(Ohm)"], errors="coerce").dropna()
            out["Internal_Resistance_max"] = float(r.max()) if len(r) else np.nan
            out["Internal_Resistance_median"] = float(r.median()) if len(r) else np.nan
            out["Internal_Resistance_p90"] = float(r.quantile(0.90)) if len(r) else np.nan
        else:
            out["Internal_Resistance_max"] = np.nan
            out["Internal_Resistance_median"] = np.nan
            out["Internal_Resistance_p90"] = np.nan

        dvdt = pd.to_numeric(g["dVdt_V_per_s"], errors="coerce").dropna()
        out["dVdt_mean"] = float(dvdt.mean()) if len(dvdt) else np.nan
        out["avg_abs_dvdt"] = float(np.mean(np.abs(dvdt))) if len(dvdt) else np.nan

        dyn = pd.to_numeric(g["abs_dVdt_over_absI"], errors="coerce").dropna()
        out["mean_abs_dVdt_over_I"] = float(dyn.mean()) if len(dyn) else np.nan
        out["median_abs_dVdt_over_I"] = float(dyn.median()) if len(dyn) else np.nan

        ch = g[charge_mask(g)].copy()
        dis = g[discharge_mask(g)].copy()

        out["Charge_cumulativeCapacity"] = float(ch["dQ_ch_Ah"].sum()) if len(ch) else np.nan
        out["Charge_cumulativeEnergy"] = float(ch["dE_Wh"].sum()) if len(ch) else np.nan
        out["Charge_duration"] = float(ch["dt_h"].sum()) if len(ch) else np.nan
        out["Charge_startVoltage"] = float(ch[volt_col].iloc[0]) if len(ch) else np.nan
        out["Charge_endVoltage"] = float(ch[volt_col].iloc[-1]) if len(ch) else np.nan
        out["DeltaV_charge"] = (
            out["Charge_endVoltage"] - out["Charge_startVoltage"]
            if np.isfinite(out["Charge_endVoltage"]) and np.isfinite(out["Charge_startVoltage"])
            else np.nan
        )
        out.update(_safe_stats(ch[volt_col], "Charge_Voltage"))
        out.update(_safe_stats(ch[curr_col], "Charge_Current"))

        out["Discharge_cumulativeCapacity"] = float(dis["dQ_dis_Ah"].sum()) if len(dis) else np.nan
        out["Discharge_cumulativeEnergy"] = float(dis["dE_Wh"].sum()) if len(dis) else np.nan
        out["Discharge_duration"] = float(dis["dt_h"].sum()) if len(dis) else np.nan
        out["Discharge_startVoltage"] = float(dis[volt_col].iloc[0]) if len(dis) else np.nan
        out["Discharge_endVoltage"] = float(dis[volt_col].iloc[-1]) if len(dis) else np.nan
        out["DeltaV_discharge"] = (
            out["Discharge_startVoltage"] - out["Discharge_endVoltage"]
            if np.isfinite(out["Discharge_startVoltage"]) and np.isfinite(out["Discharge_endVoltage"])
            else np.nan
        )
        out.update(_safe_stats(dis[volt_col], "Discharge_Voltage"))
        out.update(_safe_stats(dis[curr_col], "Discharge_Current"))

        if np.isfinite(out["Charge_cumulativeCapacity"]) and out["Charge_cumulativeCapacity"] > 0 and np.isfinite(out["Discharge_cumulativeCapacity"]):
            out["Coulombic_Efficiency"] = float(out["Discharge_cumulativeCapacity"] / out["Charge_cumulativeCapacity"])
        else:
            out["Coulombic_Efficiency"] = np.nan

        if np.isfinite(out["Charge_cumulativeEnergy"]) and out["Charge_cumulativeEnergy"] > 0 and np.isfinite(out["Discharge_cumulativeEnergy"]):
            out["Energy_Efficiency"] = float(abs(out["Discharge_cumulativeEnergy"]) / out["Charge_cumulativeEnergy"])
        else:
            out["Energy_Efficiency"] = np.nan

        if step_col is not None and step_col in g.columns:
            s2 = g[(g[step_col] == step2_value)].copy()
            s2_ch = s2[s2[curr_col] > 0].copy()

            out.update(_ic_features_from_segment(
                s2_ch, time_col, volt_col, curr_col, "dt_h",
                peak_sign="positive", prefix="Charge_Step2"
            ))

            if len(s2_ch) >= 5:
                I_med = float(np.nanmedian(s2_ch[curr_col]))
                tol = cc_tol_frac * abs(I_med) if np.isfinite(I_med) and I_med != 0 else np.nan
                s2_cc = s2_ch[np.abs(s2_ch[curr_col] - I_med) <= tol].copy() if np.isfinite(tol) else s2_ch.copy()

                out["Charge_Step2_CC_duration"] = float(s2_cc["dt_h"].sum()) if len(s2_cc) else np.nan
                out["Charge_Step2_CC_currentMedian"] = float(np.nanmedian(s2_cc[curr_col])) if len(s2_cc) else np.nan
                out["Charge_Step2_CC_energy"] = float((s2_cc["dE_Wh"]).sum()) if len(s2_cc) else np.nan
                out["Charge_Step2_CC_currentStd"] = float(pd.Series(s2_cc[curr_col]).dropna().std(ddof=1)) if len(s2_cc) > 1 else (0.0 if len(s2_cc) else np.nan)
                out["Charge_Step2_CC_voltageStd"] = float(pd.Series(s2_cc[volt_col]).dropna().std(ddof=1)) if len(s2_cc) > 1 else (0.0 if len(s2_cc) else np.nan)

                ccI = pd.Series(s2_cc[curr_col]).dropna()
                out["Charge_Step2_CC_skewness"] = float(ccI.skew()) if len(ccI) > 2 else (0.0 if len(ccI) else np.nan)
                out["Charge_Step2_CC_kurtosis"] = float(ccI.kurt()) if len(ccI) > 3 else (0.0 if len(ccI) else np.nan)
            else:
                out["Charge_Step2_CC_duration"] = np.nan
                out["Charge_Step2_CC_currentMedian"] = np.nan
                out["Charge_Step2_CC_energy"] = np.nan
                out["Charge_Step2_CC_currentStd"] = np.nan
                out["Charge_Step2_CC_voltageStd"] = np.nan
                out["Charge_Step2_CC_skewness"] = np.nan
                out["Charge_Step2_CC_kurtosis"] = np.nan

            s4 = g[(g[step_col] == step4_value)].copy()
            s4_ch = s4[s4[curr_col] > 0].copy()

            if len(s4_ch) >= 5:
                V_med = float(np.nanmedian(s4_ch[volt_col]))
                vtol = cv_vtol_frac * abs(V_med) if np.isfinite(V_med) and V_med != 0 else np.nan
                s4_cv = s4_ch[np.abs(s4_ch[volt_col] - V_med) <= vtol].copy() if np.isfinite(vtol) else s4_ch.copy()

                out["Charge_Step4_CV_duration"] = float(s4_cv["dt_h"].sum()) if len(s4_cv) else np.nan
                out["Charge_Step4_CV_voltageMedian"] = float(np.nanmedian(s4_cv[volt_col])) if len(s4_cv) else np.nan

                t_h = (s4_cv[time_col].to_numpy(dtype=float) / 3600.0)
                I = s4_cv[curr_col].to_numpy(dtype=float)
                out["Charge_Step4_CV_slope"] = _linear_slope(t_h, I)

                out["Charge_Step4_CV_energy"] = float((s4_cv["dE_Wh"]).sum()) if len(s4_cv) else np.nan
                cvI = pd.Series(s4_cv[curr_col]).dropna()
                out["Charge_Step4_CV_skewness"] = float(cvI.skew()) if len(cvI) > 2 else (0.0 if len(cvI) else np.nan)
                out["Charge_Step4_CV_kurtosis"] = float(cvI.kurt()) if len(cvI) > 3 else (0.0 if len(cvI) else np.nan)

                out["Charge_Step4_CV_endCurrent"] = float(s4_cv[curr_col].iloc[-1]) if len(s4_cv) else np.nan
                out["Charge_Step4_CV_startCurrent"] = float(s4_cv[curr_col].iloc[0]) if len(s4_cv) else np.nan
                if np.isfinite(out["Charge_Step4_CV_startCurrent"]) and np.isfinite(out["Charge_Step4_CV_endCurrent"]):
                    out["Charge_Step4_CV_currentDrop"] = float(out["Charge_Step4_CV_startCurrent"] - out["Charge_Step4_CV_endCurrent"])
                else:
                    out["Charge_Step4_CV_currentDrop"] = np.nan

                if len(s4_cv) >= 5 and np.isfinite(out["Charge_Step4_CV_startCurrent"]) and out["Charge_Step4_CV_startCurrent"] > 0:
                    I0 = out["Charge_Step4_CV_startCurrent"]
                    target = 0.10 * I0
                    idx = np.where(s4_cv[curr_col].to_numpy(dtype=float) <= target)[0]
                    if len(idx) > 0:
                        t0 = float(s4_cv[time_col].iloc[0]) / 3600.0
                        tt = float(s4_cv[time_col].iloc[int(idx[0])]) / 3600.0
                        out["Charge_Step4_CV_tau10"] = float(max(tt - t0, 0.0))
                    else:
                        out["Charge_Step4_CV_tau10"] = np.nan
                else:
                    out["Charge_Step4_CV_tau10"] = np.nan
            else:
                out["Charge_Step4_CV_duration"] = np.nan
                out["Charge_Step4_CV_voltageMedian"] = np.nan
                out["Charge_Step4_CV_slope"] = np.nan
                out["Charge_Step4_CV_energy"] = np.nan
                out["Charge_Step4_CV_skewness"] = np.nan
                out["Charge_Step4_CV_kurtosis"] = np.nan
                out["Charge_Step4_CV_endCurrent"] = np.nan
                out["Charge_Step4_CV_startCurrent"] = np.nan
                out["Charge_Step4_CV_currentDrop"] = np.nan
                out["Charge_Step4_CV_tau10"] = np.nan

            s7 = g[(g[step_col] == step7_value)].copy()
            s7_dis = s7[s7[curr_col] < 0].copy()

            out.update(_ic_features_from_segment(
                s7_dis, time_col, volt_col, curr_col, "dt_h",
                peak_sign="positive", prefix="Discharge_Step7"
            ))

            if len(s7_dis) >= 5:
                I_med = float(np.nanmedian(s7_dis[curr_col]))
                tol = cc_tol_frac * abs(I_med) if np.isfinite(I_med) and I_med != 0 else np.nan
                s7_cc = s7_dis[np.abs(s7_dis[curr_col] - I_med) <= tol].copy() if np.isfinite(tol) else s7_dis.copy()

                out["Discharge_Step7_CC_duration"] = float(s7_cc["dt_h"].sum()) if len(s7_cc) else np.nan
                out["Discharge_Step7_CC_currentMedian"] = float(np.nanmedian(s7_cc[curr_col])) if len(s7_cc) else np.nan

                t_h = (s7_cc[time_col].to_numpy(dtype=float) / 3600.0)
                V = s7_cc[volt_col].to_numpy(dtype=float)
                out["Discharge_Step7_CC_slope"] = _linear_slope(t_h, V)

                out["Discharge_Step7_CC_energy"] = float((s7_cc["dE_Wh"]).sum()) if len(s7_cc) else np.nan
                out["Discharge_Step7_CC_currentStd"] = float(pd.Series(s7_cc[curr_col]).dropna().std(ddof=1)) if len(s7_cc) > 1 else (0.0 if len(s7_cc) else np.nan)
                out["Discharge_Step7_CC_voltageStd"] = float(pd.Series(s7_cc[volt_col]).dropna().std(ddof=1)) if len(s7_cc) > 1 else (0.0 if len(s7_cc) else np.nan)

                ccV = pd.Series(s7_cc[volt_col]).dropna()
                out["Discharge_Step7_CC_skewness"] = float(ccV.skew()) if len(ccV) > 2 else (0.0 if len(ccV) else np.nan)
                out["Discharge_Step7_CC_kurtosis"] = float(ccV.kurt()) if len(ccV) > 3 else (0.0 if len(ccV) else np.nan)

                out["Discharge_Step7_CC_tInv"] = float(1.0 / out["Discharge_Step7_CC_duration"]) if np.isfinite(out["Discharge_Step7_CC_duration"]) and out["Discharge_Step7_CC_duration"] > 0 else np.nan
            else:
                out["Discharge_Step7_CC_duration"] = np.nan
                out["Discharge_Step7_CC_currentMedian"] = np.nan
                out["Discharge_Step7_CC_slope"] = np.nan
                out["Discharge_Step7_CC_energy"] = np.nan
                out["Discharge_Step7_CC_currentStd"] = np.nan
                out["Discharge_Step7_CC_voltageStd"] = np.nan
                out["Discharge_Step7_CC_skewness"] = np.nan
                out["Discharge_Step7_CC_kurtosis"] = np.nan
                out["Discharge_Step7_CC_tInv"] = np.nan

        rows.append(out)

    return pd.DataFrame(rows)

def add_dataset_identity(features, battery_id, source_dataset, protocol):
    """Attach non-target dataset identity columns to cycle features."""
    output = features.copy()
    output[GROUP_COL] = battery_id
    output["source_dataset"] = source_dataset
    output["protocol"] = int(protocol)
    return output

def _robust_distribution_features(values: pd.Series, prefix: str) -> dict:
    """Return robust quantile, IQR, and MAD features for one signal."""
    s = pd.to_numeric(values, errors="coerce").dropna()
    output = {
        f"{prefix}_q10": np.nan,
        f"{prefix}_q25": np.nan,
        f"{prefix}_q50": np.nan,
        f"{prefix}_q75": np.nan,
        f"{prefix}_q90": np.nan,
        f"{prefix}_IQR": np.nan,
        f"{prefix}_MAD": np.nan,
    }
    if s.empty:
        return output

    quantiles = s.quantile([0.10, 0.25, 0.50, 0.75, 0.90])
    median = float(quantiles.loc[0.50])
    output.update(
        {
            f"{prefix}_q10": float(quantiles.loc[0.10]),
            f"{prefix}_q25": float(quantiles.loc[0.25]),
            f"{prefix}_q50": median,
            f"{prefix}_q75": float(quantiles.loc[0.75]),
            f"{prefix}_q90": float(quantiles.loc[0.90]),
            f"{prefix}_IQR": float(quantiles.loc[0.75] - quantiles.loc[0.25]),
            f"{prefix}_MAD": float(np.median(np.abs(s.to_numpy(float) - median))),
        }
    )
    return output

def _first_progress_value(progress: np.ndarray, values: np.ndarray, level: float):
    """Return the first finite value at or beyond a causal progress level."""
    valid = np.isfinite(progress) & np.isfinite(values)
    indexes = np.flatnonzero(valid & (progress >= level))
    if len(indexes) == 0:
        return np.nan
    return float(values[indexes[0]])

def build_extra_cycle_features(
    df: pd.DataFrame,
    cycle_col: str = "Cycle_Index",
    time_col: str = "Test_Time(s)",
    volt_col: str = "Voltage(V)",
    curr_col: str = "Current(A)",
    segment_col: str | None = None,
) -> pd.DataFrame:
    """Robust within-cycle discharge features with Q10..Q90 landmarks."""
    required = {cycle_col, time_col, volt_col, curr_col}
    missing = required - set(df.columns)
    if missing:
        raise KeyError("Missing columns for extra cycle features: " + ", ".join(sorted(missing)))
    d = df.copy()
    for col in [cycle_col, time_col, volt_col, curr_col]:
        d[col] = pd.to_numeric(d[col], errors="coerce")
    d["_row_order"] = np.arange(len(d))
    grouping_columns = [cycle_col]
    sorting_columns = [cycle_col]
    if segment_col is not None:
        if segment_col not in d.columns:
            raise KeyError(f"Requested segment column is absent: {segment_col}")
        d[segment_col] = pd.to_numeric(d[segment_col], errors="coerce")
        grouping_columns.append(segment_col)
        sorting_columns.append(segment_col)
    sorting_columns.extend([time_col, "_row_order"])
    d = d.dropna(subset=[cycle_col, time_col, volt_col, curr_col]).sort_values(sorting_columns)
    dt_s = d.groupby(grouping_columns, dropna=False)[time_col].diff()
    d["_dt_h"] = dt_s.fillna(0.0).clip(lower=0.0) / 3600.0
    d["_dQ_dis_Ah"] = (-d[curr_col]).clip(lower=0.0) * d["_dt_h"]
    d["_dE_dis_Wh"] = d[volt_col] * (-d[curr_col]).clip(lower=0.0) * d["_dt_h"]
    rows: list[dict] = []
    for cycle_value, cycle_df in d.groupby(cycle_col, sort=True):
        output: dict[str, Any] = {CYCLE_COL: float(cycle_value)}
        discharge = cycle_df.loc[cycle_df[curr_col] < 0].copy()
        output.update(_robust_distribution_features(discharge[volt_col], "Discharge_Voltage"))
        output.update(_robust_distribution_features(discharge[curr_col], "Discharge_Current"))
        if len(discharge) < 2:
            rows.append(output)
            continue
        voltage = discharge[volt_col].to_numpy(float)
        current = discharge[curr_col].to_numpy(float)
        dt_h = discharge["_dt_h"].to_numpy(float)
        dQ = discharge["_dQ_dis_Ah"].to_numpy(float)
        dE = discharge["_dE_dis_Wh"].to_numpy(float)
        cumulative_q = np.cumsum(np.nan_to_num(dQ, nan=0.0))
        cumulative_e = np.cumsum(np.nan_to_num(dE, nan=0.0))
        cumulative_t = np.cumsum(np.nan_to_num(dt_h, nan=0.0))
        total_q = float(cumulative_q[-1]) if len(cumulative_q) else np.nan
        total_e = float(cumulative_e[-1]) if len(cumulative_e) else np.nan
        total_t = float(cumulative_t[-1]) if len(cumulative_t) else np.nan
        if np.isfinite(total_q) and total_q > 0:
            progress = cumulative_q / total_q
        elif np.isfinite(total_t) and total_t > 0:
            progress = cumulative_t / total_t
        else:
            progress = np.linspace(0.0, 1.0, len(discharge))
        phase_masks = {
            "Early": progress <= 1 / 3,
            "Middle": (progress > 1 / 3) & (progress <= 2 / 3),
            "Late": progress > 2 / 3,
        }
        for phase_name, mask in phase_masks.items():
            phase_voltage = voltage[mask & np.isfinite(voltage)]
            phase_current = current[mask & np.isfinite(current)]
            output[f"Discharge_Voltage_{phase_name}Mean"] = (
                float(np.mean(phase_voltage)) if len(phase_voltage) else np.nan
            )
            output[f"Discharge_Voltage_{phase_name}Std"] = (
                float(np.std(phase_voltage, ddof=1)) if len(phase_voltage) > 1
                else (0.0 if len(phase_voltage) == 1 else np.nan)
            )
            output[f"Discharge_Current_{phase_name}Mean"] = (
                float(np.mean(phase_current)) if len(phase_current) else np.nan
            )
        for level in DISCHARGE_PROGRESS_LANDMARKS:
            label = int(round(level * 100))
            output[f"Discharge_VoltageAt_Q{label}"] = _first_progress_value(progress, voltage, level)
            output[f"Discharge_TimeAt_Q{label}_h"] = _first_progress_value(progress, cumulative_t, level)
            output[f"Discharge_EnergyAt_Q{label}_Wh"] = _first_progress_value(progress, cumulative_e, level)
        for threshold in DISCHARGE_VOLTAGE_LANDMARKS:
            label = str(threshold).replace(".", "p")
            indexes = np.flatnonzero(np.isfinite(voltage) & (voltage <= threshold))
            if len(indexes):
                idx = int(indexes[0])
                output[f"Discharge_TimeTo_{label}V_h"] = float(cumulative_t[idx])
                output[f"Discharge_CapacityTo_{label}V_Ah"] = float(cumulative_q[idx])
                output[f"Discharge_EnergyTo_{label}V_Wh"] = float(cumulative_e[idx])
            else:
                output[f"Discharge_TimeTo_{label}V_h"] = np.nan
                output[f"Discharge_CapacityTo_{label}V_Ah"] = np.nan
                output[f"Discharge_EnergyTo_{label}V_Wh"] = np.nan
        finite_voltage = voltage[np.isfinite(voltage)]
        voltage_drop = float(finite_voltage[0] - finite_voltage[-1]) if len(finite_voltage) >= 2 else np.nan
        output["Discharge_AverageVoltage"] = (
            float(total_e / total_q) if np.isfinite(total_e) and np.isfinite(total_q) and total_q > 0 else np.nan
        )
        output["Discharge_VoltageDropPerHour"] = (
            float(voltage_drop / total_t) if np.isfinite(voltage_drop) and np.isfinite(total_t) and total_t > 0 else np.nan
        )
        output["Discharge_VoltageDropPerAh"] = (
            float(voltage_drop / total_q) if np.isfinite(voltage_drop) and np.isfinite(total_q) and total_q > 0 else np.nan
        )
        output["Discharge_EnergyPerHour"] = (
            float(total_e / total_t) if np.isfinite(total_e) and np.isfinite(total_t) and total_t > 0 else np.nan
        )
        output["Discharge_LateMinusEarlyVoltage"] = (
            output.get("Discharge_Voltage_LateMean", np.nan)
            - output.get("Discharge_Voltage_EarlyMean", np.nan)
        )
        rows.append(output)
    return pd.DataFrame(rows)

def build_cycle_features_run2(
    df: pd.DataFrame,
    cycle_col="Cycle_Index",
    time_col="Test_Time(s)",
    volt_col="Voltage(V)",
    curr_col="Current(A)",
    step_col=None,
    segment_col=None,
    charge_mask=None,
    discharge_mask=None,
    step2_value=2,
    step4_value=4,
    step7_value=7,
    cc_tol_frac=0.05,
    cv_vtol_frac=0.005,
    eps=1e-12,
):
    """Build the original feature set and merge the Run 2 robust features."""
    base = build_cycle_features(
        df=df,
        cycle_col=cycle_col,
        time_col=time_col,
        volt_col=volt_col,
        curr_col=curr_col,
        step_col=step_col,
        segment_col=segment_col,
        charge_mask=charge_mask,
        discharge_mask=discharge_mask,
        step2_value=step2_value,
        step4_value=step4_value,
        step7_value=step7_value,
        cc_tol_frac=cc_tol_frac,
        cv_vtol_frac=cv_vtol_frac,
        eps=eps,
    )
    extra = build_extra_cycle_features(
        df=df,
        cycle_col=cycle_col,
        time_col=time_col,
        volt_col=volt_col,
        curr_col=curr_col,
        segment_col=segment_col,
    )
    return base.merge(extra, on=CYCLE_COL, how="left", validate="one_to_one")

def _causal_early_baseline(
    values: pd.Series,
    required_finite: int = EARLY_BASELINE_FINITE_OBSERVATIONS,
) -> pd.Series:
    """
    Expanding median until `required_finite` observations exist, then freeze.

    This gives early rows a baseline based only on information available up to
    that row and prevents later-life observations from altering the baseline.
    """
    s = pd.to_numeric(values, errors="coerce")
    expanding_median = s.expanding(min_periods=1).median()
    finite_count = s.notna().cumsum()
    reached = np.flatnonzero(finite_count.to_numpy() >= int(required_finite))
    baseline = expanding_median.copy()
    if len(reached):
        freeze_position = int(reached[0])
        freeze_value = expanding_median.iloc[freeze_position]
        baseline.iloc[freeze_position:] = freeze_value
    return baseline

def _rolling_slope(values: pd.Series, window: int) -> pd.Series:
    """Trailing rolling linear slope; no centered or future observations."""
    s = pd.to_numeric(values, errors="coerce")

    def slope_function(array: np.ndarray) -> float:
        y = np.asarray(array, dtype=float)
        valid = np.isfinite(y)
        if valid.sum() < 2:
            return np.nan
        x = np.arange(len(y), dtype=float)[valid]
        y = y[valid]
        x = x - x.mean()
        denominator = float(np.dot(x, x))
        if denominator <= 0:
            return np.nan
        return float(np.dot(x, y - y.mean()) / denominator)

    return s.rolling(window=int(window), min_periods=2).apply(
        slope_function, raw=True
    )

def _invariant_base_columns(df: pd.DataFrame) -> list[str]:
    columns: list[str] = []
    for col in df.columns:
        if col in INVARIANT_EXACT_BASE_FEATURES:
            columns.append(col)
            continue
        if col.startswith("Discharge_VoltageAt_Q"):
            columns.append(col)
        elif col.startswith("Discharge_TimeAt_Q"):
            columns.append(col)
        elif col.startswith("Discharge_EnergyAt_Q"):
            columns.append(col)
        elif col.startswith("Discharge_TimeTo_"):
            columns.append(col)
        elif col.startswith("Discharge_CapacityTo_"):
            columns.append(col)
        elif col.startswith("Discharge_EnergyTo_"):
            columns.append(col)
    return sorted(dict.fromkeys(columns))

def add_run2_invariant_features(combined_df: pd.DataFrame) -> pd.DataFrame:
    """
    Add target-free, causal, battery-relative features.

    The baseline for each raw physical feature is an expanding median until the
    fifth finite observation, then it is frozen. Thus a test battery's features
    use only that battery's observed physical measurements, never test SOH.
    """
    ordered = combined_df.sort_values([GROUP_COL, CYCLE_COL]).reset_index(drop=True).copy()
    base_columns = _invariant_base_columns(ordered)
    generated: dict[str, pd.Series] = {}

    for feature in base_columns:
        delta_all = pd.Series(np.nan, index=ordered.index, dtype=float)
        ratio_all = pd.Series(np.nan, index=ordered.index, dtype=float)
        pct_all = pd.Series(np.nan, index=ordered.index, dtype=float)
        lag_pct_all = pd.Series(np.nan, index=ordered.index, dtype=float)
        slope_pct_all = pd.Series(np.nan, index=ordered.index, dtype=float)
        ewm_pct_all = pd.Series(np.nan, index=ordered.index, dtype=float)

        for _, battery_df in ordered.groupby(GROUP_COL, sort=False):
            idx = battery_df.index
            s = pd.to_numeric(battery_df[feature], errors="coerce")
            baseline = _causal_early_baseline(s, EARLY_BASELINE_FINITE_OBSERVATIONS)
            safe_den = baseline.abs().where(baseline.abs() > 1e-10)
            delta = s - baseline
            ratio = s / safe_den
            pct = 100.0 * delta / safe_den
            delta_all.loc[idx] = delta.to_numpy(float)
            ratio_all.loc[idx] = ratio.to_numpy(float)
            pct_all.loc[idx] = pct.to_numpy(float)
            lag_pct_all.loc[idx] = pct.diff().to_numpy(float)
            slope_pct_all.loc[idx] = _rolling_slope(pct, 5).to_numpy(float)
            ewm_pct_all.loc[idx] = pct.ewm(span=5, adjust=False, min_periods=1).mean().to_numpy(float)

        prefix = f"INV__{feature}"
        generated[prefix + "__delta"] = delta_all
        generated[prefix + "__ratio"] = ratio_all
        generated[prefix + "__pct"] = pct_all
        generated[prefix + "__pct_lag1_delta"] = lag_pct_all
        generated[prefix + "__pct_slope5"] = slope_pct_all
        generated[prefix + "__pct_ewm5"] = ewm_pct_all

    # A physical aging coordinate based on observed discharged Ah, normalized by
    # each battery's early capacity baseline. This avoids raw cycle numbering.
    throughput = pd.Series(np.nan, index=ordered.index, dtype=float)
    log_throughput = pd.Series(np.nan, index=ordered.index, dtype=float)
    if "Discharge_cumulativeCapacity" in ordered.columns:
        for _, battery_df in ordered.groupby(GROUP_COL, sort=False):
            idx = battery_df.index
            q = pd.to_numeric(battery_df["Discharge_cumulativeCapacity"], errors="coerce")
            q_positive = q.where(q > 0)
            q0_series = _causal_early_baseline(q_positive, EARLY_BASELINE_FINITE_OBSERVATIONS)
            q0 = q0_series.abs().where(q0_series.abs() > 1e-10)
            normalized_q = q_positive / q0
            cumulative = normalized_q.fillna(0.0).cumsum()
            throughput.loc[idx] = cumulative.to_numpy(float)
            log_throughput.loc[idx] = np.log1p(cumulative.to_numpy(float))
    generated["INV__Cumulative_Throughput_EFC"] = throughput
    generated["INV__Log_Cumulative_Throughput_EFC"] = log_throughput

    if generated:
        ordered = pd.concat([ordered, pd.DataFrame(generated)], axis=1)
    ordered.replace([np.inf, -np.inf], np.nan, inplace=True)
    return ordered

def add_past_only_features(combined_df: pd.DataFrame) -> pd.DataFrame:
    """
    Add strictly previous-cycle summaries for selected full-cycle measurements.

    At cycle t, all values here are functions of cycles <= t-1. They are safe
    for the prospective partial-cycle mode even though the underlying previous
    cycle may have used a complete discharge.
    """
    ordered = combined_df.sort_values([GROUP_COL, CYCLE_COL]).reset_index(drop=True).copy()
    generated: dict[str, pd.Series] = {}

    for feature in PAST_ONLY_BASE_FEATURES:
        if feature not in ordered.columns:
            continue

        lag1_all = pd.Series(np.nan, index=ordered.index, dtype=float)
        pct_all = pd.Series(np.nan, index=ordered.index, dtype=float)
        roll3_all = pd.Series(np.nan, index=ordered.index, dtype=float)
        slope5_all = pd.Series(np.nan, index=ordered.index, dtype=float)
        ewm5_all = pd.Series(np.nan, index=ordered.index, dtype=float)

        for _, bdf in ordered.groupby(GROUP_COL, sort=False):
            idx = bdf.index
            current = pd.to_numeric(bdf[feature], errors="coerce")
            past = current.shift(1)
            baseline = _causal_early_baseline(past, EARLY_BASELINE_FINITE_OBSERVATIONS)
            den = baseline.abs().where(baseline.abs() > 1e-10)
            pct = 100.0 * (past - baseline) / den

            lag1_all.loc[idx] = past.to_numpy(float)
            pct_all.loc[idx] = pct.to_numpy(float)
            roll3_all.loc[idx] = past.rolling(3, min_periods=1).mean().to_numpy(float)
            slope5_all.loc[idx] = _rolling_slope(past, 5).to_numpy(float)
            ewm5_all.loc[idx] = past.ewm(span=5, adjust=False, min_periods=1).mean().to_numpy(float)

        prefix = f"INV__PastOnly__{feature}"
        generated[prefix + "__lag1"] = lag1_all
        generated[prefix + "__baseline_pct"] = pct_all
        generated[prefix + "__rolling_mean3"] = roll3_all
        generated[prefix + "__slope5"] = slope5_all
        generated[prefix + "__ewm5"] = ewm5_all

    if generated:
        ordered = pd.concat([ordered, pd.DataFrame(generated, index=ordered.index)], axis=1)
    ordered.replace([np.inf, -np.inf], np.nan, inplace=True)
    return ordered

def add_legacy_history_features(combined_df: pd.DataFrame) -> pd.DataFrame:
    """Target-free Run-1-style trailing history features, separately per battery."""
    ordered = combined_df.sort_values([GROUP_COL, CYCLE_COL]).reset_index(drop=True).copy()
    generated: dict[str, pd.Series] = {}

    for feature in LEGACY_HISTORY_BASE_FEATURES:
        if feature not in ordered.columns:
            continue

        outputs = {
            "baseline_delta": pd.Series(np.nan, index=ordered.index, dtype=float),
            "baseline_ratio": pd.Series(np.nan, index=ordered.index, dtype=float),
            "baseline_pct": pd.Series(np.nan, index=ordered.index, dtype=float),
            "lag1_delta": pd.Series(np.nan, index=ordered.index, dtype=float),
            "per_cycle_delta": pd.Series(np.nan, index=ordered.index, dtype=float),
            "rolling_mean_3": pd.Series(np.nan, index=ordered.index, dtype=float),
            "rolling_mean_5": pd.Series(np.nan, index=ordered.index, dtype=float),
            "rolling_std_5": pd.Series(np.nan, index=ordered.index, dtype=float),
            "rolling_slope_5": pd.Series(np.nan, index=ordered.index, dtype=float),
            "ewm_5": pd.Series(np.nan, index=ordered.index, dtype=float),
        }

        for _, bdf in ordered.groupby(GROUP_COL, sort=False):
            idx = bdf.index
            s = pd.to_numeric(bdf[feature], errors="coerce")
            cycles = pd.to_numeric(bdf[CYCLE_COL], errors="coerce")
            baseline = _causal_early_baseline(s, EARLY_BASELINE_FINITE_OBSERVATIONS)
            denominator = baseline.abs().where(baseline.abs() > 1e-12)

            outputs["baseline_delta"].loc[idx] = (s - baseline).to_numpy(float)
            outputs["baseline_ratio"].loc[idx] = (s / denominator).to_numpy(float)
            outputs["baseline_pct"].loc[idx] = (100.0 * (s - baseline) / denominator).to_numpy(float)
            outputs["lag1_delta"].loc[idx] = s.diff().to_numpy(float)
            outputs["per_cycle_delta"].loc[idx] = (
                s.diff() / cycles.diff().replace(0, np.nan)
            ).to_numpy(float)
            outputs["rolling_mean_3"].loc[idx] = s.rolling(3, min_periods=1).mean().to_numpy(float)
            outputs["rolling_mean_5"].loc[idx] = s.rolling(5, min_periods=1).mean().to_numpy(float)
            outputs["rolling_std_5"].loc[idx] = s.rolling(5, min_periods=2).std().to_numpy(float)
            outputs["rolling_slope_5"].loc[idx] = _rolling_slope(s, 5).to_numpy(float)
            outputs["ewm_5"].loc[idx] = s.ewm(span=5, adjust=False, min_periods=1).mean().to_numpy(float)

        for suffix, series in outputs.items():
            generated[f"LEGACYH__{feature}__{suffix}"] = series

    if generated:
        ordered = pd.concat([ordered, pd.DataFrame(generated)], axis=1)
    ordered.replace([np.inf, -np.inf], np.nan, inplace=True)
    return ordered

def add_joint_legacy_aliases(combined_df: pd.DataFrame) -> pd.DataFrame:
    """Copy every numeric legacy candidate under a LEGACY__ prefix."""
    out = combined_df.copy()
    generated: dict[str, pd.Series] = {}
    for column in list(out.columns):
        if column in JOINT_NON_FEATURE_COLUMNS:
            continue
        if column.startswith("INV__") or column.startswith("LEGACY__"):
            continue
        values = pd.to_numeric(out[column], errors="coerce")
        if values.notna().any():
            generated[f"LEGACY__{column}"] = values.astype(float)
    if generated:
        out = pd.concat([out, pd.DataFrame(generated, index=out.index)], axis=1)
    out.replace([np.inf, -np.inf], np.nan, inplace=True)
    return out

def load_nasa_constant_measurements_targetfree(
    csv_path,
    battery_id,
):
    """
    Same target-independent NASA constant raw loader used in Step 5.
    """
    raw = read_nasa_csv(
        csv_path,
        battery_id,
    )

    cycle_source = find_column(
        raw,
        ["discharge_cycle"],
    )

    time_source = find_column(
        raw,
        ["Time", "time"],
    )

    voltage_source = find_column(
        raw,
        [
            "Voltage_measured",
            "voltage_measured",
            "Voltage_load",
            "voltage",
        ],
    )

    current_source = find_column(
        raw,
        [
            "Current_measured",
            "current_measured",
            "Current_load",
            "current",
        ],
    )

    out = pd.DataFrame(
        {
            CYCLE_COL: pd.to_numeric(
                raw[cycle_source],
                errors="coerce",
            ),
            "Test_Time(s)": pd.to_numeric(
                raw[time_source],
                errors="coerce",
            ),
            "Voltage(V)": pd.to_numeric(
                raw[voltage_source],
                errors="coerce",
            ),
            "Current(A)": -np.abs(
                pd.to_numeric(
                    raw[current_source],
                    errors="coerce",
                )
            ),
            "Step_Index": 7.0,
        }
    )

    return (
        out.dropna(
            subset=[
                CYCLE_COL,
                "Test_Time(s)",
                "Voltage(V)",
                "Current(A)",
            ]
        )
        .sort_values(
            [
                CYCLE_COL,
                "Test_Time(s)",
            ]
        )
        .reset_index(
            drop=True
        )
    )

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

# ============================================================
# VALIDATED STEP 6 NESTED-LOBO MODEL HELPERS
# ============================================================

def rmse(y_true, y_pred):
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))

def mae(y_true, y_pred):
    return float(mean_absolute_error(y_true, y_pred))

def equal_battery_weights(frame):
    """
    Each battery contributes equal total training weight, regardless of its
    number of retained cycles.
    """
    counts = frame[BATTERY_COL].value_counts()

    weights = frame[BATTERY_COL].map(
        lambda b: 1.0 / counts.loc[b]
    ).astype(float).to_numpy()

    if np.mean(weights) > 0:
        weights = weights / np.mean(weights)

    return weights

def numeric_frame(frame, columns):
    out = frame.loc[:, columns].apply(
        pd.to_numeric,
        errors="coerce",
    )

    return out.replace(
        [np.inf, -np.inf],
        np.nan,
    )

def load_frozen_feature_list():
    frozen = pd.read_csv(FEATURE_LIST_PATH)

    if "Feature" not in frozen.columns:
        raise KeyError(
            "Frozen Step 5 feature file does not contain a Feature column."
        )

    features = frozen["Feature"].dropna().astype(str).tolist()

    if len(features) != EXPECTED_STRICT_FEATURE_COUNT:
        raise AssertionError(
            f"Expected {EXPECTED_STRICT_FEATURE_COUNT} strict features, "
            f"found {len(features)}."
        )

    if len(set(features)) != len(features):
        raise AssertionError(
            "Frozen strict feature list contains duplicate names."
        )

    return features

def training_source_coverage_filter(
    train_frame,
    columns,
    minimum=MIN_SOURCE_COVERAGE,
):
    """
    TRAINING-ONLY availability filter.

    A feature remains eligible only when:
      1. finite coverage is >= minimum within EACH source represented in the
         current training data; and
      2. it has more than one distinct finite training value.

    No held-out outer or inner test rows are inspected.
    """
    if len(columns) == 0:
        return [], []

    X = numeric_frame(
        train_frame,
        columns,
    )

    sources = sorted(
        train_frame[SOURCE_COL].dropna().unique()
    )

    keep = []
    audit = []

    for feature in columns:
        eligible = True
        coverages = {}

        for source in sources:
            mask = train_frame[SOURCE_COL].eq(source)
            coverage = float(
                X.loc[mask, feature].notna().mean()
            )

            coverages[source] = coverage

            if not np.isfinite(coverage) or coverage < minimum:
                eligible = False

        unique_values = int(
            X[feature].nunique(dropna=True)
        )

        if unique_values <= 1:
            eligible = False

        audit.append(
            {
                "Feature": feature,
                "Eligible": bool(eligible),
                "Unique_Training_Values": unique_values,
                **{
                    f"Coverage_{source}": coverages[source]
                    for source in sources
                },
            }
        )

        if eligible:
            keep.append(feature)

    return keep, audit

def prepare_feature_selector(
    train_frame,
    candidate_columns,
):
    """
    Prepare target ranking and correlation information from the current
    training split only.
    """
    eligible, coverage_audit = (
        training_source_coverage_filter(
            train_frame,
            candidate_columns,
        )
    )

    if len(eligible) == 0:
        raise RuntimeError(
            "No features remained after training-only source coverage filtering."
        )

    X_raw = numeric_frame(
        train_frame,
        eligible,
    )

    imputer = SimpleImputer(
        strategy="median",
    )

    X_imp = imputer.fit_transform(
        X_raw,
    )

    X = pd.DataFrame(
        X_imp,
        columns=eligible,
        index=train_frame.index,
    )

    # Absolute Spearman association:
    # Pearson correlation of ranked values.
    ranked_X = X.rank(
        method="average",
    )

    ranked_y = pd.Series(
        pd.to_numeric(
            train_frame[TARGET_COL],
            errors="coerce",
        ).to_numpy(),
        index=train_frame.index,
    ).rank(
        method="average",
    )

    target_scores = (
        ranked_X.corrwith(ranked_y)
        .abs()
        .replace(
            [np.inf, -np.inf],
            np.nan,
        )
        .fillna(0.0)
    )

    ranked_order = (
        target_scores
        .sort_values(
            ascending=False,
            kind="mergesort",
        )
        .index
        .tolist()
    )

    # Spearman feature-feature correlation.
    feature_corr = (
        ranked_X.loc[:, ranked_order]
        .corr()
        .abs()
    )

    return {
        "eligible_features": eligible,
        "coverage_audit": coverage_audit,
        "target_scores": target_scores,
        "ranked_order": ranked_order,
        "feature_corr": feature_corr,
    }

def choose_features(
    prepared,
    top_k,
    corr_threshold,
):
    """
    Greedy correlation pruning, in descending training-target association order.
    """
    selected = []

    for feature in prepared["ranked_order"]:

        if len(selected) >= int(top_k):
            break

        if not selected:
            selected.append(feature)
            continue

        correlations = prepared[
            "feature_corr"
        ].loc[
            feature,
            selected,
        ]

        if bool(
            (correlations >= float(corr_threshold)).any()
        ):
            continue

        selected.append(feature)

    if not selected:
        raise RuntimeError(
            "Feature selection returned zero features."
        )

    return selected

def model_features(
    selected_electrical,
    include_age,
):
    cols = list(selected_electrical)

    if include_age:
        for age_col in AGE_COLUMNS:
            if age_col not in cols:
                cols.append(age_col)

    return cols

def fit_hgb(
    train_frame,
    selected_electrical,
    config_name,
    include_age,
):
    cols = model_features(
        selected_electrical,
        include_age,
    )

    X_raw = numeric_frame(
        train_frame,
        cols,
    )

    imputer = SimpleImputer(
        strategy="median",
    )

    X = imputer.fit_transform(
        X_raw,
    )

    model = HistGradientBoostingRegressor(
        loss="squared_error",
        early_stopping=False,
        random_state=RANDOM_STATE,
        **HGB_CONFIGS[config_name],
    )

    model.fit(
        X,
        pd.to_numeric(
            train_frame[TARGET_COL],
            errors="coerce",
        ).to_numpy(),
        sample_weight=equal_battery_weights(
            train_frame
        ),
    )

    return model, imputer, cols

def predict_hgb(
    model,
    imputer,
    columns,
    test_frame,
):
    X_raw = numeric_frame(
        test_frame,
        columns,
    )

    X = imputer.transform(
        X_raw,
    )

    raw = model.predict(X)

    clipped = np.clip(
        raw,
        CLIP_LOW,
        CLIP_HIGH,
    )

    return raw, clipped

def prepare_inner_splits(
    outer_train,
    candidate_columns,
):
    """
    Prepare each of the nine inner LOBO splits for one outer fold.
    """
    prepared = {}

    batteries = sorted(
        outer_train[BATTERY_COL].unique()
    )

    for inner_holdout in batteries:
        inner_train = outer_train.loc[
            outer_train[BATTERY_COL] != inner_holdout
        ].copy()

        inner_test = outer_train.loc[
            outer_train[BATTERY_COL] == inner_holdout
        ].copy()

        selector = prepare_feature_selector(
            inner_train,
            candidate_columns,
        )

        prepared[inner_holdout] = {
            "train": inner_train,
            "test": inner_test,
            "selector": selector,
        }

    return prepared

def tune_feature_selection(
    outer_train,
    candidate_columns,
    include_age,
):
    """
    Stage A:
    select correlation threshold and top-K using inner LOBO with one fixed
    HGB configuration.
    """
    inner = prepare_inner_splits(
        outer_train,
        candidate_columns,
    )

    rows = []

    for corr_threshold in CORR_THRESHOLDS:
        for top_k in TOP_K_OPTIONS:

            fold_rmse = []
            fold_mae = []

            for inner_holdout, split in inner.items():

                selected = choose_features(
                    split["selector"],
                    top_k=top_k,
                    corr_threshold=corr_threshold,
                )

                model, imputer, columns = fit_hgb(
                    split["train"],
                    selected,
                    FEATURE_TUNING_MODEL,
                    include_age,
                )

                pred, _ = predict_hgb(
                    model,
                    imputer,
                    columns,
                    split["test"],
                )

                y = pd.to_numeric(
                    split["test"][TARGET_COL],
                    errors="coerce",
                ).to_numpy()

                fold_rmse.append(
                    rmse(y, pred)
                )

                fold_mae.append(
                    mae(y, pred)
                )

            rows.append(
                {
                    "Corr_Threshold": corr_threshold,
                    "Top_K": top_k,
                    "Inner_Macro_RMSE": float(
                        np.mean(fold_rmse)
                    ),
                    "Inner_Macro_MAE": float(
                        np.mean(fold_mae)
                    ),
                    "Feature_Tuning_Model": FEATURE_TUNING_MODEL,
                }
            )

    table = pd.DataFrame(rows).sort_values(
        [
            "Inner_Macro_RMSE",
            "Top_K",
            "Corr_Threshold",
        ],
        ascending=[
            True,
            True,
            True,
        ],
        kind="mergesort",
    ).reset_index(
        drop=True
    )

    best = table.iloc[0]

    return (
        float(best["Corr_Threshold"]),
        int(best["Top_K"]),
        table,
        inner,
    )

def tune_hgb_config(
    inner_prepared,
    corr_threshold,
    top_k,
    include_age,
):
    """
    Stage B:
    select one predefined HGB configuration using the same inner LOBO splits.
    """
    rows = []

    for config_name in HGB_CONFIGS:

        fold_rmse = []
        fold_mae = []

        for inner_holdout, split in inner_prepared.items():

            selected = choose_features(
                split["selector"],
                top_k=top_k,
                corr_threshold=corr_threshold,
            )

            model, imputer, columns = fit_hgb(
                split["train"],
                selected,
                config_name,
                include_age,
            )

            pred, _ = predict_hgb(
                model,
                imputer,
                columns,
                split["test"],
            )

            y = pd.to_numeric(
                split["test"][TARGET_COL],
                errors="coerce",
            ).to_numpy()

            fold_rmse.append(
                rmse(y, pred)
            )

            fold_mae.append(
                mae(y, pred)
            )

        rows.append(
            {
                "HGB_Config": config_name,
                "Inner_Macro_RMSE": float(
                    np.mean(fold_rmse)
                ),
                "Inner_Macro_MAE": float(
                    np.mean(fold_mae)
                ),
            }
        )

    table = pd.DataFrame(rows).sort_values(
        [
            "Inner_Macro_RMSE",
            "HGB_Config",
        ],
        ascending=[
            True,
            True,
        ],
        kind="mergesort",
    ).reset_index(
        drop=True
    )

    return (
        str(table.iloc[0]["HGB_Config"]),
        table,
    )

def run_formulation(
    frame,
    formulation_name,
    candidate_columns,
    include_age,
):
    prediction_rows = []
    fold_rows = []
    selected_rows = []
    coverage_rows = []
    feature_tuning_rows = []
    model_tuning_rows = []

    batteries = sorted(
        frame[BATTERY_COL].unique()
    )

    print("\n" + "=" * 90)
    print("FORMULATION:", formulation_name)
    print("=" * 90)

    for fold_number, outer_holdout in enumerate(
        batteries,
        start=1,
    ):
        fold_start = time.time()

        outer_train = frame.loc[
            frame[BATTERY_COL] != outer_holdout
        ].copy()

        outer_test = frame.loc[
            frame[BATTERY_COL] == outer_holdout
        ].copy()

        print(
            f"\nOuter fold {fold_number}/{len(batteries)} "
            f"- holdout {outer_holdout}"
        )

        # ----------------------------------------------------
        # Stage A: feature-selection settings
        # ----------------------------------------------------
        (
            best_corr,
            best_k,
            feature_tuning,
            inner_prepared,
        ) = tune_feature_selection(
            outer_train,
            candidate_columns,
            include_age,
        )

        feature_tuning = feature_tuning.copy()

        feature_tuning.insert(
            0,
            "Formulation",
            formulation_name,
        )

        feature_tuning.insert(
            1,
            "Outer_Holdout",
            outer_holdout,
        )

        feature_tuning_rows.append(
            feature_tuning
        )

        # ----------------------------------------------------
        # Stage B: HGB configuration
        # ----------------------------------------------------
        best_config, model_tuning = (
            tune_hgb_config(
                inner_prepared,
                best_corr,
                best_k,
                include_age,
            )
        )

        model_tuning = model_tuning.copy()

        model_tuning.insert(
            0,
            "Formulation",
            formulation_name,
        )

        model_tuning.insert(
            1,
            "Outer_Holdout",
            outer_holdout,
        )

        model_tuning[
            "Chosen_Corr_Threshold"
        ] = best_corr

        model_tuning[
            "Chosen_Top_K"
        ] = best_k

        model_tuning_rows.append(
            model_tuning
        )

        # ----------------------------------------------------
        # Rebuild feature selection using only full outer train
        # ----------------------------------------------------
        selector = prepare_feature_selector(
            outer_train,
            candidate_columns,
        )

        selected = choose_features(
            selector,
            top_k=best_k,
            corr_threshold=best_corr,
        )

        # Coverage audit
        for row in selector["coverage_audit"]:
            coverage_rows.append(
                {
                    "Formulation": formulation_name,
                    "Outer_Holdout": outer_holdout,
                    **row,
                }
            )

        # Selected-feature audit
        for rank, feature in enumerate(
            selected,
            start=1,
        ):
            selected_rows.append(
                {
                    "Formulation": formulation_name,
                    "Outer_Holdout": outer_holdout,
                    "Selected_Rank": rank,
                    "Feature": feature,
                    "Training_Target_SpearmanAbs": float(
                        selector[
                            "target_scores"
                        ].loc[feature]
                    ),
                    "Corr_Threshold": best_corr,
                    "Top_K": best_k,
                    "HGB_Config": best_config,
                }
            )

        # ----------------------------------------------------
        # Final outer fit
        # ----------------------------------------------------
        model, imputer, columns = fit_hgb(
            outer_train,
            selected,
            best_config,
            include_age,
        )

        pred_raw, pred_clipped = (
            predict_hgb(
                model,
                imputer,
                columns,
                outer_test,
            )
        )

        y = pd.to_numeric(
            outer_test[TARGET_COL],
            errors="coerce",
        ).to_numpy()

        raw_rmse = rmse(
            y,
            pred_raw,
        )

        raw_mae = mae(
            y,
            pred_raw,
        )

        clipped_rmse = rmse(
            y,
            pred_clipped,
        )

        clipped_mae = mae(
            y,
            pred_clipped,
        )

        elapsed = time.time() - fold_start

        fold_rows.append(
            {
                "Formulation": formulation_name,
                "Holdout_Battery": outer_holdout,
                "Source": str(
                    outer_test[SOURCE_COL].iloc[0]
                ),
                "N_Test_Cycles": int(
                    len(outer_test)
                ),
                "Eligible_Training_Features": int(
                    len(
                        selector[
                            "eligible_features"
                        ]
                    )
                ),
                "Selected_Electrical_Features": int(
                    len(selected)
                ),
                "Explicit_Age_Features_Added": (
                    len(AGE_COLUMNS)
                    if include_age
                    else 0
                ),
                "Final_Model_Features": int(
                    len(columns)
                ),
                "Chosen_Corr_Threshold": best_corr,
                "Chosen_Top_K": best_k,
                "Chosen_HGB_Config": best_config,
                "RMSE_Raw": raw_rmse,
                "MAE_Raw": raw_mae,
                "RMSE_Clipped70_100": clipped_rmse,
                "MAE_Clipped70_100": clipped_mae,
                "Runtime_Seconds": elapsed,
            }
        )

        for (_, row), raw_pred, clipped_pred in zip(
            outer_test.iterrows(),
            pred_raw,
            pred_clipped,
        ):
            prediction_rows.append(
                {
                    "Formulation": formulation_name,
                    "Holdout_Battery": outer_holdout,
                    BATTERY_COL: row[BATTERY_COL],
                    SOURCE_COL: row[SOURCE_COL],
                    "Cycle_Index": row["Cycle_Index"],
                    "Cycle_Age": row["Cycle_Age"],
                    TARGET_COL: row[TARGET_COL],
                    "Prediction_Raw": float(
                        raw_pred
                    ),
                    "Prediction_Clipped70_100": float(
                        clipped_pred
                    ),
                    "Residual_Raw": float(
                        row[TARGET_COL] - raw_pred
                    ),
                    "Residual_Clipped70_100": float(
                        row[TARGET_COL] - clipped_pred
                    ),
                }
            )

        print(
            f"  eligible training features: "
            f"{len(selector['eligible_features'])}"
        )

        print(
            f"  selected={len(selected)} "
            f"| corr={best_corr} "
            f"| K={best_k} "
            f"| HGB={best_config}"
        )

        print(
            f"  raw RMSE={raw_rmse:.4f} "
            f"| clipped RMSE={clipped_rmse:.4f}"
        )

        print(
            f"  runtime={elapsed:.1f} s"
        )

    predictions = pd.DataFrame(
        prediction_rows
    )

    folds = pd.DataFrame(
        fold_rows
    )

    selected_df = pd.DataFrame(
        selected_rows
    )

    coverage_df = pd.DataFrame(
        coverage_rows
    )

    feature_tuning_df = pd.concat(
        feature_tuning_rows,
        ignore_index=True,
    )

    model_tuning_df = pd.concat(
        model_tuning_rows,
        ignore_index=True,
    )

    macro = pd.DataFrame(
        [
            {
                "Formulation": formulation_name,
                "Batteries": int(
                    folds[
                        "Holdout_Battery"
                    ].nunique()
                ),
                "Macro_RMSE_Raw": float(
                    folds["RMSE_Raw"].mean()
                ),
                "Macro_MAE_Raw": float(
                    folds["MAE_Raw"].mean()
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

    frequency = (
        selected_df.groupby(
            "Feature"
        )
        .agg(
            Outer_Folds_Selected=(
                "Outer_Holdout",
                "nunique",
            ),
            Mean_Training_Target_SpearmanAbs=(
                "Training_Target_SpearmanAbs",
                "mean",
            ),
        )
        .reset_index()
        .sort_values(
            [
                "Outer_Folds_Selected",
                "Mean_Training_Target_SpearmanAbs",
            ],
            ascending=[
                False,
                False,
            ],
        )
        .reset_index(
            drop=True
        )
    )

    frequency.insert(
        0,
        "Formulation",
        formulation_name,
    )

    return {
        "predictions": predictions,
        "folds": folds,
        "macro": macro,
        "selected": selected_df,
        "selection_frequency": frequency,
        "coverage_audit": coverage_df,
        "feature_tuning": feature_tuning_df,
        "model_tuning": model_tuning_df,
    }

def combine_results(
    results,
    key,
):
    frames = [
        result[key]
        for result in results.values()
        if key in result
        and result[key] is not None
    ]

    if not frames:
        return pd.DataFrame()

    return pd.concat(
        frames,
        ignore_index=True,
    )


# ============================================================
# RAW ZIP PREPARATION
# ============================================================

def safe_extract_zip_step11g(
    zip_path,
    destination,
):
    destination = Path(
        destination
    )

    if destination.exists():
        shutil.rmtree(
            destination
        )

    destination.mkdir(
        parents=True,
        exist_ok=True,
    )

    with zipfile.ZipFile(
        zip_path,
        "r",
    ) as archive:

        root = (
            destination.resolve()
        )

        for member in archive.infolist():

            target = (
                destination
                / member.filename
            ).resolve()

            if (
                target != root
                and root not in target.parents
            ):

                raise RuntimeError(
                    f"Unsafe ZIP path: {member.filename}"
                )

        archive.extractall(
            destination
        )


def prepare_raw_data_step11g():

    if not DRIVE_ZIP.exists():

        raise FileNotFoundError(
            "Raw battery archive was not found:\n"
            f"{DRIVE_ZIP}"
        )

    drive_size = (
        DRIVE_ZIP.stat().st_size
    )

    if (
        not LOCAL_ZIP.exists()
        or LOCAL_ZIP.stat().st_size
        != drive_size
    ):

        print(
            "Copying raw collab.zip from Drive..."
        )

        shutil.copy2(
            DRIVE_ZIP,
            LOCAL_ZIP,
        )

    marker = (
        EXTRACT_ROOT
        / ".zip_size"
    )

    reuse = False

    if (
        EXTRACT_ROOT.exists()
        and marker.exists()
    ):

        try:

            reuse = (
                int(
                    marker.read_text().strip()
                )
                == drive_size
            )

        except Exception:

            reuse = False

    if not reuse:

        print(
            "Extracting raw archive..."
        )

        safe_extract_zip_step11g(
            LOCAL_ZIP,
            EXTRACT_ROOT,
        )

        marker.write_text(
            str(
                drive_size
            )
        )

    else:

        print(
            "Reusing extracted raw archive."
        )

    return EXTRACT_ROOT


def _find_exact_dir_step11g(
    root,
    name,
):

    matches = [
        path
        for path in Path(
            root
        ).rglob("*")
        if (
            path.is_dir()
            and path.name.lower()
            == str(
                name
            ).lower()
        )
    ]

    if len(
        matches
    ) != 1:

        raise RuntimeError(
            f"{name}: expected exactly one directory, "
            f"found {len(matches)}."
        )

    return matches[
        0
    ]


def _find_exact_file_step11g(
    root,
    filename,
):

    matches = [
        path
        for path in Path(
            root
        ).rglob(
            filename
        )
        if path.is_file()
    ]

    if len(
        matches
    ) != 1:

        raise RuntimeError(
            f"{filename}: expected exactly one file, "
            f"found {len(matches)}."
        )

    return matches[
        0
    ]


def build_constant_manifest_step11g(
    root,
):

    manifest = {}
    audit_rows = []

    for (
        battery_id,
        required_sheets,
    ) in CALCE_CONSTANT_BATTERIES:

        folder = (
            _find_exact_dir_step11g(
                root,
                battery_id,
            )
        )

        workbooks = [
            Path(
                path
            )
            for path in extract_files(
                folder
            )
        ]

        sheet_union = set()

        for workbook in workbooks:

            with pd.ExcelFile(
                workbook
            ) as excel:

                sheet_union.update(
                    excel.sheet_names
                )

        missing_sheets = [
            sheet
            for sheet in required_sheets
            if sheet not in sheet_union
        ]

        if missing_sheets:

            raise RuntimeError(
                f"{battery_id}: missing required sheet(s): "
                f"{missing_sheets}"
            )

        manifest[
            battery_id
        ] = {
            "kind":
                "CALCE",

            "folder":
                folder,

            "xlsx_files":
                workbooks,

            "required_sheets":
                list(
                    required_sheets
                ),
        }

        audit_rows.append(
            {
                "Battery":
                    battery_id,

                "Source":
                    "CALCE",

                "Resolved_Path":
                    str(
                        folder
                    ),

                "File_Count":
                    len(
                        workbooks
                    ),

                "Status":
                    "OK",
            }
        )

    for battery_id in (
        NASA_CONSTANT_BATTERIES
    ):

        csv_path = (
            _find_exact_file_step11g(
                root,
                f"{battery_id}.csv",
            )
        )

        sample = pd.read_csv(
            csv_path,
            nrows=10,
            low_memory=False,
        )

        # Validate only fields needed by the target-free loader.
        find_column(
            sample,
            [
                "discharge_cycle",
            ],
        )

        find_column(
            sample,
            [
                "Time",
                "time",
            ],
        )

        find_column(
            sample,
            [
                "Voltage_measured",
                "voltage_measured",
                "Voltage_load",
                "voltage",
            ],
        )

        find_column(
            sample,
            [
                "Current_measured",
                "current_measured",
                "Current_load",
                "current",
            ],
        )

        manifest[
            battery_id
        ] = {
            "kind":
                "NASA_CONSTANT",

            "csv_path":
                csv_path,
        }

        audit_rows.append(
            {
                "Battery":
                    battery_id,

                "Source":
                    "NASA",

                "Resolved_Path":
                    str(
                        csv_path
                    ),

                "File_Count":
                    1,

                "Status":
                    "OK",
            }
        )

    return (
        manifest,
        pd.DataFrame(
            audit_rows
        ),
    )




# ============================================================
# STEP 11L: FIXED-ELAPSED-DISCHARGE-TIME OBSERVATION EXPERIMENT
# ============================================================
# This block extends the original validated Step 11G raw and Step 6
# modelling helpers above.  NO original full-cycle predictor columns
# are copied into the early-observation feature table.
#
# Prespecified primary observation setting: first 600 seconds of the
# discharge segment in EVERY battery cycle (10 minutes).
# The code never reads a future raw sample to interpolate a value at
# the boundary. It uses only samples observed <= 600 seconds.
#
# Only current-cycle DISCHARGE samples within the window are provided
# for feature engineering. No full discharge, subsequent reference
# discharge, or full-cycle charge is available to the model.
# History features are reconstructed from earlier CENSORED cycles.
#
# The previous 70-100% study population and target labels remain frozen.
# The fixed 600-s window is defined before any held-out error is examined.
# ============================================================

import matplotlib.pyplot as plt

WINDOW_SECONDS = 600.0
MIN_WINDOW_ROWS = 10
MIN_LAST_SAMPLE_FRACTION = 0.90  # last sample must occur >= 540 seconds
MAX_SAMPLE_GAP_S = 120.0
MIN_PRIMARY_BATTERY_CYCLES = 10
ENABLE_DOWNLOAD = False

STEP11L_OUT = PROJECT_ROOT / 'reviewer_response' / 'step11l_fixed_elapsed_time'
STEP11L_OUT.mkdir(parents=True, exist_ok=True)
STEP11L_LOCAL = Path('work/step11l_local')
STEP11L_LOCAL.mkdir(parents=True, exist_ok=True)
STEP11L_WORKBOOK = STEP11L_LOCAL / 'Reviewer_Step11L_Fixed_Duration_Prediction.xlsx'


def cycle_key(value):
    z = pd.to_numeric(pd.Series([value]), errors='coerce').iloc[0]
    return format(float(z), '.12g') if np.isfinite(z) else str(value)


def censor_one_cycle_fixed_duration(cycle_frame, source):
    """Return (discharge-only raw prefix, audit), using no samples after 600 s."""
    g = cycle_frame.copy()
    for name in [CYCLE_COL, 'Test_Time(s)', 'Voltage(V)', 'Current(A)']:
        g[name] = pd.to_numeric(g[name], errors='coerce')
    g = g.dropna(subset=['Test_Time(s)', 'Voltage(V)', 'Current(A)'])

    if str(source).upper() == 'CALCE' and 'Step_Index' in g.columns:
        steps = pd.to_numeric(g['Step_Index'], errors='coerce')
        st7 = g.loc[steps.eq(7) & g['Current(A)'].lt(-0.02)].copy()
        # Use validated Step 11G discharge-segment preference.
        dis = st7 if len(st7) >= 2 else g.loc[g['Current(A)'].lt(-0.02)].copy()
        segment_rule = 'STEP7_DISCHARGE' if len(st7) >= 2 else 'NEGATIVE_CURRENT_FALLBACK'
    else:
        dis = g.loc[g['Current(A)'].lt(-0.02)].copy()
        segment_rule = 'NASA_DISCHARGE_ONLY'

    dis = dis.sort_values('Test_Time(s)').drop_duplicates(subset=['Test_Time(s)'], keep='first')
    audit = {'Segment_Rule': segment_rule, 'Window_Seconds': WINDOW_SECONDS,
             'Full_Discharge_Duration_s_AuditOnly': np.nan, 'Window_Rows': 0,
             'Last_Observed_Time_s': np.nan, 'Maximum_Sample_Gap_s': np.nan,
             'Window_Coverage_Ratio': 0.0, 'Status': 'NO_VALID_DISCHARGE'}
    if len(dis) < 2:
        return None, audit

    t0 = float(dis['Test_Time(s)'].iloc[0])
    dt = dis['Test_Time(s)'].to_numpy(float) - t0
    if not np.isfinite(dt).all() or (dt < -1e-8).any():
        audit['Status'] = 'INVALID_TIME'
        return None, audit
    audit['Full_Discharge_Duration_s_AuditOnly'] = float(np.max(dt))

    # No interpolation using samples observed after the prediction time.
    prefix = dis.loc[(dt >= 0) & (dt <= WINDOW_SECONDS + 1e-9)].copy()
    if len(prefix) < 2:
        audit['Status'] = 'INSUFFICIENT_PREFIX_ROWS'
        audit['Window_Rows'] = len(prefix)
        return None, audit

    elapsed = prefix['Test_Time(s)'].to_numpy(float) - t0
    last_t = float(elapsed[-1])
    gaps = np.diff(elapsed)
    max_gap = float(np.max(gaps)) if len(gaps) else np.nan
    audit.update({'Window_Rows': len(prefix), 'Last_Observed_Time_s': last_t,
                  'Maximum_Sample_Gap_s': max_gap,
                  'Window_Coverage_Ratio': min(1.0, last_t / WINDOW_SECONDS)})

    if len(prefix) < MIN_WINDOW_ROWS:
        audit['Status'] = 'INSUFFICIENT_PREFIX_ROWS'
        return None, audit
    if last_t < MIN_LAST_SAMPLE_FRACTION * WINDOW_SECONDS:
        audit['Status'] = 'INCOMPLETE_FIXED_WINDOW'
        return None, audit
    if max_gap > MAX_SAMPLE_GAP_S:
        audit['Status'] = 'LONG_SAMPLE_GAP'
        return None, audit

    if (prefix['Current(A)'] >= -0.02).any() or (elapsed > WINDOW_SECONDS + 1e-9).any():
        raise AssertionError('Post-boundary or non-discharge sample entered the prefix.')
    audit['Status'] = 'VALID_FIXED_WINDOW'
    return prefix, audit


def censor_battery_fixed_duration(raw, battery_id, source):
    parts, audit = [], []
    for cycle, g in raw.groupby(CYCLE_COL, sort=True):
        prefix, a = censor_one_cycle_fixed_duration(g, source)
        audit.append({'Battery': battery_id, 'Source': source, CYCLE_COL: cycle, **a})
        if prefix is not None:
            parts.append(prefix)
    censored = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=raw.columns)
    return censored, pd.DataFrame(audit)


def align_fixed_time_to_step5(rebuilt, frozen_battery, strict_features, audit):
    """Frozen Step 5 target and age only; all electrical features from censored raw."""
    f = frozen_battery.copy()
    f['_Cycle_Key'] = f[CYCLE_COL].map(cycle_key)
    rr = rebuilt.copy()
    rr['_Cycle_Key'] = rr[CYCLE_COL].map(cycle_key)
    a = audit.loc[audit['Status'].eq('VALID_FIXED_WINDOW')].copy()
    a['_Cycle_Key'] = a[CYCLE_COL].map(cycle_key)
    assert f['_Cycle_Key'].is_unique, 'Duplicate frozen cycle keys'
    assert rr['_Cycle_Key'].is_unique, 'Duplicate reconstructed cycle keys'
    assert a['_Cycle_Key'].is_unique, 'Duplicate audit cycle keys'

    eligible = f.merge(a.drop(columns=[CYCLE_COL]), on='_Cycle_Key', how='inner', validate='one_to_one')
    if len(eligible) == 0:
        return pd.DataFrame()

    # Include only reconstructed early-observation columns from the frozen
    # strict schema.  NEVER fall back to frozen full-discharge features.
    candidates = [x for x in strict_features if x in rr.columns and x not in EXPLICIT_AGE_STRICT_FEATURES]
    p = rr[['_Cycle_Key'] + candidates].copy()
    aligned = eligible[[BATTERY_COL, SOURCE_COL, CYCLE_COL, TARGET_COL,
                        'Cycle_Age', 'Log_Cycle_Age', '_Cycle_Key',
                        'Window_Seconds', 'Window_Rows',
                        'Last_Observed_Time_s', 'Maximum_Sample_Gap_s',
                        'Window_Coverage_Ratio',
                        'Full_Discharge_Duration_s_AuditOnly']].merge(
                            p, on='_Cycle_Key', how='left', validate='one_to_one')
    for feature in strict_features:
        if feature not in aligned.columns:
            aligned[feature] = np.nan
    # A full-duration value may be carried for auditing but is never passed
    # to the feature-selection or HGB functions.
    return aligned.sort_values(CYCLE_COL).reset_index(drop=True)


def build_fixed_time_dataset(manifest, frozen, strict_features):
    all_frames, all_audits = [], []
    for battery_id in [b for b, _ in CALCE_CONSTANT_BATTERIES] + NASA_CONSTANT_BATTERIES:
        print(f'\nReading {battery_id}...')
        raw, source = load_clean_raw_battery(manifest, battery_id)
        fixed_raw, audit = censor_battery_fixed_duration(raw, battery_id, source)
        all_audits.append(audit)
        f = frozen.loc[frozen[BATTERY_COL].eq(battery_id)].copy()
        if not fixed_raw.empty:
            rebuilt = rebuild_full_battery_features_from_raw(fixed_raw, battery_id, source)
            aligned = align_fixed_time_to_step5(rebuilt, f, strict_features, audit)
        else:
            aligned = pd.DataFrame()
        if not aligned.empty:
            all_frames.append(aligned)
        print(f'  frozen primary={len(f)} | valid 600s={len(aligned)}')

    audit = pd.concat(all_audits, ignore_index=True)
    frame = pd.concat(all_frames, ignore_index=True) if all_frames else pd.DataFrame()
    if not frame.empty and frame.duplicated([BATTERY_COL, '_Cycle_Key']).any():
        raise AssertionError('Duplicate model observation keys')
    coverage = frozen.groupby([BATTERY_COL, SOURCE_COL]).size().rename('Primary_Step5_Count').reset_index()
    passing = frame.groupby(BATTERY_COL).size().rename('Eligible_600s_Count').reset_index() if not frame.empty else pd.DataFrame(columns=[BATTERY_COL, 'Eligible_600s_Count'])
    coverage = coverage.merge(passing, on=BATTERY_COL, how='left')
    coverage['Eligible_600s_Count'] = coverage['Eligible_600s_Count'].fillna(0).astype(int)
    coverage['Coverage_Pct'] = 100 * coverage['Eligible_600s_Count'] / coverage['Primary_Step5_Count']
    return frame, audit, coverage


def score_battery_predictions(df, formulation, label):
    subset = df.loc[df['Formulation'].eq(formulation)]
    if subset.empty:
        raise ValueError(f'No predictions for {formulation}')
    rows=[]
    for battery, part in subset.groupby(BATTERY_COL):
        rows.append({'Formulation': label, BATTERY_COL: battery,
                     'N': len(part), 'RMSE': rmse(part[TARGET_COL], part['Prediction_Raw']),
                     'MAE': mae(part[TARGET_COL], part['Prediction_Raw'])})
    return pd.DataFrame(rows)


def common_subset_step6_predictions(frame):
    full = pd.read_csv(STEP6_PREDICTIONS_PATH, low_memory=False)
    needed = ['Formulation', BATTERY_COL, CYCLE_COL, TARGET_COL, 'Prediction_Raw']
    missing = [n for n in needed if n not in full.columns]
    if missing:
        raise KeyError(f'Step6 predictions lack required columns: {missing}')
    full['_Cycle_Key'] = full[CYCLE_COL].map(cycle_key)
    elig = frame[[BATTERY_COL,'_Cycle_Key']].drop_duplicates()
    result = full.merge(elig, on=[BATTERY_COL,'_Cycle_Key'], how='inner',validate='many_to_one')
    for formulation in ['STRICT_ELECTRICAL_HGB', 'STRICT_ELECTRICAL_PLUS_AGE_HGB']:
        f = result.loc[result['Formulation'].eq(formulation)]
        if len(f) != len(frame):
            raise AssertionError(f'Expected {len(frame)} matched Step 6 rows for {formulation}; saw {len(f)}')
        compare = f.merge(frame[[BATTERY_COL,'_Cycle_Key',TARGET_COL]], on=[BATTERY_COL,'_Cycle_Key'], suffixes=('_STEP6','_FROZEN'), validate='one_to_one')
        a = pd.to_numeric(compare[TARGET_COL+'_STEP6'], errors='coerce')
        b = pd.to_numeric(compare[TARGET_COL+'_FROZEN'], errors='coerce')
        if not np.allclose(a,b,rtol=0,atol=1e-10):
            raise AssertionError('Frozen labels do not match Step6 reference.')
    return result


def bootstrap_per_battery_difference(paired, nrep=20000, seed=42):
    """Fixed-prediction battery bootstrap; excludes model-refitting uncertainty."""
    rng = np.random.default_rng(seed)
    a = paired['RMSE_Fixed_600s'].to_numpy(float)
    b = paired['RMSE_Step6_Full'].to_numpy(float)
    delta = a - b
    n = len(delta)
    draws = rng.integers(0, n, size=(nrep,n))
    means = delta[draws].mean(axis=1)
    return {'Batteries':n, 'Bootstrap_Replicates':nrep,
            'Mean_Delta_FixedMinusFull':float(delta.mean()),
            'CI95_Lower':float(np.quantile(means,0.025)),
            'CI95_Upper':float(np.quantile(means,0.975)),
            'Uncertainty_Type':'Resample frozen per-battery RMSE differences; no model retraining'}


def make_methods():
    items = [
        ('Purpose','Sensitivity to fixed elapsed-time measurement windows; addresses reviewer concern about full-discharge dV/dt encoding total discharge duration.'),
        ('Primary window','600 seconds since start of discharge; fixed in advance.'),
        ('Observation boundary','Only actual discharge samples at or before 600 s are ever used. No future interpolation.'),
        ('CALCE segment','Prefer Step_Index=7 with discharge current < -0.02 A; otherwise negative-current fallback.'),
        ('NASA segment','Reconstructed constant-current discharge samples; current set negative by validated raw loader.'),
        ('Quality','At least 10 samples, last timestamp >= 540 s, no individual observed gap > 120 s.'),
        ('Current-cycle charge','Not used; discharge prefix only.'),
        ('History','Constructed only from earlier fixed-window cycles, never earlier full-discharge features or prior target capacity.'),
        ('Target','Original Step 5 corrected reference-discharge SoH labels, unchanged; no recomputation from truncated current.'),
        ('Population','Frozen 70-100% Step 5 population intersected with cycles possessing a valid 600-s prefix.'),
        ('Strict feature schema','Same frozen Step 5 candidate feature names (335); unavailable early features are NaN, never copied from Step 5 full-cycle predictors.'),
        ('Feature filtering','Training-only >=95% finite coverage in each source and nonconstant, from Step 6 helper.'),
        ('Validation','Two-stage nested leave-one-battery-out HGB, equal battery weights, training-only tuning and selection.'),
        ('Comparison','Frozen full-discharge Step 6 predictions matched to identical eligible battery cycles.'),
        ('Uncertainty','Fixed-prediction paired battery bootstrap, 20,000 resamples; no model retraining.'),
        ('Limitation','One predeclared window, not a universal device-agnostic early-observation guarantee; constant protocol only.'),
    ]
    return pd.DataFrame(items, columns=['Item','Definition'])


def make_figures(coverage, macro, matched_summary):
    fig, ax=plt.subplots(figsize=(8.7,4.2))
    vals=coverage.sort_values(BATTERY_COL)
    ax.bar(vals[BATTERY_COL],vals['Coverage_Pct'])
    ax.set_ylabel('Eligible cycles (%)')
    ax.set_xlabel('Held-out battery')
    ax.set_title('Fixed 600-s discharge window: sample coverage')
    ax.set_ylim(0,105)
    ax.tick_params(axis='x',rotation=35)
    fig.tight_layout()
    fig.savefig(STEP11L_LOCAL/'Figure_11L_Coverage.png',dpi=220)
    fig.savefig(STEP11L_LOCAL/'Figure_11L_Coverage.pdf')
    plt.close(fig)

    fig, ax=plt.subplots(figsize=(9.3,4.7))
    ct=matched_summary.sort_values('Setting')
    ax.barh(ct['Setting'], ct['Macro_RMSE'])
    ax.set_xlabel('Battery-macro RMSE (SoH percentage points; lower is better)')
    ax.set_title('Fixed-time vs full-discharge: same battery-cycle subset')
    fig.tight_layout()
    fig.savefig(STEP11L_LOCAL/'Figure_11L_Matched_RMSE.png',dpi=220)
    fig.savefig(STEP11L_LOCAL/'Figure_11L_Matched_RMSE.pdf')
    plt.close(fig)


def main_step11l():
    print('='*80)
    print('STEP 11L — FIXED 600-s DISCHARGE OBSERVATION, NESTED LOBO')
    print('='*80)
    for path in [DRIVE_ZIP,STEP5_DATA_PATH,STEP5_FEATURE_LIST_PATH,STEP6_PREDICTIONS_PATH]:
        if not Path(path).exists():
            raise FileNotFoundError(f'Missing required project input: {path}')
    frozen = pd.read_csv(STEP5_DATA_PATH, low_memory=False)
    if len(frozen) != 4590 or frozen[BATTERY_COL].nunique()!=10:
        raise AssertionError('Frozen primary Step 5 dataset does not match validated 4,590 cycles / 10 batteries.')
    strict_features = load_frozen_feature_list()
    electrical_features = [f for f in strict_features if f not in EXPLICIT_AGE_STRICT_FEATURES]
    raw_root = prepare_raw_data_step11g()
    manifest, raw_manifest = build_constant_manifest_step11g(raw_root)
    fixed, audit, coverage = build_fixed_time_dataset(manifest, frozen, strict_features)

    # Export feasibility evidence even if not enough complete battery folds exist.
    coverage.to_csv(STEP11L_LOCAL/'Step11L_Coverage.csv',index=False)
    audit.to_csv(STEP11L_LOCAL/'Step11L_Window_Audit.csv',index=False)
    print('\nCoverage (the definition is fixed before viewing model errors):')
    print(coverage.to_string(index=False))
    if fixed.empty or (coverage['Eligible_600s_Count'] < MIN_PRIMARY_BATTERY_CYCLES).any():
        auditfile = STEP11L_LOCAL/'Step11L_Window_Feasibility_Only.xlsx'
        with pd.ExcelWriter(auditfile, engine='openpyxl') as xw:
            make_methods().to_excel(xw,sheet_name='Methods',index=False)
            raw_manifest.to_excel(xw,sheet_name='Raw_Manifest',index=False)
            coverage.to_excel(xw,sheet_name='Battery_Coverage',index=False)
            audit.to_excel(xw,sheet_name='Raw_Window_Audit',index=False)
        if auditfile.resolve() != (STEP11L_OUT/auditfile.name).resolve():
            shutil.copy2(auditfile,STEP11L_OUT/auditfile.name)
        raise RuntimeError('Insufficient eligible 600-s cycles for ten-battery LOBO. Raw feasibility report saved. Do not silently change the predeclared window based on errors; decide a revised window using only this coverage report.')

    # Strict safeguard: raw full-capacity variables are NEVER candidates.
    assert not any(f in ['Measured_Discharge_Capacity_Ah',TARGET_COL] for f in electrical_features)
    print('\nEarly observation rows:',len(fixed))
    print('Strict candidate features:',len(electrical_features))

    results = {}
    for name, age in [('FIXED600_ELECTRICAL_HGB',False),('FIXED600_ELECTRICAL_PLUS_AGE_HGB',True)]:
        results[name] = run_formulation(frame=fixed, formulation_name=name,
                                        candidate_columns=electrical_features,
                                        include_age=age)

    macro=combine_results(results,'macro')
    folds=combine_results(results,'folds')
    predictions=combine_results(results,'predictions')
    selected=combine_results(results,'selected')
    frequency=combine_results(results,'selection_frequency')
    cov_audit=combine_results(results,'coverage_audit')
    tune_fs=combine_results(results,'feature_tuning')
    tune_hgb=combine_results(results,'model_tuning')

    full=common_subset_step6_predictions(fixed)
    ref_form={'FIXED600_ELECTRICAL_HGB':'STRICT_ELECTRICAL_HGB',
              'FIXED600_ELECTRICAL_PLUS_AGE_HGB':'STRICT_ELECTRICAL_PLUS_AGE_HGB'}
    paired_rows=[]; boot_rows=[]; summary_rows=[]
    for fixed_form, full_form in ref_form.items():
        f_batt=score_battery_predictions(predictions,fixed_form,fixed_form)
        full_batt=score_battery_predictions(full,full_form,full_form)
        paired=f_batt.merge(full_batt,on=BATTERY_COL,validate='one_to_one',suffixes=('_fixed','_full'))
        paired=paired.rename(columns={'RMSE_fixed':'RMSE_Fixed_600s',
                                     'RMSE_full':'RMSE_Step6_Full',
                                     'MAE_fixed':'MAE_Fixed_600s','MAE_full':'MAE_Step6_Full'})
        paired['Formulation']=fixed_form
        paired['RMSE_Difference_FixedMinusFull']=paired['RMSE_Fixed_600s']-paired['RMSE_Step6_Full']
        paired_rows.append(paired)
        b=bootstrap_per_battery_difference(paired)
        boot_rows.append({'Formulation':fixed_form,**b})
        summary_rows += [
            {'Setting':fixed_form,'Macro_RMSE':float(paired['RMSE_Fixed_600s'].mean()),
             'Macro_MAE':float(paired['MAE_Fixed_600s'].mean()),
             'Batteries':len(paired),'Matched_Observations':len(fixed)},
            {'Setting':'MATCHED_FULL_STEP6_'+full_form,'Macro_RMSE':float(paired['RMSE_Step6_Full'].mean()),
             'Macro_MAE':float(paired['MAE_Step6_Full'].mean()),
             'Batteries':len(paired),'Matched_Observations':len(fixed)}]
    matched=pd.DataFrame(summary_rows)
    paired=pd.concat(paired_rows, ignore_index=True)
    boot=pd.DataFrame(boot_rows)

    # Optional true-3.6-V comparator, evaluated on the actual COMMON subset.
    cross_comparison=pd.DataFrame()
    step11g_path=(PROJECT_ROOT/'reviewer_response'/'step11g_true_3p6_censoring'/
                  'Reviewer_Step11G_True_3p6V_Censored_Prediction.xlsx')
    if step11g_path.exists():
        try:
            z=pd.read_excel(step11g_path,sheet_name='True3p6_Predictions')
            z['_Cycle_Key']=z[CYCLE_COL].map(cycle_key)
            z=z.merge(fixed[[BATTERY_COL,'_Cycle_Key']].drop_duplicates(),
                      on=[BATTERY_COL,'_Cycle_Key'],how='inner',validate='many_to_one')
            rows=[]
            for formulation,grp in z.groupby('Formulation'):
                if grp[BATTERY_COL].nunique()==10:
                    b=score_battery_predictions(grp,formulation,formulation)
                    rows.append({'Formulation':formulation,'Observations':len(grp),
                                 'Macro_RMSE':float(b['RMSE'].mean()), 'Macro_MAE':float(b['MAE'].mean())})
            cross_comparison=pd.DataFrame(rows)
        except Exception as ex:
            print('Optional Step11G same-subset comparison skipped:',str(ex))

    # Audit data columns; no full discharge properties from frozen Step5 become features.
    selected['Observation_Boundary_s']=WINDOW_SECONDS
    with pd.ExcelWriter(STEP11L_WORKBOOK, engine='openpyxl') as writer:
        for sheet,table in [
            ('Methods',make_methods()),('Raw_Manifest',raw_manifest),
            ('Raw_Window_Audit',audit),('Battery_Coverage',coverage),
            ('FixedTime_Macro',macro),('FixedTime_PerBattery',folds),
            ('FixedTime_Predictions',predictions),('Selected_Features',selected),
            ('Selection_Frequency',frequency),('Feature_Coverage',cov_audit),
            ('Inner_FS_Tuning',tune_fs),('Inner_HGB_Tuning',tune_hgb),
            ('Matched_Full_Predictions',full),('Matched_Summary',matched),
            ('Paired_Battery_Deltas',paired),('Paired_Bootstrap',boot),
            ('Step11G_Common_Subset',cross_comparison),
        ]:
            if table is not None and not table.empty:
                table.to_excel(writer,sheet_name=sheet,index=False)

    make_figures(coverage,macro,matched)
    if STEP11L_WORKBOOK.resolve() != (STEP11L_OUT/STEP11L_WORKBOOK.name).resolve():
        shutil.copy2(STEP11L_WORKBOOK,STEP11L_OUT/STEP11L_WORKBOOK.name)
    for name in ['Figure_11L_Coverage.png','Figure_11L_Coverage.pdf',
                 'Figure_11L_Matched_RMSE.png','Figure_11L_Matched_RMSE.pdf',
                 'Step11L_Coverage.csv','Step11L_Window_Audit.csv']:
        if (STEP11L_LOCAL/name).resolve() != (STEP11L_OUT/name).resolve():
            shutil.copy2(STEP11L_LOCAL/name,STEP11L_OUT/name)

    print('\nMatched macro results (SoH percentage points):')
    print(matched.to_string(index=False))
    print('\nPaired fixed-time minus full-discharge bootstrap:')
    print(boot.to_string(index=False))
    print('\nSaved to Drive:',STEP11L_OUT)
    print('\nWorkbook:',STEP11L_WORKBOOK.name)
    return {'macro':macro,'matched':matched,'bootstrap':boot,'coverage':coverage}


if __name__ == '__main__':
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--raw-zip',type=Path,required=True)
    parser.add_argument('--frozen-data',type=Path,required=True)
    parser.add_argument('--frozen-features',type=Path,required=True)
    parser.add_argument('--frozen-predictions',type=Path,required=True)
    parser.add_argument('--output-dir',type=Path,default=Path('outputs/600s_archived'))
    parser.add_argument('--work-dir',type=Path,default=Path('work/600s_archived'))
    cli=parser.parse_args()
    DRIVE_ZIP=cli.raw_zip.resolve()
    STEP5_DATA_PATH=cli.frozen_data.resolve()
    STEP5_FEATURE_LIST_PATH=FEATURE_LIST_PATH=cli.frozen_features.resolve()
    STEP6_PREDICTIONS_PATH=cli.frozen_predictions.resolve()
    STEP11L_OUT=STEP11L_LOCAL=cli.output_dir.resolve()
    STEP11L_OUT.mkdir(parents=True,exist_ok=True)
    STEP11L_WORKBOOK=STEP11L_LOCAL/'Reviewer_Step11L_Fixed_Duration_Prediction.xlsx'
    LOCAL_ZIP=cli.work_dir.resolve()/'collab.zip'
    EXTRACT_ROOT=cli.work_dir.resolve()/'raw'
    LOCAL_ZIP.parent.mkdir(parents=True,exist_ok=True)
    main_step11l()

