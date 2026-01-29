# -*- coding: utf-8 -*-
"""
@author: pco30

Adds SOH-useful features (no temperature data available (Total of 109 features))
- Internal resistance: max/median/p90
- dV/dt mean + mean(|dV/dt|) + dynamic resistance proxies (|dV/dt|/|I|)
- End voltages + delta-V features
- Coulombic & energy efficiency
- Step 2 / Step 7 CC stability (std) + voltage std
- Step 4 CV taper metrics (end current, current drop, tau proxy)
- IC extras: peak count, peak-to-peak distance (if >=2 peaks), IC area vs voltage
"""

# import relevant libraries 
import numpy as np
import pandas as pd
import os
from sklearn.model_selection import KFold, GroupKFold
from sklearn.inspection import permutation_importance
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

# Function to get the creation time of a file
def get_file_creation_time(file_path):
    return os.path.getctime(file_path)

# Function to extract files from a folder path
def extract_files(folder_path):
    excel_files = [os.path.join(folder_path, file) for file in os.listdir(folder_path) if file.endswith('.xlsx')]
    return excel_files

# Iterate over each file path and alter the cycle index with addition of more files
def process_files(folder_path, sheet_names_to_import):
    file_paths = extract_files(folder_path)
    file_paths.sort(key=get_file_creation_time)
    dfs = []
    last_cycle_index = 0
    for file_path in file_paths:
        sheet_names = pd.ExcelFile(file_path).sheet_names
        # Process sheets in the order they appear in sheet_names_to_import if they exist in the file
        for target_sheet in sheet_names_to_import:
            if target_sheet in sheet_names:
                df = pd.read_excel(file_path, sheet_name=target_sheet)
                df['Cycle_Index'] += last_cycle_index
                last_cycle_index = df['Cycle_Index'].max()
                dfs.append(df)
    return pd.concat(dfs, ignore_index=True)

# Extract battery name from folder path
def battery_name_from_path(folder_path):
    return os.path.basename(os.path.normpath(folder_path))

# scipy is for IC peak metrics (my code will still run without it so optional)
try:
    from scipy.signal import find_peaks, peak_widths, peak_prominences
    _HAS_SCIPY = True
except Exception:
    _HAS_SCIPY = False

# Function to calculate soh for variable discharge in each cycle
def calculate_soh_variable_discharge(df):
    # Convert time to hours
    df['Test_Time(h)'] = df['Test_Time(s)'] / 3600
    # Sort data just in case
    df = df.sort_values(['Cycle_Index', 'Test_Time(h)']).reset_index(drop=True)
    # Compute time difference (delta_t) between consecutive readings
    df['delta_t(h)'] = df['Test_Time(h)'].diff()
    df.loc[df['Cycle_Index'] != df['Cycle_Index'].shift(), 'delta_t(h)'] = 0  # Reset at cycle boundaries
    # Initialize discharge capacity column with float type
    df['Discharge_Capacity(mAh)'] = 0.0
    # Identify discharge rows (Current < 0)
    discharge_mask = df['Current(A)'] < 0
    # Calculate discharge capacity: I × Δt × 1000
    df.loc[discharge_mask, 'Discharge_Capacity(mAh)'] = (
        abs(df.loc[discharge_mask, 'Current(A)']) * df.loc[discharge_mask, 'delta_t(h)'] * 1000
    )
    # Total battery capacity per cycle
    df['Battery_Capacity(mAh)'] = df.groupby('Cycle_Index')['Discharge_Capacity(mAh)'].transform('sum')
    df['Battery_Capacity(mAh)'] = df['Battery_Capacity(mAh)']/5 # 5 cycles within each cycle
    # Calculate SOH as a percentage of the maximum capacity
    max_capacity = df['Battery_Capacity(mAh)'].max()
    df['Calculated_SOH(%)'] = (df['Battery_Capacity(mAh)'] / max_capacity) * 100
    # Compute charge and discharge times per cycle
    df['Time_charged(h)'] = 0.0
    df['Time_discharged(h)'] = 0.0
    df.loc[df['Current(A)'] >= 0, 'Time_charged(h)'] = df['delta_t(h)']
    df.loc[df['Current(A)'] < 0,  'Time_discharged(h)'] = df['delta_t(h)']
    # Total time charged/discharged per cycle
    df['Test_Time_charged(h)'] = df.groupby('Cycle_Index')['Time_charged(h)'].transform('sum')
    df['Test_Time_Discharged(h)'] = df.groupby('Cycle_Index')['Time_discharged(h)'].transform('sum')
    return df

# Function to calculate SOH for constant discharge
def soh_from_current_integration(df, cycle_col="Cycle_Index",time_col="Test_Time(s)", current_col="Current(A)",rated_capacity_Ah=None,reference="first"):  # "first" or "max"
    d = df.copy().sort_values([cycle_col, time_col])
    d["dt_h"] = d.groupby(cycle_col)[time_col].diff().fillna(0) / 3600.0
    d["dt_h"] = d["dt_h"].clip(lower=0)
    d["I_dis_A"] = (-d[current_col]).clip(lower=0)
    d["dQ_Ah"] = d["I_dis_A"] * d["dt_h"]
    d["Qdis_Ah_cum"] = d.groupby(cycle_col)["dQ_Ah"].cumsum()
    cap_by_cycle = d.groupby(cycle_col)["Qdis_Ah_cum"].max()

    if rated_capacity_Ah is not None:
        ref = float(rated_capacity_Ah)
    elif reference == "first":
        ref = float(cap_by_cycle.iloc[0])
    else:
        ref = float(cap_by_cycle.max())

    d["Battery_Capacity(Ah)"] = d[cycle_col].map(cap_by_cycle)          # Not explicity needed
    d["Discharge_Capacity_Cycle(Ah)"] = d["Battery_Capacity(Ah)"]
    d["Calculated_SOH(%)"] = d["Battery_Capacity(Ah)"] / ref * 100.0
    return d

def _safe_stats(s: pd.Series, prefix: str):
    """Return basic stats with skew/kurt using pandas (handles empty/NaN)."""
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
    """Slope of least squares. Returns NaN if insufficient."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    m = np.isfinite(x) & np.isfinite(y)
    x = x[m]; y = y[m]
    if len(x) < 2:
        return np.nan
    x0 = x - x.mean()
    denom = np.dot(x0, x0)
    if denom == 0:
        return np.nan
    return float(np.dot(x0, y - y.mean()) / denom)


def _ic_features_from_segment(seg: pd.DataFrame, time_col: str, volt_col: str,curr_col: str, dt_h_col: str,peak_sign: str,prefix: str):
    """
    Compute IC curve features from a segment.
    IC ≈ dQ/dV where Q is cumulative (Ah) over time.

    Adds extra features:
    - {prefix}_IC_areaVoltage : integral of IC over Voltage (more meaningful than index-area)
    - {prefix}_IC_peakCount
    - {prefix}_IC_peakToPeakDistance (Voltage distance between top-2 prominent peaks)
    """
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

    m = np.isfinite(I) & np.isfinite(V) & np.isfinite(dt_h)
    I = I[m]; V = V[m]; dt_h = dt_h[m]
    if len(I) < 5:
        return out

    dQ = np.abs(I) * dt_h
    Q = np.cumsum(dQ)

    # IC = dQ/dV (gradient is smoother than diff)
    dQ_dV = np.gradient(Q, V)

    s = pd.Series(dQ_dV).replace([np.inf, -np.inf], np.nan).dropna()
    if len(s) >= 1:
        y = s.to_numpy(dtype=float)

        out[f"{prefix}_IC_area"] = float(np.trapz(y, dx=1.0))  # index-area proxy
        out[f"{prefix}_IC_max"] = float(np.nanmax(y))
        out[f"{prefix}_IC_min"] = float(np.nanmin(y))
        out[f"{prefix}_IC_mean"] = float(np.nanmean(y))
        out[f"{prefix}_IC_std"] = float(np.nanstd(y, ddof=1)) if len(y) > 1 else 0.0
        out[f"{prefix}_IC_skewness"] = float(pd.Series(y).skew()) if len(y) > 2 else 0.0
        out[f"{prefix}_IC_kurtosis"] = float(pd.Series(y).kurt()) if len(y) > 3 else 0.0

        # IC area vs voltage (more interpretable)
        # if V is messy, this becomes noisy but still computed.
        # Align x with y length as best-effort.
        xV = V[:len(y)]
        ok = np.isfinite(xV) & np.isfinite(y)
        if ok.sum() >= 2:
            out[f"{prefix}_IC_areaVoltage"] = float(np.trapz(y[ok], x=xV[ok]))

    # Peak features (requires scipy)
    if not _HAS_SCIPY or len(s) < 5:
        return out

    y = s.to_numpy(dtype=float)
    xV = V[:len(y)]
    y_for_peaks = -y if peak_sign == "negative" else y

    peaks, _ = find_peaks(y_for_peaks)
    out[f"{prefix}_IC_peakCount"] = float(len(peaks))

    if len(peaks) == 0:
        return out

    prominences = peak_prominences(y_for_peaks, peaks)[0]
    k = int(np.argmax(prominences))
    p = int(peaks[k])

    out[f"{prefix}_IC_peak"] = float(y[p])
    out[f"{prefix}_IC_peakLocation"] = float(xV[p]) if np.isfinite(xV[p]) else np.nan
    out[f"{prefix}_IC_peakProminence"] = float(prominences[k])

    widths_res = peak_widths(y_for_peaks, [p], rel_height=0.5)
    out[f"{prefix}_IC_peakWidth"] = float(widths_res[0][0])

    left = int(np.floor(widths_res[2][0]))
    right = int(np.ceil(widths_res[3][0]))
    left = max(left, 0); right = min(right, len(y) - 1)
    if right > left:
        out[f"{prefix}_IC_peaksArea"] = float(np.trapz(y[left:right+1], dx=1.0))

    if p - 1 >= 0:
        out[f"{prefix}_IC_peakLeftSlope"] = float(y[p] - y[p-1])
    if p + 1 < len(y):
        out[f"{prefix}_IC_peakRightSlope"] = float(y[p+1] - y[p])

    # Peak-to-peak distance between top-2 prominent peaks (Voltage distance)
    if len(peaks) >= 2 and np.isfinite(xV).any():
        order = np.argsort(prominences)[::-1]
        p1 = int(peaks[order[0]])
        p2 = int(peaks[order[1]])
        if np.isfinite(xV[p1]) and np.isfinite(xV[p2]):
            out[f"{prefix}_IC_peakToPeakDistance"] = float(abs(xV[p1] - xV[p2]))

    return out


def build_cycle_features(df: pd.DataFrame,
                         cycle_col="Cycle_Index",
                         time_col="Test_Time(s)",
                         volt_col="Voltage(V)",
                         curr_col="Current(A)",
                         step_col=None,          # "Step_Index" if present
                         charge_mask=None,       # optional custom mask function
                         discharge_mask=None,    # optional custom mask function
                         step2_value=2,
                         step4_value=4,
                         step7_value=7,
                         cc_tol_frac=0.05,       # CC detection tolerance around median current
                         cv_vtol_frac=0.005,     # CV detection tolerance around median voltage
                         eps=1e-12):
    """
    Returns one row per cycle with charge/discharge features + additional SOH predictors:
    - Internal_Resistance_max/median/p90
    - dVdt_mean, avg_abs_dvdt, mean_abs_dVdt_over_I, median_abs_dVdt_over_I
    - Charge_endVoltage, Discharge_endVoltage, DeltaV_charge, DeltaV_discharge
    - Coulombic_Efficiency, Energy_Efficiency
    - Step2/Step7 CC stability: currentStd, voltageStd
    - Step4 CV taper: endCurrent, currentDrop, tau10 (time to 10% current)
    - IC extras: peakCount, peakToPeakDistance, IC_areaVoltage
    """

    d = df.copy()

    # Ensure numeric types and sort
    d[time_col] = pd.to_numeric(d[time_col], errors="coerce")
    d[volt_col] = pd.to_numeric(d[volt_col], errors="coerce")
    d[curr_col] = pd.to_numeric(d[curr_col], errors="coerce")
    d = d.sort_values([cycle_col, time_col])

    # dt in hours per cycle
    dt_s = d.groupby(cycle_col)[time_col].diff()
    d["dt_h"] = dt_s.fillna(0) / 3600.0
    d["dt_h"] = d["dt_h"].clip(lower=0)

    # dV/dt per row (V/s) within each cycle
    dV = d.groupby(cycle_col)[volt_col].diff()
    dT = dt_s
    d["dVdt_V_per_s"] = dV / dT
    d.loc[(dT <= 0) | (~np.isfinite(d["dVdt_V_per_s"])), "dVdt_V_per_s"] = np.nan

    # Dynamic resistance proxy: |dV/dt| / |I|
    absI = np.abs(d[curr_col].to_numpy(dtype=float))
    d["abs_dVdt_over_absI"] = np.abs(d["dVdt_V_per_s"].to_numpy(dtype=float)) / np.maximum(absI, eps)
    d.loc[~np.isfinite(d["abs_dVdt_over_absI"]), "abs_dVdt_over_absI"] = np.nan

    # Default charge/discharge masks (sign convention)
    if charge_mask is None:
        charge_mask = lambda x: x[curr_col] > 0
    if discharge_mask is None:
        discharge_mask = lambda x: x[curr_col] < 0

    # Power and energy (Wh): V*I*dt_h (signed)
    d["P_W"] = d[volt_col] * d[curr_col]
    d["dE_Wh"] = d["P_W"] * d["dt_h"]

    # Capacity increments (Ah): integrate +I for charge, -I for discharge
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

        #  Internal Resistance features
        if "Internal_Resistance(Ohm)" in g.columns:
            r = pd.to_numeric(g["Internal_Resistance(Ohm)"], errors="coerce").dropna()
            out["Internal_Resistance_max"] = float(r.max()) if len(r) else np.nan
            out["Internal_Resistance_median"] = float(r.median()) if len(r) else np.nan
            out["Internal_Resistance_p90"] = float(r.quantile(0.90)) if len(r) else np.nan
        else:
            out["Internal_Resistance_max"] = np.nan
            out["Internal_Resistance_median"] = np.nan
            out["Internal_Resistance_p90"] = np.nan

        # dV/dt features (cycle-level) 
        dvdt = pd.to_numeric(g["dVdt_V_per_s"], errors="coerce").dropna()
        out["dVdt_mean"] = float(dvdt.mean()) if len(dvdt) else np.nan
        out["avg_abs_dvdt"] = float(np.mean(np.abs(dvdt))) if len(dvdt) else np.nan

        dyn = pd.to_numeric(g["abs_dVdt_over_absI"], errors="coerce").dropna()
        out["mean_abs_dVdt_over_I"] = float(dyn.mean()) if len(dyn) else np.nan
        out["median_abs_dVdt_over_I"] = float(dyn.median()) if len(dyn) else np.nan

        ch = g[charge_mask(g)].copy()
        dis = g[discharge_mask(g)].copy()

        # Charge aggregate features 
        out["Charge_cumulativeCapacity"] = float(ch["dQ_ch_Ah"].sum()) if len(ch) else np.nan
        out["Charge_cumulativeEnergy"] = float(ch["dE_Wh"].sum()) if len(ch) else np.nan
        out["Charge_duration"] = float(ch["dt_h"].sum()) if len(ch) else np.nan
        out["Charge_startVoltage"] = float(ch[volt_col].iloc[0]) if len(ch) else np.nan
        out["Charge_endVoltage"] = float(ch[volt_col].iloc[-1]) if len(ch) else np.nan
        out["DeltaV_charge"] = (out["Charge_endVoltage"] - out["Charge_startVoltage"]
                                if np.isfinite(out["Charge_endVoltage"]) and np.isfinite(out["Charge_startVoltage"])
                                else np.nan)
        out.update(_safe_stats(ch[volt_col], "Charge_Voltage"))
        out.update(_safe_stats(ch[curr_col], "Charge_Current"))

        # Discharge aggregate features 
        out["Discharge_cumulativeCapacity"] = float(dis["dQ_dis_Ah"].sum()) if len(dis) else np.nan
        out["Discharge_cumulativeEnergy"] = float(dis["dE_Wh"].sum()) if len(dis) else np.nan
        out["Discharge_duration"] = float(dis["dt_h"].sum()) if len(dis) else np.nan
        out["Discharge_startVoltage"] = float(dis[volt_col].iloc[0]) if len(dis) else np.nan
        out["Discharge_endVoltage"] = float(dis[volt_col].iloc[-1]) if len(dis) else np.nan
        out["DeltaV_discharge"] = (out["Discharge_startVoltage"] - out["Discharge_endVoltage"]
                                   if np.isfinite(out["Discharge_startVoltage"]) and np.isfinite(out["Discharge_endVoltage"])
                                   else np.nan)
        out.update(_safe_stats(dis[volt_col], "Discharge_Voltage"))
        out.update(_safe_stats(dis[curr_col], "Discharge_Current"))

        # ---- Efficiency features ----
        if np.isfinite(out["Charge_cumulativeCapacity"]) and out["Charge_cumulativeCapacity"] > 0 and np.isfinite(out["Discharge_cumulativeCapacity"]):
            out["Coulombic_Efficiency"] = float(out["Discharge_cumulativeCapacity"] / out["Charge_cumulativeCapacity"])
        else:
            out["Coulombic_Efficiency"] = np.nan

        if np.isfinite(out["Charge_cumulativeEnergy"]) and out["Charge_cumulativeEnergy"] > 0 and np.isfinite(out["Discharge_cumulativeEnergy"]):
            out["Energy_Efficiency"] = float(abs(out["Discharge_cumulativeEnergy"]) / out["Charge_cumulativeEnergy"])
        else:
            out["Energy_Efficiency"] = np.nan

        # Step-specific
        if step_col is not None and step_col in g.columns:
            # Step 2 (charge) IC features + CC features
            s2 = g[(g[step_col] == step2_value)].copy()
            s2_ch = s2[s2[curr_col] > 0].copy()

            out.update(_ic_features_from_segment(
                s2_ch, time_col, volt_col, curr_col, "dt_h",
                peak_sign="positive",
                prefix="Charge_Step2"
            ))

            # CC heuristics for Step2: current approximately constant (within cc_tol_frac of median)
            if len(s2_ch) >= 5:
                I_med = float(np.nanmedian(s2_ch[curr_col]))
                tol = cc_tol_frac * abs(I_med) if np.isfinite(I_med) and I_med != 0 else np.nan
                s2_cc = s2_ch[np.abs(s2_ch[curr_col] - I_med) <= tol].copy() if np.isfinite(tol) else s2_ch.copy()

                out["Charge_Step2_CC_duration"] = float(s2_cc["dt_h"].sum()) if len(s2_cc) else np.nan
                out["Charge_Step2_CC_currentMedian"] = float(np.nanmedian(s2_cc[curr_col])) if len(s2_cc) else np.nan
                out["Charge_Step2_CC_energy"] = float((s2_cc["dE_Wh"]).sum()) if len(s2_cc) else np.nan

                # CC stability
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

            # Step 4 (charge) CV features: voltage ~ constant near top (heuristic)
            s4 = g[(g[step_col] == step4_value)].copy()
            s4_ch = s4[s4[curr_col] > 0].copy()

            if len(s4_ch) >= 5:
                V_med = float(np.nanmedian(s4_ch[volt_col]))
                vtol = cv_vtol_frac * abs(V_med) if np.isfinite(V_med) and V_med != 0 else np.nan
                s4_cv = s4_ch[np.abs(s4_ch[volt_col] - V_med) <= vtol].copy() if np.isfinite(vtol) else s4_ch.copy()

                out["Charge_Step4_CV_duration"] = float(s4_cv["dt_h"].sum()) if len(s4_cv) else np.nan
                out["Charge_Step4_CV_voltageMedian"] = float(np.nanmedian(s4_cv[volt_col])) if len(s4_cv) else np.nan

                # CV slope: current decay slope vs time (A per hour)
                t_h = (s4_cv[time_col].to_numpy(dtype=float) / 3600.0)
                I = s4_cv[curr_col].to_numpy(dtype=float)
                out["Charge_Step4_CV_slope"] = _linear_slope(t_h, I)

                out["Charge_Step4_CV_energy"] = float((s4_cv["dE_Wh"]).sum()) if len(s4_cv) else np.nan
                cvI = pd.Series(s4_cv[curr_col]).dropna()
                out["Charge_Step4_CV_skewness"] = float(cvI.skew()) if len(cvI) > 2 else (0.0 if len(cvI) else np.nan)
                out["Charge_Step4_CV_kurtosis"] = float(cvI.kurt()) if len(cvI) > 3 else (0.0 if len(cvI) else np.nan)

                # CV taper metrics
                out["Charge_Step4_CV_endCurrent"] = float(s4_cv[curr_col].iloc[-1]) if len(s4_cv) else np.nan
                out["Charge_Step4_CV_startCurrent"] = float(s4_cv[curr_col].iloc[0]) if len(s4_cv) else np.nan
                if np.isfinite(out["Charge_Step4_CV_startCurrent"]) and np.isfinite(out["Charge_Step4_CV_endCurrent"]):
                    out["Charge_Step4_CV_currentDrop"] = float(out["Charge_Step4_CV_startCurrent"] - out["Charge_Step4_CV_endCurrent"])
                else:
                    out["Charge_Step4_CV_currentDrop"] = np.nan

                # tau10: time to drop to 10% of start current (hours)
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

            # Step 7 (discharge) IC + CC features
            s7 = g[(g[step_col] == step7_value)].copy()
            s7_dis = s7[s7[curr_col] < 0].copy()

            out.update(_ic_features_from_segment(
                s7_dis, time_col, volt_col, curr_col, "dt_h",
                peak_sign="positive",
                prefix="Discharge_Step7"
            ))

            if len(s7_dis) >= 5:
                I_med = float(np.nanmedian(s7_dis[curr_col]))
                tol = cc_tol_frac * abs(I_med) if np.isfinite(I_med) and I_med != 0 else np.nan
                s7_cc = s7_dis[np.abs(s7_dis[curr_col] - I_med) <= tol].copy() if np.isfinite(tol) else s7_dis.copy()

                out["Discharge_Step7_CC_duration"] = float(s7_cc["dt_h"].sum()) if len(s7_cc) else np.nan
                out["Discharge_Step7_CC_currentMedian"] = float(np.nanmedian(s7_cc[curr_col])) if len(s7_cc) else np.nan

                # CC slope: voltage vs time slope during constant-current discharge (V per hour)
                t_h = (s7_cc[time_col].to_numpy(dtype=float) / 3600.0)
                V = s7_cc[volt_col].to_numpy(dtype=float)
                out["Discharge_Step7_CC_slope"] = _linear_slope(t_h, V)

                out["Discharge_Step7_CC_energy"] = float((s7_cc["dE_Wh"]).sum()) if len(s7_cc) else np.nan

                # CC stability
                out["Discharge_Step7_CC_currentStd"] = float(pd.Series(s7_cc[curr_col]).dropna().std(ddof=1)) if len(s7_cc) > 1 else (0.0 if len(s7_cc) else np.nan)
                out["Discharge_Step7_CC_voltageStd"] = float(pd.Series(s7_cc[volt_col]).dropna().std(ddof=1)) if len(s7_cc) > 1 else (0.0 if len(s7_cc) else np.nan)

                ccV = pd.Series(s7_cc[volt_col]).dropna()
                out["Discharge_Step7_CC_skewness"] = float(ccV.skew()) if len(ccV) > 2 else (0.0 if len(ccV) else np.nan)
                out["Discharge_Step7_CC_kurtosis"] = float(ccV.kurt()) if len(ccV) > 3 else (0.0 if len(ccV) else np.nan)

                # tInv proxy
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

# Returns the starter shortlist we agreed on (optionally include Cycle_Index as a "shortcut" feature).
# def starter_shortlist(include_cycle_index=False):
#     exclude = {
#         "Calculated_SOH(%)",   # target
#         "battery_id",          # ID 
#     }
#     feats = [c for c in combined.columns if c not in exclude]

#     if not include_cycle_index and "Cycle_Index" in feats:
#         feats.remove("Cycle_Index")

#     return feats

def starter_shortlist(include_cycle_index=False):
    feats = [
        # Aging / resistance / dynamics  (keep + enrich)
        "Internal_Resistance_max",          
        "Internal_Resistance_median",
        "Internal_Resistance_p90",
        "dVdt_mean",                        
        "avg_abs_dvdt",
        "mean_abs_dVdt_over_I",
        "median_abs_dVdt_over_I",           

        # Global charge/discharge summaries (keep + enrich)
        "Charge_cumulativeCapacity",
        "Charge_cumulativeEnergy",          
        "Charge_duration",
        "Charge_startVoltage",              
        "Charge_endVoltage",                
        "DeltaV_charge",

        "Discharge_cumulativeCapacity",
        "Discharge_cumulativeEnergy",       
        "Discharge_duration",
        "Discharge_startVoltage",           
        "Discharge_endVoltage",             
        "DeltaV_discharge",

        "Coulombic_Efficiency",
        "Energy_Efficiency",

        # Global voltage/current distribution stats 
        "Charge_Voltage_mean",              
        "Charge_Voltage_std",               
        "Charge_Current_mean",              
        "Charge_Current_std",               

        "Discharge_Voltage_mean",           
        "Discharge_Voltage_std",            
        "Discharge_Current_mean",           
        "Discharge_Current_std",            

        # IC charge (Step2)  (keep + enrich)
        "Charge_Step2_IC_areaVoltage",      
        "Charge_Step2_IC_peakLocation",
        "Charge_Step2_IC_peakProminence",
        "Charge_Step2_IC_peakWidth",
        "Charge_Step2_IC_peakCount",
        "Charge_Step2_IC_peaksArea",
        "Charge_Step2_IC_peakToPeakDistance",  
        "Charge_Step2_IC_peakLeftSlope",        
        "Charge_Step2_IC_peakRightSlope",       

        # CC charge stability (Step2)  
        "Charge_Step2_CC_duration",
        "Charge_Step2_CC_currentMedian",
        "Charge_Step2_CC_energy",           
        "Charge_Step2_CC_currentStd",       
        "Charge_Step2_CC_voltageStd",       

        # CV charge taper (Step4)  (keep + enrich)
        "Charge_Step4_CV_duration",
        "Charge_Step4_CV_voltageMedian",    
        "Charge_Step4_CV_slope",            
        "Charge_Step4_CV_endCurrent",
        "Charge_Step4_CV_currentDrop",
        "Charge_Step4_CV_tau10",
        "Charge_Step4_CV_energy",           

        # IC discharge (Step7)  (keep + enrich)
        "Discharge_Step7_IC_areaVoltage",   
        "Discharge_Step7_IC_peakLocation",
        "Discharge_Step7_IC_peakProminence",
        "Discharge_Step7_IC_peakWidth",
        "Discharge_Step7_IC_peakCount",
        "Discharge_Step7_IC_peaksArea",
        "Discharge_Step7_IC_peakToPeakDistance",  
        "Discharge_Step7_IC_peakLeftSlope",        
        "Discharge_Step7_IC_peakRightSlope",       

        # CC discharge (Step7)  (keep + enrich)
        "Discharge_Step7_CC_slope",
        "Discharge_Step7_CC_tInv",
        "Discharge_Step7_CC_duration",
        "Discharge_Step7_CC_energy",        
        "Discharge_Step7_CC_currentStd",    
        "Discharge_Step7_CC_voltageStd",    
        "protocol",
        "cycle_frac",
    ]

    if include_cycle_index:
        feats = ["Cycle_Index"] + feats
    return feats


# Drops redundant features by removing anything highly correlated with a previously-kept feature.
def correlation_prune(X, threshold=0.95, method="spearman"):
    if X.shape[1] <= 1:
        return list(X.columns)

    # Pairwise correlation; absolute because +1 and -1 are both "duplicate information"
    corr = X.corr(method=method).abs()

    keep = []
    for col in X.columns:
        if not keep:
            keep.append(col)
            continue

        # If this feature is basically a clone of something we already kept, skip it
        if (corr.loc[col, keep] > threshold).any():
            continue

        keep.append(col)

    return keep


# Selects features that consistently matter across CV folds using permutation importance (supports grouping by battery).
def stable_permutation_selection(X, y, groups=None, n_splits=5, top_n_each_fold=15, appear_frac_thresh=0.7, corr_threshold=0.95, corr_method="spearman", n_repeats=10, random_state=42, model=None):
    # Step 1: quick redundancy reduction so importance isn't split across near-duplicates
    pruned_cols = correlation_prune(X, threshold=corr_threshold, method=corr_method)
    Xp = X[pruned_cols]

    # Step 2: choose a split strategy
    # If groups exist (battery_id / cell_id), keep whole batteries together in folds.
    if groups is not None:
        cv = GroupKFold(n_splits=n_splits)
        split_iter = cv.split(Xp, y, groups=groups)
    else:
        cv = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        split_iter = cv.split(Xp, y)

    # Step 3: default model (fast and handles non-linearities well)
    if model is None:
        model = HistGradientBoostingRegressor(random_state=random_state)

    # We'll track (a) how often a feature lands in the top-N, and (b) its average importance.
    cols = np.array(Xp.columns)
    appear_counts = pd.Series(0, index=cols, dtype=int)
    mean_importance = pd.Series(0.0, index=cols, dtype=float)
    n_folds = 0

    for tr_idx, va_idx in split_iter:
        n_folds += 1

        # Train on this fold
        X_tr, X_va = Xp.iloc[tr_idx], Xp.iloc[va_idx]
        y_tr, y_va = y.iloc[tr_idx], y.iloc[va_idx]
        model.fit(X_tr, y_tr)

        # Permutation importance: shuffle each feature and see how much validation performance drops.
        perm = permutation_importance(model, X_va, y_va, n_repeats=n_repeats, random_state=random_state, n_jobs=-1)
        imp = pd.Series(perm.importances_mean, index=cols)

        # Accumulate average importance
        mean_importance += imp

        # Count "wins" (appears in top-N for this fold)
        top_cols = imp.sort_values(ascending=False).head(min(top_n_each_fold, len(cols))).index
        appear_counts.loc[top_cols] += 1

    # Turn totals into averages
    mean_importance /= max(n_folds, 1)
    appear_frac = appear_counts / max(n_folds, 1)

    # Ranking table (handy if you ever want to inspect later; we won't print it)
    ranking_df = pd.DataFrame(
        {"appear_frac": appear_frac, "appear_count": appear_counts, "mean_perm_importance": mean_importance}
    ).sort_values(["appear_frac", "mean_perm_importance"], ascending=False)

    # Stable set = shows up in the top-N often enough
    selected = appear_frac[appear_frac >= appear_frac_thresh].index.tolist()

    # If you're too strict and nothing survives, fall back to the strongest average signals
    if len(selected) == 0:
        selected = ranking_df["mean_perm_importance"].sort_values(ascending=False).head(min(10, len(cols))).index.tolist()

    return selected, ranking_df


# Main helper: uses the starter shortlist + stability selection and returns a dataframe with only the selected features.
def select_best_feature_df(combined, target_col="Calculated_SOH(%)", group_col="battery_id", include_cycle_index=False, dropna_target=True, corr_threshold=0.95, n_splits=5, top_n_each_fold=15, appear_frac_thresh=0.7, n_repeats=10, random_state=42, keep_target=False, keep_group=False, model=None):
    df = combined.copy()
    if dropna_target:
        df = df.loc[df[target_col].notna()].copy()

    # Only keep shortlist columns that actually exist (prevents KeyErrors if a feature is missing)
    shortlist = starter_shortlist(include_cycle_index=include_cycle_index)
    feat_cols = [c for c in shortlist if c in df.columns]

    # Features/target
    X = df[feat_cols]
    y = df[target_col].astype(float)

    # Group column is optional — if it's not present, we just do normal KFold
    groups = None
    if group_col is not None and group_col in df.columns:
        groups = df[group_col]

    # Run the selection
    selected_features, _ranking = stable_permutation_selection(
        X, y, groups=groups,
        n_splits=n_splits, top_n_each_fold=top_n_each_fold,
        appear_frac_thresh=appear_frac_thresh,
        corr_threshold=corr_threshold,
        n_repeats=n_repeats,
        random_state=random_state,
        model=model
    )

    # Decide what to return
    cols_to_return = []

    # Keep IDs if you want to preserve grouping info for later
    if keep_group and group_col is not None and group_col in df.columns:
        cols_to_return.append(group_col)

    # Keep the target if you want a self-contained training dataframe
    if keep_target and target_col in df.columns:
        cols_to_return.append(target_col)

    cols_to_return += selected_features
    return df[cols_to_return].copy()


# MODELS
def fit_predict_single_regularized_model(X_train, y_train, X_test, y_test,random_state=42):
    """
    Single well-regularized model baseline.
    - Median imputation (handles NaNs)
    - HistGradientBoosting with early stopping + L2 regularization
    """
    pre = Pipeline([
        ("imputer", SimpleImputer(strategy="median"))
    ])

    X_train_p = pre.fit_transform(X_train)
    X_test_p = pre.transform(X_test)

    # Well-regularized settings (tuneable)
    model = HistGradientBoostingRegressor(
    learning_rate=0.05,
    max_iter=5000,
    max_depth=7,
    min_samples_leaf=10,
    l2_regularization=0.1,
    early_stopping=True,
    validation_fraction=0.15,
    n_iter_no_change=80,
    random_state=42
)


    model.fit(X_train_p, y_train)

    y_pred = model.predict(X_test_p)
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    return y_pred, mse, mae

# Builds OOF predictions for one base model so the meta-learner never sees leakage.
def get_oof_preds(model, X_train, y_train, X_test, groups=None, n_splits=5, random_state=42):
    # Use group-aware CV if groups are provided (prevents mixing batteries across folds)
    if groups is not None:
        cv = GroupKFold(n_splits=n_splits)
        split_iter = cv.split(X_train, y_train, groups=groups)
    else:
        cv = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        split_iter = cv.split(X_train, y_train)

    oof_train = np.zeros((X_train.shape[0],), dtype=float)
    oof_test_folds = np.zeros((n_splits, X_test.shape[0]), dtype=float)

    for i, (tr_idx, va_idx) in enumerate(split_iter):
        X_tr, X_va = X_train[tr_idx], X_train[va_idx]
        y_tr = y_train[tr_idx]

        # Fit on fold-train and predict fold-val + test
        model.fit(X_tr, y_tr)
        oof_train[va_idx] = model.predict(X_va)
        oof_test_folds[i, :] = model.predict(X_test)

    oof_test = oof_test_folds.mean(axis=0)
    return oof_train.reshape(-1, 1), oof_test.reshape(-1, 1)


# Trains a stacked ensemble and returns predictions + MSE/MAE (handles NaNs via median imputation).
def fit_predict_stacked_ensemble(X_train, y_train, X_test, y_test, groups_train=None, n_splits=5, random_state=42):
    # Tree models don't need scaling, but Ridge does; using the same transformed matrix keeps things consistent.
    pre = Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())])

    X_train_p = pre.fit_transform(X_train)
    X_test_p = pre.transform(X_test)

    # Slightly more regularized than your originals to reduce overfit on small battery datasets
    xgb = XGBRegressor(n_estimators=600, learning_rate=0.03, max_depth=6, subsample=0.8, colsample_bytree=0.8, min_child_weight=2.0, reg_lambda=1.0, random_state=101, n_jobs=-1)
    lgb = LGBMRegressor(n_estimators=1200, learning_rate=0.02, num_leaves=31, subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0, random_state=101, n_jobs=-1)
    rf = RandomForestRegressor(n_estimators=600, max_depth=14, min_samples_leaf=2, random_state=101, n_jobs=-1)

    base_models = [xgb, lgb, rf]

    # Create meta-features using OOF predictions (no leakage)
    oof_tr_list, oof_te_list = [], []
    for m in base_models:
        oof_tr, oof_te = get_oof_preds(m, X_train_p, y_train, X_test_p, groups=groups_train, n_splits=n_splits, random_state=random_state)
        oof_tr_list.append(oof_tr)
        oof_te_list.append(oof_te)

    X_meta_train = np.concatenate(oof_tr_list, axis=1)
    X_meta_test = np.concatenate(oof_te_list, axis=1)

    # Meta learner (Ridge is stable for stacking; alpha bumped for robustness)
    meta_model = Ridge(alpha=5.0, random_state=random_state)
    meta_model.fit(X_meta_train, y_train)

    y_pred = meta_model.predict(X_meta_test)

    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)

    return y_pred, mse, mae

# Process the data
datasets = []
all_features = []

# process constant discharge data
folder_paths = ["C:/Users/HYACINTH OSEJI/OneDrive - University of Bath/Documents/PYTHON STUFF/CS2_33",
                "C:/Users/HYACINTH OSEJI/OneDrive - University of Bath/Documents/PYTHON STUFF/CS2_34",
                "C:/Users/HYACINTH OSEJI/OneDrive - University of Bath/Documents/PYTHON STUFF/CS2_35",
                "C:/Users/HYACINTH OSEJI/OneDrive - University of Bath/Documents/PYTHON STUFF/CS2_36",
                "C:/Users/HYACINTH OSEJI/OneDrive - University of Bath/Documents/PYTHON STUFF/CS2_37",
                "C:/Users/HYACINTH OSEJI/OneDrive - University of Bath/Documents/PYTHON STUFF/CS2_38"]

sheet_names = [["Channel_1-006"], ["Channel_1-007"], ["Channel_1-008"], ["Channel_1-009"], ["Channel_1-010"], ["Channel_1-011"]]

for i in range(len(folder_paths)):
    battery_id = battery_name_from_path(folder_paths[i])
    df = process_files(folder_paths[i], sheet_names[i])
    df["battery_id"] = battery_id
    df_SOH = soh_from_current_integration(df)
    df_SOH = df_SOH[["Cycle_Index", "Calculated_SOH(%)"]]
    df_SOH = df_SOH.groupby("Cycle_Index", as_index=False).agg({"Calculated_SOH(%)": "max"})

    features = build_cycle_features(df,cycle_col="Cycle_Index",time_col="Test_Time(s)",volt_col="Voltage(V)",curr_col="Current(A)",step_col="Step_Index")
    features["battery_id"] = battery_id
    features["protocol"] = 1  # variable or constant
    features = features.merge(df_SOH, on="Cycle_Index", how="left")
    features = features[features['Charge_duration'] >= 2]
    datasets.append(df)
    all_features.append(features)


# Process variable discharge data
folder_paths2 = ["C:/Users/HYACINTH OSEJI/OneDrive - University of Bath/Documents/PYTHON STUFF/CS2_9",
                 "C:/Users/HYACINTH OSEJI/OneDrive - University of Bath/Documents/PYTHON STUFF/CS2_3"]

sheet_names2 = [["Channel_1-007", "Channel_1-007_1", "Channel_1-007_2"],
                ["Channel_1-008"]]


for i in range(len(folder_paths2)):
    battery_id = battery_name_from_path(folder_paths2[i])
    df2 = process_files(folder_paths2[i], sheet_names2[i])
    df2["battery_id"] = battery_id
    df_SOH = calculate_soh_variable_discharge(df2)
    df_SOH = df_SOH[["Cycle_Index", "Calculated_SOH(%)"]]
    df_SOH = df_SOH.groupby("Cycle_Index", as_index=False).agg({"Calculated_SOH(%)": "max"})
    features = build_cycle_features(df2, cycle_col="Cycle_Index", time_col="Test_Time(s)", volt_col="Voltage(V)", curr_col="Current(A)", step_col="Step_Index")
    features["battery_id"] = battery_id
    features["protocol"] = 0
    features = features.merge(df_SOH, on="Cycle_Index", how="left")

    # Spike removal based on SOH differences (cycle-level)
    soh = features["Calculated_SOH(%)"]
    prev_diff = (soh - soh.shift(1)).abs()
    next_diff = (soh - soh.shift(-1)).abs()
    mask = (prev_diff > 5) & (next_diff > 5)

    bad_cycles = features.loc[mask, "Cycle_Index"].unique()

    # Drop bad cycles from both raw and feature tables
    df2 = df2[~df2["Cycle_Index"].isin(bad_cycles)].reset_index(drop=True)
    features = features[~features["Cycle_Index"].isin(bad_cycles)].reset_index(drop=True)

    datasets.append(df2)
    all_features.append(features)

# Concatenate all features into one per-cycle dataset
combined = pd.concat(all_features, ignore_index=True)

# EOL cutoff
combined = combined[combined["Calculated_SOH(%)"] >= 70].reset_index(drop=True)

# normalized cycle progress within each cylce
combined["cycle_frac"] = combined.groupby("battery_id")["Cycle_Index"].transform(lambda s: (s - s.min()) / max((s.max() - s.min()), 1)).astype(float)
a = combined.columns

# (A) BATTERY HOLDOUT SPLIT FIRST (before feature selection)
test_batteries_holdout = ["CS2_38"]  # put as many batteries as needed

is_test_holdout = combined["battery_id"].isin(test_batteries_holdout).to_numpy()
train_df_holdout = combined.loc[~is_test_holdout].reset_index(drop=True)
test_df_holdout  = combined.loc[ is_test_holdout].reset_index(drop=True)

# Feature selection ONLY on training batteries
best_X_train_df = select_best_feature_df(train_df_holdout,target_col="Calculated_SOH(%)",group_col="battery_id",include_cycle_index=True,keep_target=False,keep_group=False)

selected_cols = best_X_train_df.columns.tolist()
feature_names = selected_cols  # keep your feature_names variable

# y and groups for holdout training
y_train_holdout = train_df_holdout["Calculated_SOH(%)"].astype(float).to_numpy()
groups_train_holdout = train_df_holdout["battery_id"].to_numpy()

# X for holdout training
X_train_holdout = best_X_train_df.to_numpy()

# X/y for holdout test USING THE SAME selected columns
# (safety: if any selected col missing in test for any reason, create it)
missing_in_test = [c for c in selected_cols if c not in test_df_holdout.columns]
for c in missing_in_test:
    test_df_holdout[c] = np.nan

X_test_holdout = test_df_holdout[selected_cols].to_numpy()
y_test_holdout = test_df_holdout["Calculated_SOH(%)"].astype(float).to_numpy()

# STACKED
y_pred_holdout, mse_holdout, mae_holdout = fit_predict_stacked_ensemble(X_train_holdout, y_train_holdout, X_test_holdout, y_test_holdout, groups_train=groups_train_holdout, n_splits=5, random_state=42)
print("HOLDOUT (by battery) -> MSE:", round(mse_holdout, 6), " MAE:", round(mae_holdout, 6))

# SINGLE
y_pred_holdout_single, mse_holdout_single, mae_holdout_single = fit_predict_single_regularized_model(X_train_holdout, y_train_holdout,X_test_holdout,  y_test_holdout,random_state=42)
print("HOLDOUT (single HGB) -> MSE:", round(mse_holdout_single, 6), " MAE:", round(mae_holdout_single, 6))



# Early life prediction graphs 
BATTERIES = ["CS2_33", "CS2_34", "CS2_35", "CS2_36", "CS2_37", "CS2_38"]
TRAIN_FRAC = 0.30
RANDOM_STATE = 42
N_SPLITS = 5
UNC_ALPHA = 0.12
Y_LIM = (70, 102)
ALIGN_X_AXES = True
SPIKE_WINDOW = 9
SPIKE_THRESH = 5.0
SPIKE_PASSES = 2
INTERP_METHOD = "linear"
TARGET_COL = "Calculated_SOH(%)"
GROUP_COL = "battery_id"
CYCLE_COL = "Cycle_Index"
PROTOCOL_COL = "protocol"


FORCED_FEATURES = [
    "Cycle_Index",
    "Internal_Resistance_max",
    "Internal_Resistance_median",
    "Internal_Resistance_p90",
    "dVdt_mean",
    "avg_abs_dvdt",
    "mean_abs_dVdt_over_I",
    "median_abs_dVdt_over_I",
    "Charge_cumulativeCapacity",
    "Charge_cumulativeEnergy",
    "Charge_duration",
    "Charge_startVoltage",
    "Charge_endVoltage",
    "DeltaV_charge",
    "Charge_Voltage_max",
    "Charge_Voltage_min",
    "Charge_Voltage_mean",
    "Charge_Voltage_std",
    "Charge_Voltage_skewness",
    "Charge_Voltage_kurtosis",
    "Charge_Current_max",
    "Charge_Current_min",
    "Charge_Current_mean",
    "Charge_Current_std",
    "Charge_Current_skewness",
    "Charge_Current_kurtosis",
    "Discharge_cumulativeCapacity",
    "Discharge_cumulativeEnergy",
    "Discharge_duration",
    "Discharge_startVoltage",
    "Discharge_endVoltage",
    "DeltaV_discharge",
    "Discharge_Voltage_max",
    "Discharge_Voltage_min",
    "Discharge_Voltage_mean",
    "Discharge_Voltage_std",
    "Discharge_Voltage_skewness",
    "Discharge_Voltage_kurtosis",
    "Discharge_Current_max",
    "Discharge_Current_min",
    "Discharge_Current_mean",
    "Discharge_Current_std",
    "Discharge_Current_skewness",
    "Discharge_Current_kurtosis",
    "Coulombic_Efficiency",
    "Energy_Efficiency",
    "protocol",
]

AGE_FEATURES = ["cycle_frac"]
PROXY_FEATURES = ["deg_rate_proxy"]


# Helper funcs
def add_cycle_frac_if_missing(df):
    d = df.copy()
    if "cycle_frac" not in d.columns:
        d["cycle_frac"] = d.groupby(GROUP_COL)[CYCLE_COL].transform(
            lambda s: (s - s.min()) / max((s.max() - s.min()), 1)
        )
    return d


def add_degradation_proxy(df):
    d = df.copy()
    dis_cap = pd.to_numeric(d.get("Discharge_cumulativeCapacity"), errors="coerce")
    dis_dur = pd.to_numeric(d.get("Discharge_duration"), errors="coerce")
    d["deg_rate_proxy"] = dis_cap / np.maximum(dis_dur, 1e-9)
    if d["deg_rate_proxy"].notna().sum() > 10:
        lo, hi = d["deg_rate_proxy"].quantile([0.01, 0.99])
        d["deg_rate_proxy"] = d["deg_rate_proxy"].clip(lo, hi)
    return d


def clean_soh_series(y):
    y = pd.Series(y).astype(float)
    for _ in range(SPIKE_PASSES):
        med = y.rolling(SPIKE_WINDOW, center=True, min_periods=3).median()
        y = y.mask((y - med).abs() > SPIKE_THRESH)
    return y.interpolate(method=INTERP_METHOD, limit_direction="both").to_numpy()


def get_early_mask(df):
    # rank/count based per battery
    rank = df.groupby(GROUP_COL)[CYCLE_COL].rank(method="first")
    n = df.groupby(GROUP_COL)[CYCLE_COL].transform("count")
    cutoff = np.maximum(1, np.ceil(TRAIN_FRAC * n)).astype(int)
    return rank <= cutoff


def stacked_predict_with_fold_uncertainty(X_train, y_train, X_test, groups_train):
    """
    Produces:
      - mean prediction across CV folds
      - std dev across CV folds (uncertainty band proxy)

    NOTE: This is a fold-wise ensemble; it’s not perfect OOF stacking,
    but it avoids test leakage and gives a reasonable uncertainty estimate.
    """
    pre = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    X_train = pre.fit_transform(X_train)
    X_test = pre.transform(X_test)

    base_models = [
        XGBRegressor(
            n_estimators=600, learning_rate=0.03, max_depth=6,
            subsample=0.8, colsample_bytree=0.8,
            reg_lambda=1.0, min_child_weight=2.0,
            random_state=101, n_jobs=-1
        ),
        LGBMRegressor(
            n_estimators=1200, learning_rate=0.02, num_leaves=31,
            subsample=0.8, colsample_bytree=0.8,
            reg_lambda=1.0,
            random_state=101, n_jobs=-1
        ),
        RandomForestRegressor(
            n_estimators=600, max_depth=14, min_samples_leaf=2,
            random_state=101, n_jobs=-1
        )
    ]

    cv = GroupKFold(n_splits=N_SPLITS)
    preds = np.zeros((N_SPLITS, X_test.shape[0]))

    for i, (tr, _) in enumerate(cv.split(X_train, y_train, groups_train)):
        meta_tr, meta_te = [], []
        for m in base_models:
            m.fit(X_train[tr], y_train[tr])
            meta_tr.append(m.predict(X_train[tr]).reshape(-1, 1))
            meta_te.append(m.predict(X_test).reshape(-1, 1))

        meta = Ridge(alpha=5.0, random_state=RANDOM_STATE)
        meta.fit(np.hstack(meta_tr), y_train[tr])
        preds[i] = meta.predict(np.hstack(meta_te))

    return preds.mean(axis=0), preds.std(axis=0, ddof=1)


def compute_battery_panel(cc_df, battery):
    """
    Leave-one-battery-out early-life forecasting:
      - Train: early cycles of all OTHER batteries
      - Test: later cycles of the target battery
    """
    is_target = (cc_df[GROUP_COL] == battery)
    is_early_all = get_early_mask(cc_df)

    train_df = cc_df[is_early_all & (~is_target)].copy()
    test_df = cc_df[is_target].copy()

    if len(train_df) < 20 or len(test_df) < 10:
        return None

    # rank/count cutoff for target battery (not quantile)
    test_rank = test_df[CYCLE_COL].rank(method="first")
    test_n = len(test_df)
    test_cut_n = int(max(1, np.ceil(TRAIN_FRAC * test_n)))

    # cycle value at cutoff boundary (for plotting the vertical line)
    cutoff_cycle_value = test_df.loc[test_rank == test_cut_n, CYCLE_COL].iloc[0] if test_cut_n <= test_n else test_df[CYCLE_COL].max()

    test_post = test_df.loc[test_rank > test_cut_n].copy()
    if len(test_post) < 5:
        return None

    # features present
    features = [c for c in (FORCED_FEATURES + AGE_FEATURES + PROXY_FEATURES) if c in cc_df.columns]
    X_train = train_df[features]
    y_train = train_df["SoH_clean"] 
    X_test = test_post[features]
    y_test = test_post["SoH_clean"]  

    y_pred, y_std = stacked_predict_with_fold_uncertainty(X_train.values, y_train.values, X_test.values, train_df[GROUP_COL].values)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    plot_df = test_df[[CYCLE_COL, "SoH_clean"]].copy()
    plot_df["pred"] = np.nan
    plot_df["std"] = np.nan
    plot_df.loc[test_post.index, "pred"] = y_pred
    plot_df.loc[test_post.index, "std"] = y_std

    return plot_df, cutoff_cycle_value, mae, rmse


# DATA PREP
cc = combined.copy()
cc = cc.loc[cc[PROTOCOL_COL] == 1].copy()
cc = add_cycle_frac_if_missing(cc)
cc = add_degradation_proxy(cc)

cc["SoH_raw"] = pd.to_numeric(cc[TARGET_COL], errors="coerce")
cc["SoH_clean"] = cc.groupby(GROUP_COL)["SoH_raw"].transform(clean_soh_series)

X_MAX = cc[CYCLE_COL].max() if ALIGN_X_AXES else None


# Plots
for b in BATTERIES:
    out = compute_battery_panel(cc, b)
    if out is None:
        continue

    plot_df, cutoff, mae, rmse = out

    plt.figure(figsize=(6.5, 4.5))
    plt.plot(plot_df[CYCLE_COL], plot_df["SoH_clean"], label="True SoH (clean)", linewidth=2)
    plt.plot(plot_df[CYCLE_COL], plot_df["pred"], "--", label="Predicted SoH", linewidth=2)

    m = plot_df["pred"].notna()
    plt.fill_between(
        plot_df.loc[m, CYCLE_COL],
        plot_df.loc[m, "pred"] - plot_df.loc[m, "std"],
        plot_df.loc[m, "pred"] + plot_df.loc[m, "std"],
        alpha=UNC_ALPHA,
        label="±1 SD (CV folds)"
    )

    plt.axvline(cutoff, linestyle=":", linewidth=1.5, label="Training cutoff (30%)")
    plt.axvspan(plot_df[CYCLE_COL].min(), cutoff, alpha=0.06)

    plt.title(f"{b} | MAE={mae:.2f}, RMSE={rmse:.2f}")
    plt.xlabel("Cycle")
    plt.ylabel("SoH (%)")
    plt.ylim(*Y_LIM)
    if X_MAX is not None:
        plt.xlim(0, X_MAX)

    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.show()
