"""Validated raw-data reconstruction and feature-engineering core.

This module consolidates the scientifically retained preprocessing logic used by
the final constant- and variable-protocol analyses. Function bodies are carried
forward from the validated Step 3/Step 5 workflows; only notebook/Google Drive
I/O has been removed.

The module intentionally exposes the broader engineered feature space. The
strict, partial-observation, and history-assisted admissibility decisions are
implemented separately in :mod:`battery_soh_repro.feature_policy`.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import re
import shutil
import time
import warnings
import zipfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.signal import find_peaks, peak_widths, peak_prominences
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error

_HAS_SCIPY = True


RANDOM_STATE = 42

SOH_LOWER_LIMIT = 70.0

SOH_UPPER_LIMIT = 100.0

EARLY_BASELINE_FINITE_OBSERVATIONS = 5

CALCE_CONSTANT_BATTERIES = [
    ("CS2_33", ["Channel_1-006"]),
    ("CS2_34", ["Channel_1-007"]),
    ("CS2_35", ["Channel_1-008"]),
    ("CS2_36", ["Channel_1-009"]),
    ("CS2_37", ["Channel_1-010"]),
    ("CS2_38", ["Channel_1-011"]),
]

CALCE_VARIABLE_BATTERIES = [
    ("CS2_9", ["Channel_1-007", "Channel_1-007_1", "Channel_1-007_2"]),
    ("CS2_3", ["Channel_1-008"]),
]

NASA_CONSTANT_BATTERIES = ["B0005", "B0006", "B0007", "B0018"]

NASA_VARIABLE_BATTERIES = ["RW3", "RW4", "RW5", "RW6"]

ALL_BATTERY_IDS = [
    *(b for b, _ in CALCE_CONSTANT_BATTERIES),
    *(b for b, _ in CALCE_VARIABLE_BATTERIES),
    *NASA_CONSTANT_BATTERIES,
    *NASA_VARIABLE_BATTERIES,
]

TARGET_COL = "Calculated_SOH(%)"

GROUP_COL = "battery_id"

CYCLE_COL = "Cycle_Index"

SOURCE_COL = "source_dataset"

PROTOCOL_COL = "protocol"

CONSTANT_PROTOCOL = 1

VARIABLE_PROTOCOL = 0

VARIABLE_REFERENCE_CURRENT_A = 0.22

VARIABLE_REFERENCE_CURRENT_TOLERANCE_A = 0.04

VARIABLE_DISCHARGE_THRESHOLD_A = 0.02

VARIABLE_MIN_DISCHARGE_SAMPLES = 5

DISCHARGE_PROGRESS_LANDMARKS = tuple(i / 10 for i in range(1, 10))

DISCHARGE_VOLTAGE_LANDMARKS = (4.0, 3.8, 3.6, 3.4, 3.2)

INVARIANT_EXACT_BASE_FEATURES = {
    "Internal_Resistance_median",
    "dVdt_mean", "avg_abs_dvdt",
    "mean_abs_dVdt_over_I", "median_abs_dVdt_over_I",
    "Discharge_cumulativeCapacity", "Discharge_cumulativeEnergy",
    "Discharge_duration", "Discharge_startVoltage", "Discharge_endVoltage",
    "DeltaV_discharge", "Discharge_Voltage_mean", "Discharge_Voltage_std",
    "Discharge_Voltage_q10", "Discharge_Voltage_q25", "Discharge_Voltage_q50",
    "Discharge_Voltage_q75", "Discharge_Voltage_q90",
    "Discharge_Voltage_IQR", "Discharge_Voltage_MAD",
    "Discharge_Current_mean", "Discharge_Current_std",
    "Discharge_Current_IQR", "Discharge_Current_MAD",
    "Discharge_AverageVoltage",
    "Discharge_Voltage_EarlyMean", "Discharge_Voltage_MiddleMean",
    "Discharge_Voltage_LateMean", "Discharge_LateMinusEarlyVoltage",
    "Discharge_VoltageDropPerHour", "Discharge_VoltageDropPerAh",
    "Discharge_EnergyPerHour",
    "Discharge_Step7_IC_peakLocation", "Discharge_Step7_IC_peakProminence",
    "Discharge_Step7_IC_peakWidth", "Discharge_Step7_IC_peakCount",
    "Discharge_Step7_CC_slope",
}

LEGACY_HISTORY_BASE_FEATURES = [
    "Internal_Resistance_median",
    "dVdt_mean", "avg_abs_dvdt",
    "mean_abs_dVdt_over_I", "median_abs_dVdt_over_I",
    "Discharge_cumulativeCapacity", "Discharge_cumulativeEnergy",
    "Discharge_duration", "Discharge_startVoltage", "Discharge_endVoltage",
    "DeltaV_discharge", "Discharge_Voltage_mean", "Discharge_Voltage_std",
    "Discharge_Voltage_IQR", "Discharge_Voltage_MAD",
    "Discharge_AverageVoltage", "Discharge_VoltageAt_Q50",
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
    TARGET_COL, GROUP_COL, SOURCE_COL, PROTOCOL_COL,
}

R5E_REF_EPOCH_COL = "R5E_Reference_Epoch"

R5E_INTERVAL_ID_COL = "R5E_Interval_ID"

R5E_INTERVAL_POSITION_COL = "R5E_Interval_Cycle_Position"

R5E_INTERVAL_TOTAL_COL = "R5E_Interval_Total_Cycles"

R5E_NORM_PREFIX = "INV__R5E_Norm__"

FROZEN_STRICT_CONSTANT_FEATURES = ['INV__DeltaV_discharge__delta', 'INV__DeltaV_discharge__pct', 'INV__DeltaV_discharge__pct_ewm5', 'INV__DeltaV_discharge__pct_lag1_delta', 'INV__DeltaV_discharge__pct_slope5', 'INV__DeltaV_discharge__ratio', 'INV__Discharge_Current_IQR__delta', 'INV__Discharge_Current_IQR__pct', 'INV__Discharge_Current_IQR__pct_ewm5', 'INV__Discharge_Current_IQR__pct_lag1_delta', 'INV__Discharge_Current_IQR__pct_slope5', 'INV__Discharge_Current_IQR__ratio', 'INV__Discharge_Current_MAD__delta', 'INV__Discharge_Current_MAD__pct', 'INV__Discharge_Current_MAD__pct_ewm5', 'INV__Discharge_Current_MAD__pct_lag1_delta', 'INV__Discharge_Current_MAD__pct_slope5', 'INV__Discharge_Current_MAD__ratio', 'INV__Discharge_Current_mean__delta', 'INV__Discharge_Current_mean__pct', 'INV__Discharge_Current_mean__pct_ewm5', 'INV__Discharge_Current_mean__pct_lag1_delta', 'INV__Discharge_Current_mean__pct_slope5', 'INV__Discharge_Current_mean__ratio', 'INV__Discharge_Current_std__delta', 'INV__Discharge_Current_std__pct', 'INV__Discharge_Current_std__pct_ewm5', 'INV__Discharge_Current_std__pct_lag1_delta', 'INV__Discharge_Current_std__pct_slope5', 'INV__Discharge_Current_std__ratio', 'INV__Discharge_Step7_CC_slope__delta', 'INV__Discharge_Step7_CC_slope__pct', 'INV__Discharge_Step7_CC_slope__pct_ewm5', 'INV__Discharge_Step7_CC_slope__pct_lag1_delta', 'INV__Discharge_Step7_CC_slope__pct_slope5', 'INV__Discharge_Step7_CC_slope__ratio', 'INV__Discharge_Voltage_IQR__delta', 'INV__Discharge_Voltage_IQR__pct', 'INV__Discharge_Voltage_IQR__pct_ewm5', 'INV__Discharge_Voltage_IQR__pct_lag1_delta', 'INV__Discharge_Voltage_IQR__pct_slope5', 'INV__Discharge_Voltage_IQR__ratio', 'INV__Discharge_Voltage_MAD__delta', 'INV__Discharge_Voltage_MAD__pct', 'INV__Discharge_Voltage_MAD__pct_ewm5', 'INV__Discharge_Voltage_MAD__pct_lag1_delta', 'INV__Discharge_Voltage_MAD__pct_slope5', 'INV__Discharge_Voltage_MAD__ratio', 'INV__Discharge_Voltage_mean__delta', 'INV__Discharge_Voltage_mean__pct', 'INV__Discharge_Voltage_mean__pct_ewm5', 'INV__Discharge_Voltage_mean__pct_lag1_delta', 'INV__Discharge_Voltage_mean__pct_slope5', 'INV__Discharge_Voltage_mean__ratio', 'INV__Discharge_Voltage_q10__delta', 'INV__Discharge_Voltage_q10__pct', 'INV__Discharge_Voltage_q10__pct_ewm5', 'INV__Discharge_Voltage_q10__pct_lag1_delta', 'INV__Discharge_Voltage_q10__pct_slope5', 'INV__Discharge_Voltage_q10__ratio', 'INV__Discharge_Voltage_q25__delta', 'INV__Discharge_Voltage_q25__pct', 'INV__Discharge_Voltage_q25__pct_ewm5', 'INV__Discharge_Voltage_q25__pct_lag1_delta', 'INV__Discharge_Voltage_q25__pct_slope5', 'INV__Discharge_Voltage_q25__ratio', 'INV__Discharge_Voltage_q50__delta', 'INV__Discharge_Voltage_q50__pct', 'INV__Discharge_Voltage_q50__pct_ewm5', 'INV__Discharge_Voltage_q50__pct_lag1_delta', 'INV__Discharge_Voltage_q50__pct_slope5', 'INV__Discharge_Voltage_q50__ratio', 'INV__Discharge_Voltage_q75__delta', 'INV__Discharge_Voltage_q75__pct', 'INV__Discharge_Voltage_q75__pct_ewm5', 'INV__Discharge_Voltage_q75__pct_lag1_delta', 'INV__Discharge_Voltage_q75__pct_slope5', 'INV__Discharge_Voltage_q75__ratio', 'INV__Discharge_Voltage_q90__delta', 'INV__Discharge_Voltage_q90__pct', 'INV__Discharge_Voltage_q90__pct_ewm5', 'INV__Discharge_Voltage_q90__pct_lag1_delta', 'INV__Discharge_Voltage_q90__pct_slope5', 'INV__Discharge_Voltage_q90__ratio', 'INV__Discharge_Voltage_std__delta', 'INV__Discharge_Voltage_std__pct', 'INV__Discharge_Voltage_std__pct_ewm5', 'INV__Discharge_Voltage_std__pct_lag1_delta', 'INV__Discharge_Voltage_std__pct_slope5', 'INV__Discharge_Voltage_std__ratio', 'INV__Discharge_endVoltage__delta', 'INV__Discharge_endVoltage__pct', 'INV__Discharge_endVoltage__pct_ewm5', 'INV__Discharge_endVoltage__pct_lag1_delta', 'INV__Discharge_endVoltage__pct_slope5', 'INV__Discharge_endVoltage__ratio', 'INV__Discharge_startVoltage__delta', 'INV__Discharge_startVoltage__pct', 'INV__Discharge_startVoltage__pct_ewm5', 'INV__Discharge_startVoltage__pct_lag1_delta', 'INV__Discharge_startVoltage__pct_slope5', 'INV__Discharge_startVoltage__ratio', 'INV__Internal_Resistance_median__delta', 'INV__Internal_Resistance_median__pct', 'INV__Internal_Resistance_median__pct_ewm5', 'INV__Internal_Resistance_median__pct_lag1_delta', 'INV__Internal_Resistance_median__pct_slope5', 'INV__Internal_Resistance_median__ratio', 'INV__avg_abs_dvdt__delta', 'INV__avg_abs_dvdt__pct', 'INV__avg_abs_dvdt__pct_ewm5', 'INV__avg_abs_dvdt__pct_lag1_delta', 'INV__avg_abs_dvdt__pct_slope5', 'INV__avg_abs_dvdt__ratio', 'INV__dVdt_mean__delta', 'INV__dVdt_mean__pct', 'INV__dVdt_mean__pct_ewm5', 'INV__dVdt_mean__pct_lag1_delta', 'INV__dVdt_mean__pct_slope5', 'INV__dVdt_mean__ratio', 'INV__mean_abs_dVdt_over_I__delta', 'INV__mean_abs_dVdt_over_I__pct', 'INV__mean_abs_dVdt_over_I__pct_ewm5', 'INV__mean_abs_dVdt_over_I__pct_lag1_delta', 'INV__mean_abs_dVdt_over_I__pct_slope5', 'INV__mean_abs_dVdt_over_I__ratio', 'INV__median_abs_dVdt_over_I__delta', 'INV__median_abs_dVdt_over_I__pct', 'INV__median_abs_dVdt_over_I__pct_ewm5', 'INV__median_abs_dVdt_over_I__pct_lag1_delta', 'INV__median_abs_dVdt_over_I__pct_slope5', 'INV__median_abs_dVdt_over_I__ratio', 'LEGACY__Charge_Current_kurtosis', 'LEGACY__Charge_Current_max', 'LEGACY__Charge_Current_mean', 'LEGACY__Charge_Current_min', 'LEGACY__Charge_Current_skewness', 'LEGACY__Charge_Current_std', 'LEGACY__Charge_Step2_CC_currentMedian', 'LEGACY__Charge_Step2_CC_currentStd', 'LEGACY__Charge_Step2_CC_kurtosis', 'LEGACY__Charge_Step2_CC_skewness', 'LEGACY__Charge_Step2_CC_voltageStd', 'LEGACY__Charge_Step4_CV_currentDrop', 'LEGACY__Charge_Step4_CV_endCurrent', 'LEGACY__Charge_Step4_CV_kurtosis', 'LEGACY__Charge_Step4_CV_skewness', 'LEGACY__Charge_Step4_CV_slope', 'LEGACY__Charge_Step4_CV_startCurrent', 'LEGACY__Charge_Step4_CV_tau10', 'LEGACY__Charge_Step4_CV_voltageMedian', 'LEGACY__Charge_Voltage_kurtosis', 'LEGACY__Charge_Voltage_max', 'LEGACY__Charge_Voltage_mean', 'LEGACY__Charge_Voltage_min', 'LEGACY__Charge_Voltage_skewness', 'LEGACY__Charge_Voltage_std', 'LEGACY__Charge_endVoltage', 'LEGACY__Charge_startVoltage', 'LEGACY__Cycle_Age', 'LEGACY__Cycle_Index', 'LEGACY__DeltaV_charge', 'LEGACY__DeltaV_discharge', 'LEGACY__Discharge_Current_IQR', 'LEGACY__Discharge_Current_MAD', 'LEGACY__Discharge_Current_kurtosis', 'LEGACY__Discharge_Current_max', 'LEGACY__Discharge_Current_mean', 'LEGACY__Discharge_Current_min', 'LEGACY__Discharge_Current_q10', 'LEGACY__Discharge_Current_q25', 'LEGACY__Discharge_Current_q50', 'LEGACY__Discharge_Current_q75', 'LEGACY__Discharge_Current_q90', 'LEGACY__Discharge_Current_skewness', 'LEGACY__Discharge_Current_std', 'LEGACY__Discharge_Step7_CC_currentMedian', 'LEGACY__Discharge_Step7_CC_currentStd', 'LEGACY__Discharge_Step7_CC_kurtosis', 'LEGACY__Discharge_Step7_CC_skewness', 'LEGACY__Discharge_Step7_CC_slope', 'LEGACY__Discharge_Step7_CC_voltageStd', 'LEGACY__Discharge_Voltage_IQR', 'LEGACY__Discharge_Voltage_MAD', 'LEGACY__Discharge_Voltage_kurtosis', 'LEGACY__Discharge_Voltage_max', 'LEGACY__Discharge_Voltage_mean', 'LEGACY__Discharge_Voltage_min', 'LEGACY__Discharge_Voltage_q10', 'LEGACY__Discharge_Voltage_q25', 'LEGACY__Discharge_Voltage_q50', 'LEGACY__Discharge_Voltage_q75', 'LEGACY__Discharge_Voltage_q90', 'LEGACY__Discharge_Voltage_skewness', 'LEGACY__Discharge_Voltage_std', 'LEGACY__Discharge_endVoltage', 'LEGACY__Discharge_startVoltage', 'LEGACY__Internal_Resistance_max', 'LEGACY__Internal_Resistance_median', 'LEGACY__Internal_Resistance_p90', 'LEGACY__LEGACYH__DeltaV_discharge__baseline_delta', 'LEGACY__LEGACYH__DeltaV_discharge__baseline_pct', 'LEGACY__LEGACYH__DeltaV_discharge__baseline_ratio', 'LEGACY__LEGACYH__DeltaV_discharge__ewm_5', 'LEGACY__LEGACYH__DeltaV_discharge__lag1_delta', 'LEGACY__LEGACYH__DeltaV_discharge__per_cycle_delta', 'LEGACY__LEGACYH__DeltaV_discharge__rolling_mean_3', 'LEGACY__LEGACYH__DeltaV_discharge__rolling_mean_5', 'LEGACY__LEGACYH__DeltaV_discharge__rolling_slope_5', 'LEGACY__LEGACYH__DeltaV_discharge__rolling_std_5', 'LEGACY__LEGACYH__Discharge_Step7_CC_slope__baseline_delta', 'LEGACY__LEGACYH__Discharge_Step7_CC_slope__baseline_pct', 'LEGACY__LEGACYH__Discharge_Step7_CC_slope__baseline_ratio', 'LEGACY__LEGACYH__Discharge_Step7_CC_slope__ewm_5', 'LEGACY__LEGACYH__Discharge_Step7_CC_slope__lag1_delta', 'LEGACY__LEGACYH__Discharge_Step7_CC_slope__per_cycle_delta', 'LEGACY__LEGACYH__Discharge_Step7_CC_slope__rolling_mean_3', 'LEGACY__LEGACYH__Discharge_Step7_CC_slope__rolling_mean_5', 'LEGACY__LEGACYH__Discharge_Step7_CC_slope__rolling_slope_5', 'LEGACY__LEGACYH__Discharge_Step7_CC_slope__rolling_std_5', 'LEGACY__LEGACYH__Discharge_Voltage_IQR__baseline_delta', 'LEGACY__LEGACYH__Discharge_Voltage_IQR__baseline_pct', 'LEGACY__LEGACYH__Discharge_Voltage_IQR__baseline_ratio', 'LEGACY__LEGACYH__Discharge_Voltage_IQR__ewm_5', 'LEGACY__LEGACYH__Discharge_Voltage_IQR__lag1_delta', 'LEGACY__LEGACYH__Discharge_Voltage_IQR__per_cycle_delta', 'LEGACY__LEGACYH__Discharge_Voltage_IQR__rolling_mean_3', 'LEGACY__LEGACYH__Discharge_Voltage_IQR__rolling_mean_5', 'LEGACY__LEGACYH__Discharge_Voltage_IQR__rolling_slope_5', 'LEGACY__LEGACYH__Discharge_Voltage_IQR__rolling_std_5', 'LEGACY__LEGACYH__Discharge_Voltage_MAD__baseline_delta', 'LEGACY__LEGACYH__Discharge_Voltage_MAD__baseline_pct', 'LEGACY__LEGACYH__Discharge_Voltage_MAD__baseline_ratio', 'LEGACY__LEGACYH__Discharge_Voltage_MAD__ewm_5', 'LEGACY__LEGACYH__Discharge_Voltage_MAD__lag1_delta', 'LEGACY__LEGACYH__Discharge_Voltage_MAD__per_cycle_delta', 'LEGACY__LEGACYH__Discharge_Voltage_MAD__rolling_mean_3', 'LEGACY__LEGACYH__Discharge_Voltage_MAD__rolling_mean_5', 'LEGACY__LEGACYH__Discharge_Voltage_MAD__rolling_slope_5', 'LEGACY__LEGACYH__Discharge_Voltage_MAD__rolling_std_5', 'LEGACY__LEGACYH__Discharge_Voltage_mean__baseline_delta', 'LEGACY__LEGACYH__Discharge_Voltage_mean__baseline_pct', 'LEGACY__LEGACYH__Discharge_Voltage_mean__baseline_ratio', 'LEGACY__LEGACYH__Discharge_Voltage_mean__ewm_5', 'LEGACY__LEGACYH__Discharge_Voltage_mean__lag1_delta', 'LEGACY__LEGACYH__Discharge_Voltage_mean__per_cycle_delta', 'LEGACY__LEGACYH__Discharge_Voltage_mean__rolling_mean_3', 'LEGACY__LEGACYH__Discharge_Voltage_mean__rolling_mean_5', 'LEGACY__LEGACYH__Discharge_Voltage_mean__rolling_slope_5', 'LEGACY__LEGACYH__Discharge_Voltage_mean__rolling_std_5', 'LEGACY__LEGACYH__Discharge_Voltage_std__baseline_delta', 'LEGACY__LEGACYH__Discharge_Voltage_std__baseline_pct', 'LEGACY__LEGACYH__Discharge_Voltage_std__baseline_ratio', 'LEGACY__LEGACYH__Discharge_Voltage_std__ewm_5', 'LEGACY__LEGACYH__Discharge_Voltage_std__lag1_delta', 'LEGACY__LEGACYH__Discharge_Voltage_std__per_cycle_delta', 'LEGACY__LEGACYH__Discharge_Voltage_std__rolling_mean_3', 'LEGACY__LEGACYH__Discharge_Voltage_std__rolling_mean_5', 'LEGACY__LEGACYH__Discharge_Voltage_std__rolling_slope_5', 'LEGACY__LEGACYH__Discharge_Voltage_std__rolling_std_5', 'LEGACY__LEGACYH__Discharge_endVoltage__baseline_delta', 'LEGACY__LEGACYH__Discharge_endVoltage__baseline_pct', 'LEGACY__LEGACYH__Discharge_endVoltage__baseline_ratio', 'LEGACY__LEGACYH__Discharge_endVoltage__ewm_5', 'LEGACY__LEGACYH__Discharge_endVoltage__lag1_delta', 'LEGACY__LEGACYH__Discharge_endVoltage__per_cycle_delta', 'LEGACY__LEGACYH__Discharge_endVoltage__rolling_mean_3', 'LEGACY__LEGACYH__Discharge_endVoltage__rolling_mean_5', 'LEGACY__LEGACYH__Discharge_endVoltage__rolling_slope_5', 'LEGACY__LEGACYH__Discharge_endVoltage__rolling_std_5', 'LEGACY__LEGACYH__Discharge_startVoltage__baseline_delta', 'LEGACY__LEGACYH__Discharge_startVoltage__baseline_pct', 'LEGACY__LEGACYH__Discharge_startVoltage__baseline_ratio', 'LEGACY__LEGACYH__Discharge_startVoltage__ewm_5', 'LEGACY__LEGACYH__Discharge_startVoltage__lag1_delta', 'LEGACY__LEGACYH__Discharge_startVoltage__per_cycle_delta', 'LEGACY__LEGACYH__Discharge_startVoltage__rolling_mean_3', 'LEGACY__LEGACYH__Discharge_startVoltage__rolling_mean_5', 'LEGACY__LEGACYH__Discharge_startVoltage__rolling_slope_5', 'LEGACY__LEGACYH__Discharge_startVoltage__rolling_std_5', 'LEGACY__LEGACYH__Internal_Resistance_median__baseline_delta', 'LEGACY__LEGACYH__Internal_Resistance_median__baseline_pct', 'LEGACY__LEGACYH__Internal_Resistance_median__baseline_ratio', 'LEGACY__LEGACYH__Internal_Resistance_median__ewm_5', 'LEGACY__LEGACYH__Internal_Resistance_median__lag1_delta', 'LEGACY__LEGACYH__Internal_Resistance_median__per_cycle_delta', 'LEGACY__LEGACYH__Internal_Resistance_median__rolling_mean_3', 'LEGACY__LEGACYH__Internal_Resistance_median__rolling_mean_5', 'LEGACY__LEGACYH__Internal_Resistance_median__rolling_slope_5', 'LEGACY__LEGACYH__Internal_Resistance_median__rolling_std_5', 'LEGACY__LEGACYH__avg_abs_dvdt__baseline_delta', 'LEGACY__LEGACYH__avg_abs_dvdt__baseline_pct', 'LEGACY__LEGACYH__avg_abs_dvdt__baseline_ratio', 'LEGACY__LEGACYH__avg_abs_dvdt__ewm_5', 'LEGACY__LEGACYH__avg_abs_dvdt__lag1_delta', 'LEGACY__LEGACYH__avg_abs_dvdt__per_cycle_delta', 'LEGACY__LEGACYH__avg_abs_dvdt__rolling_mean_3', 'LEGACY__LEGACYH__avg_abs_dvdt__rolling_mean_5', 'LEGACY__LEGACYH__avg_abs_dvdt__rolling_slope_5', 'LEGACY__LEGACYH__avg_abs_dvdt__rolling_std_5', 'LEGACY__LEGACYH__dVdt_mean__baseline_delta', 'LEGACY__LEGACYH__dVdt_mean__baseline_pct', 'LEGACY__LEGACYH__dVdt_mean__baseline_ratio', 'LEGACY__LEGACYH__dVdt_mean__ewm_5', 'LEGACY__LEGACYH__dVdt_mean__lag1_delta', 'LEGACY__LEGACYH__dVdt_mean__per_cycle_delta', 'LEGACY__LEGACYH__dVdt_mean__rolling_mean_3', 'LEGACY__LEGACYH__dVdt_mean__rolling_mean_5', 'LEGACY__LEGACYH__dVdt_mean__rolling_slope_5', 'LEGACY__LEGACYH__dVdt_mean__rolling_std_5', 'LEGACY__LEGACYH__mean_abs_dVdt_over_I__baseline_delta', 'LEGACY__LEGACYH__mean_abs_dVdt_over_I__baseline_pct', 'LEGACY__LEGACYH__mean_abs_dVdt_over_I__baseline_ratio', 'LEGACY__LEGACYH__mean_abs_dVdt_over_I__ewm_5', 'LEGACY__LEGACYH__mean_abs_dVdt_over_I__lag1_delta', 'LEGACY__LEGACYH__mean_abs_dVdt_over_I__per_cycle_delta', 'LEGACY__LEGACYH__mean_abs_dVdt_over_I__rolling_mean_3', 'LEGACY__LEGACYH__mean_abs_dVdt_over_I__rolling_mean_5', 'LEGACY__LEGACYH__mean_abs_dVdt_over_I__rolling_slope_5', 'LEGACY__LEGACYH__mean_abs_dVdt_over_I__rolling_std_5', 'LEGACY__LEGACYH__median_abs_dVdt_over_I__baseline_delta', 'LEGACY__LEGACYH__median_abs_dVdt_over_I__baseline_pct', 'LEGACY__LEGACYH__median_abs_dVdt_over_I__baseline_ratio', 'LEGACY__LEGACYH__median_abs_dVdt_over_I__ewm_5', 'LEGACY__LEGACYH__median_abs_dVdt_over_I__lag1_delta', 'LEGACY__LEGACYH__median_abs_dVdt_over_I__per_cycle_delta', 'LEGACY__LEGACYH__median_abs_dVdt_over_I__rolling_mean_3', 'LEGACY__LEGACYH__median_abs_dVdt_over_I__rolling_mean_5', 'LEGACY__LEGACYH__median_abs_dVdt_over_I__rolling_slope_5', 'LEGACY__LEGACYH__median_abs_dVdt_over_I__rolling_std_5', 'LEGACY__Log_Cycle_Age', 'LEGACY__avg_abs_dvdt', 'LEGACY__dVdt_mean', 'LEGACY__mean_abs_dVdt_over_I', 'LEGACY__median_abs_dVdt_over_I']

STRICT_VARIABLE_FEATURES = ['INV__DeltaV_discharge__delta', 'INV__DeltaV_discharge__pct', 'INV__DeltaV_discharge__pct_ewm5', 'INV__DeltaV_discharge__pct_lag1_delta', 'INV__DeltaV_discharge__pct_slope5', 'INV__DeltaV_discharge__ratio', 'INV__Discharge_Current_IQR__delta', 'INV__Discharge_Current_IQR__pct', 'INV__Discharge_Current_IQR__pct_ewm5', 'INV__Discharge_Current_IQR__pct_lag1_delta', 'INV__Discharge_Current_IQR__pct_slope5', 'INV__Discharge_Current_IQR__ratio', 'INV__Discharge_Current_MAD__delta', 'INV__Discharge_Current_MAD__pct', 'INV__Discharge_Current_MAD__pct_ewm5', 'INV__Discharge_Current_MAD__pct_lag1_delta', 'INV__Discharge_Current_MAD__pct_slope5', 'INV__Discharge_Current_MAD__ratio', 'INV__Discharge_Current_mean__delta', 'INV__Discharge_Current_mean__pct', 'INV__Discharge_Current_mean__pct_ewm5', 'INV__Discharge_Current_mean__pct_lag1_delta', 'INV__Discharge_Current_mean__pct_slope5', 'INV__Discharge_Current_mean__ratio', 'INV__Discharge_Current_std__delta', 'INV__Discharge_Current_std__pct', 'INV__Discharge_Current_std__pct_ewm5', 'INV__Discharge_Current_std__pct_lag1_delta', 'INV__Discharge_Current_std__pct_slope5', 'INV__Discharge_Current_std__ratio', 'INV__Discharge_Step7_CC_slope__delta', 'INV__Discharge_Step7_CC_slope__pct', 'INV__Discharge_Step7_CC_slope__pct_ewm5', 'INV__Discharge_Step7_CC_slope__pct_lag1_delta', 'INV__Discharge_Step7_CC_slope__pct_slope5', 'INV__Discharge_Step7_CC_slope__ratio', 'INV__Discharge_Voltage_IQR__delta', 'INV__Discharge_Voltage_IQR__pct', 'INV__Discharge_Voltage_IQR__pct_ewm5', 'INV__Discharge_Voltage_IQR__pct_lag1_delta', 'INV__Discharge_Voltage_IQR__pct_slope5', 'INV__Discharge_Voltage_IQR__ratio', 'INV__Discharge_Voltage_MAD__delta', 'INV__Discharge_Voltage_MAD__pct', 'INV__Discharge_Voltage_MAD__pct_ewm5', 'INV__Discharge_Voltage_MAD__pct_lag1_delta', 'INV__Discharge_Voltage_MAD__pct_slope5', 'INV__Discharge_Voltage_MAD__ratio', 'INV__Discharge_Voltage_mean__delta', 'INV__Discharge_Voltage_mean__pct', 'INV__Discharge_Voltage_mean__pct_ewm5', 'INV__Discharge_Voltage_mean__pct_lag1_delta', 'INV__Discharge_Voltage_mean__pct_slope5', 'INV__Discharge_Voltage_mean__ratio', 'INV__Discharge_Voltage_q10__delta', 'INV__Discharge_Voltage_q10__pct', 'INV__Discharge_Voltage_q10__pct_ewm5', 'INV__Discharge_Voltage_q10__pct_lag1_delta', 'INV__Discharge_Voltage_q10__pct_slope5', 'INV__Discharge_Voltage_q10__ratio', 'INV__Discharge_Voltage_q25__delta', 'INV__Discharge_Voltage_q25__pct', 'INV__Discharge_Voltage_q25__pct_ewm5', 'INV__Discharge_Voltage_q25__pct_lag1_delta', 'INV__Discharge_Voltage_q25__pct_slope5', 'INV__Discharge_Voltage_q25__ratio', 'INV__Discharge_Voltage_q50__delta', 'INV__Discharge_Voltage_q50__pct', 'INV__Discharge_Voltage_q50__pct_ewm5', 'INV__Discharge_Voltage_q50__pct_lag1_delta', 'INV__Discharge_Voltage_q50__pct_slope5', 'INV__Discharge_Voltage_q50__ratio', 'INV__Discharge_Voltage_q75__delta', 'INV__Discharge_Voltage_q75__pct', 'INV__Discharge_Voltage_q75__pct_ewm5', 'INV__Discharge_Voltage_q75__pct_lag1_delta', 'INV__Discharge_Voltage_q75__pct_slope5', 'INV__Discharge_Voltage_q75__ratio', 'INV__Discharge_Voltage_q90__delta', 'INV__Discharge_Voltage_q90__pct', 'INV__Discharge_Voltage_q90__pct_ewm5', 'INV__Discharge_Voltage_q90__pct_lag1_delta', 'INV__Discharge_Voltage_q90__pct_slope5', 'INV__Discharge_Voltage_q90__ratio', 'INV__Discharge_Voltage_std__delta', 'INV__Discharge_Voltage_std__pct', 'INV__Discharge_Voltage_std__pct_ewm5', 'INV__Discharge_Voltage_std__pct_lag1_delta', 'INV__Discharge_Voltage_std__pct_slope5', 'INV__Discharge_Voltage_std__ratio', 'INV__Discharge_endVoltage__delta', 'INV__Discharge_endVoltage__pct', 'INV__Discharge_endVoltage__pct_ewm5', 'INV__Discharge_endVoltage__pct_lag1_delta', 'INV__Discharge_endVoltage__pct_slope5', 'INV__Discharge_endVoltage__ratio', 'INV__Discharge_startVoltage__delta', 'INV__Discharge_startVoltage__pct', 'INV__Discharge_startVoltage__pct_ewm5', 'INV__Discharge_startVoltage__pct_lag1_delta', 'INV__Discharge_startVoltage__pct_slope5', 'INV__Discharge_startVoltage__ratio', 'INV__Internal_Resistance_median__delta', 'INV__Internal_Resistance_median__pct', 'INV__Internal_Resistance_median__pct_ewm5', 'INV__Internal_Resistance_median__pct_lag1_delta', 'INV__Internal_Resistance_median__pct_slope5', 'INV__Internal_Resistance_median__ratio', 'INV__R5E_Norm__AbsDVdt_PerAbsCurrent', 'INV__R5E_Norm__Current_CV', 'INV__R5E_Norm__Current_IQR_Rel', 'INV__R5E_Norm__Current_MAD_Rel', 'INV__R5E_Norm__VoltageDrop_PerAbsCurrent', 'INV__avg_abs_dvdt__delta', 'INV__avg_abs_dvdt__pct', 'INV__avg_abs_dvdt__pct_ewm5', 'INV__avg_abs_dvdt__pct_lag1_delta', 'INV__avg_abs_dvdt__pct_slope5', 'INV__avg_abs_dvdt__ratio', 'INV__dVdt_mean__delta', 'INV__dVdt_mean__pct', 'INV__dVdt_mean__pct_ewm5', 'INV__dVdt_mean__pct_lag1_delta', 'INV__dVdt_mean__pct_slope5', 'INV__dVdt_mean__ratio', 'INV__mean_abs_dVdt_over_I__delta', 'INV__mean_abs_dVdt_over_I__pct', 'INV__mean_abs_dVdt_over_I__pct_ewm5', 'INV__mean_abs_dVdt_over_I__pct_lag1_delta', 'INV__mean_abs_dVdt_over_I__pct_slope5', 'INV__mean_abs_dVdt_over_I__ratio', 'INV__median_abs_dVdt_over_I__delta', 'INV__median_abs_dVdt_over_I__pct', 'INV__median_abs_dVdt_over_I__pct_ewm5', 'INV__median_abs_dVdt_over_I__pct_lag1_delta', 'INV__median_abs_dVdt_over_I__pct_slope5', 'INV__median_abs_dVdt_over_I__ratio', 'LEGACY__Charge_Current_kurtosis', 'LEGACY__Charge_Current_max', 'LEGACY__Charge_Current_mean', 'LEGACY__Charge_Current_min', 'LEGACY__Charge_Current_skewness', 'LEGACY__Charge_Current_std', 'LEGACY__Charge_Step2_CC_currentMedian', 'LEGACY__Charge_Step2_CC_currentStd', 'LEGACY__Charge_Step2_CC_kurtosis', 'LEGACY__Charge_Step2_CC_skewness', 'LEGACY__Charge_Step2_CC_voltageStd', 'LEGACY__Charge_Voltage_kurtosis', 'LEGACY__Charge_Voltage_max', 'LEGACY__Charge_Voltage_mean', 'LEGACY__Charge_Voltage_min', 'LEGACY__Charge_Voltage_skewness', 'LEGACY__Charge_Voltage_std', 'LEGACY__Charge_endVoltage', 'LEGACY__Charge_startVoltage', 'LEGACY__Cycle_Age', 'LEGACY__Cycle_Index', 'LEGACY__DeltaV_charge', 'LEGACY__DeltaV_discharge', 'LEGACY__Discharge_Current_IQR', 'LEGACY__Discharge_Current_MAD', 'LEGACY__Discharge_Current_kurtosis', 'LEGACY__Discharge_Current_max', 'LEGACY__Discharge_Current_mean', 'LEGACY__Discharge_Current_min', 'LEGACY__Discharge_Current_q10', 'LEGACY__Discharge_Current_q25', 'LEGACY__Discharge_Current_q50', 'LEGACY__Discharge_Current_q75', 'LEGACY__Discharge_Current_q90', 'LEGACY__Discharge_Current_skewness', 'LEGACY__Discharge_Current_std', 'LEGACY__Discharge_Voltage_IQR', 'LEGACY__Discharge_Voltage_MAD', 'LEGACY__Discharge_Voltage_kurtosis', 'LEGACY__Discharge_Voltage_max', 'LEGACY__Discharge_Voltage_mean', 'LEGACY__Discharge_Voltage_min', 'LEGACY__Discharge_Voltage_q10', 'LEGACY__Discharge_Voltage_q25', 'LEGACY__Discharge_Voltage_q50', 'LEGACY__Discharge_Voltage_q75', 'LEGACY__Discharge_Voltage_q90', 'LEGACY__Discharge_Voltage_skewness', 'LEGACY__Discharge_Voltage_std', 'LEGACY__Discharge_endVoltage', 'LEGACY__Discharge_startVoltage', 'LEGACY__Internal_Resistance_max', 'LEGACY__Internal_Resistance_median', 'LEGACY__Internal_Resistance_p90', 'LEGACY__LEGACYH__DeltaV_discharge__baseline_delta', 'LEGACY__LEGACYH__DeltaV_discharge__baseline_pct', 'LEGACY__LEGACYH__DeltaV_discharge__baseline_ratio', 'LEGACY__LEGACYH__DeltaV_discharge__ewm_5', 'LEGACY__LEGACYH__DeltaV_discharge__lag1_delta', 'LEGACY__LEGACYH__DeltaV_discharge__per_cycle_delta', 'LEGACY__LEGACYH__DeltaV_discharge__rolling_mean_3', 'LEGACY__LEGACYH__DeltaV_discharge__rolling_mean_5', 'LEGACY__LEGACYH__DeltaV_discharge__rolling_slope_5', 'LEGACY__LEGACYH__DeltaV_discharge__rolling_std_5', 'LEGACY__LEGACYH__Discharge_Voltage_IQR__baseline_delta', 'LEGACY__LEGACYH__Discharge_Voltage_IQR__baseline_pct', 'LEGACY__LEGACYH__Discharge_Voltage_IQR__baseline_ratio', 'LEGACY__LEGACYH__Discharge_Voltage_IQR__ewm_5', 'LEGACY__LEGACYH__Discharge_Voltage_IQR__lag1_delta', 'LEGACY__LEGACYH__Discharge_Voltage_IQR__per_cycle_delta', 'LEGACY__LEGACYH__Discharge_Voltage_IQR__rolling_mean_3', 'LEGACY__LEGACYH__Discharge_Voltage_IQR__rolling_mean_5', 'LEGACY__LEGACYH__Discharge_Voltage_IQR__rolling_slope_5', 'LEGACY__LEGACYH__Discharge_Voltage_IQR__rolling_std_5', 'LEGACY__LEGACYH__Discharge_Voltage_MAD__baseline_delta', 'LEGACY__LEGACYH__Discharge_Voltage_MAD__baseline_pct', 'LEGACY__LEGACYH__Discharge_Voltage_MAD__baseline_ratio', 'LEGACY__LEGACYH__Discharge_Voltage_MAD__ewm_5', 'LEGACY__LEGACYH__Discharge_Voltage_MAD__lag1_delta', 'LEGACY__LEGACYH__Discharge_Voltage_MAD__per_cycle_delta', 'LEGACY__LEGACYH__Discharge_Voltage_MAD__rolling_mean_3', 'LEGACY__LEGACYH__Discharge_Voltage_MAD__rolling_mean_5', 'LEGACY__LEGACYH__Discharge_Voltage_MAD__rolling_slope_5', 'LEGACY__LEGACYH__Discharge_Voltage_MAD__rolling_std_5', 'LEGACY__LEGACYH__Discharge_Voltage_mean__baseline_delta', 'LEGACY__LEGACYH__Discharge_Voltage_mean__baseline_pct', 'LEGACY__LEGACYH__Discharge_Voltage_mean__baseline_ratio', 'LEGACY__LEGACYH__Discharge_Voltage_mean__ewm_5', 'LEGACY__LEGACYH__Discharge_Voltage_mean__lag1_delta', 'LEGACY__LEGACYH__Discharge_Voltage_mean__per_cycle_delta', 'LEGACY__LEGACYH__Discharge_Voltage_mean__rolling_mean_3', 'LEGACY__LEGACYH__Discharge_Voltage_mean__rolling_mean_5', 'LEGACY__LEGACYH__Discharge_Voltage_mean__rolling_slope_5', 'LEGACY__LEGACYH__Discharge_Voltage_mean__rolling_std_5', 'LEGACY__LEGACYH__Discharge_Voltage_std__baseline_delta', 'LEGACY__LEGACYH__Discharge_Voltage_std__baseline_pct', 'LEGACY__LEGACYH__Discharge_Voltage_std__baseline_ratio', 'LEGACY__LEGACYH__Discharge_Voltage_std__ewm_5', 'LEGACY__LEGACYH__Discharge_Voltage_std__lag1_delta', 'LEGACY__LEGACYH__Discharge_Voltage_std__per_cycle_delta', 'LEGACY__LEGACYH__Discharge_Voltage_std__rolling_mean_3', 'LEGACY__LEGACYH__Discharge_Voltage_std__rolling_mean_5', 'LEGACY__LEGACYH__Discharge_Voltage_std__rolling_slope_5', 'LEGACY__LEGACYH__Discharge_Voltage_std__rolling_std_5', 'LEGACY__LEGACYH__Discharge_endVoltage__baseline_delta', 'LEGACY__LEGACYH__Discharge_endVoltage__baseline_pct', 'LEGACY__LEGACYH__Discharge_endVoltage__baseline_ratio', 'LEGACY__LEGACYH__Discharge_endVoltage__ewm_5', 'LEGACY__LEGACYH__Discharge_endVoltage__lag1_delta', 'LEGACY__LEGACYH__Discharge_endVoltage__per_cycle_delta', 'LEGACY__LEGACYH__Discharge_endVoltage__rolling_mean_3', 'LEGACY__LEGACYH__Discharge_endVoltage__rolling_mean_5', 'LEGACY__LEGACYH__Discharge_endVoltage__rolling_slope_5', 'LEGACY__LEGACYH__Discharge_endVoltage__rolling_std_5', 'LEGACY__LEGACYH__Discharge_startVoltage__baseline_delta', 'LEGACY__LEGACYH__Discharge_startVoltage__baseline_pct', 'LEGACY__LEGACYH__Discharge_startVoltage__baseline_ratio', 'LEGACY__LEGACYH__Discharge_startVoltage__ewm_5', 'LEGACY__LEGACYH__Discharge_startVoltage__lag1_delta', 'LEGACY__LEGACYH__Discharge_startVoltage__per_cycle_delta', 'LEGACY__LEGACYH__Discharge_startVoltage__rolling_mean_3', 'LEGACY__LEGACYH__Discharge_startVoltage__rolling_mean_5', 'LEGACY__LEGACYH__Discharge_startVoltage__rolling_slope_5', 'LEGACY__LEGACYH__Discharge_startVoltage__rolling_std_5', 'LEGACY__LEGACYH__Internal_Resistance_median__baseline_delta', 'LEGACY__LEGACYH__Internal_Resistance_median__ewm_5', 'LEGACY__LEGACYH__Internal_Resistance_median__lag1_delta', 'LEGACY__LEGACYH__Internal_Resistance_median__per_cycle_delta', 'LEGACY__LEGACYH__Internal_Resistance_median__rolling_mean_3', 'LEGACY__LEGACYH__Internal_Resistance_median__rolling_mean_5', 'LEGACY__LEGACYH__Internal_Resistance_median__rolling_slope_5', 'LEGACY__LEGACYH__Internal_Resistance_median__rolling_std_5', 'LEGACY__LEGACYH__avg_abs_dvdt__baseline_delta', 'LEGACY__LEGACYH__avg_abs_dvdt__baseline_pct', 'LEGACY__LEGACYH__avg_abs_dvdt__baseline_ratio', 'LEGACY__LEGACYH__avg_abs_dvdt__ewm_5', 'LEGACY__LEGACYH__avg_abs_dvdt__lag1_delta', 'LEGACY__LEGACYH__avg_abs_dvdt__per_cycle_delta', 'LEGACY__LEGACYH__avg_abs_dvdt__rolling_mean_3', 'LEGACY__LEGACYH__avg_abs_dvdt__rolling_mean_5', 'LEGACY__LEGACYH__avg_abs_dvdt__rolling_slope_5', 'LEGACY__LEGACYH__avg_abs_dvdt__rolling_std_5', 'LEGACY__LEGACYH__dVdt_mean__baseline_delta', 'LEGACY__LEGACYH__dVdt_mean__baseline_pct', 'LEGACY__LEGACYH__dVdt_mean__baseline_ratio', 'LEGACY__LEGACYH__dVdt_mean__ewm_5', 'LEGACY__LEGACYH__dVdt_mean__lag1_delta', 'LEGACY__LEGACYH__dVdt_mean__per_cycle_delta', 'LEGACY__LEGACYH__dVdt_mean__rolling_mean_3', 'LEGACY__LEGACYH__dVdt_mean__rolling_mean_5', 'LEGACY__LEGACYH__dVdt_mean__rolling_slope_5', 'LEGACY__LEGACYH__dVdt_mean__rolling_std_5', 'LEGACY__LEGACYH__mean_abs_dVdt_over_I__baseline_delta', 'LEGACY__LEGACYH__mean_abs_dVdt_over_I__baseline_pct', 'LEGACY__LEGACYH__mean_abs_dVdt_over_I__baseline_ratio', 'LEGACY__LEGACYH__mean_abs_dVdt_over_I__ewm_5', 'LEGACY__LEGACYH__mean_abs_dVdt_over_I__lag1_delta', 'LEGACY__LEGACYH__mean_abs_dVdt_over_I__per_cycle_delta', 'LEGACY__LEGACYH__mean_abs_dVdt_over_I__rolling_mean_3', 'LEGACY__LEGACYH__mean_abs_dVdt_over_I__rolling_mean_5', 'LEGACY__LEGACYH__mean_abs_dVdt_over_I__rolling_slope_5', 'LEGACY__LEGACYH__mean_abs_dVdt_over_I__rolling_std_5', 'LEGACY__LEGACYH__median_abs_dVdt_over_I__baseline_delta', 'LEGACY__LEGACYH__median_abs_dVdt_over_I__baseline_pct', 'LEGACY__LEGACYH__median_abs_dVdt_over_I__baseline_ratio', 'LEGACY__LEGACYH__median_abs_dVdt_over_I__ewm_5', 'LEGACY__LEGACYH__median_abs_dVdt_over_I__lag1_delta', 'LEGACY__LEGACYH__median_abs_dVdt_over_I__per_cycle_delta', 'LEGACY__LEGACYH__median_abs_dVdt_over_I__rolling_mean_3', 'LEGACY__LEGACYH__median_abs_dVdt_over_I__rolling_mean_5', 'LEGACY__LEGACYH__median_abs_dVdt_over_I__rolling_slope_5', 'LEGACY__LEGACYH__median_abs_dVdt_over_I__rolling_std_5', 'LEGACY__Log_Cycle_Age', 'LEGACY__avg_abs_dvdt', 'LEGACY__dVdt_mean', 'LEGACY__mean_abs_dVdt_over_I', 'LEGACY__median_abs_dVdt_over_I']

def _natural_key(value: str):
    return [int(piece) if piece.isdigit() else piece.lower()
            for piece in re.split(r"(\d+)", str(value))]

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

def safe_extract_zip(zip_path: Path, destination: Path) -> None:
    if destination.exists():
        shutil.rmtree(destination)
    destination.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as archive:
        root = destination.resolve()
        for info in archive.infolist():
            candidate = (destination / info.filename).resolve()
            if root != candidate and root not in candidate.parents:
                raise ValueError(f"Unsafe ZIP member path: {info.filename}")
        archive.extractall(destination)

def _find_exact_dirs(root: Path, name: str) -> list[Path]:
    return sorted(
        [p for p in root.rglob("*") if p.is_dir() and p.name.lower() == name.lower()],
        key=lambda p: _natural_key(str(p)),
    )

def _find_named_files(root: Path, filename: str) -> list[Path]:
    return sorted(
        [p for p in root.rglob("*") if p.is_file() and p.name.lower() == filename.lower()],
        key=lambda p: _natural_key(str(p)),
    )

def _validate_nasa_columns(csv_path: Path, battery_id: str, random_walk: bool) -> list[str]:
    sample = pd.read_csv(csv_path, nrows=8, low_memory=False)
    if random_walk:
        required_groups = [
            ["operation_class"], ["random_walk_cycle"],
            ["random_walk_segment", "source_step_index"],
            ["relativeTime", "relative_time"], ["voltage", "Voltage"],
            ["current", "Current"], ["SOH"],
        ]
    else:
        required_groups = [
            ["discharge_cycle"], ["Time", "time"],
            ["Voltage_measured", "voltage_measured", "Voltage_load", "voltage"],
            ["Current_measured", "current_measured", "Current_load", "current"],
            ["SOH"],
        ]
    found = []
    for candidates in required_groups:
        found.append(find_column(sample, candidates, required=True))
    return found

def validate_extracted_dataset(root: Path) -> tuple[dict[str, dict], pd.DataFrame]:
    manifest: dict[str, dict] = {}
    rows: list[dict] = []
    errors: list[str] = []

    for battery_id, required_sheets in CALCE_CONSTANT_BATTERIES + CALCE_VARIABLE_BATTERIES:
        dirs = _find_exact_dirs(root, battery_id)
        if len(dirs) != 1:
            errors.append(
                f"{battery_id}: expected exactly one folder named {battery_id}; found {len(dirs)}."
            )
            continue
        folder = dirs[0]
        try:
            xlsx_files = [Path(p) for p in extract_files(folder)]
        except Exception as exc:
            errors.append(f"{battery_id}: {exc}")
            continue
        sheet_union: set[str] = set()
        workbook_order = []
        workbook_dates = []
        try:
            for workbook in xlsx_files:
                stamp = parse_calce_workbook_date(workbook)
                workbook_order.append(str(workbook.relative_to(folder)))
                workbook_dates.append(stamp.strftime("%Y-%m-%d") if stamp is not None else "UNPARSED")
                with pd.ExcelFile(workbook) as excel:
                    sheet_union.update(excel.sheet_names)
        except Exception as exc:
            errors.append(f"{battery_id}: could not inspect workbook sheets: {exc}")
            continue
        missing_sheets = [s for s in required_sheets if s not in sheet_union]
        if missing_sheets:
            errors.append(f"{battery_id}: missing required sheet(s): {missing_sheets}")
            continue
        manifest[battery_id] = {
            "kind": "CALCE",
            "folder": folder,
            "xlsx_files": xlsx_files,
            "required_sheets": list(required_sheets),
        }
        rows.append({
            "Battery": battery_id, "Kind": "CALCE", "Status": "OK",
            "File_Count": len(xlsx_files), "Required_Sheets": required_sheets,
            "Detected_Columns": "",
            "Chronological_Workbook_Order": workbook_order,
            "Parsed_Workbook_Dates": workbook_dates,
            "Resolved_Path": str(folder),
        })

    for battery_id in NASA_CONSTANT_BATTERIES + NASA_VARIABLE_BATTERIES:
        matches = _find_named_files(root, f"{battery_id}.csv")
        if len(matches) != 1:
            errors.append(
                f"{battery_id}: expected exactly one {battery_id}.csv; found {len(matches)}."
            )
            continue
        csv_path = matches[0]
        random_walk = battery_id in NASA_VARIABLE_BATTERIES
        try:
            detected = _validate_nasa_columns(csv_path, battery_id, random_walk=random_walk)
        except Exception as exc:
            errors.append(f"{battery_id}: NASA CSV validation failed: {exc}")
            continue
        manifest[battery_id] = {
            "kind": "NASA_RW" if random_walk else "NASA_CONSTANT",
            "csv_path": csv_path,
        }
        rows.append({
            "Battery": battery_id,
            "Kind": "NASA_RW" if random_walk else "NASA_CONSTANT",
            "Status": "OK", "File_Count": 1, "Required_Sheets": "",
            "Detected_Columns": detected, "Chronological_Workbook_Order": "",
            "Parsed_Workbook_Dates": "", "Resolved_Path": str(csv_path),
        })

    missing = sorted(set(ALL_BATTERY_IDS) - set(manifest))
    if missing:
        errors.append("Unresolved batteries: " + ", ".join(missing))
    validation_df = pd.DataFrame(rows)
    if errors:
        print("\n" + "=" * 78)
        print("DATA VALIDATION FAILED")
        print("=" * 78)
        for error in errors:
            print(" - " + error)
        print("\nTraining has NOT started.")
        raise RuntimeError("ZIP validation failed. Fix the items above and upload a new ZIP.")

    print("\n" + "=" * 78)
    print("DATA VALIDATION PASSED")
    print("=" * 78)
    print(validation_df[["Battery", "Kind", "File_Count", "Resolved_Path"]].to_string(index=False))
    print("\nCALCE workbook order parsed from filename dates (important in Colab):")
    for battery_id, _ in CALCE_CONSTANT_BATTERIES + CALCE_VARIABLE_BATTERIES:
        folder = manifest[battery_id]["folder"]
        entries = []
        for p in manifest[battery_id]["xlsx_files"]:
            stamp = parse_calce_workbook_date(p)
            date_label = stamp.strftime("%Y-%m-%d") if stamp is not None else "UNPARSED"
            entries.append(f"{date_label} :: {p.relative_to(folder)}")
        print(f"  {battery_id}:")
        for entry in entries:
            print(f"      {entry}")
    return manifest, validation_df

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

def _integrate_discharge_block_capacity(
    block,
    time_col="Test_Time(s)",
    current_col="Current(A)",
):
    """
    Integrate one contiguous discharge block using trapezoidal
    current integration.

    Returns
    -------
    capacity_ah : float
        Integrated discharged capacity in ampere-hours.
    duration_s : float
        Sum of valid positive time intervals in seconds.
    """
    b = block[[time_col, current_col]].copy()
    b[time_col] = pd.to_numeric(b[time_col], errors="coerce")
    b[current_col] = pd.to_numeric(b[current_col], errors="coerce")
    b = b.dropna(subset=[time_col, current_col]).sort_values(time_col)

    if len(b) < 2:
        return np.nan, np.nan

    time_s = b[time_col].to_numpy(dtype=float)
    current_a = np.abs(b[current_col].to_numpy(dtype=float))

    dt_s = np.diff(time_s)
    interval_current_a = 0.5 * (current_a[:-1] + current_a[1:])

    valid = (
        np.isfinite(dt_s)
        & np.isfinite(interval_current_a)
        & (dt_s > 0)
    )

    if not valid.any():
        return np.nan, np.nan

    capacity_ah = float(
        np.sum(interval_current_a[valid] * dt_s[valid]) / 3600.0
    )
    duration_s = float(np.sum(dt_s[valid]))

    return capacity_ah, duration_s

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

def joint_candidate_features(df: pd.DataFrame) -> list[str]:
    return sorted(
        c for c in df.columns
        if c.startswith("LEGACY__") or c.startswith("INV__")
    )

def _feature_body(feature: str) -> str:
    f = str(feature)
    if f.startswith("LEGACY__"):
        return f[len("LEGACY__"):]
    if f.startswith("INV__"):
        return f[len("INV__"):]
    return f

def is_direct_capacity_proxy(feature: str) -> bool:
    body = _feature_body(feature)
    return (
        ("Discharge_cumulativeCapacity" in body and "PastOnly" not in body)
        or ("Throughput_EFC" in body)
    )

def is_total_duration_proxy(feature: str) -> bool:
    body = _feature_body(feature)
    return "Discharge_duration" in body and "PastOnly" not in body

def is_total_energy_proxy(feature: str) -> bool:
    body = _feature_body(feature)
    return "Discharge_cumulativeEnergy" in body and "PastOnly" not in body

def is_derived_full_cycle_scale_proxy(feature: str) -> bool:
    body = _feature_body(feature)
    if "PastOnly" in body:
        return False
    return any(token in body for token in (
        "Discharge_VoltageDropPerAh",
        "Discharge_VoltageDropPerHour",
        "Discharge_EnergyPerHour",
    ))

def uses_current_cycle_q_fraction(feature: str) -> bool:
    body = _feature_body(feature)
    return any(token in body for token in (
        "Discharge_VoltageAt_Q",
        "Discharge_TimeAt_Q",
        "Discharge_EnergyAt_Q",
    ))

def is_age_feature(feature: str) -> bool:
    body = _feature_body(feature)
    return body in {"Cycle_Index", "Cycle_Age", "Log_Cycle_Age"}

def is_dataset_identity_feature(feature: str) -> bool:
    body = _feature_body(feature)
    return body in {"source_is_nasa", "protocol"}

def proxy_safe_feature(feature: str, regime_kind: str) -> bool:
    """
    Shared feature policy used by every base model and stack.

    Allowed examples:
      * fixed-voltage capacity/time/energy landmarks (4.0, 3.8, 3.6, 3.4, 3.2 V)
      * voltage/current distribution features and IC/CC features
      * charge-stage features and resistance
      * causal Legacy/Invariant history features
      * strictly PastOnly previous-cycle full-cycle summaries

    Banned current-cycle proxies:
      * total discharge capacity
      * total discharge duration
      * total discharge energy
      * Q10-Q90 landmarks defined using eventual current-cycle total Q
      * current-cycle capacity-throughput/EFC quantities
      * dataset identity

    Raw cycle age is permitted only for LOBO if ALLOW_CAUSAL_AGE_IN_LOBO=True.
    """
    if is_dataset_identity_feature(feature):
        return False
    if BAN_SAME_CYCLE_TOTAL_CAPACITY and is_direct_capacity_proxy(feature):
        return False
    if BAN_SAME_CYCLE_TOTAL_DURATION and is_total_duration_proxy(feature):
        return False
    if BAN_SAME_CYCLE_TOTAL_ENERGY and is_total_energy_proxy(feature):
        return False
    if BAN_CURRENT_CYCLE_Q_FRACTION_LANDMARKS and uses_current_cycle_q_fraction(feature):
        return False
    if is_derived_full_cycle_scale_proxy(feature):
        return False
    if BAN_CURRENT_CAPACITY_THROUGHPUT and "Throughput_EFC" in _feature_body(feature):
        return False
    if is_age_feature(feature):
        if regime_kind == "LOBO":
            return bool(ALLOW_CAUSAL_AGE_IN_LOBO)
        return bool(ALLOW_CAUSAL_AGE_IN_TRANSFER)
    return True

def proxy_safe_candidates(df: pd.DataFrame, regime_kind: str) -> list[str]:
    return [
        f for f in joint_candidate_features(df)
        if proxy_safe_feature(f, regime_kind)
    ]

def load_nasa_constant_measurements_targetfree(csv_path, battery_id):
    """
    Load B0005/B0006/B0007/B0018 without using the source SOH column.

    The revised target is reconstructed solely from raw discharge
    current and time.
    """
    raw = read_nasa_csv(csv_path, battery_id)

    cycle_source = find_column(raw, ["discharge_cycle"])
    time_source = find_column(raw, ["Time", "time"])
    voltage_source = find_column(
        raw,
        ["Voltage_measured", "voltage_measured", "Voltage_load", "voltage"],
    )
    current_source = find_column(
        raw,
        ["Current_measured", "current_measured", "Current_load", "current"],
    )

    out = pd.DataFrame(
        {
            CYCLE_COL: pd.to_numeric(raw[cycle_source], errors="coerce"),
            "Test_Time(s)": pd.to_numeric(raw[time_source], errors="coerce"),
            "Voltage(V)": pd.to_numeric(raw[voltage_source], errors="coerce"),
            "Current(A)": -np.abs(
                pd.to_numeric(raw[current_source], errors="coerce")
            ),
            "Step_Index": 7.0,
        }
    )

    out = (
        out.dropna(
            subset=[CYCLE_COL, "Test_Time(s)", "Voltage(V)", "Current(A)"]
        )
        .sort_values([CYCLE_COL, "Test_Time(s)"])
        .reset_index(drop=True)
    )

    if out.empty:
        raise ValueError(
            f"No usable NASA constant-discharge measurements for {battery_id}."
        )

    return out

def reconstruct_constant_targets(raw, battery_id, source):
    """
    Reconstruct one measured capacity-based SoH target per constant-discharge
    cycle using raw current/time integration.

    IMPORTANT:
    - No existing SOH column is used.
    - No visual target-spike filtering is performed.
    - No Charge_duration >= 2 h rule is applied.
    - The 70-100% scope restriction is applied later, after feature/history
      construction.
    """
    required = [CYCLE_COL, "Test_Time(s)", "Current(A)"]

    missing = [c for c in required if c not in raw.columns]
    if missing:
        raise KeyError(
            f"{battery_id}: missing target reconstruction columns: {missing}"
        )

    d = raw[required].copy()

    d[CYCLE_COL] = pd.to_numeric(d[CYCLE_COL], errors="coerce")
    d["Test_Time(s)"] = pd.to_numeric(d["Test_Time(s)"], errors="coerce")
    d["Current(A)"] = pd.to_numeric(d["Current(A)"], errors="coerce")

    d = (
        d.dropna(subset=[CYCLE_COL, "Test_Time(s)", "Current(A)"])
        .sort_values([CYCLE_COL, "Test_Time(s)"])
        .reset_index(drop=True)
    )

    rows = []

    for cycle, g in d.groupby(CYCLE_COL, sort=True):
        t = g["Test_Time(s)"].to_numpy(dtype=float)
        i = g["Current(A)"].to_numpy(dtype=float)

        finite = np.isfinite(t) & np.isfinite(i)
        t = t[finite]
        i = i[finite]

        n = int(len(t))

        if n < 2:
            rows.append(
                {
                    GROUP_COL: battery_id,
                    SOURCE_COL: source,
                    CYCLE_COL: cycle,
                    "N_Raw_Samples": n,
                    "Duration_s": np.nan,
                    "Measured_Discharge_Capacity_Ah": np.nan,
                    "Raw_QC_Pass": False,
                    "Raw_QC_Reason": "fewer_than_2_finite_samples",
                }
            )
            continue

        order = np.argsort(t)
        t = t[order]
        i = i[order]

        dt = np.diff(t, prepend=t[0])
        dt = np.where(np.isfinite(dt) & (dt > 0), dt, 0.0)

        i_dis = np.maximum(-i, 0.0)

        q_ah = float(np.sum(i_dis * dt) / 3600.0)
        duration_s = float(np.max(t) - np.min(t))

        qc = (
            np.isfinite(q_ah)
            and q_ah > 0
            and np.isfinite(duration_s)
            and duration_s > 0
        )

        reason = "" if qc else "nonpositive_or_nonfinite_capacity_or_duration"

        rows.append(
            {
                GROUP_COL: battery_id,
                SOURCE_COL: source,
                CYCLE_COL: cycle,
                "N_Raw_Samples": n,
                "Duration_s": duration_s,
                "Measured_Discharge_Capacity_Ah": q_ah,
                "Raw_QC_Pass": bool(qc),
                "Raw_QC_Reason": reason,
            }
        )

    audit = pd.DataFrame(rows).sort_values(CYCLE_COL).reset_index(drop=True)

    valid = audit.loc[audit["Raw_QC_Pass"]].copy()

    if valid.empty:
        raise RuntimeError(
            f"{battery_id}: no valid raw current-integration target cycles."
        )

    # First valid measured discharge defines 100% reference.
    q_ref = float(valid["Measured_Discharge_Capacity_Ah"].iloc[0])

    if not np.isfinite(q_ref) or q_ref <= 0:
        raise RuntimeError(
            f"{battery_id}: invalid first reference capacity: {q_ref}"
        )

    audit["Reference_Capacity_Ah"] = q_ref
    audit[TARGET_COL] = (
        audit["Measured_Discharge_Capacity_Ah"] / q_ref * 100.0
    )

    # This is an analysis-scope flag, NOT an outlier rule.
    audit["In_Primary_70_100_Scope"] = (
        audit["Raw_QC_Pass"]
        & audit[TARGET_COL].between(
            SOH_LOWER_LIMIT,
            SOH_UPPER_LIMIT,
            inclusive="both",
        )
    )

    target = audit.loc[
        audit["Raw_QC_Pass"],
        [
            GROUP_COL,
            SOURCE_COL,
            CYCLE_COL,
            "Measured_Discharge_Capacity_Ah",
            "Reference_Capacity_Ah",
            TARGET_COL,
        ],
    ].copy()

    return target, audit

def build_revised_constant_cycle_table(manifest):
    """
    Build target-independent features for all ten constant-protocol batteries,
    then create the target and apply the 70-100% study scope only AFTER
    causal/history feature construction.
    """
    feature_parts = []
    target_audits = []

    # --------------------------------------------------------
    # CALCE constant cells
    # --------------------------------------------------------
    for battery_id, sheets in CALCE_CONSTANT_BATTERIES:
        print(f"\n[CALCE] building target-independent constant features: {battery_id}")

        raw = process_files(
            str(manifest[battery_id]["folder"]),
            sheets,
        )

        raw[GROUP_COL] = battery_id

        target, audit = reconstruct_constant_targets(
            raw,
            battery_id=battery_id,
            source="CALCE",
        )

        f = build_cycle_features_run2(
            raw,
            cycle_col=CYCLE_COL,
            time_col="Test_Time(s)",
            volt_col="Voltage(V)",
            curr_col="Current(A)",
            step_col="Step_Index",
            segment_col=None,
        )

        f = add_dataset_identity(
            f,
            battery_id,
            "CALCE",
            CONSTANT_PROTOCOL,
        )

        f = f.merge(
            target[
                [
                    CYCLE_COL,
                    "Measured_Discharge_Capacity_Ah",
                    "Reference_Capacity_Ah",
                    TARGET_COL,
                ]
            ],
            on=CYCLE_COL,
            how="left",
            validate="one_to_one",
        )

        feature_parts.append(f)
        target_audits.append(audit)

        del raw, f
        gc.collect()

    # --------------------------------------------------------
    # NASA classic constant cells
    # --------------------------------------------------------
    for battery_id in NASA_CONSTANT_BATTERIES:
        print(f"\n[NASA] building target-independent constant features: {battery_id}")

        raw = load_nasa_constant_measurements_targetfree(
            manifest[battery_id]["csv_path"],
            battery_id,
        )

        raw[GROUP_COL] = battery_id

        target, audit = reconstruct_constant_targets(
            raw,
            battery_id=battery_id,
            source="NASA",
        )

        f = build_cycle_features_run2(
            raw,
            cycle_col=CYCLE_COL,
            time_col="Test_Time(s)",
            volt_col="Voltage(V)",
            curr_col="Current(A)",
            step_col="Step_Index",
            segment_col=None,
        )

        f = add_dataset_identity(
            f,
            battery_id,
            "NASA",
            CONSTANT_PROTOCOL,
        )

        f = f.merge(
            target[
                [
                    CYCLE_COL,
                    "Measured_Discharge_Capacity_Ah",
                    "Reference_Capacity_Ah",
                    TARGET_COL,
                ]
            ],
            on=CYCLE_COL,
            how="left",
            validate="one_to_one",
        )

        feature_parts.append(f)
        target_audits.append(audit)

        del raw, f
        gc.collect()

    combined = pd.concat(
        feature_parts,
        ignore_index=True,
        sort=False,
    )

    combined[CYCLE_COL] = pd.to_numeric(
        combined[CYCLE_COL],
        errors="coerce",
    )
    combined[TARGET_COL] = pd.to_numeric(
        combined[TARGET_COL],
        errors="coerce",
    )

    combined = (
        combined.sort_values([GROUP_COL, CYCLE_COL])
        .reset_index(drop=True)
    )

    # --------------------------------------------------------
    # Define age BEFORE the target scope is applied.
    # No target is consulted.
    # --------------------------------------------------------
    first_cycle = combined.groupby(GROUP_COL)[CYCLE_COL].transform("min")

    combined["Cycle_Age"] = (
        combined[CYCLE_COL] - first_cycle
    ).clip(lower=0)

    combined["Log_Cycle_Age"] = np.log1p(
        combined["Cycle_Age"]
    )

    combined["source_is_nasa"] = (
        combined[SOURCE_COL]
        .astype(str)
        .str.upper()
        .eq("NASA")
        .astype(int)
    )

    # --------------------------------------------------------
    # Generate all causal/history representations BEFORE
    # restricting the target range.
    # --------------------------------------------------------
    print("\nBuilding target-independent causal/history representations...")

    combined = add_legacy_history_features(combined)
    combined = add_run2_invariant_features(combined)
    combined = add_past_only_features(combined)
    combined = add_joint_legacy_aliases(combined)

    combined = (
        combined.sort_values([GROUP_COL, CYCLE_COL])
        .reset_index(drop=True)
    )

    # --------------------------------------------------------
    # Primary study scope.
    # This is not outlier filtering.
    # --------------------------------------------------------
    primary_mask = (
        np.isfinite(combined[TARGET_COL])
        & combined[TARGET_COL].between(
            SOH_LOWER_LIMIT,
            SOH_UPPER_LIMIT,
            inclusive="both",
        )
    )

    primary = combined.loc[primary_mask].copy()

    if primary.empty:
        raise RuntimeError("No constant observations remained in the 70-100% scope.")

    # --------------------------------------------------------
    # Freeze exact Step 1 strict-safe feature schema.
    # Missing columns are explicitly represented as NaN.
    # --------------------------------------------------------
    for feature in FROZEN_STRICT_CONSTANT_FEATURES:
        if feature not in primary.columns:
            primary[feature] = np.nan

    meta_cols = [
        GROUP_COL,
        SOURCE_COL,
        CYCLE_COL,
        TARGET_COL,
        "Measured_Discharge_Capacity_Ah",
        "Reference_Capacity_Ah",
        "Cycle_Age",
        "Log_Cycle_Age",
    ]

    primary = primary[
        meta_cols + FROZEN_STRICT_CONSTANT_FEATURES
    ].copy()

    target_audit = pd.concat(
        target_audits,
        ignore_index=True,
        sort=False,
    )

    return combined, primary, target_audit

def equal_battery_row_weights(frame):
    counts = frame[GROUP_COL].value_counts()

    w = frame[GROUP_COL].map(
        lambda b: 1.0 / counts.loc[b]
    ).astype(float).to_numpy()

    if np.mean(w) > 0:
        w = w / np.mean(w)

    return w

def weighted_age_linear_fit(train):
    """
    Equal-battery weighted OLS:
        SoH = intercept + slope * Cycle_Age
    """
    x = pd.to_numeric(
        train["Cycle_Age"],
        errors="coerce",
    ).to_numpy(dtype=float)

    y = pd.to_numeric(
        train[TARGET_COL],
        errors="coerce",
    ).to_numpy(dtype=float)

    w = equal_battery_row_weights(train)

    valid = (
        np.isfinite(x)
        & np.isfinite(y)
        & np.isfinite(w)
        & (w > 0)
    )

    x = x[valid]
    y = y[valid]
    w = w[valid]

    X = np.column_stack(
        [
            np.ones(len(x)),
            x,
        ]
    )

    sw = np.sqrt(w)

    beta, *_ = np.linalg.lstsq(
        X * sw[:, None],
        y * sw,
        rcond=None,
    )

    return float(beta[0]), float(beta[1])

def run_constant_trivial_baselines(primary):
    """
    LOBO trivial baselines for the corrected constant dataset.
    """
    perf_rows = []
    pred_rows = []

    batteries = sorted(primary[GROUP_COL].unique())

    for battery in batteries:
        train = primary.loc[
            primary[GROUP_COL] != battery
        ].copy()

        test = primary.loc[
            primary[GROUP_COL] == battery
        ].copy()

        y = test[TARGET_COL].to_numpy(dtype=float)

        # ----------------------------------------------------
        # Equal-battery training mean
        # ----------------------------------------------------
        battery_means = (
            train.groupby(GROUP_COL)[TARGET_COL]
            .mean()
        )

        train_mean = float(battery_means.mean())

        pred_mean = np.full(
            len(test),
            train_mean,
            dtype=float,
        )

        # ----------------------------------------------------
        # Age-only weighted linear regression
        # ----------------------------------------------------
        intercept, slope = weighted_age_linear_fit(train)

        age = test["Cycle_Age"].to_numpy(dtype=float)

        pred_age = intercept + slope * age
        pred_age_clip = np.clip(
            pred_age,
            SOH_LOWER_LIMIT,
            SOH_UPPER_LIMIT,
        )

        models = [
            ("EqualBatteryTrainMean", pred_mean),
            ("AgeOnlyLinear", pred_age),
            ("AgeOnlyLinear_Clipped70_100", pred_age_clip),
        ]

        for name, pred in models:
            rmse_value = float(
                np.sqrt(np.mean((y - pred) ** 2))
            )
            mae_value = float(
                np.mean(np.abs(y - pred))
            )

            perf_rows.append(
                {
                    "Baseline": name,
                    "Holdout_Battery": battery,
                    "Source": str(test[SOURCE_COL].iloc[0]),
                    "N_Cycle_Observations": int(len(test)),
                    "RMSE": rmse_value,
                    "MAE": mae_value,
                    "Actual_Min": float(np.min(y)),
                    "Actual_Max": float(np.max(y)),
                    "Pred_Min": float(np.min(pred)),
                    "Pred_Max": float(np.max(pred)),
                }
            )

            for (_, row), p in zip(test.iterrows(), pred):
                pred_rows.append(
                    {
                        "Baseline": name,
                        "Holdout_Battery": battery,
                        GROUP_COL: battery,
                        SOURCE_COL: row[SOURCE_COL],
                        CYCLE_COL: row[CYCLE_COL],
                        "Cycle_Age": row["Cycle_Age"],
                        "Actual_SOH": row[TARGET_COL],
                        "Predicted_SOH": float(p),
                    }
                )

        # ----------------------------------------------------
        # History-assisted previous-target persistence
        # ----------------------------------------------------
        ordered = test.sort_values(CYCLE_COL).reset_index(drop=True)

        if len(ordered) >= 2:
            actual = ordered[TARGET_COL].to_numpy(dtype=float)[1:]
            pred = ordered[TARGET_COL].to_numpy(dtype=float)[:-1]

            rmse_value = float(
                np.sqrt(np.mean((actual - pred) ** 2))
            )
            mae_value = float(
                np.mean(np.abs(actual - pred))
            )

            perf_rows.append(
                {
                    "Baseline": "PreviousTargetPersistence_HISTORY_ASSISTED",
                    "Holdout_Battery": battery,
                    "Source": str(test[SOURCE_COL].iloc[0]),
                    "N_Cycle_Observations": int(len(actual)),
                    "RMSE": rmse_value,
                    "MAE": mae_value,
                    "Actual_Min": float(np.min(actual)),
                    "Actual_Max": float(np.max(actual)),
                    "Pred_Min": float(np.min(pred)),
                    "Pred_Max": float(np.max(pred)),
                }
            )

            for idx in range(1, len(ordered)):
                pred_rows.append(
                    {
                        "Baseline": "PreviousTargetPersistence_HISTORY_ASSISTED",
                        "Holdout_Battery": battery,
                        GROUP_COL: battery,
                        SOURCE_COL: ordered.loc[idx, SOURCE_COL],
                        CYCLE_COL: ordered.loc[idx, CYCLE_COL],
                        "Cycle_Age": ordered.loc[idx, "Cycle_Age"],
                        "Actual_SOH": ordered.loc[idx, TARGET_COL],
                        "Predicted_SOH": float(
                            ordered.loc[idx - 1, TARGET_COL]
                        ),
                    }
                )

    perf = pd.DataFrame(perf_rows)
    preds = pd.DataFrame(pred_rows)

    macro = (
        perf.groupby("Baseline", as_index=False)
        .agg(
            Batteries=("Holdout_Battery", "nunique"),
            Macro_RMSE=("RMSE", "mean"),
            Macro_MAE=("MAE", "mean"),
        )
        .sort_values(
            ["Macro_RMSE", "Macro_MAE"],
            ascending=True,
        )
        .reset_index(drop=True)
    )

    return perf, macro, preds

def constant_feature_coverage(primary):
    rows = []

    for feature in FROZEN_STRICT_CONSTANT_FEATURES:
        values = pd.to_numeric(
            primary[feature],
            errors="coerce",
        )

        calce = primary[SOURCE_COL].eq("CALCE")
        nasa = primary[SOURCE_COL].eq("NASA")

        rows.append(
            {
                "Feature": feature,
                "Overall_Coverage": float(values.notna().mean()),
                "CALCE_Coverage": (
                    float(values.loc[calce].notna().mean())
                    if calce.any()
                    else np.nan
                ),
                "NASA_Coverage": (
                    float(values.loc[nasa].notna().mean())
                    if nasa.any()
                    else np.nan
                ),
                "N_Batteries_With_Any": int(
                    primary.assign(
                        _finite=values.notna()
                    )
                    .groupby(GROUP_COL)["_finite"]
                    .any()
                    .sum()
                ),
            }
        )

    return pd.DataFrame(rows)

def build_constant_dataset_summary(combined, primary):
    frames = []

    for label, frame in [
        ("ALL_RECONSTRUCTED_TARGETS", combined),
        ("PRIMARY_70_100_SCOPE", primary),
    ]:
        for (source, battery), g in frame.groupby(
            [SOURCE_COL, GROUP_COL],
            sort=True,
        ):
            target = pd.to_numeric(
                g[TARGET_COL],
                errors="coerce",
            )

            frames.append(
                {
                    "Dataset": label,
                    "Source": source,
                    "Battery": battery,
                    "Cycle_Observations": int(target.notna().sum()),
                    "Target_Min": float(target.min()),
                    "Target_Max": float(target.max()),
                    "Target_Median": float(target.median()),
                    "Cycle_Min": float(
                        pd.to_numeric(
                            g[CYCLE_COL],
                            errors="coerce",
                        ).min()
                    ),
                    "Cycle_Max": float(
                        pd.to_numeric(
                            g[CYCLE_COL],
                            errors="coerce",
                        ).max()
                    ),
                }
            )

    return pd.DataFrame(frames)

def add_run5e_protocol_normalized_cycle_features(df: pd.DataFrame) -> pd.DataFrame:
    """Target-free load-normalized features from proxy-safe cycle quantities."""
    out = df.copy()
    generated = {}
    def num(name):
        return pd.to_numeric(out[name], errors="coerce") if name in out.columns else pd.Series(np.nan, index=out.index)
    eps = 1e-9
    i_mean = num("Discharge_Current_mean").abs()
    i_std = num("Discharge_Current_std").abs()
    i_iqr = num("Discharge_Current_IQR").abs()
    i_mad = num("Discharge_Current_MAD").abs()
    dv = num("DeltaV_discharge").abs()
    dvdt = num("avg_abs_dvdt").abs()
    generated[R5E_NORM_PREFIX + "Current_CV"] = i_std / i_mean.where(i_mean > eps)
    generated[R5E_NORM_PREFIX + "Current_IQR_Rel"] = i_iqr / i_mean.where(i_mean > eps)
    generated[R5E_NORM_PREFIX + "Current_MAD_Rel"] = i_mad / i_mean.where(i_mean > eps)
    generated[R5E_NORM_PREFIX + "VoltageDrop_PerAbsCurrent"] = dv / i_mean.where(i_mean > eps)
    generated[R5E_NORM_PREFIX + "AbsDVdt_PerAbsCurrent"] = dvdt / i_mean.where(i_mean > eps)
    for high, low in [("3p8", "3p2"), ("3p6", "3p2"), ("3p6", "3p4")]:
        th, tl = num(f"Discharge_TimeTo_{high}V_h"), num(f"Discharge_TimeTo_{low}V_h")
        qh, ql = num(f"Discharge_CapacityTo_{high}V_Ah"), num(f"Discharge_CapacityTo_{low}V_Ah")
        eh, el = num(f"Discharge_EnergyTo_{high}V_Wh"), num(f"Discharge_EnergyTo_{low}V_Wh")
        generated[R5E_NORM_PREFIX + f"TimeRatio_{high}_to_{low}"] = th / tl.where(tl.abs() > eps)
        generated[R5E_NORM_PREFIX + f"CapacityRatio_{high}_to_{low}"] = qh / ql.where(ql.abs() > eps)
        generated[R5E_NORM_PREFIX + f"EnergyRatio_{high}_to_{low}"] = eh / el.where(el.abs() > eps)
    for level in ["4p0", "3p8", "3p6", "3p4", "3p2"]:
        q = num(f"Discharge_CapacityTo_{level}V_Ah")
        e = num(f"Discharge_EnergyTo_{level}V_Wh")
        generated[R5E_NORM_PREFIX + f"PartialAverageVoltageTo_{level}V"] = e / q.where(q.abs() > eps)
    return pd.concat([out, pd.DataFrame(generated, index=out.index).replace([np.inf, -np.inf], np.nan)], axis=1)

def run5e_source_features(cycle_table: pd.DataFrame) -> list[str]:
    safe = proxy_safe_candidates(cycle_table, "LOBO")
    normalized = [c for c in cycle_table.columns if c.startswith(R5E_NORM_PREFIX)]
    features = sorted(dict.fromkeys([*safe, *normalized]))
    if len(features) < 8:
        raise RuntimeError(f"Only {len(features)} proxy-safe cycle features available for Run 5E.")
    return features

def run5e_candidate_features(df: pd.DataFrame, formulation: str) -> list[str]:
    safe = proxy_safe_candidates(df, "LOBO")
    normalized = [c for c in df.columns if c.startswith(R5E_NORM_PREFIX)]
    cycle = [f for f in [*safe, *normalized] if not str(f).startswith(R5E_INTERVAL_PREFIX)]
    banned = {R5E_REF_EPOCH_COL, R5E_INTERVAL_POSITION_COL, R5E_INTERVAL_TOTAL_COL}
    cycle = [f for f in cycle if f not in banned]
    if formulation == "A_COVERAGE_SAFE_CYCLE_ONLY":
        return sorted(dict.fromkeys(cycle))
    if formulation == "B_COVERAGE_SAFE_HYBRID":
        interval = [c for c in df.columns if c.startswith(R5E_INTERVAL_PREFIX)]
        return sorted(dict.fromkeys([*cycle, *interval]))
    raise ValueError(f"Unknown Run 5E formulation={formulation}")

def load_nasa_rw_measurements_targetfree(csv_path, battery_id):
    """Return only raw random-walk discharge measurements. No SOH column is read."""
    raw = read_nasa_csv(csv_path, battery_id)
    op_col = find_column(raw, ["operation_class"])
    rw = raw.loc[
        raw[op_col].astype(str).str.strip().str.lower().eq("random_walk_discharge")
    ].copy()
    if rw.empty:
        raise ValueError(f"{battery_id}: no random_walk_discharge rows found.")

    cycle_col = find_column(rw, ["random_walk_cycle"])
    seg_col = find_column(rw, ["random_walk_segment", "source_step_index"])
    time_col = find_column(rw, ["relativeTime", "relative_time"])
    volt_col = find_column(rw, ["voltage", "Voltage"])
    curr_col = find_column(rw, ["current", "Current"])

    out = pd.DataFrame({
        CYCLE_COL: pd.to_numeric(rw[cycle_col], errors="coerce"),
        "Segment_Index": pd.to_numeric(rw[seg_col], errors="coerce"),
        "Test_Time(s)": pd.to_numeric(rw[time_col], errors="coerce"),
        "Voltage(V)": pd.to_numeric(rw[volt_col], errors="coerce"),
        "Current(A)": -np.abs(pd.to_numeric(rw[curr_col], errors="coerce")),
    })
    out = (
        out.dropna(subset=[CYCLE_COL, "Segment_Index", "Test_Time(s)", "Voltage(V)", "Current(A)"])
           .sort_values([CYCLE_COL, "Segment_Index", "Test_Time(s)"])
           .reset_index(drop=True)
    )
    if out.empty:
        raise ValueError(f"{battery_id}: no usable target-free random-walk samples.")
    return out

def calculate_calce_variable_reference_soh_unfiltered(
    df,
    battery_id,
    cycle_col=CYCLE_COL,
    time_col="Test_Time(s)",
    current_col="Current(A)",
    target_current_a=VARIABLE_REFERENCE_CURRENT_A,
    current_tolerance_a=VARIABLE_REFERENCE_CURRENT_TOLERANCE_A,
    discharge_threshold_a=VARIABLE_DISCHARGE_THRESHOLD_A,
    minimum_samples=VARIABLE_MIN_DISCHARGE_SAMPLES,
):
    """Reconstruct CALCE Type-3 reference SoH without target-value cleaning.

    Detection of the recurring 0.22 A reference discharge is based only on current/time.
    All finite positive capacities are retained. SoH >100 and SoH <70 are NOT treated as
    outliers; they are merely marked out-of-scope later if the 70-100 study window is used.
    """
    required = {cycle_col, time_col, current_col}
    missing = required - set(df.columns)
    if missing:
        raise KeyError(f"{battery_id}: missing columns for CALCE target reconstruction: {sorted(missing)}")

    w = df[[cycle_col, time_col, current_col]].copy()
    w[cycle_col] = pd.to_numeric(w[cycle_col], errors="coerce")
    w[time_col] = pd.to_numeric(w[time_col], errors="coerce")
    w[current_col] = pd.to_numeric(w[current_col], errors="coerce")
    w["_original_row"] = np.arange(len(w))
    w = w.dropna(subset=[cycle_col, time_col, current_col]).sort_values([cycle_col, time_col, "_original_row"])

    rows = []
    for cyc, g in w.groupby(cycle_col, sort=True):
        g = g.copy()
        g["_is_discharge"] = g[current_col] < -abs(discharge_threshold_a)
        g["_block_start"] = g["_is_discharge"] & ~g["_is_discharge"].shift(fill_value=False)
        g["_block"] = g["_block_start"].cumsum()
        candidates = []
        for block_id, b in g.loc[g["_is_discharge"]].groupby("_block", sort=True):
            if len(b) < minimum_samples:
                continue
            med_i = float(np.nanmedian(np.abs(pd.to_numeric(b[current_col], errors="coerce").to_numpy(float))))
            cap, dur = _integrate_discharge_block_capacity(b, time_col=time_col, current_col=current_col)
            candidates.append({
                "block": int(block_id), "samples": int(len(b)), "median_abs_current_a": med_i,
                "current_error_a": abs(med_i-target_current_a) if np.isfinite(med_i) else np.nan,
                "capacity_ah": cap, "duration_s": dur,
            })
        valid = [x for x in candidates if np.isfinite(x["capacity_ah"]) and x["capacity_ah"] > 0 and np.isfinite(x["median_abs_current_a"])]
        chosen = min(valid, key=lambda x: x["current_error_a"]) if valid else None
        if chosen is None:
            status = "no_valid_discharge_block"
            cap = np.nan
            med_i = np.nan
            err = np.nan
            block = np.nan
            ns = np.nan
            dur = np.nan
        elif chosen["current_error_a"] > current_tolerance_a:
            status = "closest_block_outside_current_tolerance"
            cap = np.nan
            med_i = chosen["median_abs_current_a"]
            err = chosen["current_error_a"]
            block = chosen["block"]
            ns = chosen["samples"]
            dur = chosen["duration_s"]
        else:
            status = "valid_0.22A_reference"
            cap = chosen["capacity_ah"]
            med_i = chosen["median_abs_current_a"]
            err = chosen["current_error_a"]
            block = chosen["block"]
            ns = chosen["samples"]
            dur = chosen["duration_s"]
        rows.append({
            GROUP_COL: battery_id, CYCLE_COL: float(cyc),
            "Reference_Capacity_Ah": cap, "Reference_Target_Current_A": target_current_a,
            "Reference_Median_Current_A": med_i, "Reference_Current_Error_A": err,
            "Selected_Block": block, "Selected_Samples": ns, "Selected_Duration_s": dur,
            "Selection_Status": status,
        })

    audit = pd.DataFrame(rows).sort_values(CYCLE_COL).reset_index(drop=True)
    valid = np.isfinite(audit["Reference_Capacity_Ah"]) & (audit["Reference_Capacity_Ah"] > 0)
    if not valid.any():
        raise RuntimeError(f"{battery_id}: no valid CALCE 0.22 A reference capacity.")
    first_idx = audit.index[valid][0]
    baseline = float(audit.loc[first_idx, "Reference_Capacity_Ah"])
    audit["Reconstructed_SOH_pct"] = np.nan
    audit.loc[valid, "Reconstructed_SOH_pct"] = audit.loc[valid, "Reference_Capacity_Ah"] / baseline * 100.0
    audit.loc[first_idx, "Reconstructed_SOH_pct"] = 100.0
    audit["Baseline_Reference_Capacity_Ah"] = baseline
    audit["In_70_100_Window"] = audit["Reconstructed_SOH_pct"].between(SOH_LOWER_LIMIT, SOH_UPPER_LIMIT, inclusive="both")
    audit["Target_Construction"] = "RAW_0p22A_REFERENCE_CURRENT_INTEGRATION_NO_TARGET_CLEANING"
    return audit

def integrate_reference_event_raw(part, time_col, current_col):
    t = pd.to_numeric(part[time_col], errors="coerce").to_numpy(float)
    i = np.abs(pd.to_numeric(part[current_col], errors="coerce").to_numpy(float))
    finite = np.isfinite(t) & np.isfinite(i)
    t, i = t[finite], i[finite]
    if len(t) < 2:
        return np.nan, 0, 0, len(t)
    dt = np.diff(t)
    imid = 0.5 * (i[:-1] + i[1:])
    ok = np.isfinite(dt) & np.isfinite(imid) & (dt > 0)
    cap = float(np.sum(imid[ok] * dt[ok]) / 3600.0) if ok.any() else np.nan
    return cap, int(ok.sum()), int((np.isfinite(dt) & (dt <= 0)).sum()), int(len(t))

def build_nasa_reconstructed_reference_targets(csv_path, battery_id):
    """Build raw-capacity reference targets and preceding-RW interval mapping.

    Neither SOH nor SOH_interpolated is used anywhere in this function.
    """
    raw = read_nasa_csv(csv_path, battery_id).reset_index(drop=True).copy()
    op_col = find_column(raw, ["operation_class"])
    cycle_col = find_column(raw, ["random_walk_cycle"])
    time_col = find_column(raw, ["relativeTime", "relative_time"])
    curr_col = find_column(raw, ["current", "Current"])
    volt_col = find_column(raw, ["voltage", "Voltage"], required=False)

    op = raw[op_col].astype(str).str.strip().str.lower()
    is_ref = op.str.contains("reference", regex=False) & op.str.contains("discharge", regex=False)
    is_rw = op.eq("random_walk_discharge")
    if not is_ref.any() or not is_rw.any():
        raise RuntimeError(f"{battery_id}: missing reference or RW operations.")

    event_start = is_ref & ~is_ref.shift(1, fill_value=False)
    event_id = event_start.cumsum().astype(int)
    ref_event_on_rows = event_id.where(is_ref)
    bounds = []
    for eid in sorted(pd.unique(ref_event_on_rows.dropna().astype(int))):
        idx = np.flatnonzero(ref_event_on_rows.eq(eid).to_numpy())
        bounds.append((int(eid), int(idx.min()), int(idx.max())))

    event_rows = []
    cycle_rows = []
    capacities = []
    for j, (eid, start, end) in enumerate(bounds):
        prev_end = bounds[j-1][2] if j > 0 else -1
        ref = raw.iloc[start:end+1].copy()
        cap, npos, nnonpos, nfinite = integrate_reference_event_raw(ref, time_col, curr_col)
        qc = bool(np.isfinite(cap) and cap > 0 and npos > 0 and nfinite >= 2)
        capacities.append(cap if qc else np.nan)
        preceding_mask = is_rw & (np.arange(len(raw)) > prev_end) & (np.arange(len(raw)) < start)
        preceding = raw.loc[preceding_mask, [cycle_col]].copy()
        preceding[cycle_col] = pd.to_numeric(preceding[cycle_col], errors="coerce")
        cycle_first = preceding.dropna().drop_duplicates(cycle_col, keep="first")
        cycle_first = cycle_first.sort_index()
        iid = f"{battery_id}__REF_{eid:04d}"
        for pos, cyc in enumerate(cycle_first[cycle_col].tolist(), start=1):
            cycle_rows.append({
                GROUP_COL: battery_id, CYCLE_COL: float(cyc),
                R5E_REF_EPOCH_COL: eid, R5E_INTERVAL_ID_COL: iid,
                R5E_INTERVAL_POSITION_COL: pos,
                R5E_INTERVAL_TOTAL_COL: int(len(cycle_first)),
            })
        event_rows.append({
            GROUP_COL: battery_id, R5E_REF_EPOCH_COL: eid, R5E_INTERVAL_ID_COL: iid,
            "Reference_Start_Row": start, "Reference_End_Row": end,
            "Reference_Raw_Rows": int(end-start+1), "Finite_Time_Current_Samples": nfinite,
            "Positive_DT_Intervals": npos, "Nonpositive_DT_Intervals": nnonpos,
            "Reference_Capacity_Ah": cap, "Raw_QC_Pass": qc,
            "PreReference_RW_Cycles": int(len(cycle_first)),
            "Reference_Start_V": float(pd.to_numeric(ref[volt_col], errors="coerce").iloc[0]) if volt_col and len(ref) else np.nan,
            "Reference_End_V": float(pd.to_numeric(ref[volt_col], errors="coerce").iloc[-1]) if volt_col and len(ref) else np.nan,
        })

    events = pd.DataFrame(event_rows)
    valid = events["Raw_QC_Pass"] & np.isfinite(events["Reference_Capacity_Ah"]) & (events["Reference_Capacity_Ah"] > 0)
    if not valid.any():
        raise RuntimeError(f"{battery_id}: no QC-valid NASA reference capacity.")
    baseline_idx = events.index[valid][0]
    baseline = float(events.loc[baseline_idx, "Reference_Capacity_Ah"])
    events["Reconstructed_SOH_pct"] = np.nan
    events.loc[valid, "Reconstructed_SOH_pct"] = events.loc[valid, "Reference_Capacity_Ah"] / baseline * 100.0
    events.loc[baseline_idx, "Reconstructed_SOH_pct"] = 100.0
    events["Baseline_Reference_Capacity_Ah"] = baseline
    events["In_70_100_Window"] = events["Reconstructed_SOH_pct"].between(SOH_LOWER_LIMIT, SOH_UPPER_LIMIT, inclusive="both")
    events["Target_Construction"] = "RAW_REFERENCE_DISCHARGE_CURRENT_INTEGRATION"

    cycle_map = pd.DataFrame(cycle_rows)
    if not cycle_map.empty:
        cycle_map = cycle_map.merge(
            events[[GROUP_COL, R5E_REF_EPOCH_COL, R5E_INTERVAL_ID_COL, "Reconstructed_SOH_pct", "Raw_QC_Pass", "In_70_100_Window"]],
            on=[GROUP_COL, R5E_REF_EPOCH_COL, R5E_INTERVAL_ID_COL], how="left", validate="many_to_one"
        )
    return events, cycle_map

def build_revised_variable_cycle_features(manifest):
    frames = []
    calce_target_audits = []
    nasa_event_audits = []
    nasa_maps = []

    # CALCE variable: build features for all cycles before any target-range restriction.
    for battery_id, sheets in CALCE_VARIABLE_BATTERIES:
        print(f"\n[CALCE] building target-independent features: {battery_id}")
        raw = process_files(str(manifest[battery_id]["folder"]), sheets)
        raw[GROUP_COL] = battery_id
        target_audit = calculate_calce_variable_reference_soh_unfiltered(raw, battery_id)
        f = build_cycle_features_run2(
            raw, cycle_col=CYCLE_COL, time_col="Test_Time(s)", volt_col="Voltage(V)", curr_col="Current(A)",
            step_col="Step_Index", segment_col=None,
        )
        f = add_dataset_identity(f, battery_id, "CALCE", VARIABLE_PROTOCOL)
        f = f.merge(
            target_audit[[CYCLE_COL, "Reconstructed_SOH_pct", "Reference_Capacity_Ah", "Selection_Status", "In_70_100_Window"]],
            on=CYCLE_COL, how="left", validate="one_to_one"
        )
        frames.append(f)
        calce_target_audits.append(target_audit)
        del raw
        gc.collect()

    # NASA variable: features are built only from random-walk measurements; target is reconstructed separately.
    for battery_id in NASA_VARIABLE_BATTERIES:
        print(f"\n[NASA] building target-free RW features: {battery_id}")
        rw = load_nasa_rw_measurements_targetfree(manifest[battery_id]["csv_path"], battery_id)
        f = build_cycle_features_run2(
            rw, cycle_col=CYCLE_COL, time_col="Test_Time(s)", volt_col="Voltage(V)", curr_col="Current(A)",
            step_col=None, segment_col="Segment_Index",
        )
        f = add_dataset_identity(f, battery_id, "NASA", VARIABLE_PROTOCOL)
        events, cmap = build_nasa_reconstructed_reference_targets(manifest[battery_id]["csv_path"], battery_id)
        frames.append(f)
        nasa_event_audits.append(events)
        nasa_maps.append(cmap)
        del rw
        gc.collect()

    combined = pd.concat(frames, ignore_index=True, sort=False)
    combined[CYCLE_COL] = pd.to_numeric(combined[CYCLE_COL], errors="coerce")
    combined = combined.sort_values([GROUP_COL, CYCLE_COL]).reset_index(drop=True)

    # Age/history transforms are constructed BEFORE target-range filtering.
    first_cycle = combined.groupby(GROUP_COL)[CYCLE_COL].transform("min")
    combined["Cycle_Age"] = (combined[CYCLE_COL] - first_cycle).clip(lower=0)
    combined["Log_Cycle_Age"] = np.log1p(combined["Cycle_Age"])
    combined["source_is_nasa"] = combined[SOURCE_COL].astype(str).str.upper().eq("NASA").astype(int)
    combined = add_legacy_history_features(combined)
    combined = add_run2_invariant_features(combined)
    combined = add_past_only_features(combined)
    combined = add_joint_legacy_aliases(combined)
    combined = add_run5e_protocol_normalized_cycle_features(combined)
    combined = combined.sort_values([GROUP_COL, CYCLE_COL]).reset_index(drop=True)

    return (
        combined,
        pd.concat(calce_target_audits, ignore_index=True),
        pd.concat(nasa_event_audits, ignore_index=True),
        pd.concat(nasa_maps, ignore_index=True),
    )

def build_independent_observation_table(cycle_features, calce_targets, nasa_events, nasa_map):
    missing = [f for f in STRICT_VARIABLE_FEATURES if f not in cycle_features.columns]
    if missing:
        raise RuntimeError(f"{len(missing)} frozen strict features are missing from rebuilt cycle table. First 20: {missing[:20]}")

    # Audit that no target column is accidentally in the feature list.
    forbidden_exact = {TARGET_COL, "SOH", "SOH_interpolated", "Reconstructed_SOH_pct", "Reference_Capacity_Ah"}
    leaked = [f for f in STRICT_VARIABLE_FEATURES if f in forbidden_exact or "SOH" in f.upper()]
    if leaked:
        raise RuntimeError(f"Target-named features found in strict allowlist: {leaked}")

    rows_meta = []
    rows_feat = []

    # CALCE: one macro-cycle = one independent reference observation.
    for battery_id, _sheets in CALCE_VARIABLE_BATTERIES:
        cf = cycle_features.loc[cycle_features[GROUP_COL].astype(str).eq(battery_id)].copy()
        ta = calce_targets.loc[calce_targets[GROUP_COL].astype(str).eq(battery_id)].copy()
        merged = cf.merge(
            ta[[CYCLE_COL, "Reconstructed_SOH_pct", "Reference_Capacity_Ah", "Selection_Status", "In_70_100_Window"]],
            on=CYCLE_COL, how="inner", suffixes=("", "_target"), validate="one_to_one"
        )
        # Keep only valid measured reference capacities; range filtering is explicitly a study-scope step.
        merged = merged.loc[np.isfinite(merged["Reconstructed_SOH_pct"])].copy()
        for _, r in merged.iterrows():
            iid = f"{battery_id}__CALCE_REF_{int(r[CYCLE_COL]):05d}"
            meta = {
                GROUP_COL: battery_id, SOURCE_COL: "CALCE", R5E_INTERVAL_ID_COL: iid,
                R5E_REF_EPOCH_COL: float(r[CYCLE_COL]), "Target_SOH": float(r["Reconstructed_SOH_pct"]),
                "Reference_Capacity_Ah": float(r["Reference_Capacity_Ah"]),
                "In_70_100_Window": bool(r["In_70_100_Window"]),
                "N_Cycles_In_Interval": 1, "Start_Cycle": float(r[CYCLE_COL]), "End_Cycle": float(r[CYCLE_COL]),
                "Age_Coord": float(r.get("Cycle_Age", np.nan)), "Log_Age_Coord": float(r.get("Log_Cycle_Age", np.nan)),
                "Target_Provenance": "RAW_0p22A_REFERENCE_CURRENT_INTEGRATION_NO_TARGET_CLEANING",
            }
            feat = {}
            for f in STRICT_VARIABLE_FEATURES:
                val = pd.to_numeric(pd.Series([r[f]]), errors="coerce").iloc[0]
                feat[f"MED__{f}"] = val
                feat[f"LAST__{f}"] = val
            rows_meta.append(meta); rows_feat.append(feat)

    # NASA: one reference event = one independent observation. Aggregate preceding RW cycles.
    nasa_cf = cycle_features.loc[cycle_features[SOURCE_COL].astype(str).eq("NASA")].copy()
    mapped = nasa_cf.merge(nasa_map, on=[GROUP_COL, CYCLE_COL], how="inner", validate="one_to_one")
    for (battery_id, iid), g in mapped.groupby([GROUP_COL, R5E_INTERVAL_ID_COL], sort=True):
        g = g.sort_values(R5E_INTERVAL_POSITION_COL)
        ev = nasa_events.loc[
            nasa_events[GROUP_COL].astype(str).eq(str(battery_id)) & nasa_events[R5E_INTERVAL_ID_COL].astype(str).eq(str(iid))
        ]
        if len(ev) != 1:
            raise RuntimeError(f"{battery_id} {iid}: expected exactly one reference-event row, found {len(ev)}")
        ev = ev.iloc[0]
        if not bool(ev["Raw_QC_Pass"]):
            continue
        if len(g) == 0 or not np.isfinite(ev["Reconstructed_SOH_pct"]):
            continue
        meta = {
            GROUP_COL: str(battery_id), SOURCE_COL: "NASA", R5E_INTERVAL_ID_COL: str(iid),
            R5E_REF_EPOCH_COL: int(ev[R5E_REF_EPOCH_COL]), "Target_SOH": float(ev["Reconstructed_SOH_pct"]),
            "Reference_Capacity_Ah": float(ev["Reference_Capacity_Ah"]),
            "In_70_100_Window": bool(ev["In_70_100_Window"]),
            "N_Cycles_In_Interval": int(len(g)), "Start_Cycle": float(g[CYCLE_COL].min()), "End_Cycle": float(g[CYCLE_COL].max()),
            "Age_Coord": float(pd.to_numeric(g["Cycle_Age"], errors="coerce").max()),
            "Log_Age_Coord": float(np.log1p(pd.to_numeric(g["Cycle_Age"], errors="coerce").max())),
            "Target_Provenance": "RAW_REFERENCE_DISCHARGE_CURRENT_INTEGRATION",
        }
        feat = {}
        for f in STRICT_VARIABLE_FEATURES:
            s = pd.to_numeric(g[f], errors="coerce")
            finite = s.dropna()
            feat[f"MED__{f}"] = float(finite.median()) if len(finite) else np.nan
            feat[f"LAST__{f}"] = float(finite.iloc[-1]) if len(finite) else np.nan
        rows_meta.append(meta); rows_feat.append(feat)

    meta_df = pd.DataFrame(rows_meta)
    feat_df = pd.DataFrame(rows_feat)
    table = pd.concat([meta_df.reset_index(drop=True), feat_df.reset_index(drop=True)], axis=1)
    table = table.sort_values([GROUP_COL, "Age_Coord", R5E_INTERVAL_ID_COL]).reset_index(drop=True)
    table["Observation_Index"] = table.groupby(GROUP_COL).cumcount().astype(int)
    table["Log_Observation_Index"] = np.log1p(table["Observation_Index"].astype(float))
    table["Study_Scope_Status"] = np.where(table["In_70_100_Window"], "IN_70_100_PRIMARY_SCOPE", "OUTSIDE_70_100_RETAINED_FOR_AUDIT")

    primary = table.loc[table["In_70_100_Window"]].copy().reset_index(drop=True)
    return table, primary

def _rmse(y, p):
    return float(np.sqrt(mean_squared_error(np.asarray(y, float), np.asarray(p, float))))

def _mae(y, p):
    return float(mean_absolute_error(np.asarray(y, float), np.asarray(p, float)))

def run_variable_lobo_trivial_baselines(primary):
    batteries = sorted(primary[GROUP_COL].astype(str).unique())
    pred_rows = []
    perf_rows = []

    for heldout in batteries:
        train = primary.loc[~primary[GROUP_COL].astype(str).eq(heldout)].copy()
        test = primary.loc[primary[GROUP_COL].astype(str).eq(heldout)].copy()
        if train.empty or test.empty:
            continue
        y_train = pd.to_numeric(train["Target_SOH"], errors="coerce").to_numpy(float)
        y_test = pd.to_numeric(test["Target_SOH"], errors="coerce").to_numpy(float)

        # Equal-battery mean: average the training battery means rather than pooling rows.
        train_battery_means = train.groupby(GROUP_COL)["Target_SOH"].mean()
        mean_pred = float(train_battery_means.mean())
        p_mean = np.full(len(test), mean_pred, dtype=float)

        # Age-only weighted linear regression. Equal battery -> equal total fitting weight.
        age_cols = ["Age_Coord", "Log_Age_Coord"]
        Xtr = train[age_cols].apply(pd.to_numeric, errors="coerce").to_numpy(float)
        Xte = test[age_cols].apply(pd.to_numeric, errors="coerce").to_numpy(float)
        good_tr = np.isfinite(Xtr).all(axis=1) & np.isfinite(y_train)
        good_te = np.isfinite(Xte).all(axis=1) & np.isfinite(y_test)
        if good_tr.sum() < 3 or not good_te.all():
            raise RuntimeError(f"{heldout}: insufficient finite age data for baseline.")
        lr = LinearRegression()
        lr.fit(Xtr[good_tr], y_train[good_tr], sample_weight=equal_battery_row_weights(train.loc[good_tr]))
        p_age = lr.predict(Xte)
        p_age_clip = np.clip(p_age, SOH_LOWER_LIMIT, SOH_UPPER_LIMIT)

        for model_name, pred in [
            ("EqualBatteryTrainMean", p_mean),
            ("AgeOnlyLinear", p_age),
            ("AgeOnlyLinear_Clipped70_100", p_age_clip),
        ]:
            perf_rows.append({
                "Baseline": model_name, "Holdout_Battery": heldout,
                "Source": str(test[SOURCE_COL].iloc[0]), "N_Independent_Observations": int(len(test)),
                "RMSE": _rmse(y_test, pred), "MAE": _mae(y_test, pred),
                "Actual_Min": float(np.min(y_test)), "Actual_Max": float(np.max(y_test)),
                "Pred_Min": float(np.min(pred)), "Pred_Max": float(np.max(pred)),
            })
            for rec, pp in zip(test.to_dict("records"), pred):
                pred_rows.append({
                    "Baseline": model_name, "Holdout_Battery": heldout,
                    GROUP_COL: rec[GROUP_COL], SOURCE_COL: rec[SOURCE_COL],
                    R5E_INTERVAL_ID_COL: rec[R5E_INTERVAL_ID_COL],
                    "Age_Coord": rec["Age_Coord"], "Actual_SOH": rec["Target_SOH"], "Predicted_SOH": float(pp),
                })

    # History-assisted persistence baseline: previous measured independent target in the SAME battery.
    for battery, g in primary.groupby(GROUP_COL, sort=True):
        g = g.sort_values(["Age_Coord", "Observation_Index"]).copy()
        g["Predicted_SOH"] = g["Target_SOH"].shift(1)
        gg = g.loc[np.isfinite(g["Predicted_SOH"])].copy()
        if gg.empty:
            continue
        perf_rows.append({
            "Baseline": "PreviousTargetPersistence_HISTORY_ASSISTED",
            "Holdout_Battery": str(battery), "Source": str(gg[SOURCE_COL].iloc[0]),
            "N_Independent_Observations": int(len(gg)),
            "RMSE": _rmse(gg["Target_SOH"], gg["Predicted_SOH"]),
            "MAE": _mae(gg["Target_SOH"], gg["Predicted_SOH"]),
            "Actual_Min": float(gg["Target_SOH"].min()), "Actual_Max": float(gg["Target_SOH"].max()),
            "Pred_Min": float(gg["Predicted_SOH"].min()), "Pred_Max": float(gg["Predicted_SOH"].max()),
        })
        for _, rec in gg.iterrows():
            pred_rows.append({
                "Baseline": "PreviousTargetPersistence_HISTORY_ASSISTED",
                "Holdout_Battery": str(battery), GROUP_COL: str(battery), SOURCE_COL: rec[SOURCE_COL],
                R5E_INTERVAL_ID_COL: rec[R5E_INTERVAL_ID_COL], "Age_Coord": rec["Age_Coord"],
                "Actual_SOH": float(rec["Target_SOH"]), "Predicted_SOH": float(rec["Predicted_SOH"]),
            })

    perf = pd.DataFrame(perf_rows)
    preds = pd.DataFrame(pred_rows)
    macro = (
        perf.groupby("Baseline", as_index=False)
            .agg(Batteries=("Holdout_Battery", "nunique"), Macro_RMSE=("RMSE", "mean"), Macro_MAE=("MAE", "mean"))
            .sort_values("Macro_RMSE")
            .reset_index(drop=True)
    )
    return perf, macro, preds

def feature_coverage_table(primary):
    feature_cols = [c for c in primary.columns if c.startswith("MED__") or c.startswith("LAST__")]
    rows = []
    for f in feature_cols:
        s = pd.to_numeric(primary[f], errors="coerce")
        rows.append({
            "Feature": f, "Finite_Coverage": float(s.notna().mean()),
            "CALCE_Coverage": float(pd.to_numeric(primary.loc[primary[SOURCE_COL].eq("CALCE"), f], errors="coerce").notna().mean()) if (primary[SOURCE_COL].eq("CALCE")).any() else np.nan,
            "NASA_Coverage": float(pd.to_numeric(primary.loc[primary[SOURCE_COL].eq("NASA"), f], errors="coerce").notna().mean()) if (primary[SOURCE_COL].eq("NASA")).any() else np.nan,
        })
    return pd.DataFrame(rows).sort_values(["Finite_Coverage", "Feature"], ascending=[False, True]).reset_index(drop=True)

def dataset_summary(table_all, primary):
    rows = []
    for label, df in [("ALL_RECONSTRUCTED_TARGETS", table_all), ("PRIMARY_70_100_SCOPE", primary)]:
        for (source, battery), g in df.groupby([SOURCE_COL, GROUP_COL], sort=True):
            rows.append({
                "Dataset": label, "Source": source, "Battery": battery,
                "Independent_Observations": int(len(g)),
                "Target_Min": float(g["Target_SOH"].min()), "Target_Max": float(g["Target_SOH"].max()),
                "Target_Median": float(g["Target_SOH"].median()),
                "Mean_Cycles_Per_Observation": float(g["N_Cycles_In_Interval"].mean()),
                "Min_Cycles_Per_Observation": int(g["N_Cycles_In_Interval"].min()),
                "Max_Cycles_Per_Observation": int(g["N_Cycles_In_Interval"].max()),
            })
    return pd.DataFrame(rows)

def prepare_raw_archive(raw_zip: str | Path, work_dir: str | Path):
    """Extract and validate the original combined CALCE/NASA archive.

    Parameters
    ----------
    raw_zip:
        Path to the user-supplied archive containing the original public
        CALCE and NASA data in the layout used by the validated experiments.
    work_dir:
        Scratch directory for extraction. It is replaced on each call.

    Returns
    -------
    manifest, validation_table
        The same manifest structure used by the validated preprocessing code.
    """
    raw_zip = Path(raw_zip).expanduser().resolve()
    work_dir = Path(work_dir).expanduser().resolve()
    if not raw_zip.is_file():
        raise FileNotFoundError(f"Raw data archive not found: {raw_zip}")
    if work_dir.exists():
        shutil.rmtree(work_dir)
    safe_extract_zip(raw_zip, work_dir)
    return validate_extracted_dataset(work_dir)
