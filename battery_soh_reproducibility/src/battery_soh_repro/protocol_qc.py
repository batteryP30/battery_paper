"""Post hoc CALCE charging/discharge completeness audit.

Reimplements the verified tester-step screen on ALL raw constant-protocol
CALCE cycles without reading SoH or modelling residuals. A failing screen means
'completion not verifiable from the logged sequence', not a proven defective
battery or a physical charging failure.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

ACTIVE_CURRENT_A = 0.02
CV_VOLTAGE_RANGE_V = (4.15, 4.25)
CV_END_CURRENT_A = 0.06
DISCHARGE_END_V = 2.75


def audit_cycle(cycle: pd.DataFrame) -> tuple[dict, pd.DataFrame]:
    """Evaluate consecutive tester-step RUNS, not disjoint Step_Index groups.

    ``cycle`` has columns Test_Time(s), Current(A), Voltage(V), Step_Index.
    """
    required = ['Test_Time(s)', 'Current(A)', 'Voltage(V)', 'Step_Index']
    if not set(required).issubset(cycle):
        raise ValueError(f'Missing CALCE fields {set(required) - set(cycle.columns)}')
    z = cycle[required].apply(pd.to_numeric, errors='coerce')
    z = z.loc[np.isfinite(z['Test_Time(s)']) & np.isfinite(z['Current(A)'])]
    z = z.sort_values('Test_Time(s)', kind='stable').reset_index(drop=True)
    if len(z) < 2: raise ValueError('Insufficient time/current measurements for cycle')
    t = z['Test_Time(s)'].to_numpy(float)
    i = z['Current(A)'].to_numpy(float)
    v = z['Voltage(V)'].to_numpy(float)
    step = z['Step_Index'].to_numpy(float)
    dt = np.diff(t, prepend=t[0]); dt = np.where(np.isfinite(dt) & (dt > 0), dt, 0.)
    old_capacity = float(np.sum(np.maximum(-i, 0) * dt / 3600))
    changes = np.r_[True, (step[1:] != step[:-1]) |
                    ~np.isfinite(step[1:]) | ~np.isfinite(step[:-1])]
    run_id = np.cumsum(changes)
    runs = []
    for run in np.unique(run_id):
        ix = np.flatnonzero(run_id == run)
        ti, ci, vi = t[ix], i[ix], v[ix]
        positive = np.isfinite(ci) & (ci > ACTIVE_CURRENT_A)
        negative = np.isfinite(ci) & (ci < -ACTIVE_CURRENT_A)
        cp = ci[positive]
        median_positive_voltage = float(np.nanmedian(vi[positive])) if len(cp) else np.nan
        cv_like = bool(len(cp) >= 3 and np.isfinite(median_positive_voltage) and
                       CV_VOLTAGE_RANGE_V[0] <= median_positive_voltage <= CV_VOLTAGE_RANGE_V[1] and
                       cp[0] > cp[-1] + .005)
        step_num = int(step[ix[0]]) if np.isfinite(step[ix[0]]) else -1
        runs.append({'Run': int(run), 'Step_Index': step_num, 'Rows': len(ix),
                     'Positive_Charge_Rows': int(positive.sum()),
                     'Negative_Discharge_Rows': int(negative.sum()),
                     'First_Positive_Charge_A': float(cp[0]) if len(cp) else np.nan,
                     'Last_Positive_Charge_A': float(cp[-1]) if len(cp) else np.nan,
                     'Median_Positive_Charge_V': median_positive_voltage,
                     'Voltage_Start_V': float(vi[0]), 'Voltage_End_V': float(vi[-1]),
                     'Current_Median_A': float(np.nanmedian(ci)),
                     'Duration_s': float(ti[-1] - ti[0]),
                     'CV_Like_Shape': cv_like,
                     'CV_Taper_Screen': bool(cv_like and cp[-1] <= CV_END_CURRENT_A) if len(cp) else False})
    r = pd.DataFrame(runs)
    r4 = r[r.Step_Index == 4]
    positive4 = r4[r4.Positive_Charge_Rows >= 3]
    verified4 = positive4[positive4.CV_Taper_Screen]
    candidates = r[(r.Positive_Charge_Rows >= 3) & r.CV_Like_Shape]
    alternative = candidates[candidates.Step_Index != 4]
    r7 = r[(r.Step_Index == 7) & (r.Negative_Discharge_Rows >= 3)]
    end7 = float(r7.Voltage_End_V.iloc[-1]) if len(r7) else np.nan
    if len(verified4):
        cv = 'LOGGED_STEP4_CV_TAPER_CONSISTENT'
    elif not len(r4):
        cv = 'NO_STEP4_RUN'
    elif not len(positive4):
        cv = 'STEP4_NO_SUSTAINED_POSITIVE_CURRENT'
    else:
        cv = 'STEP4_PRESENT_BUT_CV_TAPER_NOT_VERIFIED'
    if cv != 'LOGGED_STEP4_CV_TAPER_CONSISTENT' and len(alternative):
        cv += '_ALTERNATIVE_CV_LIKE_RUN_FOUND'
    passed_d = bool(np.isfinite(end7) and end7 <= DISCHARGE_END_V)
    row = {'Raw_Rows': len(cycle), 'Usable_Rows': len(z),
           'Original_RightSample_Discharge_Ah': old_capacity,
           'Run_Sequence': ','.join(map(str, r.Step_Index.tolist())),
           'Step2_Runs': int((r.Step_Index == 2).sum()),
           'Step4_Runs': len(r4), 'Step7_Runs': int((r.Step_Index == 7).sum()),
           'Step4_Positive_Charge_Rows': int(r4.Positive_Charge_Rows.sum()),
           'Step4_End_Current_A': float(positive4.Last_Positive_Charge_A.iloc[-1]) if len(positive4) else np.nan,
           'CV_Interpretation': cv, 'Step7_Last_V': end7,
           'Step7_End_Cutoff_Screen': passed_d,
           'CV_Pass': cv == 'LOGGED_STEP4_CV_TAPER_CONSISTENT',
           'Discharge_Pass': passed_d,
           'Quality_Pass': (cv == 'LOGGED_STEP4_CV_TAPER_CONSISTENT') and passed_d}
    return row, r


def screen_battery(raw: pd.DataFrame, battery: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    if 'Cycle_Index' not in raw: raise ValueError('Cycle_Index missing')
    cycle_rows, run_rows = [], []
    for cycle, subset in raw.groupby('Cycle_Index', sort=True):
        row, runs = audit_cycle(subset)
        cyc = int(round(float(cycle)))
        if abs(float(cycle) - cyc) > 1e-6: raise ValueError('Nonintegral raw cycle index')
        cycle_rows.append({'battery_id': battery, 'Cycle_Index': cyc, **row})
        runs = runs.copy()
        runs.insert(0, 'Cycle_Index', cyc)
        runs.insert(0, 'battery_id', battery)
        run_rows.append(runs)
    result = pd.DataFrame(cycle_rows)
    if result.duplicated(['battery_id', 'Cycle_Index']).any():
        raise ValueError('Duplicate cycle keys')
    return result, pd.concat(run_rows, ignore_index=True)
