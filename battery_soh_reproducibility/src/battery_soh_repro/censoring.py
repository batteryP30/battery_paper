"""Leakage-safe fixed elapsed-time and first-voltage-crossing discharge prefixes.

Only samples physically observed at or before the chosen boundary enter the
model. In particular, the 600-s setting never interpolates from future samples.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def select_discharge(cycle: pd.DataFrame, source: str, current_threshold: float = 0.02) -> pd.DataFrame:
    z = cycle.copy()
    for c in ('Test_Time(s)', 'Voltage(V)', 'Current(A)'):
        z[c] = pd.to_numeric(z[c], errors='coerce')
    z = z.dropna(subset=['Test_Time(s)', 'Voltage(V)', 'Current(A)'])
    neg = z['Current(A)'].lt(-current_threshold)
    if source.upper() == 'CALCE' and 'Step_Index' in z:
        steps = pd.to_numeric(z.Step_Index, errors='coerce')
        st7 = z.loc[neg & steps.eq(7)].copy()
        if len(st7) >= 2:
            z = st7
        else:
            z = z.loc[neg]
    else:
        z = z.loc[neg]
    return z.sort_values('Test_Time(s)').drop_duplicates(subset=['Test_Time(s)'], keep='first')


def censored_prefix(cycle: pd.DataFrame, source: str, *,
                    seconds: float | None = None, voltage: float | None = None,
                    min_samples: int = 10, max_gap_seconds: float = 120,
                    min_time_coverage: float = .90) -> tuple[pd.DataFrame | None, dict]:
    """Exactly one of ``seconds`` or ``voltage`` must be set.

    The voltage boundary is the FIRST observed downward threshold crossing; a
    crossing sample is included but no post-crossing measurement is accessed.
    """
    if (seconds is None) == (voltage is None):
        raise ValueError('Specify exactly one of seconds or voltage')
    dis = select_discharge(cycle, source)
    audit = {'Rule': 'fixed_seconds' if seconds is not None else 'first_voltage_crossing',
             'Boundary': seconds if seconds is not None else voltage,
             'Full_Discharge_Rows_AuditOnly': len(dis), 'Rows': 0,
             'Last_Observed_s': np.nan, 'Maximum_Gap_s': np.nan,
             'Status': 'NO_VALID_DISCHARGE'}
    if len(dis) < 2:
        return None, audit
    tt = dis['Test_Time(s)'].to_numpy(float)
    t = tt - tt[0]
    if seconds is not None:
        boundary = np.flatnonzero(t <= seconds + 1e-9)
        z = dis.iloc[boundary].copy()
    else:
        v = dis['Voltage(V)'].to_numpy(float)
        crosses = np.flatnonzero((v[:-1] > voltage) & (v[1:] <= voltage))
        if not len(crosses):
            audit['Status'] = 'NO_DOWNWARD_CROSSING'
            return None, audit
        z = dis.iloc[:int(crosses[0])+2].copy()
    if len(z) < min_samples:
        audit['Status'] = 'INSUFFICIENT_PREFIX_ROWS'; audit['Rows'] = len(z)
        return None, audit
    elapsed = z['Test_Time(s)'].to_numpy(float) - tt[0]
    gap = float(np.diff(elapsed).max()) if len(elapsed) > 1 else np.nan
    audit.update({'Rows': len(z), 'Last_Observed_s': float(elapsed[-1]),
                  'Maximum_Gap_s': gap})
    if gap > max_gap_seconds:
        audit['Status'] = 'LONG_SAMPLE_GAP'
        return None, audit
    if seconds is not None and elapsed[-1] < seconds * min_time_coverage:
        audit['Status'] = 'INCOMPLETE_FIXED_WINDOW'
        return None, audit
    audit['Status'] = 'VALID_PREFIX'
    return z, audit


def temporal_resample(discharge_prefix: pd.DataFrame, length: int = 96) -> np.ndarray:
    """Return normalized-progress [voltage,current,elapsed-time] sequence."""
    if length < 2: raise ValueError('Sequence length must be >=2')
    d = discharge_prefix.sort_values('Test_Time(s)').drop_duplicates('Test_Time(s)')
    if len(d) < 2: raise ValueError('At least two time samples are needed')
    t = d['Test_Time(s)'].to_numpy(float)
    elapsed = t - t[0]
    if elapsed[-1] <= 0 or not np.all(np.diff(t) > 0):
        raise ValueError('Nonincreasing discharge timestamps')
    x = elapsed / elapsed[-1]
    grid = np.linspace(0, 1, length)
    v = np.interp(grid, x, d['Voltage(V)'].to_numpy(float))
    i = np.interp(grid, x, d['Current(A)'].to_numpy(float))
    e = np.interp(grid, x, elapsed)
    out = np.stack([v, i, e], axis=-1).astype(np.float32)
    if not np.isfinite(out).all(): raise ValueError('Non-finite sequence')
    return out


def centered_segment_filter(frame: pd.DataFrame, window: int = 5) -> pd.DataFrame:
    """Five-sample boxcar within each cycle and tester step, not across boundaries."""
    if window < 1 or window % 2 != 1:
        raise ValueError('Window must be a positive odd number')
    out = frame.copy()
    keys = ['Cycle_Index'] + (['Step_Index'] if 'Step_Index' in out else [])
    for c in ('Voltage(V)', 'Current(A)'):
        if c in out:
            out[c] = out.groupby(keys, dropna=False, sort=False)[c].transform(
                lambda x: x.rolling(window, center=True, min_periods=1).mean())
    return out
