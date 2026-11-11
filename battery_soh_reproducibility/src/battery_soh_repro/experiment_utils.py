"""Statistical and data-integrity primitives shared by later manuscript experiments.

All paired error differences are computed on identical held-out battery/cycle
keys. Whole batteries, not individual cycles, are the unit of bootstrap sampling.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


def file_sha256(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def integer_cycles(values: pd.Series) -> pd.Series:
    a = pd.to_numeric(values, errors='coerce')
    if not a.notna().all() or not np.allclose(a, np.rint(a), rtol=0, atol=1e-6):
        raise ValueError('Missing or non-integral battery cycle keys')
    return a.round().astype('int64')


def ensure_unique(frame: pd.DataFrame, keys: list[str]) -> None:
    if frame.duplicated(keys).any():
        raise ValueError(f'Duplicate prediction/observation keys: {keys}')


def metric_by_battery(frame: pd.DataFrame, actual: str, predicted: str,
                      battery: str = 'battery_id') -> pd.DataFrame:
    records = []
    for b, group in frame.groupby(battery, sort=True):
        y = pd.to_numeric(group[actual], errors='raise').to_numpy(float)
        p = pd.to_numeric(group[predicted], errors='raise').to_numpy(float)
        if not len(y) or not np.isfinite(y).all() or not np.isfinite(p).all():
            raise ValueError(f'{b}: invalid numeric data for RMSE')
        e = y - p
        records.append({'battery_id': str(b), 'N': len(y),
                        'RMSE': float(np.sqrt(np.mean(e * e))),
                        'MAE': float(np.mean(np.abs(e)))})
    return pd.DataFrame(records)


def paired_battery_bootstrap(paired: pd.DataFrame, left: str, right: str,
                             reps: int = 20000, seed: int = 42) -> dict:
    """Negative differences mean the ``right`` method has lower RMSE."""
    if paired['battery_id'].duplicated().any() or not len(paired):
        raise ValueError('Requires one matched row per held-out battery')
    diff = paired[right].to_numpy(float) - paired[left].to_numpy(float)
    rng = np.random.default_rng(seed)
    means = diff[rng.integers(0, len(diff), size=(reps, len(diff)))].mean(axis=1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    return {'Batteries': len(diff), 'Mean_Paired_RMSE_Difference': float(diff.mean()),
            'CI95_Lower': float(lo), 'CI95_Upper': float(hi),
            'Bootstrap_Repetitions': reps, 'Bootstrap_Seed': seed}


def compare_frozen_predictions(reference: pd.DataFrame, new: pd.DataFrame,
                               keys: list[str], target: str,
                               reference_prediction: str = 'Prediction_Raw',
                               new_prediction: str = 'Prediction_Raw',
                               reps: int = 20000, seed: int = 42) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    left = reference[keys + [target, reference_prediction]].rename(
        columns={target: 'Target_Frozen', reference_prediction: 'Original_Prediction'})
    right = new[keys + [target, new_prediction]].rename(
        columns={target: 'Target_New', new_prediction: 'New_Prediction'})
    ensure_unique(left, keys); ensure_unique(right, keys)
    merged = right.merge(left, on=keys, how='left', indicator=True, validate='one_to_one')
    if not merged['_merge'].eq('both').all():
        raise ValueError('New predictions cannot all be matched to frozen predictions')
    if not np.allclose(merged.Target_Frozen, merged.Target_New, atol=1e-9, rtol=0):
        raise ValueError('Original and new targets differ; refusing invalid comparison')
    per = metric_by_battery(merged, 'Target_New', 'Original_Prediction').rename(
        columns={'RMSE': 'RMSE_Frozen', 'MAE': 'MAE_Frozen', 'N': 'N_Observations'})
    new_per = metric_by_battery(merged, 'Target_New', 'New_Prediction')
    per = per.merge(new_per[['battery_id', 'RMSE', 'MAE']], on='battery_id', validate='one_to_one').rename(
        columns={'RMSE': 'RMSE_New', 'MAE': 'MAE_New'})
    stat = paired_battery_bootstrap(per, 'RMSE_Frozen', 'RMSE_New', reps=reps, seed=seed)
    stat.update({'N_Observations': len(merged),
                 'RMSE_Frozen_EqualBattery': per.RMSE_Frozen.mean(),
                 'RMSE_New_EqualBattery': per.RMSE_New.mean()})
    return merged.drop(columns='_merge'), per, pd.DataFrame([stat])


def record_inputs(output_dir: Path, files: dict[str, Path], settings: dict) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    items = {name: {'path': str(p), 'sha256': file_sha256(p)} for name, p in files.items()}
    (output_dir / 'run_manifest.json').write_text(json.dumps({'inputs': items, 'settings': settings},
                                                              indent=2, default=str), encoding='utf-8')
