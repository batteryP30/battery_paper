#!/usr/bin/env python3
"""Step 11O: target-unrestricted constant-protocol evaluation.

Reuses the archived battery_soh_repro data reconstruction and strict HGB
train/validation code; does NOT select cycles by measured SoH. The main cohort
is every instrument-recorded constant-protocol cycle with a reconstructible
reference-capacity target and usable time/current observations, irrespective of
SoH. A second, OPTIONAL, post-hoc CALCE protocol-verified cohort can be trained
using its own causally rebuilt feature histories.

The method remains a calibrated laboratory, post-discharge diagnostic study,
NOT a validated in-vehicle or early prediction use case.

Run from an environment containing battery_soh_repro and the original
scripts/modeling/constant_strict_lobo.py (bundled in the Colab notebook).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import sys
import time
import warnings
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer

from battery_soh_repro.data_pipeline import (
    CALCE_CONSTANT_BATTERIES, FROZEN_STRICT_CONSTANT_FEATURES,
    TARGET_COL, GROUP_COL, SOURCE_COL, CYCLE_COL,
    add_legacy_history_features, add_run2_invariant_features,
    add_past_only_features, add_joint_legacy_aliases,
    build_revised_constant_cycle_table, prepare_raw_archive, process_files,
)
from battery_soh_repro.protocol_qc import audit_cycle
from battery_soh_repro.data_pipeline import weighted_age_linear_fit

# Older published data_pipeline.py refers to gc.collect() but does not import gc.
# Supply the standard-library module explicitly without altering its algorithms.
import gc
import battery_soh_repro.data_pipeline as _archived_pipeline
_archived_pipeline.gc = gc

MODEL_DIR = Path(__file__).resolve().parents[1] / 'modeling'  # repository-local scripts/modeling/
if not (MODEL_DIR / 'constant_strict_lobo.py').is_file():
    # Local installation alternative. Point --repo-root at the updated repository.
    alt = os.environ.get('BATTERY_SOH_MODEL_DIR')
    if alt:
        MODEL_DIR = Path(alt)
if not (MODEL_DIR / 'constant_strict_lobo.py').is_file():
    raise FileNotFoundError('Cannot locate original constant_strict_lobo.py; see --repo-root/README')
sys.path.insert(0, str(MODEL_DIR))
import constant_strict_lobo as strict_hgb

# Older archived strict HGB script references CLIP_LOW/CLIP_HIGH in an unused
# diagnostic return branch without defining them at module scope. Set the
# historical 70–100 display-only bounds here. ALL scores use raw predictions.
strict_hgb.CLIP_LOW = 70.0
strict_hgb.CLIP_HIGH = 100.0

VERSION = 'step11o_unrestricted_v1_2026_11_13'
KEYS = [GROUP_COL, CYCLE_COL]
EXPECTED_BATTERIES = {'CS2_33', 'CS2_34', 'CS2_35', 'CS2_36', 'CS2_37', 'CS2_38',
                      'B0005', 'B0006', 'B0007', 'B0018'}
PRIMARY_NAME = 'TARGET_UNRESTRICTED_RAW_QC'
SCREENED_NAME = 'TARGET_UNRESTRICTED_PROTOCOL_SCREENED_POSTHOC'
FORMULATIONS = [('STRICT_ELECTRICAL_HGB', False), ('STRICT_ELECTRICAL_PLUS_AGE_HGB', True)]
BASELINES = ['TRAINING_MEAN', 'AGE_ONLY_LINEAR', 'AGE_ONLY_HGB']


def log(*parts):
    print('[' + time.strftime('%Y-%m-%d %H:%M:%S') + ']', *parts, flush=True)


def sha256(path: Path):
    digest = hashlib.sha256()
    with open(path, 'rb') as fh:
        for chunk in iter(lambda: fh.read(4 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def integer_cycle(s):
    x = pd.to_numeric(s, errors='coerce')
    if x.isna().any() or not np.allclose(x, np.rint(x), atol=1e-6, rtol=0):
        raise ValueError('Cycle_Index contains missing or non-integral keys.')
    return x.round().astype('int64')


def check_keys(df: pd.DataFrame, context: str):
    if not set(KEYS).issubset(df.columns):
        raise ValueError(f'{context}: required {KEYS} are missing')
    if df.duplicated(KEYS).any():
        raise ValueError(f'{context}: duplicate battery/cycle keys')
    if df[GROUP_COL].isna().any():
        raise ValueError(f'{context}: missing battery id')


def atomic_csv(frame: pd.DataFrame, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.tmp')
    frame.to_csv(tmp, index=False)
    os.replace(tmp, path)


def atomic_json(obj, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(obj, indent=2, sort_keys=True, default=str))
    os.replace(tmp, path)


def manuscript_reconstruction_gate(candidate: pd.DataFrame, reference_path: Path):
    """Check archived original targets ONLY; never use them as selection criteria."""
    if not reference_path.exists():
        raise FileNotFoundError(f'Archived target gate file not found: {reference_path}')
    ref = pd.read_csv(reference_path, low_memory=False)
    ref[CYCLE_COL] = integer_cycle(ref[CYCLE_COL]); check_keys(ref, 'archived reference')
    subset = candidate.copy(); subset[CYCLE_COL] = integer_cycle(subset[CYCLE_COL])
    check_keys(subset, 'new reconstructed full population')
    cmp = ref.merge(subset, on=KEYS, how='left', validate='one_to_one',
                    suffixes=('_archived', '_unrestricted'), indicator=True)
    if not cmp['_merge'].eq('both').all():
        bad = cmp.loc[cmp['_merge']!='both', KEYS].head(10).to_dict('records')
        raise RuntimeError(f'Archived 4590 cohort keys not recovered from raw data: {bad}')
    rows = []
    for col in [TARGET_COL, 'Measured_Discharge_Capacity_Ah', 'Reference_Capacity_Ah']:
        left = pd.to_numeric(cmp[col + '_archived'], errors='coerce').to_numpy(float)
        right = pd.to_numeric(cmp[col + '_unrestricted'], errors='coerce').to_numpy(float)
        eq = np.isclose(left, right, rtol=0, atol=1e-7, equal_nan=True)
        rows.append({'Checked_Column': col, 'N_Compared': len(eq),
                     'Mismatches': int((~eq).sum()), 'Max_Abs_Difference': float(np.nanmax(np.abs(left-right)))})
        if not eq.all():
            raise RuntimeError(f'Original target reconstruction FAILED for {col}: {(~eq).sum()} differences')
    if len(ref) != 4590 or int(ref[GROUP_COL].nunique()) != 10:
        raise RuntimeError('Reference gate is not the archived 4590-row, ten-cell source')
    return pd.DataFrame(rows)


def categorize_true_soh(s):
    y = pd.to_numeric(s, errors='coerce')
    conditions = [y < 50, (y >= 50) & (y < 70), (y >= 70) & (y <= 100), y > 100]
    return np.select(conditions, ['Below50', '50toBelow70', '70to100', 'Above100'],
                     default='No_Reference_Target').astype(str)


def strict_cols(frame: pd.DataFrame):
    for col in FROZEN_STRICT_CONSTANT_FEATURES:
        if col not in frame:
            frame[col] = np.nan
    return frame


def available_input_flags(audit: pd.DataFrame):
    """Input availability is judged without target SoH or its 70–100 range.

    Target availability (positive reconstructible discharge Q) is recorded
    separately: it is needed for *scoring*, not to define the input cohort.
    """
    x = audit.copy()
    x[CYCLE_COL] = integer_cycle(x[CYCLE_COL]); check_keys(x, 'raw target audit')
    samples = pd.to_numeric(x['N_Raw_Samples'], errors='coerce')
    duration = pd.to_numeric(x['Duration_s'], errors='coerce')
    x['Input_Record_Eligible'] = samples.ge(2) & duration.gt(0) & np.isfinite(duration)
    # Ground truth may be unavailable even when an input record is present.
    q = pd.to_numeric(x['Measured_Discharge_Capacity_Ah'], errors='coerce')
    qr = pd.to_numeric(x['Reference_Capacity_Ah'], errors='coerce')
    x['Reference_Target_Available'] = q.gt(0) & np.isfinite(q) & qr.gt(0) & np.isfinite(qr)
    x['Input_Cohort_But_Unscorable'] = x.Input_Record_Eligible & ~x.Reference_Target_Available
    x['Evaluable_Reference_Cycle'] = x.Input_Record_Eligible & x.Reference_Target_Available
    x['Observed_SoH_Stratum_PostEvaluation'] = categorize_true_soh(x[TARGET_COL])
    x['In_Original_70_100_Scope_PostEvaluation'] = pd.to_numeric(
        x[TARGET_COL], errors='coerce').between(70, 100, inclusive='both')
    return x


def obtain_protocol_qc(manifest, out_dir: Path):
    """Audit all raw CALCE cycles. Exceptions remain explicitly unverified."""
    rows = []
    for bid, sheets in CALCE_CONSTANT_BATTERIES:
        log('Auditing CALCE charging/discharge steps:', bid)
        raw = process_files(str(manifest[bid]['folder']), sheets)
        raw[CYCLE_COL] = pd.to_numeric(raw[CYCLE_COL], errors='coerce')
        for cyc, z in raw.groupby(CYCLE_COL, sort=True):
            if not np.isfinite(cyc):
                continue
            try:
                diag, _ = audit_cycle(z)
                record = {'battery_id': bid, CYCLE_COL: int(round(float(cyc))), **diag,
                          'Protocol_Screen_Exception': ''}
            except (ValueError, IndexError, TypeError) as exc:
                record = {'battery_id': bid, CYCLE_COL: int(round(float(cyc))),
                          'Quality_Pass': False, 'Protocol_Screen_Exception': str(exc),
                          'CV_Interpretation': 'UNVERIFIABLE_RAW_RECORD',
                          'Step7_End_Cutoff_Screen': False}
            rows.append(record)
        del raw
    qc = pd.DataFrame(rows)
    qc[CYCLE_COL] = integer_cycle(qc[CYCLE_COL]); check_keys(qc, 'protocol QC')
    qc['Quality_Pass'] = qc['Quality_Pass'].fillna(False).astype(bool)
    atomic_csv(qc, out_dir/'CALCE_Protocol_Screen_AllRaw.csv')
    return qc


def recompute_filtered_histories(combined: pd.DataFrame, eligible_ids: pd.MultiIndex):
    """Rebuild every causal/history feature after removing failed CALCE records.

    The first measured reference Q and original absolute Cycle_Age coordinate
    remain unchanged; discarded rows cannot contribute to trailing histories.
    """
    z = combined.set_index(KEYS, drop=False)
    if not eligible_ids.isin(z.index).all():
        raise ValueError('Protocol screened keys missing from full reconstructed history')
    reduced = z.loc[eligible_ids].copy().reset_index(drop=True)
    # Drop all previously generated lag/invariant/alias columns before rebuilding.
    prebuilt = [c for c in reduced.columns if c.startswith(('INV__', 'LEGACY__', 'LEGACYH__'))]
    reduced = reduced.drop(columns=prebuilt)
    # Non-feature metadata and base measurement features remain unchanged.
    for fn in [add_legacy_history_features, add_run2_invariant_features,
               add_past_only_features, add_joint_legacy_aliases]:
        reduced = fn(reduced)
    reduced = strict_cols(reduced)
    return reduced


def prepare_data(raw_zip: Path, work_dir: Path, output_dir: Path,
                 archived_reference: Path, apply_protocol_qc: bool,
                 raw_sha: str):
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest, validation = prepare_raw_archive(raw_zip, work_dir)
    log('Reconstructing targets and feature histories, without SoH range filtering')
    combined, primary_old, raw_audit = build_revised_constant_cycle_table(manifest)
    combined[CYCLE_COL] = integer_cycle(combined[CYCLE_COL])
    raw_audit[CYCLE_COL] = integer_cycle(raw_audit[CYCLE_COL])
    check_keys(combined, 'combined full-cycle features')
    audit = available_input_flags(raw_audit)
    both = audit[['battery_id',CYCLE_COL,'Input_Record_Eligible','Reference_Target_Available',
                  'Evaluable_Reference_Cycle','In_Original_70_100_Scope_PostEvaluation']]
    combined_check = combined.merge(both, on=KEYS, how='left', validate='one_to_one')
    if combined_check['Evaluable_Reference_Cycle'].isna().any():
        raise RuntimeError('Full feature table contains cycles absent from raw target audit')
    sel = combined_check['Evaluable_Reference_Cycle'].eq(True)
    full = combined_check.loc[sel].copy()
    full = strict_cols(full)
    full = full[[GROUP_COL,SOURCE_COL,CYCLE_COL,TARGET_COL,
                 'Measured_Discharge_Capacity_Ah','Reference_Capacity_Ah',
                 'Cycle_Age','Log_Cycle_Age'] + list(FROZEN_STRICT_CONSTANT_FEATURES)]
    check_keys(full, 'target unrestricted scored population')
    if set(full[GROUP_COL]) != EXPECTED_BATTERIES:
        raise RuntimeError(f'Unexpected battery set: {set(full[GROUP_COL])}')
    if not np.isfinite(pd.to_numeric(full[TARGET_COL], errors='coerce')).all():
        raise RuntimeError('Target unrestricted population still has invalid labels')
    if (full[TARGET_COL]<70).sum() == 0:
        raise RuntimeError('No below-70 SoH observations found. Target restriction may still be active.')
    # Scope gate verifies unchanged historical target construction, but is never
    # used to select full-cohort training or testing observations.
    gate = manuscript_reconstruction_gate(full, archived_reference)
    atomic_csv(gate, output_dir/'Original4590_Target_Reconstruction_Gate.csv')
    atomic_csv(audit, output_dir/'Raw_Eligibility_and_Label_Audit.csv')
    atomic_csv(validation, output_dir/'Raw_Archive_Validation.csv')
    atomic_csv(full, output_dir/'Unrestricted_All_QC_Valid_Cycles.csv')
    per = (audit.groupby([GROUP_COL,SOURCE_COL],as_index=False)
           .agg(Raw_Cycles=(CYCLE_COL,'size'),Input_Eligible=('Input_Record_Eligible','sum'),
                With_Reference_Target=('Reference_Target_Available','sum'),
                Evaluable=('Evaluable_Reference_Cycle','sum'),
                Input_Eligible_But_Unscorable=('Input_Cohort_But_Unscorable','sum'),
                In_Original_Scope=('In_Original_70_100_Scope_PostEvaluation','sum')))
    atomic_csv(per, output_dir/'Population_Coverage_By_Battery.csv')
    log('Unrestricted labelled study population:',len(full),'observations across',full[GROUP_COL].nunique(),'batteries')
    log('Below 70%:',int((full[TARGET_COL]<70).sum()),'| 70–100%:',int(full[TARGET_COL].between(70,100).sum()),
        '| Above 100%:',int((full[TARGET_COL]>100).sum()))
    if len(full) != 6299:
        warnings.warn(f'Expected archived raw-QC total 6299; found {len(full)}. Review raw QA before interpreting.')
    # A verified CALCE protocol cohort is optional and post hoc, not mistaken for
    # a pre-existing eligibility definition or an early-time deployment filter.
    if apply_protocol_qc:
        qc = obtain_protocol_qc(manifest, output_dir)
        is_calce = audit[SOURCE_COL].eq('CALCE')
        x = audit.merge(qc[[GROUP_COL,CYCLE_COL,'Quality_Pass','CV_Interpretation',
                            'Step7_End_Cutoff_Screen','Protocol_Screen_Exception']],
                        on=KEYS, how='left', validate='one_to_one')
        if x.loc[is_calce, 'Quality_Pass'].isna().any():
            raise RuntimeError('CALCE raw cycle missing from protocol screen')
        x['TargetFree_Protocol_Eligible'] = np.where(x[SOURCE_COL].eq('NASA'),
                                                      x['Input_Record_Eligible'],
                                                      x['Input_Record_Eligible'] & x['Quality_Pass'].fillna(False))
        atomic_csv(x, output_dir/'Eligibility_With_Protocol_Diagnostics.csv')
        input_keys = x.loc[x.TargetFree_Protocol_Eligible & x.Reference_Target_Available, KEYS]
        index = pd.MultiIndex.from_frame(input_keys)
        filtered = recompute_filtered_histories(combined, index)
        filtered = filtered[[GROUP_COL,SOURCE_COL,CYCLE_COL,TARGET_COL,
                             'Measured_Discharge_Capacity_Ah','Reference_Capacity_Ah',
                             'Cycle_Age','Log_Cycle_Age'] + list(FROZEN_STRICT_CONSTANT_FEATURES)]
        if len(filtered) >= len(full):
            warnings.warn('Protocol screen removed no records. Check step interpretation.')
        if not np.allclose(filtered[TARGET_COL].to_numpy(float),
                           filtered.merge(full[KEYS+[TARGET_COL]],on=KEYS,validate='one_to_one',
                                          suffixes=('_q','_full'))[TARGET_COL+'_full'].to_numpy(float),
                           atol=1e-9,rtol=0):
            raise RuntimeError('Protocol sensitivity unexpectedly changed targets')
        atomic_csv(filtered, output_dir/'Unrestricted_Protocol_Screened_PostHoc.csv')
        log('Post-hoc protocol verified labelled population:',len(filtered))
    # Mark this prep version tied to the raw archive and reference.
    atomic_json({'version':VERSION,'raw_zip_sha256':raw_sha,
                 'archived_reference_sha256':sha256(archived_reference),
                 'protocol_audit_generated':bool(apply_protocol_qc),
                 'all_eligible_rows':len(full)}, output_dir/'PREPARED.json')
    return full


def valid_cached_preparation(output_dir: Path, raw_sha: str, reference_sha: str, need_qc: bool):
    path = output_dir/'PREPARED.json'
    if not path.is_file() or not (output_dir/'Unrestricted_All_QC_Valid_Cycles.csv').exists():
        return False
    try:
        z = json.loads(path.read_text())
        return (z.get('version') == VERSION and z.get('raw_zip_sha256')==raw_sha
                and z.get('archived_reference_sha256')==reference_sha
                and (not need_qc or z.get('protocol_audit_generated') is True)
                and (not need_qc or (output_dir/'Unrestricted_Protocol_Screened_PostHoc.csv').exists()))
    except (ValueError, OSError):
        return False


def load_cohort(path: Path):
    frame = pd.read_csv(path,low_memory=False)
    frame[CYCLE_COL] = integer_cycle(frame[CYCLE_COL])
    check_keys(frame, 'model cohort')
    if set(frame[GROUP_COL].unique()) != EXPECTED_BATTERIES:
        raise RuntimeError('A battery is missing; cannot run 10-battery LOBO')
    if frame[TARGET_COL].isna().any() or (pd.to_numeric(frame[TARGET_COL],errors='coerce') <= 0).any():
        raise RuntimeError('Model cohort must contain finite, positive reference targets')
    for c in FROZEN_STRICT_CONSTANT_FEATURES:
        if c not in frame: frame[c] = np.nan
    return frame


def predict_baselines(train: pd.DataFrame, test: pd.DataFrame):
    y = pd.to_numeric(train[TARGET_COL],errors='coerce').to_numpy(float)
    cell_mean = train.groupby(GROUP_COL)[TARGET_COL].mean()
    mean = float(cell_mean.mean())
    intercept,slope = weighted_age_linear_fit(train)
    age_test = pd.to_numeric(test['Cycle_Age'],errors='coerce').to_numpy(float)
    return {'TRAINING_MEAN':np.full(len(test),mean,dtype=float),
            'AGE_ONLY_LINEAR': intercept + slope * age_test}


def train_age_hgb(train: pd.DataFrame, test: pd.DataFrame):
    inner = []
    for config in strict_hgb.HGB_CONFIGS:
        errs=[]; maes=[]
        for b in sorted(train[GROUP_COL].unique()):
            tr = train.loc[train[GROUP_COL]!=b]
            va = train.loc[train[GROUP_COL]==b]
            estimator, imp, cols = strict_hgb.fit_hgb(tr, [], config, True)
            pred, _ = strict_hgb.predict_hgb(estimator, imp, cols, va)
            errs.append(float(np.sqrt(np.mean((va[TARGET_COL].to_numpy(float) - pred)**2))))
            maes.append(float(np.mean(np.abs(va[TARGET_COL].to_numpy(float) - pred))))
        inner.append({'HGB_Config':config,'Inner_Mean_Battery_RMSE':float(np.mean(errs)),
                      'Inner_Mean_Battery_MAE':float(np.mean(maes))})
    tune=pd.DataFrame(inner).sort_values(['Inner_Mean_Battery_RMSE','HGB_Config'],kind='stable')
    best=str(tune.iloc[0]['HGB_Config'])
    est, imp, cols = strict_hgb.fit_hgb(train,[],best,True)
    pred,_=strict_hgb.predict_hgb(est,imp,cols,test)
    return pred,best,tune


def subset_signature(dataset_sha: str, name: str, formulation: str, battery: str):
    payload=json.dumps({'version':VERSION,'dataset_sha':dataset_sha,'cohort':name,
                        'formulation':formulation,'holdout':battery,
                        'strict_cfg':strict_hgb.HGB_CONFIGS,
                        'corr':strict_hgb.CORR_THRESHOLDS,'topk':strict_hgb.TOP_K_OPTIONS},sort_keys=True)
    return hashlib.sha256(payload.encode()).hexdigest()


def one_outer_fold(frame: pd.DataFrame, battery: str, cohort_name: str, dataset_sha: str,
                   output_dir: Path):
    train=frame.loc[frame[GROUP_COL]!=battery].copy()
    test=frame.loc[frame[GROUP_COL]==battery].copy()
    if len(test)<2: raise RuntimeError(f'{battery}: fewer than two held-out cycles')
    if len(train[GROUP_COL].unique()) != 9: raise RuntimeError(f'{battery}: must have 9 training batteries')
    p_rows=[]; f_rows=[]; selected_rows=[]; tuning_rows=[]
    base=predict_baselines(train,test)
    age_pred,age_conf,age_tune = train_age_hgb(train,test)
    base['AGE_ONLY_HGB']=age_pred
    age_tune=age_tune.copy();age_tune.insert(0,'Holdout_Battery',battery)
    age_tune.insert(0,'Formulation','AGE_ONLY_HGB')
    tuning_rows.append(age_tune)
    for name, pred in base.items():
        for i, yp in enumerate(pred):
            r = test.iloc[i]
            p_rows.append({GROUP_COL: battery, SOURCE_COL: str(r[SOURCE_COL]),
                           CYCLE_COL: int(r[CYCLE_COL]), TARGET_COL: float(r[TARGET_COL]),
                           'Formulation': name, 'Prediction_Raw': float(yp)})
    f_rows += [{'Holdout_Battery':battery,'Formulation':'AGE_ONLY_HGB',
                'Chosen_Config':age_conf,'N_Test':len(test)}]
    # The archived strict HGB procedure does training-only inner split feature
    # selection, HGB configuration tuning, and battery-balanced final fitting.
    candidate=[c for c in FROZEN_STRICT_CONSTANT_FEATURES if c not in strict_hgb.EXPLICIT_AGE_STRICT_FEATURES]
    for form,with_age in FORMULATIONS:
        log('Outer fold',battery,':',form)
        corr,k,feature_tuning,inner=strict_hgb.tune_feature_selection(train,candidate,with_age)
        chosen,model_tuning = strict_hgb.tune_hgb_config(inner,corr,k,with_age)
        prep=strict_hgb.prepare_feature_selector(train,candidate)
        selected=strict_hgb.choose_features(prep,k,corr)
        model,imp,cols=strict_hgb.fit_hgb(train,selected,chosen,with_age)
        pred,_=strict_hgb.predict_hgb(model,imp,cols,test)
        for i,yp in enumerate(pred):
            r=test.iloc[i]
            p_rows.append({GROUP_COL:battery,SOURCE_COL:str(r[SOURCE_COL]),CYCLE_COL:int(r[CYCLE_COL]),
                           TARGET_COL:float(r[TARGET_COL]),'Formulation':form,'Prediction_Raw':float(yp)})
        f_rows.append({'Holdout_Battery':battery,'Formulation':form,'N_Test':len(test),
                       'Chosen_Corr':corr,'Chosen_TopK':k,'Chosen_HGB_Config':chosen,
                       'N_Selected':len(selected),'Selected_Features':' | '.join(selected)})
        for rank,col in enumerate(selected,start=1):
            selected_rows.append({'Holdout_Battery':battery,'Formulation':form,'Rank':rank,
                                  'Feature':col,'Abs_Training_Spearman':float(prep['target_scores'].loc[col])})
        ft=feature_tuning.copy();ft.insert(0,'Holdout_Battery',battery)
        ft.insert(0,'Formulation',form); ft['Stage']='FeatureSelection';tuning_rows.append(ft)
        ht=model_tuning.copy();ht.insert(0,'Holdout_Battery',battery)
        ht.insert(0,'Formulation',form);ht['Stage']='HGBModel';tuning_rows.append(ht)
    paths={
        'pred':output_dir/f'pred_{battery}.csv',
        'fold':output_dir/f'config_{battery}.csv',
        'selected':output_dir/f'selected_{battery}.csv',
        'tuning':output_dir/f'tuning_{battery}.csv'
    }
    atomic_csv(pd.DataFrame(p_rows),paths['pred'])
    atomic_csv(pd.DataFrame(f_rows),paths['fold'])
    atomic_csv(pd.DataFrame(selected_rows),paths['selected'])
    atomic_csv(pd.concat(tuning_rows,ignore_index=True,sort=False),paths['tuning'])
    atomic_json({'signature':subset_signature(dataset_sha,cohort_name,'all_models',battery),
                 'test_rows':len(test),'predictions':len(p_rows),'version':VERSION},
                output_dir/f'done_{battery}.json')
    log('Completed',battery,'test cycles:',len(test))


def train_models(cohort_file: Path, cohort_name: str, output_dir: Path):
    frame=load_cohort(cohort_file)
    output_dir.mkdir(parents=True,exist_ok=True)
    dataset_sha=sha256(cohort_file)
    for battery in sorted(EXPECTED_BATTERIES):
        meta=output_dir/f'done_{battery}.json'
        expected=subset_signature(dataset_sha,cohort_name,'all_models',battery)
        required=[output_dir/f'{a}_{battery}.csv' for a in ['pred','config','selected','tuning']]
        if meta.is_file() and all(x.is_file() for x in required):
            z=json.loads(meta.read_text())
            if z.get('signature')==expected:
                log('Reusing checkpoint:',battery)
                continue
        one_outer_fold(frame,battery,cohort_name,dataset_sha,output_dir)
    log('All 10 held-out battery folds complete for:',cohort_name)


def bootstrap_battery_deltas(per_battery: pd.DataFrame, seed: int, reps: int):
    rng=np.random.default_rng(seed)
    results=[]
    piv=per_battery.pivot(index='battery_id',columns='Formulation',values='RMSE')
    comparisons=[('STRICT_ELECTRICAL_HGB','AGE_ONLY_HGB'),
                 ('STRICT_ELECTRICAL_PLUS_AGE_HGB','AGE_ONLY_HGB'),
                 ('STRICT_ELECTRICAL_HGB','TRAINING_MEAN'),
                 ('STRICT_ELECTRICAL_PLUS_AGE_HGB','TRAINING_MEAN'),
                 ('STRICT_ELECTRICAL_PLUS_AGE_HGB','STRICT_ELECTRICAL_HGB')]
    for a,b in comparisons:
        dif=(piv[a]-piv[b]).dropna().to_numpy(float)
        if len(dif)!=10: raise RuntimeError('Paired comparison must contain exactly ten batteries')
        loc=rng.integers(0,len(dif),size=(reps,len(dif)))
        bootstrap=dif[loc].mean(axis=1)
        lo,hi=np.quantile(bootstrap,[.025,.975])
        results.append({'Model_A':a,'Model_B':b,'Metric':'Mean_Battery_RMSE_Difference_A_Minus_B',
                        'N_Batteries':len(dif),'Point_Delta':float(dif.mean()),
                        'CI_95_Lower':float(lo),'CI_95_Upper':float(hi),
                        'Replicates':int(reps),'Seed':int(seed)})
    return pd.DataFrame(results)


def summary_tables(pred: pd.DataFrame, cohort_name: str):
    x=pred.copy()
    x['Absolute_Error']=abs(x.Prediction_Raw-x[TARGET_COL])
    x['Squared_Error']=(x.Prediction_Raw-x[TARGET_COL])**2
    x['Measured_SoH_Stratum_PostEvaluation']=categorize_true_soh(x[TARGET_COL])
    per_battery=[]
    for (form,bid),g in x.groupby(['Formulation',GROUP_COL]):
        per_battery.append({'Formulation':form,GROUP_COL:bid,'Source':g[SOURCE_COL].iloc[0],
                            'N':len(g),'RMSE':float(np.sqrt(g.Squared_Error.mean())),
                            'MAE':float(g.Absolute_Error.mean()),'Mean_Bias':float((g.Prediction_Raw-g[TARGET_COL]).mean())})
    pb=pd.DataFrame(per_battery)
    total=[]
    for form,g in x.groupby('Formulation'):
        p=pb.loc[pb.Formulation==form]
        total.append({'Population':cohort_name,'Formulation':form,'Batteries':len(p),
                      'N_Observations':len(g),'Equal_Battery_Mean_RMSE':float(p.RMSE.mean()),
                      'Equal_Battery_Mean_MAE':float(p.MAE.mean()),
                      'Pooled_RMSE_Descriptive':float(np.sqrt(g.Squared_Error.mean())),
                      'Pooled_MAE_Descriptive':float(g.Absolute_Error.mean()),
                      'N_Below70':int((g[TARGET_COL]<70).sum()),
                      'N_70to100':int(g[TARGET_COL].between(70,100).sum()),
                      'N_Above100':int((g[TARGET_COL]>100).sum())})
    overall=pd.DataFrame(total)
    strat=[]
    for (form,binname),g in x.groupby(['Formulation','Measured_SoH_Stratum_PostEvaluation']):
        per_cell=[]
        for battery,zz in g.groupby(GROUP_COL):
            per_cell.append({'battery':battery,'rmse':float(np.sqrt(zz.Squared_Error.mean())),
                             'mae':float(zz.Absolute_Error.mean())})
        cell=pd.DataFrame(per_cell)
        strat.append({'Formulation':form,'Observed_SoH_Stratum_Only_After_Prediction':binname,
                      'N_Observations':len(g),'Batteries_With_Observations':len(cell),
                      'Equal_Represented_Battery_RMSE_Descriptive':float(cell.rmse.mean()),
                      'Pooled_RMSE_Descriptive':float(np.sqrt(g.Squared_Error.mean())),
                      'Pooled_MAE_Descriptive':float(g.Absolute_Error.mean()),
                      'Nonzero_Error_Samples':int((g.Absolute_Error>1e-10).sum())})
    return x,pb,overall,pd.DataFrame(strat)


def report(cohort_file: Path, cohort_name: str, fold_dir: Path, output_dir: Path, bootstrap_reps: int):
    output_dir.mkdir(parents=True,exist_ok=True)
    frame=load_cohort(cohort_file)
    result_frames=[]
    aux={name:[] for name in ['config','selected','tuning']}
    for battery in sorted(EXPECTED_BATTERIES):
        for name in ['pred']:
            p=fold_dir/f'{name}_{battery}.csv'
            if not p.exists(): raise FileNotFoundError(f'Missing fold prediction: {p}')
            result_frames.append(pd.read_csv(p,low_memory=False))
        for name in aux:
            aux[name].append(pd.read_csv(fold_dir/f'{name}_{battery}.csv',low_memory=False))
    pred=pd.concat(result_frames,ignore_index=True)
    pred[CYCLE_COL]=integer_cycle(pred[CYCLE_COL])
    if pred.duplicated([GROUP_COL,CYCLE_COL,'Formulation']).any():
        raise RuntimeError('Duplicate battery-cycle/formulation predictions')
    all_models=set(BASELINES + [f for f,_ in FORMULATIONS])
    if set(pred.Formulation.unique()) != all_models: raise RuntimeError('Incorrect set of fitted model formulations')
    for mod,g in pred.groupby('Formulation'):
        keys=frame[KEYS+[TARGET_COL]].merge(g[KEYS+[TARGET_COL]],on=KEYS,
                     how='left',validate='one_to_one',indicator=True,suffixes=('_true','_pred'))
        if not keys['_merge'].eq('both').all():
            raise RuntimeError(f'Predictions incomplete for {mod}')
        if not np.allclose(keys[TARGET_COL+'_true'],keys[TARGET_COL+'_pred'],rtol=0,atol=1e-8):
            raise RuntimeError(f'Target mismatch in saved predictions: {mod}')
    scored, per_battery, overall, strat = summary_tables(pred,cohort_name)
    ci=bootstrap_battery_deltas(per_battery,seed=42,reps=bootstrap_reps)
    atomic_csv(scored, output_dir/'Predictions_All_Models_Unclipped.csv')
    atomic_csv(per_battery, output_dir/'RMSE_PerBattery.csv')
    atomic_csv(overall, output_dir/'RMSE_EqualBattery_And_Pooled.csv')
    atomic_csv(strat, output_dir/'MeasuredSoH_Strata_PostEvaluation.csv')
    atomic_csv(ci,output_dir/'Paired_Battery_Bootstrap.csv')
    for name, frames in aux.items():
        atomic_csv(pd.concat(frames,ignore_index=True,sort=False),
                   output_dir/f'OuterFold_{name.capitalize()}.csv')
    # Compact workbook with results for review. Full wide feature dataset stays CSV.
    wb_path=output_dir/'Reviewer_Step11O_Target_Unrestricted.xlsx'
    with pd.ExcelWriter(wb_path,engine='openpyxl') as writer:
        overall.to_excel(writer,'Overall',index=False)
        per_battery.to_excel(writer,'PerBattery',index=False)
        strat.to_excel(writer,'SoH_Strata_Diagnostic',index=False)
        ci.to_excel(writer,'Paired_Bootstrap',index=False)
        pd.concat(aux['config'],ignore_index=True).to_excel(writer,'Outer_Configs',index=False)
        scored.to_excel(writer,'All_Model_Predictions',index=False)
    method=(
        'TARGET-UNRESTRICTED LABORATORY CONSTANT-PROTOCOL STUDY\n'
        'Population selection: instrument-recorded constant-protocol cycle with >=2 finite time/current samples, positive time span and reconstructible reference target.\n'
        'The 70%-100% SoH bounds are NOT used for training, model selection or test membership.\n'
        'Missing/invalid reference targets are enumerated as unscorable; no arbitrary target bound is imposed.\n'
        'Batteries: six CALCE CS2 and four NASA B0005/B0006/B0007/B0018.\n'
        'Validation: ten outer battery holdouts; training-only 9-battery inner feature/grid selection.\n'
        'Target: full recorded discharge / battery-specific initial reference capacity, identical to original source pipeline.\n'
        'Features: original 335 historical strict-candidate schema; explicit throughput/total duration, energy excluded, indirect capacity proxies may remain.\n'
        'Metrics: mean of ten battery RMSEs (unclipped) for main metric; pooled RMSE descriptive only.\n'
        'Age baselines: training mean, battery-weighted age linear, retuned age-only HGB.\n'
        'Bootstrap: 20,000 (or specified number) paired resamples of ten observed battery RMSE differences, no refitting.\n'
        'SoH-stratified reports are conditional diagnostic AFTER predictions; stratum is NOT a deployment eligibility rule.\n'
        'Any optional CALCE protocol-completeness screen was chosen post hoc and applies only to CALCE.\n'
        'Scope: calibrated, controlled laboratory post-discharge health characterization only; no proof of general BMS deployment.\n'
        'No estimates outside the studied chemistry/protocol/source distributions are established by this evaluation.\n'
    )
    (output_dir/'METHODS_AND_INFERENCE_SCOPE.txt').write_text(method)
    result_zip=output_dir/f'Step11O_{cohort_name}_Results.zip'
    with zipfile.ZipFile(result_zip,'w',zipfile.ZIP_DEFLATED,compresslevel=7) as z:
        for fn in ['RMSE_EqualBattery_And_Pooled.csv','RMSE_PerBattery.csv',
                   'MeasuredSoH_Strata_PostEvaluation.csv','Paired_Battery_Bootstrap.csv',
                   'OuterFold_Config.csv','OuterFold_Selected.csv','METHODS_AND_INFERENCE_SCOPE.txt',
                   'Reviewer_Step11O_Target_Unrestricted.xlsx']:
            p=output_dir/fn
            if p.exists(): z.write(p,arcname=fn)
    log('RESULTS',cohort_name,'\n'+overall.to_string(index=False))
    log('Result ZIP:',result_zip)
    return result_zip


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage',choices=['all','prepare','train','report'],default='all')
    parser.add_argument('--raw-zip',type=Path,required=True)
    parser.add_argument('--project-output',type=Path,required=True)
    parser.add_argument('--work-dir',type=Path,default=Path('/content/step11o_work'))
    parser.add_argument('--reference-gate',type=Path,required=True)
    parser.add_argument('--protocol-qc',action='store_true',help='Audit tester-step completeness for all CALCE cycles')
    parser.add_argument('--train-protocol-cohort',action='store_true',help='Also run full second nested LOBO on optional post-hoc protocol-screened cohort')
    parser.add_argument('--bootstrap-reps',type=int,default=20000)
    parser.add_argument('--force-prepare',action='store_true')
    a=parser.parse_args()
    if a.bootstrap_reps < 100: raise ValueError('bootstrap-reps must be >=100')
    if not a.raw_zip.is_file(): raise FileNotFoundError(f'Raw archive not found: {a.raw_zip}')
    if not a.reference_gate.is_file(): raise FileNotFoundError(f'Reference gate not found: {a.reference_gate}')
    raw_sha=sha256(a.raw_zip); ref_sha=sha256(a.reference_gate)
    root=a.project_output; prep=root/'prepared'; prep.mkdir(parents=True,exist_ok=True)
    audit_protocol = a.protocol_qc or a.train_protocol_cohort
    if a.stage in ('all','prepare'):
        if not a.force_prepare and valid_cached_preparation(prep,raw_sha,ref_sha,audit_protocol):
            log('Reusing prepared dataset from Drive (input hashes matched).')
        else:
            prepare_data(a.raw_zip,a.work_dir,prep,a.reference_gate,audit_protocol,raw_sha)
    if a.stage in ('all','train'):
        mainfile=prep/'Unrestricted_All_QC_Valid_Cycles.csv'
        if not mainfile.is_file(): raise FileNotFoundError('Run --stage prepare first')
        train_models(mainfile,PRIMARY_NAME,root/'folds_raw_qc')
        if a.train_protocol_cohort:
            another=prep/'Unrestricted_Protocol_Screened_PostHoc.csv'
            if not another.is_file(): raise FileNotFoundError('Run --stage prepare --protocol-qc first')
            train_models(another,SCREENED_NAME,root/'folds_protocol_qc')
    if a.stage in ('all','report'):
        mainfile=prep/'Unrestricted_All_QC_Valid_Cycles.csv'
        report(mainfile,PRIMARY_NAME,root/'folds_raw_qc',root/'results_raw_qc',a.bootstrap_reps)
        if a.train_protocol_cohort:
            other=prep/'Unrestricted_Protocol_Screened_PostHoc.csv'
            report(other,SCREENED_NAME,root/'folds_protocol_qc',root/'results_protocol_qc',a.bootstrap_reps)


if __name__=='__main__':
    main()
