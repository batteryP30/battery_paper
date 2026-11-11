"""Rebuild post hoc CALCE-screened features from original raw cycle histories.

The quality screen is defined from tester-step behavior alone. Failed CALCE
cycles are removed BEFORE causal history features are rebuilt; NASA cycles and
all retained targets remain unchanged. The primary 4590-row data are untouched.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd

from battery_soh_repro.data_pipeline import (
    CALCE_CONSTANT_BATTERIES, NASA_CONSTANT_BATTERIES, CYCLE_COL,
    SOURCE_COL, TARGET_COL, build_cycle_features_run2, add_dataset_identity,
    add_legacy_history_features, add_run2_invariant_features, add_past_only_features,
    add_joint_legacy_aliases, prepare_raw_archive, process_files,
    load_nasa_constant_measurements_targetfree)
from battery_soh_repro.experiment_utils import ensure_unique, integer_cycles, record_inputs


def add_history(instant: pd.DataFrame, first_raw_cycle: int) -> pd.DataFrame:
    out=instant.sort_values(CYCLE_COL).copy().reset_index(drop=True)
    out[CYCLE_COL]=integer_cycles(out[CYCLE_COL])
    out['Cycle_Age']=(out[CYCLE_COL]-int(first_raw_cycle)).clip(lower=0)
    out['Log_Cycle_Age']=np.log1p(out.Cycle_Age)
    out['source_is_nasa']=(out[SOURCE_COL] == 'NASA').astype(int)
    for fn in [add_legacy_history_features, add_run2_invariant_features,
               add_past_only_features, add_joint_legacy_aliases]:
        out=fn(out)
    return out


def rebuild(instant: pd.DataFrame, first_raw_cycle: int,
            frozen_battery: pd.DataFrame, candidate_features: list[str], battery: str,
            verification: bool = False):
    transformed=add_history(instant, first_raw_cycle)
    if verification:
        cols=[c for c in candidate_features if c in transformed.columns and c in frozen_battery.columns]
        aligned=frozen_battery[['battery_id',CYCLE_COL]+cols].merge(
            transformed[['battery_id',CYCLE_COL]+cols], on=['battery_id',CYCLE_COL],
            validate='one_to_one', how='left', indicator=True, suffixes=('_frozen','_rebuilt'))
        if not aligned['_merge'].eq('both').all(): raise ValueError(f'{battery}: raw features fail to cover frozen keys')
        nbad=0; checked=0; bad_features=[]
        for c in cols:
            a=pd.to_numeric(aligned[c+'_frozen'], errors='coerce').to_numpy(float)
            b=pd.to_numeric(aligned[c+'_rebuilt'], errors='coerce').to_numpy(float)
            eq=np.isclose(a,b,rtol=1e-5,atol=1e-6,equal_nan=True)
            checked+=len(eq);nbad+=int((~eq).sum())
            if eq.mean()<0.995:bad_features.append(c)
        if nbad/max(checked,1) >.005 or bad_features:
            raise ValueError(f'{battery}: frozen-feature reproduction gate failed: '
                             f'{nbad}/{checked} cells, features {bad_features[:10]}')
        return {'Battery':battery,'Compared_Cells':checked,'Mismatch_Cells':nbad,
                'Mismatch_Fraction':nbad/max(checked,1)}
    return transformed


def run(raw_zip: Path, frozen_data: Path, frozen_features: Path, protocol_screen: Path,
        work_dir: Path, output_dir: Path, strict_counts: bool=True) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    frozen=pd.read_csv(frozen_data, low_memory=False)
    frozen[CYCLE_COL]=integer_cycles(frozen[CYCLE_COL]);ensure_unique(frozen,['battery_id',CYCLE_COL])
    ff=pd.read_csv(frozen_features)
    col='Feature' if 'Feature' in ff else ff.columns[0]
    candidate_features=ff[col].astype(str).tolist()
    qc=pd.read_csv(protocol_screen)
    qc[CYCLE_COL]=integer_cycles(qc[CYCLE_COL])
    ensure_unique(qc,['battery_id',CYCLE_COL])
    if qc.Quality_Pass.dtype==object:
        mapping={'true':True,'false':False}
        qc['Quality_Pass']=qc.Quality_Pass.astype(str).str.lower().map(mapping)
        if qc.Quality_Pass.isna().any():raise ValueError('Non-boolean protocol screen entries')
    lookup=qc.set_index(['battery_id',CYCLE_COL]).Quality_Pass
    man,_=prepare_raw_archive(raw_zip,work_dir)
    batches=[];verifications=[];coverage=[]
    for battery,sheets in CALCE_CONSTANT_BATTERIES+[(b,None) for b in NASA_CONSTANT_BATTERIES]:
        is_calce=sheets is not None
        source='CALCE' if is_calce else 'NASA'
        print('Rebuilding screened histories:',battery, flush=True)
        raw=(process_files(str(man[battery]['folder']),sheets) if is_calce else
             load_nasa_constant_measurements_targetfree(man[battery]['csv_path'], battery))
        instant=build_cycle_features_run2(raw,cycle_col=CYCLE_COL,time_col='Test_Time(s)',
            volt_col='Voltage(V)',curr_col='Current(A)',step_col='Step_Index')
        instant=add_dataset_identity(instant,battery,source,1)
        instant[CYCLE_COL]=integer_cycles(instant[CYCLE_COL]);ensure_unique(instant,[CYCLE_COL])
        first_raw=int(instant[CYCLE_COL].min())
        orig=frozen[frozen.battery_id==battery].copy()
        verifications.append(rebuild(instant,first_raw,orig,candidate_features,battery,verification=True))
        if is_calce:
            keys=pd.MultiIndex.from_arrays([[battery]*len(instant),instant[CYCLE_COL]])
            if not keys.isin(lookup.index).all():raise ValueError(f'{battery}: screen missing raw cycle keys')
            retain=lookup.reindex(keys).to_numpy(dtype=bool)
            subset=instant.loc[retain].copy()
        else:
            subset=instant.copy()
        generated=rebuild(subset,first_raw,orig,candidate_features,battery)
        out=generated.merge(orig[['battery_id',CYCLE_COL,TARGET_COL,
                                   'Measured_Discharge_Capacity_Ah','Reference_Capacity_Ah']],
                            on=['battery_id',CYCLE_COL],how='inner',validate='one_to_one')
        cols=['battery_id',SOURCE_COL,CYCLE_COL,TARGET_COL,
              'Measured_Discharge_Capacity_Ah','Reference_Capacity_Ah',
              'Cycle_Age','Log_Cycle_Age']+candidate_features
        out=out.reindex(columns=cols)
        batches.append(out)
        coverage.append({'Battery':battery,'Source':source,'Original_Study':len(orig),
                         'Retained_Study':len(out),'Excluded_Study':len(orig)-len(out),
                         'Raw_Cycles':len(instant)})
    study=pd.concat(batches,ignore_index=True)
    ensure_unique(study,['battery_id',CYCLE_COL])
    original=study[['battery_id',CYCLE_COL,TARGET_COL]].merge(
        frozen[['battery_id',CYCLE_COL,TARGET_COL]],on=['battery_id',CYCLE_COL],
        validate='one_to_one',suffixes=('_new','_old'))
    if not np.allclose(original[TARGET_COL+'_new'],original[TARGET_COL+'_old'],atol=1e-10,rtol=0):
        raise ValueError('Retained SoH targets differ from frozen labels')
    cov=pd.DataFrame(coverage)
    if strict_counts and (len(study)!=4501 or int(cov.Excluded_Study.sum())!=89 or
                          len(cov)!=10 or int(study[SOURCE_COL].eq('NASA').sum())!=567):
        raise ValueError('Unexpected 4501-cycle/89-failure sensitivity population; inspect screen')
    path=output_dir/'screened_constant_observations.csv'
    study.to_csv(path,index=False)
    cov.to_csv(output_dir/'quality_screened_coverage.csv',index=False)
    pd.DataFrame(verifications).to_csv(output_dir/'frozen_feature_reconstruction_gate.csv',index=False)
    record_inputs(output_dir,{'raw_zip':raw_zip,'frozen_data':frozen_data,
                              'frozen_features':frozen_features,'screen':protocol_screen},
                  {'post_hoc':True,'strict_counts':strict_counts,'original_cycle_age_preserved':True})
    return path


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--raw-zip',type=Path,required=True)
    p.add_argument('--frozen-data',type=Path,default=Path('outputs/constant_dataset/constant_observations.csv'))
    p.add_argument('--frozen-features',type=Path,default=Path('outputs/constant_dataset/constant_strict_feature_names.csv'))
    p.add_argument('--protocol-screen',type=Path,default=Path('outputs/calce_qc/calce_protocol_screen.csv'))
    p.add_argument('--work-dir',type=Path,default=Path('work/screened_raw'))
    p.add_argument('--output-dir',type=Path,default=Path('outputs/quality_screened_dataset'))
    p.add_argument('--exploratory-counts',action='store_true',help='Disable frozen-study population count gate')
    a=p.parse_args()
    print(run(a.raw_zip,a.frozen_data,a.frozen_features,a.protocol_screen,a.work_dir,a.output_dir,
              not a.exploratory_counts))
if __name__=='__main__':main()
