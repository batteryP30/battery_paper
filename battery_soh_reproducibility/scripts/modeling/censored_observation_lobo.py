"""True prediction-time censored HGB: first 600 s or first downward 3.6 V.

Data construction NEVER copies a full-discharge electrical feature into an
observation-time predictor. Historical transformations use censored past cycles
only; frozen SoH labels are joined strictly after feature construction.

Examples:
    python scripts/modeling/censored_observation_lobo.py --raw-zip raw_data/collab.zip --boundary seconds
    python scripts/modeling/censored_observation_lobo.py --raw-zip raw_data/collab.zip --boundary voltage
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import constant_strict_lobo as model
from battery_soh_repro.censoring import censored_prefix
from battery_soh_repro.data_pipeline import (
    CALCE_CONSTANT_BATTERIES,NASA_CONSTANT_BATTERIES,
    prepare_raw_archive,process_files,load_nasa_constant_measurements_targetfree,
    build_cycle_features_run2,add_dataset_identity)
from battery_soh_repro.experiment_utils import integer_cycles,ensure_unique,compare_frozen_predictions,record_inputs

# Reuse validated original feature-history code without importing the screened
# pipeline through the application/script path.
from battery_soh_repro.data_pipeline import (add_legacy_history_features,
    add_run2_invariant_features,add_past_only_features,add_joint_legacy_aliases)


def early_features(raw: pd.DataFrame, battery: str, source: str,
                   boundary: str) -> tuple[pd.DataFrame,pd.DataFrame]:
    prefixes=[];audits=[]
    for cycle,sub in raw.groupby('Cycle_Index',sort=True):
        prefix,a=censored_prefix(sub,source,seconds=600 if boundary=='seconds' else None,
                                 voltage=3.6 if boundary=='voltage' else None,
                                 min_samples=10,
                                 max_gap_seconds=120 if boundary=='seconds' else float('inf'))
        audits.append({'battery_id':battery,'Cycle_Index':int(cycle),**a})
        if prefix is not None: prefixes.append(prefix)
    audit=pd.DataFrame(audits)
    if not prefixes:return pd.DataFrame(),audit
    joined=pd.concat(prefixes,ignore_index=True)
    f=build_cycle_features_run2(joined,cycle_col='Cycle_Index',time_col='Test_Time(s)',
                                volt_col='Voltage(V)',curr_col='Current(A)',step_col='Step_Index')
    f=add_dataset_identity(f,battery,source,1)
    f['Cycle_Index']=integer_cycles(f['Cycle_Index']);f=f.sort_values('Cycle_Index')
    # Age must retain raw cycle-coordinate spacing, never renumber eligible observations.
    first_raw=int(pd.to_numeric(raw['Cycle_Index'],errors='raise').min())
    f['Cycle_Age']=(f['Cycle_Index']-first_raw).clip(lower=0)
    f['Log_Cycle_Age']=np.log1p(f['Cycle_Age'])
    f['source_is_nasa']=int(source=='NASA')
    for fn in [add_legacy_history_features,add_run2_invariant_features,
               add_past_only_features,add_joint_legacy_aliases]:f=fn(f)
    return f,audit


def run(raw_zip:Path,frozen_data:Path,frozen_features:Path,
        frozen_predictions:Path,work_dir:Path,output_dir:Path,
        boundary:str,strict_counts:bool=True):
    output_dir.mkdir(parents=True,exist_ok=True)
    frozen=pd.read_csv(frozen_data,low_memory=False)
    frozen['Cycle_Index']=integer_cycles(frozen['Cycle_Index'])
    feat=pd.read_csv(frozen_features)
    features=feat['Feature'].astype(str).tolist()
    if len(features)!=335:raise ValueError('Expected archived strict feature list of 335 columns')
    manifest,_=prepare_raw_archive(raw_zip,work_dir)
    frames=[];audits=[]
    for b,sheets in CALCE_CONSTANT_BATTERIES+[(b,None) for b in NASA_CONSTANT_BATTERIES]:
        source='CALCE' if sheets is not None else 'NASA'
        raw=(process_files(str(manifest[b]['folder']),sheets) if sheets is not None else
             load_nasa_constant_measurements_targetfree(manifest[b]['csv_path'],b))
        f,a=early_features(raw,b,source,boundary)
        audits.append(a)
        if f.empty:continue
        old=frozen[frozen.battery_id==b]
        # Join targets and age only, NEVER previous full-cycle feature values.
        joined=f.merge(old[['battery_id','Cycle_Index','Calculated_SOH(%)']],
                       on=['battery_id','Cycle_Index'],how='inner',validate='one_to_one')
        keep=['battery_id','source_dataset','Cycle_Index','Calculated_SOH(%)',
              'Cycle_Age','Log_Cycle_Age']+features
        frames.append(joined.reindex(columns=keep))
    frame=pd.concat(frames,ignore_index=True)
    ensure_unique(frame,['battery_id','Cycle_Index'])
    count=4590 if boundary=='seconds' else 4589
    if strict_counts and (len(frame)!=count or frame.battery_id.nunique()!=10):
        raise ValueError(f'Expected {count} cycles across ten batteries for {boundary}; got {len(frame)}')
    frame.to_csv(output_dir/'censored_observations.csv',index=False)
    pd.concat(audits,ignore_index=True).to_csv(output_dir/'observation_window_audit.csv',index=False)
    electric=[f for f in features if f not in model.EXPLICIT_AGE_STRICT_FEATURES]
    results={}
    for name,with_age in [('STRICT_ELECTRICAL_HGB',False),('STRICT_ELECTRICAL_PLUS_AGE_HGB',True)]:
        results[name]=model.run_formulation(frame,name,electric,with_age)
    for name,result in results.items():
        for key,value in result.items():
            if isinstance(value,pd.DataFrame):value.to_csv(output_dir/f'{name}_{key}.csv',index=False)
    frozen_pred=pd.read_csv(frozen_predictions)
    paired=[];by_battery=[];summaries=[]
    for name,result in results.items():
        original=frozen_pred[frozen_pred.Formulation.eq(name)]
        new=result['predictions']
        pair,per,stat=compare_frozen_predictions(original,new,['battery_id','Cycle_Index'],'Calculated_SOH(%)')
        pair.insert(0,'Formulation',name);per.insert(0,'Formulation',name);stat.insert(0,'Formulation',name)
        paired.append(pair);by_battery.append(per);summaries.append(stat)
    pd.concat(paired,ignore_index=True).to_csv(output_dir/'paired_predictions.csv',index=False)
    pd.concat(by_battery,ignore_index=True).to_csv(output_dir/'paired_by_battery.csv',index=False)
    pd.concat(summaries,ignore_index=True).to_csv(output_dir/'paired_summary.csv',index=False)
    record_inputs(output_dir,{'raw_zip':raw_zip,'frozen_data':frozen_data,
                              'features':frozen_features,'predictions':frozen_predictions},
                  {'boundary':boundary,'expected_cycles':count,'post_boundary_samples_used':False})
    return frame


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--raw-zip',required=True,type=Path)
    p.add_argument('--frozen-data',type=Path,default=Path('outputs/constant_dataset/constant_observations.csv'))
    p.add_argument('--frozen-features',type=Path,default=Path('outputs/constant_dataset/constant_strict_feature_names.csv'))
    p.add_argument('--frozen-predictions',type=Path,default=Path('outputs/constant_hgb/constant_hgb_predictions.csv'))
    p.add_argument('--boundary',choices=['seconds','voltage'],required=True)
    p.add_argument('--work-dir',type=Path,default=Path('work/censored'))
    p.add_argument('--output-dir',type=Path,default=Path('outputs/censored_hgb'))
    p.add_argument('--exploratory-counts',action='store_true')
    a=p.parse_args()
    run(a.raw_zip,a.frozen_data,a.frozen_features,a.frozen_predictions,a.work_dir,a.output_dir,
        a.boundary,not a.exploratory_counts)
if __name__=='__main__':main()
