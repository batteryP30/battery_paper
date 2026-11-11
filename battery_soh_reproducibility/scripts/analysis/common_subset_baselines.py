"""Compare baseline/HGB predictions only where a previous measured target exists.

The first observation of every held-out battery is excluded for every model;
no persistence initialization is manufactured and no models are retrained.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from battery_soh_repro.experiment_utils import ensure_unique,metric_by_battery,record_inputs


def run(data:Path,baseline_predictions:Path,hgb_predictions:Path,
        age_predictions:Path|None,protocol:str,output_dir:Path):
    output_dir.mkdir(parents=True,exist_ok=True)
    key='Cycle_Index' if protocol=='constant' else 'R5E_Reference_Epoch'
    y='Calculated_SOH(%)' if protocol=='constant' else 'Target_SOH'
    source=pd.read_csv(data,low_memory=False).sort_values(['battery_id',key]).copy()
    source['_Previous_SoH']=source.groupby('battery_id')[y].shift(1)
    common=source[source._Previous_SoH.notna()].copy()
    keycols=['battery_id',key]
    ensure_unique(common,keycols)
    if len(common)!=len(source)-source.battery_id.nunique():
        raise ValueError('Common-subset first-observation restriction inconsistent')
    records=[]
    def add_model(label,frame,actual,pred):
        z=common[keycols+[y]].merge(frame[keycols+[pred]],on=keycols,how='left',validate='one_to_one')
        if z[pred].isna().any(): raise ValueError(f'Missing {label} predictions on common subset')
        per=metric_by_battery(z,y,pred)
        per.insert(0,'Model',label)
        for _,r in per.iterrows():records.append(r.to_dict())
    raw=pd.read_csv(baseline_predictions)
    if protocol=='variable' and key not in raw.columns:
        # Original variable baseline predictions are keyed by interval ID;
        # the HGB predictions instead use the reference-epoch coordinate.
        if 'R5E_Interval_ID' not in raw.columns:
            raise ValueError('Variable baseline lacks interval ID and reference epoch')
        mapping=source[['battery_id','R5E_Interval_ID',key]].drop_duplicates()
        raw=raw.merge(mapping,on=['battery_id','R5E_Interval_ID'],validate='many_to_one')
    for name,g in raw.groupby('Baseline'):
        if 'PERSISTENCE' in name.upper():
            continue
        add_model(name,g,'Actual_SOH','Predicted_SOH')
    # Persistence is recomputed directly from prior frozen target on each battery;
    # this avoids relying on a potentially different baseline file convention.
    add_model('PreviousTargetPersistence_HISTORY_ASSISTED',
              common.rename(columns={'_Previous_SoH':'Prior_Prediction'}),y,'Prior_Prediction')
    hgb=pd.read_csv(hgb_predictions)
    for name,g in hgb.groupby('Formulation'):
        add_model(name,g,y,'Prediction_Raw')
    if age_predictions is not None:
        add_model('AgeOnlyHGB',pd.read_csv(age_predictions),y,'Prediction_Raw')
    per=pd.DataFrame(records)
    summary=per.groupby('Model').agg(Batteries=('battery_id','nunique'),N=('N','sum'),
                                     Equal_Battery_RMSE=('RMSE','mean'),
                                     Equal_Battery_MAE=('MAE','mean')).reset_index()
    per.to_csv(output_dir/f'{protocol}_common_subset_per_battery.csv',index=False)
    summary.to_csv(output_dir/f'{protocol}_common_subset_summary.csv',index=False)
    record_inputs(output_dir,{'data':data,'baseline_predictions':baseline_predictions,
                              'hgb_predictions':hgb_predictions,
                              **({'age_predictions':age_predictions} if age_predictions else {})},
                  {'protocol':protocol,'rows_common':len(common),'no_retraining':True})
    print(summary.to_string(index=False))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['data','baseline-predictions','hgb-predictions']:
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--age-predictions',type=Path)
    p.add_argument('--protocol',choices=['constant','variable'],required=True)
    p.add_argument('--output-dir',type=Path,default=Path('outputs/common_subset_baselines'))
    a=p.parse_args();run(a.data,a.baseline_predictions,a.hgb_predictions,a.age_predictions,a.protocol,a.output_dir)
if __name__=='__main__':main()
