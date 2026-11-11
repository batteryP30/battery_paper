"""Nested battery-held-out age-only HGB benchmark (constant and variable)."""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from battery_soh_repro.baselines import age_only_lobo
from battery_soh_repro.experiment_utils import metric_by_battery,record_inputs


def run(data:Path,protocol:str,output_dir:Path):
    output_dir.mkdir(parents=True,exist_ok=True)
    frame=pd.read_csv(data,low_memory=False)
    target='Calculated_SOH(%)' if protocol=='constant' else 'Target_SOH'
    cols=['Cycle_Age','Log_Cycle_Age'] if protocol=='constant' else ['Age_Coord','Log_Age_Coord']
    key='Cycle_Index' if protocol=='constant' else 'R5E_Reference_Epoch'
    preds,folds=age_only_lobo(frame,cols,target)
    orig=frame.reset_index()[['index',key]]
    preds=preds.merge(orig,left_on='_Original_Row',right_on='index',validate='one_to_one').drop(columns=['_Original_Row','index'])
    per=metric_by_battery(preds,target,'Prediction_Raw')
    summary={'Protocol':protocol,'Model':'AgeOnlyHGB','Batteries':len(per),'Observations':len(preds),
             'Equal_Battery_RMSE':per.RMSE.mean(),'Equal_Battery_MAE':per.MAE.mean()}
    preds.to_csv(output_dir/f'{protocol}_age_only_hgb_predictions.csv',index=False)
    folds.to_csv(output_dir/f'{protocol}_age_only_hgb_selection.csv',index=False)
    per.to_csv(output_dir/f'{protocol}_age_only_hgb_per_battery.csv',index=False)
    pd.DataFrame([summary]).to_csv(output_dir/f'{protocol}_age_only_hgb_summary.csv',index=False)
    record_inputs(output_dir,{'study_data':data},{'protocol':protocol,'strict_target_free':True,
                'inner_lobo_config_selection':True,'model':'AgeOnlyHGB'})
    print(summary)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--protocol',choices=['constant','variable'],required=True)
    p.add_argument('--data',required=True,type=Path)
    p.add_argument('--output-dir',default=Path('outputs/age_only_hgb'),type=Path)
    a=p.parse_args();run(a.data,a.protocol,a.output_dir)
if __name__=='__main__':main()
