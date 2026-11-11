"""Paired battery-level confidence interval for two held-out prediction methods.

The number of batteries is 10 or 6, not the number of cycles. Resampling
operates on paired per-battery RMSE differences without model refitting.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd
from battery_soh_repro.experiment_utils import compare_frozen_predictions,record_inputs


def run(reference:Path,comparison:Path,left_form:str,right_form:str,
        target:str,key:str,output_dir:Path,repetitions:int=20000):
    output_dir.mkdir(parents=True,exist_ok=True)
    l=pd.read_csv(reference);r=pd.read_csv(comparison)
    if 'Formulation' in l:l=l[l.Formulation.eq(left_form)]
    if 'Formulation' in r:r=r[r.Formulation.eq(right_form)]
    pair,per,stat=compare_frozen_predictions(l,r,['battery_id',key],target,reps=repetitions)
    pair.to_csv(output_dir/'paired_predictions.csv',index=False)
    per.to_csv(output_dir/'paired_battery_metrics.csv',index=False)
    stat.to_csv(output_dir/'paired_battery_bootstrap.csv',index=False)
    record_inputs(output_dir,{'reference':reference,'comparison':comparison},
                  {'left_form':left_form,'right_form':right_form,'key':key,
                   'repetitions':repetitions,'no_refitting':True})
    print(stat.to_string(index=False))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['reference','comparison']:p.add_argument('--'+name,required=True,type=Path)
    p.add_argument('--left-form',default='STRICT_ELECTRICAL_HGB')
    p.add_argument('--right-form',default='STRICT_ELECTRICAL_PLUS_AGE_HGB')
    p.add_argument('--target',default='Calculated_SOH(%)')
    p.add_argument('--key',default='Cycle_Index')
    p.add_argument('--repetitions',type=int,default=20000)
    p.add_argument('--output-dir',type=Path,default=Path('outputs/paired_uncertainty'))
    a=p.parse_args();run(a.reference,a.comparison,a.left_form,a.right_form,
                        a.target,a.key,a.output_dir,a.repetitions)
if __name__=='__main__':main()
