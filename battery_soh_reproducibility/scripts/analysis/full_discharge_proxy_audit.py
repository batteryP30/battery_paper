"""Audit indirect capacity proxies among original full-discharge feature columns.

Associations are descriptive *within battery*. A strong inverse-capacity
association is not proof of numerical target leakage, but prohibits an
unqualified claim of statistical capacity independence.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from battery_soh_repro.experiment_utils import record_inputs


def run(data:Path,features:Path,output_dir:Path):
    output_dir.mkdir(parents=True,exist_ok=True)
    frame=pd.read_csv(data,low_memory=False)
    f=pd.read_csv(features)
    names=f['Feature'].dropna().astype(str).tolist()
    q=pd.to_numeric(frame['Measured_Discharge_Capacity_Ah'],errors='coerce')
    if not (q>0).all():raise ValueError('Capacity reference must be finite and positive')
    frame['Reciprocal_Capacity_Ah_inv']=1/q
    candidate=[name for name in names if name in frame.columns and
               any(term in name.lower() for term in ['dvdt','slope','dynamic','dropperhour','differential'])]
    results=[]
    for name in candidate:
        for battery,group in frame.groupby('battery_id'):
            x=pd.to_numeric(group[name],errors='coerce').to_numpy(float)
            y=group['Reciprocal_Capacity_Ah_inv'].to_numpy(float)
            mask=np.isfinite(x)&np.isfinite(y)
            score=(float(spearmanr(x[mask],y[mask]).statistic) if (mask.sum()>3 and
                np.unique(x[mask]).size>1 and np.unique(y[mask]).size>1) else np.nan)
            results.append({'Feature':name,'battery_id':battery,'N':int(mask.sum()),
                            'Spearman_With_Reciprocal_Q':score,
                            'Absolute_Spearman':abs(score) if np.isfinite(score) else np.nan})
    per=pd.DataFrame(results)
    summary=per.groupby('Feature').agg(Batteries=('battery_id','nunique'),
                         Median_Within_Battery_AbsSpearman=('Absolute_Spearman','median'),
                         Min_Within_Battery_AbsSpearman=('Absolute_Spearman','min'),
                         Max_Within_Battery_AbsSpearman=('Absolute_Spearman','max')).reset_index()
    summary=summary.sort_values('Median_Within_Battery_AbsSpearman',ascending=False)
    per.to_csv(output_dir/'full_discharge_proxy_by_battery.csv',index=False)
    summary.to_csv(output_dir/'full_discharge_proxy_summary.csv',index=False)
    record_inputs(output_dir,{'data':data,'features':features},
                  {'post_hoc_diagnostic':True,'within_battery_correlations':True})
    print(summary.head(12).to_string(index=False))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',type=Path,default=Path('outputs/constant_dataset/constant_observations.csv'))
    p.add_argument('--features',type=Path,default=Path('outputs/constant_dataset/constant_strict_feature_names.csv'))
    p.add_argument('--output-dir',type=Path,default=Path('outputs/full_discharge_proxy_audit'))
    a=p.parse_args();run(a.data,a.features,a.output_dir)
if __name__=='__main__':main()
