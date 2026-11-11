"""Within-source constant-protocol HGB LOBO using common transfer feature schema.

This is a target-source-only control for zero-shot transfer interpretation,
not an all-source LOBO estimate. A held-out battery never informs its fit.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd
import constant_strict_lobo as hgb
import constant_cross_dataset_transfer as transfer
from battery_soh_repro.experiment_utils import record_inputs


def run(data:Path,features:Path,output_dir:Path):
    output_dir.mkdir(parents=True,exist_ok=True)
    frame=pd.read_csv(data,low_memory=False)
    f=pd.read_csv(features)['Feature'].tolist()
    common,_=transfer.define_interoperable_schema(frame,f)
    candidate=[x for x in common if x not in hgb.EXPLICIT_AGE_STRICT_FEATURES]
    summary=[]
    for source in ['CALCE','NASA']:
        subset=frame[frame.source_dataset.eq(source)]
        for name,include_age in [('STRICT_ELECTRICAL_HGB',False),('STRICT_ELECTRICAL_PLUS_AGE_HGB',True)]:
            result=hgb.run_formulation(subset,name,candidate,include_age)
            for key,val in result.items():
                if isinstance(val,pd.DataFrame):val.to_csv(output_dir/f'{source}_{name}_{key}.csv',index=False)
            tab=result['macro'].iloc[0]
            summary.append({'Source':source,'Formulation':name,'N':len(subset),
                            'Batteries':subset.battery_id.nunique(),
                            'Macro_RMSE_Raw':float(tab.Macro_RMSE_Raw),
                            'Macro_MAE_Raw':float(tab.Macro_MAE_Raw)})
    pd.DataFrame(summary).to_csv(output_dir/'source_only_hgb_summary.csv',index=False)
    record_inputs(output_dir,{'data':data,'features':features},
                  {'same_common_transfer_candidate_schema':True,'within_source_lobo':True})


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',type=Path,default=Path('outputs/constant_dataset/constant_observations.csv'))
    p.add_argument('--features',type=Path,default=Path('outputs/constant_dataset/constant_strict_feature_names.csv'))
    p.add_argument('--output-dir',type=Path,default=Path('outputs/source_only_hgb'))
    a=p.parse_args();run(a.data,a.features,a.output_dir)
if __name__=='__main__':main()
