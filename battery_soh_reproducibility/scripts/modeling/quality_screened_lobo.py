"""Post hoc quality-screened nested LOBO HGB and paired frozen-model comparison.

Requires screened features rebuilt from raw first; these results supplement and
never replace the primary unscreened 4590-observation analysis.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd
import constant_strict_lobo as model
from battery_soh_repro.experiment_utils import compare_frozen_predictions, record_inputs


def run(screened: Path, features: Path, frozen_predictions: Path, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    frame=pd.read_csv(screened,low_memory=False)
    frozen=pd.read_csv(frozen_predictions)
    if len(frame)!=4501 or frame.battery_id.nunique()!=10:
        raise ValueError('Screened cohort must contain 4501 observations and ten batteries')
    strict=model.load_frozen_feature_list(features)
    electric=[f for f in strict if f not in model.EXPLICIT_AGE_STRICT_FEATURES]
    formulations=[('STRICT_ELECTRICAL_HGB',False),('STRICT_ELECTRICAL_PLUS_AGE_HGB',True)]
    merged=[];p_rows=[];summ=[]
    for name,age in formulations:
        print('Retraining post hoc:',name)
        result=model.run_formulation(frame,name,electric,age)
        for field,table in result.items():
            if isinstance(table,pd.DataFrame):
                table.to_csv(output_dir/f'screened_{name}_{field}.csv',index=False)
        pred=result['predictions'].copy()
        orig=frozen[frozen.Formulation.eq(name)]
        pair,per,stat=compare_frozen_predictions(orig,pred,
            keys=['battery_id','Cycle_Index'],target='Calculated_SOH(%)')
        pair.insert(0,'Formulation',name);per.insert(0,'Formulation',name);stat.insert(0,'Formulation',name)
        merged.append(pair);p_rows.append(per);summ.append(stat)
    pd.concat(merged,ignore_index=True).to_csv(output_dir/'paired_screened_predictions.csv',index=False)
    pd.concat(p_rows,ignore_index=True).to_csv(output_dir/'paired_screened_by_battery.csv',index=False)
    pd.concat(summ,ignore_index=True).to_csv(output_dir/'paired_screened_summary.csv',index=False)
    record_inputs(output_dir,{'screened':screened,'features':features,
                              'frozen_predictions':frozen_predictions},
                  {'post_hoc':True,'nested_battery_level':True,'n_expected':4501})


def main():
    a=argparse.ArgumentParser(description=__doc__)
    a.add_argument('--screened',type=Path,default=Path('outputs/quality_screened_dataset/screened_constant_observations.csv'))
    a.add_argument('--features',type=Path,default=Path('outputs/constant_dataset/constant_strict_feature_names.csv'))
    a.add_argument('--frozen-predictions',type=Path,default=Path('outputs/constant_hgb/constant_hgb_predictions.csv'))
    a.add_argument('--output-dir',type=Path,default=Path('outputs/quality_screened_lobo'))
    x=a.parse_args();run(x.screened,x.features,x.frozen_predictions,x.output_dir)
if __name__=='__main__':main()
