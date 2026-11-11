"""Sensitivity: rank electrical features by equal-battery mean Spearman strength.

The original nested LOBO pipeline is reused; only the selector's training-only
ranking changes from pooled cycles to an equal average across training cells.
This is a post hoc alternative and never overwrites original model results.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'modeling'))
import constant_strict_lobo as base
from battery_soh_repro.experiment_utils import compare_frozen_predictions,record_inputs


def balanced_selector(train_frame,candidate_columns):
    eligible,coverage=base.training_source_coverage_filter(train_frame,candidate_columns)
    if not eligible:raise ValueError('No eligible electrical features')
    raw=base.numeric_frame(train_frame,eligible)
    X=pd.DataFrame(SimpleImputer(strategy='median').fit_transform(raw),
                   columns=eligible,index=train_frame.index)
    ranked=X.rank(method='average')
    scores=[]
    for b,z in train_frame.groupby('battery_id'):
        yy=pd.to_numeric(z[base.TARGET_COL],errors='coerce').rank()
        corr=X.loc[z.index].rank().corrwith(yy).abs().fillna(0.)
        scores.append(corr)
    target_scores=pd.concat(scores,axis=1).mean(axis=1)
    order=target_scores.sort_values(ascending=False,kind='mergesort').index.tolist()
    # Preserve pooled interfeature redundancy-pruning from the original pipeline.
    fc=ranked[order].corr().abs()
    return {'eligible_features':eligible,'coverage_audit':coverage,
            'target_scores':target_scores,'ranked_order':order,'feature_corr':fc}


def run(data:Path,features:Path,frozen_predictions:Path,output_dir:Path):
    output_dir.mkdir(parents=True,exist_ok=True)
    df=pd.read_csv(data,low_memory=False)
    ff=base.load_frozen_feature_list(features)
    allowed=[x for x in ff if x not in base.EXPLICIT_AGE_STRICT_FEATURES]
    frozen=pd.read_csv(frozen_predictions)
    original=base.prepare_feature_selector
    try:
        base.prepare_feature_selector=balanced_selector
        for name,with_age in [('STRICT_ELECTRICAL_HGB',False),('STRICT_ELECTRICAL_PLUS_AGE_HGB',True)]:
            result=base.run_formulation(df,name,allowed,with_age)
            for k,v in result.items():
                if isinstance(v,pd.DataFrame):v.to_csv(output_dir/f'{name}_{k}.csv',index=False)
            matched,per,stat=compare_frozen_predictions(
                frozen[frozen.Formulation.eq(name)],result['predictions'],
                ['battery_id','Cycle_Index'],base.TARGET_COL)
            for title,table in [('paired_predictions',matched),('paired_per_battery',per),('paired_summary',stat)]:
                table.to_csv(output_dir/f'{name}_{title}.csv',index=False)
    finally:
        base.prepare_feature_selector=original
    record_inputs(output_dir,{'data':data,'features':features,'frozen_predictions':frozen_predictions},
                  {'ranking':'Equal mean of within-battery absolute Spearman associations',
                   'other_hgb_procedure':'Original nested battery-level pipeline'})


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',type=Path,default=Path('outputs/constant_dataset/constant_observations.csv'))
    p.add_argument('--features',type=Path,default=Path('outputs/constant_dataset/constant_strict_feature_names.csv'))
    p.add_argument('--frozen-predictions',type=Path,default=Path('outputs/constant_hgb/constant_hgb_predictions.csv'))
    p.add_argument('--output-dir',type=Path,default=Path('outputs/battery_balanced_ranking'))
    a=p.parse_args();run(a.data,a.features,a.frozen_predictions,a.output_dir)
if __name__=='__main__':main()
