"""Build plotting-input CSV for Figures 3, 4, 5 from local result summaries.

The values are read from executable experiment outputs; no published number
is silently inserted when a result CSV is absent.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd

ELECTRICAL='STRICT_ELECTRICAL_HGB'
WITH_AGE='STRICT_ELECTRICAL_PLUS_AGE_HGB'


def from_macro(path:Path,form:str):
    df=pd.read_csv(path)
    row=df[df.Formulation.eq(form)]
    if len(row)!=1:raise ValueError(f'{path}: not exactly one row for {form}')
    for col in ['Macro_RMSE_Raw','Equal_Battery_RMSE','Macro_RMSE']:
        if col in row:
            return float(row.iloc[0][col])
    raise ValueError(f'{path}: no RMSE field recognized')


def run(primary:Path,fixed600:Path,voltage36:Path,source_only:Path,
        transfer:Path,sequence:Path,out:Path):
    records=[]
    for with_age,form,panel in [(False,ELECTRICAL,'Electrical inputs only'),
                                (True,WITH_AGE,'With explicit age')]:
        for setting,file in [('Full discharge',primary),('First 600 s',fixed600),('True 3.6 V',voltage36)]:
            # Censored scripts save per-formulation CSVs rather than one combined file.
            if setting!='Full discharge':
                file=file/f'{form}_macro.csv'
            records.append({'Figure':3,'Panel':panel,'Model':setting,
                            'RMSE':from_macro(file,form)})
    s=pd.read_csv(source_only);tr=pd.read_csv(transfer)
    for destination,origin in [('CALCE','NASA'),('NASA','CALCE')]:
        panel=f'{origin} → {destination}'
        for form,label in [(ELECTRICAL,'Electrical only'),(WITH_AGE,'Electrical + age')]:
            src=s[(s.Source==destination)&(s.Formulation==form)]
            ts=tr[(tr.Train_Source==origin)&(tr.Test_Source==destination)&(tr.Formulation==form)]
            if len(src)!=1 or len(ts)!=1:raise ValueError('Missing source-only / transfer metric row')
            records.extend([
                {'Figure':4,'Panel':panel,'Model':f'{label}: within source',
                 'RMSE':float(src.iloc[0]['Macro_RMSE_Raw'])},
                {'Figure':4,'Panel':panel,'Model':f'{label}: zero shot',
                 'RMSE':float(ts.iloc[0]['Macro_RMSE_Raw'])}])
    nn=pd.read_csv(sequence)
    for with_age,form,panel in [(False,ELECTRICAL,'Electrical inputs only'),
                                (True,WITH_AGE,'With explicit age')]:
        records.append({'Figure':5,'Panel':panel,'Model':'HGB engineered',
                        'RMSE':from_macro(voltage36/f'{form}_macro.csv',form)})
        for family in ['LSTM','Transformer']:
            key=f'{family}_{"WITH_AGE" if with_age else "ELECTRICAL"}'
            sub=nn[nn.Formulation.eq(key)]
            if len(sub)!=1:raise ValueError('Missing neural sequence model: '+key)
            records.append({'Figure':5,'Panel':panel,'Model':family+' raw prefix',
                            'RMSE':float(sub.iloc[0].Equal_Battery_RMSE)})
    out.parent.mkdir(parents=True,exist_ok=True)
    pd.DataFrame(records).to_csv(out,index=False)
    return out


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--primary-macro',type=Path,default=Path('outputs/constant_hgb/constant_hgb_macro.csv'))
    p.add_argument('--fixed-600s-dir',type=Path,default=Path('outputs/censored_600s'))
    p.add_argument('--voltage-3p6-dir',type=Path,default=Path('outputs/censored_3p6'))
    p.add_argument('--source-only',type=Path,default=Path('outputs/source_only_hgb/source_only_hgb_summary.csv'))
    p.add_argument('--transfer',type=Path,default=Path('outputs/constant_transfer/constant_transfer_summary.csv'))
    p.add_argument('--sequence',type=Path,default=Path('outputs/sequence_3p6_results/sequence_3p6_macro.csv'))
    p.add_argument('--output',type=Path,default=Path('outputs/manuscript_figures/figure_comparison_metrics.csv'))
    a=p.parse_args()
    print(run(a.primary_macro,a.fixed_600s_dir,a.voltage_3p6_dir,a.source_only,a.transfer,a.sequence,a.output))
if __name__=='__main__':main()
