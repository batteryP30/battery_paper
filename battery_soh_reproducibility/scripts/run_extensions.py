"""Opt-in local workflows for additional published battery SoH experiments.

The original `scripts/run_all.py` remains unchanged. Run it first to create
primary `outputs/constant_dataset`, `outputs/variable_dataset`, and the frozen
HGB predictions; this runner then extends those results without overwriting.
All stages execute as separate Python processes and preserve their CSV outputs.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import subprocess
import sys

STAGES=['proxy','age','transfer_baselines','common_subset','calce_qc','screened_features',
        'screened_hgb','censored_600s','censored_3p6','sequence_prepare',
        'sequence_train','temperature','battery_balanced_ranking','electrical_linear']


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--stages',nargs='+',choices=STAGES,required=True)
    p.add_argument('--raw-zip',type=Path,
                   help='Required for raw-derived stages: QC, screened, censored, sequence preparation, temperature')
    p.add_argument('--output-root',type=Path,default=Path('outputs'))
    p.add_argument('--work-root',type=Path,default=Path('work/extensions'))
    p.add_argument('--device',default='cpu')
    a=p.parse_args()
    out=a.output_root.resolve(); work=a.work_root.resolve()
    out.mkdir(parents=True,exist_ok=True);work.mkdir(parents=True,exist_ok=True)
    root=Path(__file__).resolve().parents[1]
    scripts=root/'scripts'
    constant=out/'constant_dataset/constant_observations.csv'
    variable=out/'variable_dataset/variable_independent_observations.csv'
    const_feats=out/'constant_dataset/constant_strict_feature_names.csv'
    primary_pred=out/'constant_hgb/constant_hgb_predictions.csv'
    variable_pred=out/'variable_hgb/variable_hgb_predictions.csv'
    baseline_const=out/'constant_dataset/constant_baseline_predictions.csv'
    baseline_variable=out/'variable_dataset/variable_baseline_predictions.csv'
    screen=out/'calce_qc/calce_protocol_screen.csv'
    screened=out/'quality_screened_dataset/screened_constant_observations.csv'
    raw_stages={'calce_qc','screened_features','censored_600s','censored_3p6',
                'sequence_prepare','temperature'}
    if raw_stages.intersection(a.stages) and (a.raw_zip is None or not a.raw_zip.is_file()):
        p.error('--raw-zip must point to a real raw-data ZIP for the requested stages')
    raw=str(a.raw_zip.resolve()) if a.raw_zip else ''
    def launch(script:str,*args):
        cmd=[sys.executable,str(scripts/script),*map(str,args)]
        print('\nRunning:', ' '.join(cmd),flush=True)
        subprocess.run(cmd,check=True,cwd=root)
    for stage in a.stages:
        if stage=='proxy':
            launch('analysis/full_discharge_proxy_audit.py','--data',constant,'--features',const_feats,
                   '--output-dir',out/'full_discharge_proxy_audit')
        elif stage=='age':
            for protocol,d in [('constant',constant),('variable',variable)]:
                launch('modeling/age_only_hgb.py','--protocol',protocol,'--data',d,
                       '--output-dir',out/'age_only_hgb')
        elif stage=='transfer_baselines':
            launch('modeling/transfer_age_mean_baselines.py','--data',constant,'--output-dir',out/'age_mean_transfer')
        elif stage=='common_subset':
            for protocol,d,b,h,age in [('constant',constant,baseline_const,primary_pred,
                     out/'age_only_hgb/constant_age_only_hgb_predictions.csv'),
                     ('variable',variable,baseline_variable,variable_pred,
                     out/'age_only_hgb/variable_age_only_hgb_predictions.csv')]:
                launch('analysis/common_subset_baselines.py','--protocol',protocol,'--data',d,
                       '--baseline-predictions',b,'--hgb-predictions',h,
                       '--age-predictions',age,'--output-dir',out/'common_subset')
        elif stage=='calce_qc':
            launch('data/calce_protocol_completeness.py','--raw-zip',raw,'--frozen-data',constant,
                   '--work-dir',work/'calce_qc','--output-dir',out/'calce_qc')
        elif stage=='screened_features':
            launch('data/build_quality_screened_dataset.py','--raw-zip',raw,
                   '--frozen-data',constant,'--frozen-features',const_feats,
                   '--protocol-screen',screen,'--work-dir',work/'screened',
                   '--output-dir',out/'quality_screened_dataset')
        elif stage=='screened_hgb':
            launch('modeling/quality_screened_lobo.py','--screened',screened,
                   '--features',const_feats,'--frozen-predictions',primary_pred,
                   '--output-dir',out/'quality_screened_lobo')
        elif stage in ('censored_600s','censored_3p6'):
            which='seconds' if stage=='censored_600s' else 'voltage'
            launch('modeling/censored_observation_lobo.py','--raw-zip',raw,
                   '--frozen-data',constant,'--frozen-features',const_feats,
                   '--frozen-predictions',primary_pred,'--boundary',which,
                   '--work-dir',work/stage,'--output-dir',out/stage)
        elif stage=='sequence_prepare':
            launch('modeling/sequence_3p6.py','--stage','prepare','--raw-zip',raw,
                   '--observations',constant,'--work-dir',work/'sequence',
                   '--data-dir',out/'sequence_3p6_input')
        elif stage=='sequence_train':
            launch('modeling/sequence_3p6.py','--stage','train',
                   '--data-dir',out/'sequence_3p6_input','--output-dir',out/'sequence_3p6_results',
                   '--device',a.device)
        elif stage=='temperature':
            launch('analysis/variable_temperature_feasibility.py','--raw-zip',raw,
                   '--output-dir',out/'variable_temperature')
        elif stage=='battery_balanced_ranking':
            launch('analysis/battery_balanced_ranking.py','--data',constant,'--features',const_feats,
                   '--frozen-predictions',primary_pred,'--output-dir',out/'battery_balanced_ranking')
        elif stage=='electrical_linear':
            launch('modeling/electrical_linear_ridge.py','--project-root',out,
                   '--output-dir',out/'electrical_linear_baseline')
    print('\nSelected extension stages completed. Unrequested stages were not run.')

if __name__=='__main__':main()
