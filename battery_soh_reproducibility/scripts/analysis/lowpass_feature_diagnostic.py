"""Inspect matched five-sample step-wise smoothing prior to derivative extraction.

This is a raw-data diagnostic, NOT a replacement for the archived full
Step 11HB matched-training/noise-retraining experiment. It compares electrical
features obtained from the same raw cycle before/after the fixed filter.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from battery_soh_repro.censoring import centered_segment_filter
from battery_soh_repro.data_pipeline import (prepare_raw_archive,
                 CALCE_CONSTANT_BATTERIES,NASA_CONSTANT_BATTERIES,
                 process_files,load_nasa_constant_measurements_targetfree,build_cycle_features_run2)
from battery_soh_repro.experiment_utils import record_inputs


def run(raw_zip:Path,work_dir:Path,output_dir:Path,window:int=5):
    output_dir.mkdir(parents=True,exist_ok=True)
    man,_=prepare_raw_archive(raw_zip,work_dir)
    reports=[]
    for b,sheets in CALCE_CONSTANT_BATTERIES+[(b,None) for b in NASA_CONSTANT_BATTERIES]:
        raw=(process_files(str(man[b]['folder']),sheets) if sheets is not None else
             load_nasa_constant_measurements_targetfree(man[b]['csv_path'],b))
        filt=centered_segment_filter(raw,window=window)
        clean=build_cycle_features_run2(raw,step_col='Step_Index')
        filtered=build_cycle_features_run2(filt,step_col='Step_Index')
        derivative=[c for c in clean.columns if c in filtered and
                    any(t in c.lower() for t in ['dvdt','slope','dynamic'])]
        joined=clean[['Cycle_Index']+derivative].merge(
            filtered[['Cycle_Index']+derivative],on='Cycle_Index',validate='one_to_one',
            suffixes=('_raw','_filtered'))
        for c in derivative:
            x=pd.to_numeric(joined[c+'_raw'],errors='coerce').to_numpy(float)
            y=pd.to_numeric(joined[c+'_filtered'],errors='coerce').to_numpy(float)
            good=np.isfinite(x)&np.isfinite(y)
            if good.any():
                reports.append({'Battery':b,'Feature':c,'Compared_Cycles':int(good.sum()),
                   'Median_Absolute_Change':float(np.median(abs(x[good]-y[good]))),
                   'Median_Absolute_Raw':float(np.median(abs(x[good]))),
                   'Median_Absolute_Filtered':float(np.median(abs(y[good])))})
    pd.DataFrame(reports).to_csv(output_dir/'filtered_derivative_change_by_battery.csv',index=False)
    record_inputs(output_dir,{'raw_zip':raw_zip},
                  {'filter':'centered stepwise boxcar','window_samples':window,
                   'only_descriptive_no_model_scores':True})


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--raw-zip',type=Path,required=True)
    p.add_argument('--work-dir',type=Path,default=Path('work/lowpass'))
    p.add_argument('--output-dir',type=Path,default=Path('outputs/lowpass_feature_diagnostic'))
    p.add_argument('--window',type=int,default=5)
    a=p.parse_args();run(a.raw_zip,a.work_dir,a.output_dir,a.window)
if __name__=='__main__':main()
