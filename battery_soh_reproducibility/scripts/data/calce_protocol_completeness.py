"""Verify all CALCE constant-protocol tester-step cycles from original raw ZIP.

This is a post hoc feasibility screen. No target/SoH value influences eligibility.
The result is an auditable CSV, not an automatic recommendation to delete rows.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd
from battery_soh_repro.data_pipeline import (CALCE_CONSTANT_BATTERIES,
                                               prepare_raw_archive, process_files)
from battery_soh_repro.protocol_qc import screen_battery
from battery_soh_repro.experiment_utils import record_inputs


def run(raw_zip: Path, work_dir: Path, output_dir: Path,
        frozen_data: Path | None = None) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest, _ = prepare_raw_archive(raw_zip, work_dir)
    summaries, runs = [], []
    for battery, sheets in CALCE_CONSTANT_BATTERIES:
        print('Protocol screen:', battery, flush=True)
        raw = process_files(str(manifest[battery]['folder']), sheets)
        qc, steps = screen_battery(raw, battery)
        summaries.append(qc); runs.append(steps)
    table = pd.concat(summaries, ignore_index=True)
    checks = pd.concat(runs, ignore_index=True)
    record_inputs(output_dir, {'raw_zip': raw_zip, **({'frozen_data': frozen_data} if frozen_data else {})},
                  {'CV_end_current_A': .06, 'discharge_end_V': 2.75,
                   'active_current_A': .02, 'post_hoc': True})
    paths = {'cycles': output_dir/'calce_protocol_screen.csv',
             'runs': output_dir/'calce_tester_step_runs.csv',
             'summary': output_dir/'calce_screen_summary.csv'}
    table.to_csv(paths['cycles'], index=False)
    checks.to_csv(paths['runs'], index=False)
    counts = table.groupby('battery_id').Quality_Pass.agg(['count', 'sum']).reset_index()
    counts['failed'] = counts['count'] - counts['sum']
    counts.to_csv(paths['summary'], index=False)
    if frozen_data:
        frozen = pd.read_csv(frozen_data, usecols=['battery_id','source_dataset','Cycle_Index',
                                                    'Measured_Discharge_Capacity_Ah'])
        calce = frozen[frozen.source_dataset == 'CALCE'].copy()
        a = calce.merge(table, on=['battery_id','Cycle_Index'], how='left',
                        validate='one_to_one', indicator=True)
        if not a['_merge'].eq('both').all(): raise ValueError('Frozen CALCE observations missing from protocol audit')
        a['Absolute_Difference_Ah'] = abs(a.Measured_Discharge_Capacity_Ah - a.Original_RightSample_Discharge_Ah)
        a.to_csv(output_dir/'frozen_capacity_reproduction.csv', index=False)
        if a.Absolute_Difference_Ah.max() > 1e-5:
            raise ValueError('Raw capacity reconstruction differs from frozen reference')
        print('Audited study cycles:', len(a), 'Excluded:', int((~a.Quality_Pass).sum()))
    return paths


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--raw-zip', type=Path, required=True)
    p.add_argument('--work-dir', type=Path, default=Path('work/calce_qc'))
    p.add_argument('--output-dir', type=Path, default=Path('outputs/calce_qc'))
    p.add_argument('--frozen-data', type=Path)
    a=p.parse_args()
    print(run(a.raw_zip,a.work_dir,a.output_dir,a.frozen_data))

if __name__=='__main__':main()
