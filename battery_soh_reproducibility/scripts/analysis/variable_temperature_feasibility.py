# Reviewer temperature feasibility audit: CS2_3, CS2_9, RW3-RW6
# Local Python source.
# NO model training. Reads raw source archive, never uses SOH to select channels.


import io
import os
import re
import zipfile
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from openpyxl import load_workbook

# Local CLI settings, populated in __main__.
PROJECT_ROOT = Path('.')
DRIVE_ZIP = Path('raw_data/collab.zip')
OUTPUT_DIR = Path('outputs/variable_temperature')
LOCAL_OUTPUT = OUTPUT_DIR
AUTO_DOWNLOAD = False
ZIP_OVERRIDE = ''

CALCE_SHEETS = {
    'CS2_3': {'Channel_1-008'},
    'CS2_9': {'Channel_1-007', 'Channel_1-007_1', 'Channel_1-007_2'},
}
NASA_BATTERIES = ['RW3', 'RW4', 'RW5', 'RW6']
BATTERIES = ['CS2_3', 'CS2_9'] + NASA_BATTERIES
EXPECTED_PRIMARY_OBS = {'CS2_3': 26, 'CS2_9': 92, 'RW3': 13, 'RW4': 12, 'RW5': 13, 'RW6': 9}

# These are broad quality-screening bounds, NOT a calibration or rejection rule.
PLAUSIBLE_TEMP_C = (-40.0, 100.0)
MIN_VALID_COVERAGE = 0.95
RANDOM_SEED = 42


def normalized(s):
    return ''.join(ch for ch in str(s).lower() if ch.isalnum())


def temperature_semantics(col):
    key = normalized(col)
    if not ('temp' in key or 'thermistor' in key or 'thermocouple' in key):
        return None
    if any(t in key for t in ('timestamp', 'temperaturetime', 'temptime')):
        return None
    if any(t in key for t in ('ambient', 'chamber', 'room', 'environment')):
        return 'AMBIENT_OR_CHAMBER'
    if any(t in key for t in ('cell', 'battery', 'measured', 'surface', 'thermistor', 'thermocouple')):
        return 'CELL_OR_MEASURED'
    return 'GENERIC_UNSPECIFIED'


def candidate_columns(cols):
    return [(str(c), temperature_semantics(c)) for c in cols if temperature_semantics(c)]


def select_best(candidates):
    order = {'CELL_OR_MEASURED': 0, 'GENERIC_UNSPECIFIED': 1, 'AMBIENT_OR_CHAMBER': 2}
    if not candidates:
        return None
    return sorted(candidates, key=lambda v: (order.get(v[1], 9), v[0].lower()))[0]


def find_column(columns, options):
    lookup = {normalized(c): str(c) for c in columns}
    return next((lookup[normalized(c)] for c in options if normalized(c) in lookup), None)


class RunningStats:
    """Exact finite mean/std/min/max and sample counts, bounded-memory quantile sample."""
    def __init__(self, seed=42, keep=5000):
        self.rows = 0
        self.finite_n = 0
        self.plausible_n = 0
        self.zero_n = 0
        self.n = 0
        self.mean = 0.0
        self.m2 = 0.0
        self.minimum = np.inf
        self.maximum = -np.inf
        self.keep = keep
        self.sample = []
        self.seen = 0
        self.rng = np.random.default_rng(seed)

    def add(self, values):
        x = pd.to_numeric(pd.Series(values), errors='coerce').to_numpy(dtype=float)
        self.rows += len(x)
        x = x[np.isfinite(x)]
        if not len(x):
            return
        self.finite_n += len(x)
        self.plausible_n += int(((x >= PLAUSIBLE_TEMP_C[0]) & (x <= PLAUSIBLE_TEMP_C[1])).sum())
        self.zero_n += int((np.abs(x) < 0.001).sum())
        self.minimum = min(self.minimum, float(x.min()))
        self.maximum = max(self.maximum, float(x.max()))
        nx = len(x)
        mx = float(x.mean())
        m2x = float(np.square(x - mx).sum())
        combined = self.n + nx
        delta = mx - self.mean
        self.mean += delta * nx / combined
        self.m2 += m2x + delta**2 * self.n * nx / combined
        self.n = combined
        # Bounded reservoir, deterministic across a fixed input order.
        for val in x:
            self.seen += 1
            if len(self.sample) < self.keep:
                self.sample.append(float(val))
            else:
                pos = int(self.rng.integers(self.seen))
                if pos < self.keep:
                    self.sample[pos] = float(val)

    def summary(self):
        samp = np.asarray(self.sample, dtype=float)
        return {
            'Rows_Examined': self.rows,
            'Finite_Samples': self.finite_n,
            'Finite_Coverage': self.finite_n / self.rows if self.rows else np.nan,
            'Plausible_Sample_Fraction': self.plausible_n / self.finite_n if self.finite_n else np.nan,
            'Near_Zero_Fraction': self.zero_n / self.finite_n if self.finite_n else np.nan,
            'Mean_C_Assumed': self.mean if self.n else np.nan,
            'Std_C_Assumed': np.sqrt(self.m2 / (self.n - 1)) if self.n > 1 else np.nan,
            'Min_C_Assumed': self.minimum if self.n else np.nan,
            'Max_C_Assumed': self.maximum if self.n else np.nan,
            'Median_C_Est': float(np.median(samp)) if len(samp) else np.nan,
            'IQR_C_Est': float(np.quantile(samp, .75) - np.quantile(samp, .25)) if len(samp) else np.nan,
        }


def setup_input():
    source=Path(ZIP_OVERRIDE) if ZIP_OVERRIDE else DRIVE_ZIP
    if not source.is_file():
        raise FileNotFoundError(f'Raw source archive missing: {source}')
    return source,False


def source_files(archive):
    inventory = {b: [] for b in BATTERIES}
    for item in archive.infolist():
        if item.is_dir():
            continue
        p = Path(item.filename.replace('\\', '/'))
        parent_names = {x.upper() for x in p.parts[:-1]}
        for b in CALCE_SHEETS:
            if b.upper() in parent_names and p.suffix.lower() in ('.xlsx', '.xlsm', '.xls') and not p.name.startswith('~$'):
                inventory[b].append(item)
        for b in NASA_BATTERIES:
            if p.name.lower() == f'{b.lower()}.csv':
                inventory[b].append(item)
    return inventory


def scan_calce(archive, bat, entries, candidates_out, file_out):
    all_found = []
    model_found = []
    stats = RunningStats(seed=RANDOM_SEED + len(bat))
    for file_no, entry in enumerate(sorted(entries, key=lambda e: e.filename.lower()), start=1):
        try:
            with archive.open(entry) as f:
                # Each CALCE Excel workbook is itself a ZIP. Read only one at a time.
                content = io.BytesIO(f.read())
            wb = load_workbook(content, read_only=True, data_only=True)
            for sheet in wb.worksheets:
                head = next(sheet.iter_rows(min_row=1, max_row=1, values_only=True), ())
                columns = [str(v) if v is not None else f'Unnamed_{i}' for i, v in enumerate(head)]
                found = candidate_columns(columns)
                is_model_sheet = sheet.title in CALCE_SHEETS[bat]
                best = select_best(found)
                all_found.extend(found)
                if is_model_sheet:
                    model_found.extend(found)
                file_out.append({
                    'Battery': bat, 'Source': 'CALCE', 'File': entry.filename,
                    'Sheet': sheet.title, 'Model_Sheet': is_model_sheet,
                    'Temp_Column_Count': len(found),
                    'Temp_Columns': '; '.join(c for c, _ in found),
                    'Status': 'CHECKED', 'Error': ''
                })
                for col, semantic in found:
                    candidates_out.append({
                        'Battery': bat, 'Source': 'CALCE', 'File': entry.filename,
                        'Sheet': sheet.title, 'Model_Sheet': is_model_sheet,
                        'Column': col, 'Semantic': semantic
                    })
                # Read values only if a channel exists in a model sheet.
                if is_model_sheet and best:
                    idx = columns.index(best[0])
                    for row in sheet.iter_rows(min_row=2, values_only=True):
                        stats.add([row[idx] if idx < len(row) else None])
            wb.close()
        except Exception as exc:
            file_out.append({
                'Battery': bat, 'Source': 'CALCE', 'File': entry.filename,
                'Sheet': '', 'Model_Sheet': '', 'Temp_Column_Count': np.nan,
                'Temp_Columns': '', 'Status': 'ERROR', 'Error': str(exc)[:400]
            })
    return model_found, all_found, stats


def scan_nasa(archive, bat, entries, candidates_out, file_out, cycle_out, operation_out):
    if len(entries) != 1:
        return [], {}, 'MISSING_OR_DUPLICATE_CSV'
    entry = entries[0]
    stats_by_col_op = defaultdict(RunningStats)
    stats_by_cycle_col = defaultdict(RunningStats)
    counts_op = defaultdict(int)
    try:
        with archive.open(entry) as fh:
            for chunk_no, chunk in enumerate(pd.read_csv(fh, chunksize=100000, low_memory=False)):
                if chunk_no == 0:
                    candidates = candidate_columns(chunk.columns)
                    operation_col = find_column(chunk.columns, ['operation_class'])
                    rw_cycle_col = find_column(chunk.columns, ['random_walk_cycle'])
                    file_out.append({
                        'Battery': bat, 'Source': 'NASA', 'File': entry.filename,
                        'Sheet': '(CSV)', 'Model_Sheet': True,
                        'Temp_Column_Count': len(candidates),
                        'Temp_Columns': '; '.join(c for c, _ in candidates),
                        'Status': 'CHECKED', 'Error': ''
                    })
                    for col, sem in candidates:
                        candidates_out.append({
                            'Battery': bat, 'Source': 'NASA', 'File': entry.filename,
                            'Sheet': '(CSV)', 'Model_Sheet': True,
                            'Column': col, 'Semantic': sem
                        })
                if operation_col is None:
                    ops = pd.Series(['OPERATION_UNSPECIFIED'] * len(chunk), index=chunk.index)
                else:
                    ops = chunk[operation_col].astype(str).str.strip().str.lower()
                for op, sub_idx in ops.groupby(ops).groups.items():
                    sub = chunk.loc[sub_idx]
                    counts_op[op] += len(sub)
                    for col, sem in candidates:
                        stats_by_col_op[(col, op)].add(sub[col])
                        if op == 'random_walk_discharge' and rw_cycle_col:
                            # Only operating-interval signals, not reference targets.
                            for cycle, block in sub.groupby(rw_cycle_col, dropna=True):
                                stats_by_cycle_col[(col, str(cycle))].add(block[col])
        for (col, op), stat in stats_by_col_op.items():
            operation_out.append({'Battery': bat, 'Column': col, 'Operation_Class': op, **stat.summary()})
        for (col, cycle), stat in stats_by_cycle_col.items():
            cycle_out.append({'Battery': bat, 'Cycle': cycle, 'Column': col, **stat.summary()})
        return candidates, stats_by_col_op, ''
    except Exception as exc:
        file_out.append({
            'Battery': bat, 'Source': 'NASA', 'File': entry.filename,
            'Sheet': '(CSV)', 'Model_Sheet': True,
            'Temp_Column_Count': np.nan, 'Temp_Columns': '',
            'Status': 'ERROR', 'Error': str(exc)[:400]
        })
        return [], {}, str(exc)


def quality_flag(summary):
    if not summary or not summary.get('Finite_Samples'):
        return 'NO_FINITE_SAMPLES'
    coverage = summary.get('Finite_Coverage', np.nan)
    plausible = summary.get('Plausible_Sample_Fraction', np.nan)
    zeros = summary.get('Near_Zero_Fraction', np.nan)
    iqr = summary.get('IQR_C_Est', np.nan)
    if not np.isfinite(coverage) or coverage < MIN_VALID_COVERAGE:
        return 'LOW_FINITE_COVERAGE'
    if not np.isfinite(plausible) or plausible < .99:
        return 'IMPLAUSIBLE_TEMPERATURE_VALUES'
    if np.isfinite(zeros) and zeros > .10:
        return 'POSSIBLE_ZERO_SENTINELS'
    if np.isfinite(iqr) and iqr < .05:
        return 'NEAR_CONSTANT_TEMPERATURE'
    return 'PLAUSIBILITY_SCREEN_PASSED'


def main():
    source, drive_ready = setup_input()
    LOCAL_OUTPUT.mkdir(parents=True, exist_ok=True)
    print('Raw archive:', source)
    print('No SOH is read. No model is fitted. NASA reference segments are excluded from cycle-level metrics.')

    candidate_rows, inventory_rows, cycles, operations, batteries = [], [], [], [], []
    with zipfile.ZipFile(source, 'r') as arc:
        manifest = source_files(arc)
        for bat in BATTERIES:
            entries = manifest[bat]
            print(f'\n{bat}: {len(entries)} raw source file(s)')
            if bat in CALCE_SHEETS:
                model_found, all_found, stats = scan_calce(arc, bat, entries, candidate_rows, inventory_rows)
                errors = [r for r in inventory_rows if r['Battery'] == bat and r['Status'] == 'ERROR']
                if not entries:
                    status = 'FILES_NOT_LOCATED'
                elif errors:
                    status = 'INCOMPLETE_FILE_SCAN'
                elif not model_found:
                    status = 'NO_TEMPERATURE_IN_MODEL_CHANNELS'
                elif all(s == 'AMBIENT_OR_CHAMBER' for _, s in model_found):
                    status = 'AMBIENT_ONLY'
                else:
                    status = quality_flag(stats.summary())
                chosen = select_best(model_found)
                summary = stats.summary()
                batteries.append({
                    'Battery': bat, 'Source': 'CALCE', 'Expected_Primary_Observations': EXPECTED_PRIMARY_OBS[bat],
                    'Raw_Files_Found': len(entries), 'Columns_In_Model_Sheets': '; '.join(sorted(set(c for c, _ in model_found))),
                    'Columns_In_Any_Sheet': '; '.join(sorted(set(c for c, _ in all_found))),
                    'Selected_Channel': chosen[0] if chosen else '',
                    'Channel_Semantics': chosen[1] if chosen else '',
                    'Feasibility_Status': status, **summary
                })
            else:
                found, per_operation, error = scan_nasa(arc, bat, entries, candidate_rows, inventory_rows, cycles, operations)
                selected = select_best(found)
                if not entries:
                    status = 'FILES_NOT_LOCATED'
                    summary = {}
                elif error:
                    status = 'INCOMPLETE_FILE_SCAN'
                    summary = {}
                elif not selected:
                    status = 'NO_TEMPERATURE_FIELD'
                    summary = {}
                elif selected[1] == 'AMBIENT_OR_CHAMBER':
                    status = 'AMBIENT_ONLY'
                    summary = per_operation.get((selected[0], 'random_walk_discharge'), RunningStats()).summary()
                else:
                    summary = per_operation.get((selected[0], 'random_walk_discharge'), RunningStats()).summary()
                    status = quality_flag(summary)
                if bat == 'RW3' and status == 'PLAUSIBILITY_SCREEN_PASSED':
                    status = 'PASSES_NUMERIC_CHECKS_BUT_RW3_SENSOR_REPORTED_CORRUPT'
                batteries.append({
                    'Battery': bat, 'Source': 'NASA', 'Expected_Primary_Observations': EXPECTED_PRIMARY_OBS[bat],
                    'Raw_Files_Found': len(entries),
                    'Columns_In_Model_Sheets': '; '.join(sorted(set(c for c, _ in found))),
                    'Columns_In_Any_Sheet': '; '.join(sorted(set(c for c, _ in found))),
                    'Selected_Channel': selected[0] if selected else '',
                    'Channel_Semantics': selected[1] if selected else '',
                    'Feasibility_Status': status, **summary
                })
            print('  Status:', batteries[-1]['Feasibility_Status'])
            print('  Selected:', batteries[-1]['Selected_Channel'] or '(none)')

    summary_df = pd.DataFrame(batteries)
    incomplete = summary_df['Feasibility_Status'].isin(['FILES_NOT_LOCATED', 'INCOMPLETE_FILE_SCAN']).any()
    calce_good = summary_df.loc[summary_df['Source'] == 'CALCE', 'Feasibility_Status'].eq('PLAUSIBILITY_SCREEN_PASSED').all()
    nasa_good = summary_df.loc[summary_df['Source'] == 'NASA', 'Feasibility_Status'].eq('PLAUSIBILITY_SCREEN_PASSED').all()
    if incomplete:
        decision = 'INCOMPLETE: one or more raw sources could not be checked; do not claim missing temperature.'
    elif not calce_good:
        decision = 'NO: six-battery temperature-augmented LOBO is not comparable; CALCE model observations lack validated measured-cell temperature.'
    elif not nasa_good:
        decision = 'NO: six-battery temperature-augmented LOBO lacks validated comparable NASA operating-interval temperature.'
    else:
        decision = 'POTENTIALLY: inspect sensor semantics, units, chronology, and 70-100% population alignment before a model.'

    notes = [
        ('Objective', 'Reviewer 1: temperature mean/max/std feasibility for variable-protocol LOBO.'),
        ('Raw source', str(source)),
        ('Populations', 'Existing Step 11A expected primary observation counts are context only; raw audit does not align to 70-100% SoH subset.'),
        ('CALCE', 'All worksheets are scanned for temperature headers; only specified model channel sheets count toward valid predictor availability.'),
        ('NASA', 'Random-walk-discharge operating intervals are audited separately from reference/other operations.'),
        ('Statistics', 'Finite coverage, exact mean/std/min/max, plausible fraction; median/IQR estimated from bounded deterministic reservoir.'),
        ('Units', 'Temperature statistics are labelled Celsius-assumed; verify source units before interpreting magnitudes.'),
        ('Corruption', 'Passing numeric plausibility checks does not certify sensor validity; RW3 has published temperature-quality concerns.'),
        ('No leakage', 'No SOH or reference-discharge temperature is used to choose features or compute RW cycle summaries.'),
        ('Limitation', 'No cross-source augmented model is run by this code, even if some channels exist.'),
        ('Conservative rule', 'No temperature field is not equivalent to zero degrees or an ambient setting.'),
    ]
    local_xlsx = LOCAL_OUTPUT / 'Reviewer_Variable_Temperature_Feasibility_Audit.xlsx'
    with pd.ExcelWriter(local_xlsx, engine='xlsxwriter') as writer:
        sheets = [
            ('Battery_Summary', summary_df),
            ('Raw_File_Inventory', pd.DataFrame(inventory_rows)),
            ('Temperature_Columns', pd.DataFrame(candidate_rows)),
            ('NASA_Operation_Stats', pd.DataFrame(operations)),
            ('NASA_RW_Cycle_Stats', pd.DataFrame(cycles)),
            ('Methods', pd.DataFrame(notes, columns=['Topic', 'Details'])),
            ('Final_Decision', pd.DataFrame([{'Decision': decision, 'Source_File': str(source)}])),
        ]
        for sheet_name, table in sheets:
            if table.empty and not len(table.columns):
                table = pd.DataFrame({'Note': ['No records in this category']})
            table.to_excel(writer, sheet_name=sheet_name, index=False)
            worksheet = writer.sheets[sheet_name]
            worksheet.freeze_panes(1, 0)
            worksheet.autofilter(0, 0, len(table), max(0, len(table.columns) - 1))
            for j, col in enumerate(table.columns):
                width = min(50, max(13, len(str(col)) + 2))
                worksheet.set_column(j, j, width)

    # A graphic about presence/absence, not about predictive performance.
    fig, ax = plt.subplots(figsize=(10, 4.5))
    vals = summary_df['Finite_Coverage'].fillna(0).to_numpy(dtype=float)
    ax.bar(summary_df['Battery'], vals * 100)
    ax.set_ylim(0, 105)
    ax.set_ylabel('Finite raw temperature coverage (%)')
    ax.set_xlabel('Battery')
    ax.set_title('Temperature availability in model observation channels')
    for idx, value in enumerate(vals):
        label = f'{value * 100:.1f}%' if pd.notna(summary_df.iloc[idx]['Selected_Channel']) and summary_df.iloc[idx]['Selected_Channel'] else 'No channel'
        ax.text(idx, max(2, value * 100 + 2), label, ha='center', fontsize=9)
    fig.tight_layout()
    local_png = LOCAL_OUTPUT / 'Variable_Temperature_Availability.png'
    fig.savefig(local_png, dpi=200)
    plt.close(fig)

    print('\n' + '=' * 80)
    print('FEASIBILITY DECISION:', decision)
    print(summary_df[['Battery', 'Raw_Files_Found', 'Selected_Channel', 'Feasibility_Status', 'Finite_Coverage']].to_string(index=False))
    print('Results:', local_xlsx, local_png)

    return summary_df


if __name__ == '__main__':
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--raw-zip',type=Path,required=True)
    parser.add_argument('--output-dir',type=Path,default=Path('outputs/variable_temperature'))
    args=parser.parse_args()
    ZIP_OVERRIDE=str(args.raw_zip.resolve())
    LOCAL_OUTPUT=OUTPUT_DIR=args.output_dir.resolve()
    main()

