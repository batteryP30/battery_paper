#!/usr/bin/env python3
"""Run the four supported archived-evidence verifiers and summarize the current scope."""
from pathlib import Path
import argparse, subprocess, sys

ap = argparse.ArgumentParser()
ap.add_argument('--repo-root', type=Path, default=Path(__file__).resolve().parents[2])
args = ap.parse_args()
root = args.repo_root.resolve()
for name in ['verify_calce_xlsx_evidence.py', 'verify_nasa_v2_evidence.py',
             'verify_editor_chain.py', 'verify_release_portable.py']:
    print(f'\n===== {name} =====', flush=True)
    rc=subprocess.call([sys.executable, str(root/'research_release/tools'/name), '--repo-root', str(root)], cwd=root)
    if rc != 0:
        print(f'FAIL {name} exited {rc}', file=sys.stderr)
        sys.exit(rc)
print('\nALL CURRENT EDITOR EVIDENCE CHECKS PASSED')
print('SOURCE LEVEL: 231 CALCE XLSX and 8 NASA study CSVs were reconstructed and compared in Colab; preserved output tables independently verified here.')
print('ARCHIVED RESULTS: capacity/SoH labels, held-out splits, saved model choices, and Step 11N/11O out-of-fold predictions verified.')
print('BOUNDARIES: native NASA-to-study-CSV converter, logger-to-CALCE-XLSX provenance, and complete historical model refits not independently verified.')
print('Prior v0.9.1 reports that CALCE was not rerun are historical; the present addendum documents the 2026-11-19 source-workbook verification.')
