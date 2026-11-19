#!/usr/bin/env python3
"""Portable v0.9.0 manifest validation + archived numerical checks.

The manifest has 12 known CSVs originally hashed with CRLF bytes. Git's
Linux checkout may normalize them to LF. Only the 12 verified paths can pass
an EXACT-reversal check. Does not change the v0.9.0 manifest or any file.
All unexpected differences raise an error.
"""
from __future__ import annotations
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

KNOWN_CRLF_CONVERSION = {
 'research_release/ARTIFACT_INVENTORY.csv',
 'research_release/MANUSCRIPT_ARTIFACT_MAP.csv',
 'research_release/data/step11o_constant_6299_soh_target_values.csv',
 'research_release/results/step11o/Step11O_AllFive_Model_Predictions.csv',
 'research_release/results/step11o/Step11O_EqualBattery_Summary.csv',
 'research_release/results/step11o/Step11O_OuterFold_Configs_and_Selected_Features.csv',
 'research_release/results/step11o/Step11O_Paired_Battery_Bootstrap.csv',
 'research_release/results/step11o/Step11O_PerBattery.csv',
 'research_release/results/step11o/Step11O_SoH_Strata_PostEvaluation.csv',
 'research_release/splits/step11o_constant_inner_design.csv',
 'research_release/splits/step11o_constant_outer_design.csv',
 'research_release/splits/variable_outer_design.csv',
}

def check_manifest(root:Path):
    results=[]
    for line in (root/'research_release/MANIFEST.sha256').read_text().splitlines():
        if not line.strip():continue
        expected,rel=line.split('  ',1)
        file=root/rel
        if not file.is_file():raise FileNotFoundError(file)
        data=file.read_bytes()
        got=hashlib.sha256(data).hexdigest()
        if got==expected: status='EXACT'
        elif rel in KNOWN_CRLF_CONVERSION and b'\r\n' not in data and hashlib.sha256(data.replace(b'\n',b'\r\n')).hexdigest()==expected:
            status='PROVEN_EOL_ONLY_LF_TO_CRLF'
        else: raise ValueError(f'Unexpected SHA256 mismatch: {rel}')
        results.append({'path':rel,'status':status})
    if len(results)!=57:raise ValueError('Expected v0.9.0 manifest of 57 entries')
    exact=sum(x['status']=='EXACT' for x in results)
    print(f'Manifest: {exact} exact, {57-exact} proven LF/CRLF reversal, 0 unexplained')
    return results

def main(repo:Path,output:Path|None):
    repo=repo.resolve()
    paths=check_manifest(repo)
    path=repo/'research_release/tools/verify_release.py'
    spec=importlib.util.spec_from_file_location('original_verify_release',path)
    old=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(old)
    # Use the unchanged original numerical tests, replacing only its manifest
    # callback with a validated, controlled EOL proof (not a disabled check).
    old.check_manifest=lambda: None  # already checked above: aborts on any unexplained file change
    old.main()
    result={'status':'PASS','manifest_classifications':paths,'archived_numerical_checks':'PASS','raw_refits_verified':False}
    if output:
        output.parent.mkdir(parents=True,exist_ok=True)
        output.write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    return result

if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--repo-root',type=Path,default=Path(__file__).resolve().parents[2])
    p.add_argument('--json-output',type=Path)
    a=p.parse_args()
    try:main(a.repo_root,a.json_output)
    except Exception as e:
        print(f'FAIL: {e}',file=sys.stderr)
        raise
