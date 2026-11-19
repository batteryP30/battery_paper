#!/usr/bin/env python3
"""Verify frozen CALCE XLSX reconstruction evidence against Step 11A + Step 11O.

This is an archived evidence verification. The full source-workbook reconstruction
is performed by notebooks/CALCE_Source_to_Target_Verification_Colab.ipynb using
user-provided collab.zip.
"""
from __future__ import annotations
import argparse, hashlib, json, sys
from pathlib import Path

import numpy as np
import pandas as pd

BATTERIES = ['CS2_33','CS2_34','CS2_35','CS2_36','CS2_37','CS2_38','CS2_9','CS2_3']
CONST = set(BATTERIES[:6]); EPS=1e-9


def need(condition,msg):
    if not condition: raise AssertionError(msg)


def compare_frames(a:pd.DataFrame, b:pd.DataFrame, key, desc):
    need(not a.duplicated(key).any(), f'{desc}: duplicate source keys')
    need(not b.duplicated(key).any(), f'{desc}: duplicate archived keys')
    need(set(a.columns)==set(b.columns),f'{desc}: columns differ: {set(a.columns)^set(b.columns)}')
    join=a.merge(b,on=key,how='outer',suffixes=('_source','_archived'),indicator=True,validate='one_to_one')
    need(join['_merge'].eq('both').all(),f'{desc}: unmatched source/archived keys: {join["_merge"].value_counts().to_dict()}')
    for c in (set(a.columns)-set(key)):
        x,y=join[f'{c}_source'],join[f'{c}_archived']
        if pd.api.types.is_numeric_dtype(x) and pd.api.types.is_numeric_dtype(y):
            dx=pd.to_numeric(x,errors='coerce').astype(float)
            dy=pd.to_numeric(y,errors='coerce').astype(float)
            ok=np.isclose(dx,dy,rtol=1e-11,atol=EPS,equal_nan=True)
        else:
            ok=x.fillna('<MISSING>').astype(str).eq(y.fillna('<MISSING>').astype(str)).to_numpy()
        need(bool(np.all(ok)),f'{desc}: mismatch in {c} for {int(np.sum(~ok))} rows')
    return len(join)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--repo-root',type=Path,default=Path(__file__).resolve().parents[2]); a=ap.parse_args()
    repo=a.repo_root.resolve()
    proof=repo/'research_release/editor_traceability/calce_xlsx_v1'
    parent=repo/'research_release/editor_traceability'
    manifest=proof/'EVIDENCE.sha256'
    need(manifest.is_file(),'Evidence hash manifest not found')
    files=[]
    for line in manifest.read_text(encoding='utf-8').splitlines():
        if not line.strip(): continue
        expected,name=line.split('  ',1); file=proof/name
        need(file.is_file(),f'file missing: {name}')
        need(hashlib.sha256(file.read_bytes()).hexdigest()==expected,f'file modified: {name}')
        files.append(name)
    need(len(files)==24,f'expected 24 immutable Colab evidence files, received {len(files)}')
    manifest_file=pd.read_csv(proof/'CALCE_Source_Workbook_SHA256.csv')
    need(len(manifest_file)==231,'expected 231 workbook checksums')
    need(manifest_file.archive_member.is_unique,'duplicate workbook archive member')
    need(manifest_file.sha256.astype(str).str.fullmatch('[0-9a-f]{64}').all(),'invalid SHA-256 workbook fingerprint')
    counts=manifest_file.groupby('battery_id').size().to_dict()
    need(set(counts)==set(BATTERIES),'unexpected/missing CALCE battery IDs')
    report=json.loads((proof/'CALCE_Validation_Report.json').read_text(encoding='utf-8'))
    need(report.get('all_pass') is True and report.get('calce_workbook_count')==231,'Colab report not in passed state')
    summary=pd.read_csv(proof/'CALCE_Source_Verification_Summary.csv')
    need(set(summary.battery_id)==set(BATTERIES),'wrong summary battery population')
    total_const=total_var=qc_const=qc_var=prim_const=prim_var=0
    for battery in BATTERIES:
        out=pd.read_csv(proof/f'{battery}_regenerated_targets.csv')
        original=pd.read_csv(parent/('step11a_constant_cycle_capacity_audit.csv' if battery in CONST else 'step11a_calce_variable_reference_audit.csv'))
        original=original.loc[original.battery_id.eq(battery)].copy()
        n=compare_frames(out,original,['battery_id','Cycle_Index'],battery)
        summ=summary.loc[summary.battery_id.eq(battery)].iloc[0]
        need(bool(summ.passed) and n==int(summ.regenerated_rows)==int(summ.archived_rows),f'{battery}: row count mismatch')
        qref_col='Reference_Capacity_Ah' if battery in CONST else 'Baseline_Reference_Capacity_Ah'
        qref=float(out[qref_col].dropna().iloc[0]); need(abs(qref-float(summ.qref_ah))<EPS,f'{battery}: Qref mismatch')
        qc=int(out.Raw_QC_Pass.sum()); primary=int(out.In_Primary_70_100_Scope.sum())
        need(qc==int(summ.pre_scope_observations) and primary==int(summ.primary_70_100),f'{battery}: QC and primary count mismatch')
        if battery in CONST: total_const+=n;qc_const+=qc;prim_const+=primary
        else: total_var+=n;qc_var+=qc;prim_var+=primary
        print(f'PASS {battery}: {n} cycles, QC={qc}, 70-100={primary}, Qref={qref:.9f} Ah, workbooks={counts[battery]}')
    need((total_const,qc_const,prim_const)==(5686,5663,4023),'constant population mismatch')
    need((total_var,qc_var,prim_var)==(251,238,118),'variable population mismatch')
    link=pd.read_csv(proof/'CALCE_Constant_Step11O_Label_Comparison.csv')
    step11o=pd.read_csv(repo/'research_release/data/step11o_constant_6299_soh_target_values.csv')
    need(len(link)==5663 and link['_merge'].eq('both').all(),'Step11O Colab linkage invalid')
    need(not link[['battery_id','Cycle_Index']].duplicated().any(),'duplicate CALCE Step11O key')
    need(np.isclose(link['Calculated_SOH(%)_source'],link['Calculated_SOH(%)_step11o'],rtol=0,atol=EPS).all(),'saved linkage target mismatch')
    # Cross-check against independently loaded repo target table, not only the Colab generated comparison.
    candidate_names=[x for x in step11o.columns if x.lower().replace(' ','') in ('calculated_soh(%)','calculated_soh','soh','soh_pct','soh_percent')]
    need('battery_id' in step11o and 'Cycle_Index' in step11o and bool(candidate_names),'Step11O labels missing expected columns')
    final_name='Calculated_SOH(%)' if 'Calculated_SOH(%)' in step11o.columns else candidate_names[0]
    t=step11o[['battery_id','Cycle_Index',final_name]].copy()
    j=link.merge(t,on=['battery_id','Cycle_Index'],how='left',validate='one_to_one',indicator='_link_status')
    need(j['_link_status'].eq('both').all(), 'Step11O target keys missing from repository')
    need(np.isclose(j['Calculated_SOH(%)_source'],j[final_name],rtol=0,atol=EPS).all(),'Step11O target differs from verified workbook data')
    print(f'PASS workbook manifest: {len(manifest_file)} members, unique SHA-256 values present')
    print(f'PASS immutable evidence: {len(files)} recorded file checksums')
    print('PASS Step11A: 5,686 constant and 251 variable target audits match all archived columns')
    print('PASS Step11O: 5,663 CALCE keys and target values match repository')
    print('SCOPE: verified archived evidence from 231 XLSX files; run included Colab notebook for fresh raw-workbook recreation')
    print('NOT VERIFIED: original acquisition-to-XLSX export, NASA native-to-CSV conversion, all historical model refits')

if __name__=='__main__':
    try:main()
    except Exception as exc:
        print('FAIL:',exc,file=sys.stderr);sys.exit(1)
