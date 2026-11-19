#!/usr/bin/env python3
"""Independently cross-check archived Colab NASA study-CSV audit outputs.

Verifies saved audit tables against independently regenerated Colab outputs;
checks evidence hashes and numerical-linkage metadata. Does NOT re-read NASA
measurement CSVs, verify NASA's native MATLAB->CSV conversion, or refit models.
For source-level rerun, use the published Colab notebook with the original CSVs.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
import pandas as pd

CONST = ['B0005','B0006','B0007','B0018']
RW = ['RW3','RW4','RW5','RW6']
FLOAT_TOL=1e-10

def ensure(condition, message):
    if not bool(condition): raise AssertionError(message)

def csvread(path):
    ensure(path.is_file(),f'Missing {path}')
    return pd.read_csv(path,low_memory=False)

def normbool(v):
    return v.astype(str).str.lower().map({'true':True,'false':False,'1':True,'0':False})

def compare(actual, archived, keys, floats, ints=(), bools=(), desc=''):
    aa=actual.copy();bb=archived.copy()
    for key in keys:
        ensure(key in aa and key in bb, f'{desc}: missing key {key}')
        if key=='Cycle_Index':
            aa[key]=pd.to_numeric(aa[key],errors='raise').astype('int64')
            bb[key]=pd.to_numeric(bb[key],errors='raise').astype('int64')
        else:
            aa[key]=aa[key].astype(str);bb[key]=bb[key].astype(str)
    ensure(not aa.duplicated(keys).any() and not bb.duplicated(keys).any(),f'{desc}: duplicate keys')
    j=aa.merge(bb,on=keys,how='outer',indicator=True,suffixes=('_computed','_archived'),validate='one_to_one')
    ensure(j._merge.eq('both').all() and len(j)==len(aa)==len(bb),f'{desc}: unmatched records computed={len(aa)}, archived={len(bb)}')
    result={'rows':len(j),'max_abs_diff':{}}
    for field in floats:
        ac=pd.to_numeric(j[field+'_computed'],errors='coerce'); ar=pd.to_numeric(j[field+'_archived'],errors='coerce')
        ensure(ac.notna().all() and ar.notna().all(),f'{desc}: non-numeric {field}')
        err=(ac-ar).abs().max()
        ensure(float(err) <= FLOAT_TOL, f'{desc}: {field} max delta={err}')
        result['max_abs_diff'][field]=float(err)
    for field in ints:
        a=pd.to_numeric(j[field+'_computed'],errors='coerce');b=pd.to_numeric(j[field+'_archived'],errors='coerce')
        ensure(a.notna().all() and b.notna().all() and (a==b).all(),f'{desc}: integer column differs: {field}')
    for f1,f2 in bools:
        a=normbool(j[f1+'_computed']); b=normbool(j[f2+'_archived'])
        ensure(a.notna().all() and b.notna().all() and (a==b).all(),f'{desc}: boolean column differs: {f1}/{f2}')
    return result

def main(repo:Path,output:Path|None=None):
    edir=repo/'research_release/editor_traceability'
    ev=edir/'nasa_csv_v2'
    # Verify that evidence files have not been altered after capture.
    hfile=ev/'EVIDENCE.sha256'
    ensure(hfile.is_file(),'Evidence manifest missing')
    pairs=[]
    for ln in hfile.read_text(encoding='utf-8').splitlines():
        if not ln.strip():continue
        digest,name=ln.split('  ',1)
        ensure('/' not in name and '\\' not in name and name not in {'.','..'},'unsafe filename in evidence manifest')
        p=ev/name
        ensure(p.is_file(),f'evidence missing: {name}')
        actual=hashlib.sha256(p.read_bytes()).hexdigest()
        ensure(actual==digest,f'evidence altered: {name}')
        pairs.append(name)
    ensure(len(pairs)==37 and len(pairs)==len(set(pairs)),'expected exactly 37 unique Colab report files')
    # Exact-byte hashes for the seven preserved Step11A table exports. They are
    # set to -text in the directory-specific .gitattributes for Git portability.
    audit_manifest=edir/'STEP11A_EXPORTED_TABLES.sha256'
    ensure(audit_manifest.is_file(),'Step11A table export manifest absent')
    audit_names=[]
    for ln in audit_manifest.read_text(encoding='utf-8').splitlines():
        if not ln.strip(): continue
        digest,name=ln.split('  ',1)
        ensure(name.startswith('step11a_') and name.endswith('.csv') and '/' not in name,'Invalid Step11A table manifest path')
        file=edir/name
        ensure(file.is_file() and hashlib.sha256(file.read_bytes()).hexdigest()==digest,f'Step11A table export altered: {name}')
        audit_names.append(name)
    ensure(len(audit_names)==7 and len(set(audit_names))==7,'Expected seven frozen Step11A table exports')
    summary=json.loads((ev/'NASA_Verification_Summary.json').read_text())
    ensure(len(summary)==8 and {x['battery'] for x in summary}==set(CONST+RW),'not eight battery results')
    const_audit=csvread(edir/'step11a_constant_cycle_capacity_audit.csv')
    refs=csvread(edir/'step11a_nasa_rw_reference_events.csv')
    rwmap=csvread(edir/'step11a_nasa_rw_pre_reference_cycle_map.csv')
    statuses=[]
    for item in summary:
        b=item['battery']
        ensure(item['status']=='PASS' and item.get('worker_returncode')==0,f'{b}: Colab worker did not pass')
        check=json.loads((ev/f'{b}_check.json').read_text())
        ensure(check['status']=='PASS' and abs(float(check['reference_Ah'])-float(item['reference_Ah']))<FLOAT_TOL,f'{b}: inconsistent reports')
        ensure((ev/f'{b}_worker.log').is_file(),f'{b}: worker log absent')
        if b in CONST:
            actual=csvread(ev/f'{b}_reintegrated_constant_targets.csv')
            old=const_audit.loc[const_audit.battery_id.eq(b)].copy()
            ret=compare(actual,old,['battery_id','Cycle_Index'],['Measured_Discharge_Capacity_Ah','Reference_Capacity_Ah','Calculated_SOH(%)','Duration_s'],bools=[('Raw_QC_Pass','Raw_QC_Pass')],desc=b)
            ensure(len(actual)==item['valid_targets']==item['tests']['rows'],f'{b}: constant counts differ')
            qref=float(actual.iloc[0].Reference_Capacity_Ah)
        else:
            actual=csvread(ev/f'{b}_reintegrated_reference_events.csv')
            old=refs.loc[refs.battery_id.eq(b)].copy()
            ret=compare(actual,old,['battery_id','Reference_Interval_ID'],['Reference_Capacity_Ah','Baseline_Reference_Capacity_Ah','Calculated_SOH(%)'],['Reference_Epoch','Reference_Start_Row','Reference_End_Row','PreReference_RW_Cycles'],[('Raw_QC_Pass','Raw_QC_Pass')],f'{b} reference events')
            amap=csvread(ev/f'{b}_regenerated_cycle_reference_map.csv')
            oldmap=rwmap.loc[rwmap.battery_id.eq(b)].copy()
            m=compare(amap,oldmap,['battery_id','Cycle_Index'],['Calculated_SOH(%)'],['Reference_Epoch','Interval_Position','Interval_Total_Cycles'],[('Raw_QC_Pass','Raw_QC_Pass')],f'{b} RW map')
            ensure(len(amap)==item['mapped_cycles']==item['tests']['cycle_map']['rows'],f'{b}: mapping counts differ')
            ensure(len(actual)==item['reference_event_count']==item['tests']['reference_events']['rows'],f'{b}: reference counts differ')
            qref=float(actual.iloc[0].Baseline_Reference_Capacity_Ah)
            ret['cycle_map']=m
        ensure(abs(qref-float(item['reference_Ah']))<FLOAT_TOL,f'{b}: Qref mismatch')
        statuses.append({'battery':b,'reference_capacity_Ah':qref,'result':'PASS','comparison':ret})
    cls=json.loads((ev/'Archive_Checks_Classification.json').read_text())
    diag=csvread(ev/'Release_Manifest_Diagnostics.csv')
    counts=diag.classification.value_counts().to_dict()
    ensure(len(diag)==57 and len(diag.path.unique())==57,'release diagnostic row/key count unexpected')
    ensure(counts.get('EXACT',0)==45 and counts.get('CHECKOUT_LF_VS_FROZEN_CRLF',0)==12 and sum(v for k,v in counts.items() if k not in {'EXACT','CHECKOUT_LF_VS_FROZEN_CRLF'})==0,'unexplained release manifest difference')
    ensure(cls['manifest_exact_files']==45 and cls['manifest_eol_equivalent_files']==12 and cls['manifest_unexplained_files']==0,'checksum metadata inconsistent')
    src=csvread(ev/'NASA_Source_Manifest_SHA256.csv')
    ensure(set(src.battery_id)==set(CONST+RW) and len(src)==8,'source CSV hash registry not eight')
    for item in summary:
        row=src.loc[src.battery_id.eq(item['battery'])].iloc[0]
        ensure(str(row['sha256'])==item['source_sha256'] and int(row['size_bytes'])==item['source_size_bytes'],f"{item['battery']}: source hash metadata differs")
    archived=json.loads((ev/'Editor_Archived_Chain_Report.json').read_text())
    ensure(archived['status']=='PASS' and archived['constant_targets_matching_step11o']==6299 and archived['nasa_rw_cycles_mapped']==2607, 'archived chain report inconsistent')
    count_refs=sum(s['comparison']['rows'] for s in statuses if s['battery'] in RW)
    count_rw=sum(s['comparison']['cycle_map']['rows'] for s in statuses if s['battery'] in RW)
    ensure((count_refs,count_rw)==(73,2607),f'wrong aggregated RW counts {(count_refs,count_rw)}')
    report={'status':'PASS','scope':'Independent comparison of SAVED Colab source-reconstruction outputs to archived Step11A tables; no raw CSV included in release, no native .mat converter and no model refits',
           'evidence_files_hashed':len(pairs),'archived_step11a_table_exports_hashed':len(audit_names),'source_battery_csvs_recorded':len(src),'source_bytes_sha256_values_verified_against_report':True,
           'constant_cells_pass':len(CONST),'rw_cells_pass':len(RW),'rw_reference_events':count_refs,'rw_cycles_mapped':count_rw,'manifest_exact_on_colab':45,'manifest_EOL_only_on_colab':12,'manifest_unexplained':0,'batteries':statuses}
    print('PASS: eight reconstructed NASA study-CSV audits match preserved Step11A tables')
    print('Reference discharges:',count_refs,'RW cycle/reference maps:',count_rw)
    print('Evidence files SHA256:',len(pairs),'integrity checks passed')
    print('Recorded v0.9.0 checksum status: 45 exact, 12 proven LF/CRLF transformations, 0 unexplained')
    print('NOT VERIFIED: NASA native MATLAB -> study CSV conversion, raw files omitted, CALCE source rerun, historical refits')
    if output:
        output.parent.mkdir(parents=True,exist_ok=True)
        output.write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
        print('Saved:',output)
    return report

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--repo-root',type=Path,default=Path(__file__).resolve().parents[2])
    parser.add_argument('--json-output',type=Path)
    a=parser.parse_args()
    try:main(a.repo_root,a.json_output)
    except Exception as e:
        print(f'FAIL: {e}',file=sys.stderr)
        raise
