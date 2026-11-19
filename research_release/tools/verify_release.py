#!/usr/bin/env python3
"""Check integrity and archived predictions"""
import csv
import hashlib
import math
import sys
from collections import defaultdict
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
R=ROOT/'research_release'
N=R/'results/step11n'
O=R/'results/step11o'

def assertclose(a,b,abs_tol=1e-7):
    if not math.isclose(float(a),float(b),rel_tol=0,abs_tol=abs_tol):
        raise AssertionError(f'Expected {b}, observed {a}')

def read(name):
    with open(name,encoding='utf-8',newline='') as f:
        yield from csv.DictReader(f)

def check_manifest():
    entries=0
    for line in (R/'MANIFEST.sha256').read_text().splitlines():
        if not line.strip():continue
        expected,relative=line.split('  ',1)
        p=ROOT/relative
        assert p.is_file(),f'File missing: {relative}'
        value=hashlib.sha256(p.read_bytes()).hexdigest()
        assert value==expected,f'SHA256 changed: {relative}'
        entries+=1
    print(f'SHA-256: {entries} files match')

def check_preds(path,model_column,total,expected_rmse,target_column):
    key_targets={}; counts=defaultdict(int); errors=defaultdict(lambda:defaultdict(list))
    seen=set()
    for row in read(path):
        bid=row['battery_id'];cycle=int(row['Cycle_Index']);mod=row[model_column]
        key=(bid,cycle);full=(mod,bid,cycle)
        if full in seen:raise AssertionError(f'Duplicate model prediction {full}')
        seen.add(full)
        y=float(row[target_column]);p=float(row['Prediction_Raw'])
        if key in key_targets: assertclose(y,key_targets[key],1e-8)
        key_targets[key]=y
        errors[mod][bid].append((p-y)**2)
        counts[mod]+=1
    assert len(key_targets)==total,(len(key_targets),total)
    assert len(errors)==5,errors.keys()
    assert all(v==total for v in counts.values()),counts
    metrics={}
    for mod,cells in errors.items():
        assert len(cells)==10,(mod,len(cells))
        metrics[mod]=sum(math.sqrt(sum(es)/len(es)) for es in cells.values())/len(cells)
    for mod,gold in expected_rmse.items(): assertclose(metrics[mod],gold,1e-8)
    print(f'{path.parent.name}: {total} unique observations, 5 models, 10 held-out batteries')
    for m in sorted(metrics):print(f'  {m:45s} RMSE {metrics[m]:.6f}')
    return key_targets

def main():
    check_manifest()
    n=check_preds(N/'Step11N_AllFive_Model_Predictions.csv','Model',4589,{
       'AGE_ONLY_HGB_TUNED':5.635523393541547,
       'AGE_ONLY_LINEAR':6.549172540268637,
       'TRUE_3P6V_CENSORED_ELECTRICAL_HGB':2.282087480660201,
       'TRUE_3P6V_CENSORED_ELECTRICAL_PLUS_AGE_HGB':1.715946556127656,
    },'Calculated_SOH(%)')
    o=check_preds(O/'Step11O_AllFive_Model_Predictions.csv','Formulation',6299,{
       'AGE_ONLY_HGB':8.464806622833519,
       'AGE_ONLY_LINEAR':12.00487049650829,
       'STRICT_ELECTRICAL_HGB':2.288309473252453,
       'STRICT_ELECTRICAL_PLUS_AGE_HGB':2.194821177418254,
       'TRAINING_MEAN':17.69978833642335,  # from archived workbook
    },'Calculated_SOH(%)')
    for k,v in n.items():
        assert k in o,f'No unrestricted target at {k}'
        assertclose(v,o[k],1e-8)
    under70=sum(y<70 for y in o.values())
    in70to100=sum(70<=y<=100 for y in o.values())
    assert (under70,in70to100)==(1709,4590),(under70,in70to100)
    assert len(list(read(R/'splits/step11o_constant_outer_design.csv')))==10
    assert len(list(read(R/'splits/step11o_constant_inner_design.csv')))==90
    conf=list(read(O/'Step11O_OuterFold_Configs_and_Selected_Features.csv'))
    assert len(conf)==30
    assert sum(bool(c['Selected_Features']) for c in conf)==20
    print('Step11N vs Step11O labels: identical on 4589 overlapping keys')
    print('Step11O SoH population: 1709 below 70%; 4590 within 70-100%')
    print('Split design: 10 outer holds, 90 nested inner holds')
    print('Step11O selected feature lists: 20 nonempty outer model records')
    print('PASS: archived-artifact checks. Raw-data reproduction NOT verified.')

if __name__=='__main__':
    try:main()
    except Exception as e:print(f'FAIL: {e}',file=sys.stderr);raise
