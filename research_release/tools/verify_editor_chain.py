#!/usr/bin/env python3
"""Editor traceability checks for preserved Step 11A/11N/11O data.

Runs without original raw archive and never refits a model. The facts checked
are target-capacity algebra, battery/cycle keys, NASA reference-interval maps,
LOBO split DESIGN, archived configurations and OOF predictions. Raw -> CSV
conversion and training cannot be certified without original source files.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd

EXPECTED_NASA_QREF = {
    'B0005': 1.861951950365522, 'B0006': 2.05197810816213,
    'B0007': 1.918661234856553, 'B0018': 1.868527668789764,
    'RW3': 4.011886890277778, 'RW4': 3.948668852777778,
    'RW5': 3.932915277777778, 'RW6': 3.848791011111112,
}

def check(test: bool, msg: str):
    if not bool(test):
        raise AssertionError(msg)

def approx(a,b,tol=1e-8):
    x=np.asarray(a,dtype=float);y=np.asarray(b,dtype=float)
    return bool(np.isfinite(x).all() and np.isfinite(y).all() and np.max(np.abs(x-y))<tol)

def read(root,rel):
    p=root/rel
    check(p.is_file(),f'missing: {p}')
    return pd.read_csv(p,low_memory=False)

def run(root: Path)->dict:
    root=root.resolve()
    t='research_release/editor_traceability/'
    summary=read(root,t+'step11a_battery_reference_summary.csv')
    audit=read(root,t+'step11a_constant_cycle_capacity_audit.csv')
    events=read(root,t+'step11a_nasa_rw_reference_events.csv')
    rwmap=read(root,t+'step11a_nasa_rw_pre_reference_cycle_map.csv')
    input_reg=read(root,t+'step11a_source_input_registry.csv')
    old_checks=read(root,t+'step11a_regression_checks.csv')
    labels=read(root,'research_release/data/step11o_constant_6299_soh_target_values.csv')
    preds=read(root,'research_release/results/step11o/Step11O_AllFive_Model_Predictions.csv')
    p11n=read(root,'research_release/results/step11n/Step11N_AllFive_Model_Predictions.csv')
    configs=read(root,'research_release/results/step11o/Step11O_OuterFold_Configs_and_Selected_Features.csv')
    outer=read(root,'research_release/splits/step11o_constant_outer_design.csv')
    inner=read(root,'research_release/splits/step11o_constant_inner_design.csv')
    step11o_battery=read(root,'research_release/results/step11o/Step11O_PerBattery.csv')
    check(len(summary)==16 and summary.Battery.nunique()==16,'expected 16 battery source summaries')
    check(len(input_reg)==16 and input_reg.Battery.nunique()==16,'input provenance registry incomplete')
    check(approx(summary.set_index('Battery').loc[list(EXPECTED_NASA_QREF),'Qref_Ah'].astype(float),list(EXPECTED_NASA_QREF.values()),1e-10),'NASA Qref differs from Step11A archived first-valid reference')
    check(len(audit)==6322,'unexpected constant target candidate count')
    good=audit.loc[audit.Raw_QC_Pass.astype(str).str.lower().eq('true')].copy()
    check(len(good)==6299 and len(audit)-len(good)==23,'constant raw QC counts mismatch')
    key=['battery_id','Cycle_Index']
    for d,name in [(good,'constant QC'),(labels,'Step11O labels')]:
        check(not d.duplicated(key).any(),f'duplicate {name} keys')
    merged=labels.merge(good[key+['Measured_Discharge_Capacity_Ah','Reference_Capacity_Ah','Calculated_SOH(%)']],on=key,how='outer',indicator=True,suffixes=('_saved','_audit'),validate='one_to_one')
    check(len(merged)==6299 and merged._merge.eq('both').all(),'Step11O keys are not identical to archived Step11A QC-valid targets')
    check(approx(merged['Calculated_SOH(%)_saved'],merged['Calculated_SOH(%)_audit'],1e-9),'Step11O target values disagree with Step11A archived target reconstruction')
    check(approx(100*merged.Measured_Discharge_Capacity_Ah/merged.Reference_Capacity_Ah,merged['Calculated_SOH(%)_audit'],1e-9),'capacity-to-target algebra fails')
    check(int((merged['Calculated_SOH(%)_saved']<70).sum())==1709,'unexpected below-70 cycle population')
    check(int(merged['Calculated_SOH(%)_saved'].between(70,100).sum())==4590,'unexpected 70-100 cycle population')
    checks_true=old_checks.Passed.astype(str).str.lower().eq('true')
    check(checks_true.all(),'archived Step11A regression check indicates failure')

    # NASA variable reference events and relationship to preceding RW cycles
    check(len(events)==73 and len(rwmap)==2607,'NASA reference events / RW rows changed')
    ekey=['battery_id','Reference_Interval_ID']
    check(not events.duplicated(ekey).any(),'duplicate NASA reference event IDs')
    ev=events.copy()
    ev_qc=ev.Raw_QC_Pass.astype(str).str.lower().eq('true')
    check(ev_qc.all(),'unexpected invalid archived NASA reference event')
    check(int(ev.Primary_Independent_Observation.astype(str).str.lower().eq('true').sum())==47,'expected 47 primary NASA RW reference intervals')
    for battery,q in EXPECTED_NASA_QREF.items():
        if not battery.startswith('RW'): continue
        rows=ev.loc[ev.battery_id.eq(battery)].sort_values('Reference_Epoch')
        check(len(rows)>1,'no NASA RW reference events for '+battery)
        check(approx([rows['Reference_Capacity_Ah'].iloc[0]],[q],1e-10),'NASA RW Qref event mismatch for '+battery)
        check(int(rows.iloc[0].PreReference_RW_Cycles)==0,'first NASA RW reference event must lack prior usage interval')
        check(approx(100*rows.Reference_Capacity_Ah.astype(float)/q,rows['Calculated_SOH(%)'].astype(float),1e-9),'NASA RW reference-target normalization mismatch for '+battery)
    j=rwmap.merge(ev[ekey+['Calculated_SOH(%)','PreReference_RW_Cycles']],on=ekey,how='left',indicator=True,suffixes=('_map','_event'),validate='many_to_one')
    check(len(j)==len(rwmap) and j._merge.eq('both').all(),'RW cycle-to-reference map contains unknown events')
    check(approx(j['Calculated_SOH(%)_map'],j['Calculated_SOH(%)_event'],1e-9),'RW-to-reference target matching mismatch')
    check(not rwmap.duplicated(['battery_id','Cycle_Index']).any(),'duplicate RW cycles across reference intervals')
    count=rwmap.groupby(ekey,dropna=False).size().rename('mapped_n').reset_index().merge(ev[ekey+['PreReference_RW_Cycles']],on=ekey,validate='one_to_one')
    check((count.mapped_n==count.PreReference_RW_Cycles).all(),'RW interval size differs from archived source-event audit')
    check(len(count)==69,'expected 69 NASA RW intervals with preceding cycling')

    # Step11O observations, model OOF, named fold identity and outer feature selections
    models_o=preds.Formulation.unique().tolist()
    check(len(models_o)==5 and len(preds)==6299*5,'Step11O model records missing')
    check(not preds.duplicated(key+['Formulation']).any(),'duplicate Step11O OOF rows')
    pp=preds.merge(labels,on=key,how='left',suffixes=('_pred','_saved'),validate='many_to_one')
    check(not pp['Calculated_SOH(%)_saved'].isna().any(),'Step11O prediction lacks reconstructed target key')
    check(approx(pp['Calculated_SOH(%)_saved'],pp['Calculated_SOH(%)_pred'],1e-9),'Step11O prediction labels disagree with archived targets')
    for m in models_o:
        d=preds.loc[preds.Formulation.eq(m)]
        check(len(d)==6299,'Step11O model has incomplete coverage: '+m)
    batts=set(labels.battery_id.astype(str))
    check(len(batts)==10,'expected 10 held-out constant batteries')
    check(len(outer)==10 and set(outer.outer_holdout_battery)==batts,'outer-fold design mismatch')
    check(len(inner)==90,'expected 90 inner-fold design rows')
    for _,row in outer.iterrows():
        b=str(row.outer_holdout_battery)
        train=set(str(row.outer_train_batteries).split('|'))
        check(train==batts-{b},'outer train set unexpectedly includes held-out battery '+b)
        check(int(row.n_outer_test_cycles_step11o)==int(labels.battery_id.eq(b).sum()),'held-out count mismatch for '+b)
        ii=inner.loc[inner.outer_holdout_battery.eq(b)]
        check(len(ii)==9 and set(ii.inner_validation_battery)==train,'inner fold design mismatch for '+b)
        for _,rr in ii.iterrows():
            check(set(str(rr.inner_training_batteries).split('|'))==train-{str(rr.inner_validation_battery)},'inner design contains wrong training batteries')
    ect=configs.loc[configs.Formulation.isin(['STRICT_ELECTRICAL_HGB','STRICT_ELECTRICAL_PLUS_AGE_HGB'])].copy()
    check(len(ect)==20 and not ect.duplicated(['Holdout_Battery','Formulation']).any(),'expected 20 saved electrical HGB outer configs')
    check(ect.Selected_Features.notna().all() and ect.Selected_Features.astype(str).str.len().gt(3).all(),'missing selected features for an outer fold')
    check(ect.Chosen_HGB_Config.notna().all(),'missing HGB configurations')
    check(ect.apply(lambda x:int(x.N_Selected)==len([a for a in str(x.Selected_Features).split(' | ') if a]),axis=1).all(),'feature selection counts disagree')
    # Verify reconstructed battery-level RMSEs from saved OOF, avoiding pooled RMSE confusion
    for _,row in step11o_battery.iterrows():
        d=preds.loc[preds.Formulation.eq(row.Formulation)&preds.battery_id.eq(row.battery_id)]
        check(len(d)==int(row.N),'Step11O N mismatch for '+str(row.Formulation)+str(row.battery_id))
        rmse=float(np.sqrt(np.mean((pd.to_numeric(d.Prediction_Raw)-pd.to_numeric(d['Calculated_SOH(%)']))**2)))
        check(abs(rmse-float(row.RMSE))<1e-8,'Step11O per-battery RMSE mismatch')

    # Step11N observation-label link to Step11O (different training population)
    check(len(p11n)==4589*5 and p11n.Model.nunique()==5,'Step11N coverage invalid')
    check(not p11n.duplicated(key+['Model']).any(),'duplicate Step11N prediction')
    p=p11n.merge(labels,on=key,how='left',validate='many_to_one',suffixes=('_n','_o'))
    check(p['Calculated_SOH(%)_o'].notna().all(),'Step11N contains unknown keys')
    check(approx(p['Calculated_SOH(%)_n'],p['Calculated_SOH(%)_o'],1e-9),'Step11N targets differ from Step11A/11O')
    report={
      'status':'PASS',
      'scope':'archived Step11A -> reconstructed targets -> Step11O/Step11N OOF and split/config checks; NOT raw-input rebuild',
      'constant_raw_candidate_cycles':len(audit),'constant_raw_qc_valid_cycles':len(good),
      'constant_targets_matching_step11o':len(merged),
      'nasa_reference_events':len(events),'nasa_primary_pre_reference_intervals':47,
      'nasa_rw_cycles_mapped':len(rwmap),'nasa_reference_ah':EXPECTED_NASA_QREF,
      'step11o_observations':len(labels),'step11o_models':len(models_o),
      'step11n_observations':4589,'step11n_models':p11n.Model.nunique(),
      'outer_design_folds':len(outer),'inner_design_folds':len(inner),
      'saved_selected_feature_configurations':len(ect),
      'limitations':['NASA original .mat to exported CSV conversion not independently reconstructed','Archived split design is deterministic, not original inner-CV run history','Step11O saved OOF can be audited but raw model refitting cannot be certified without original measurement exports'],
    }
    return report

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo-root',type=Path,default=Path.cwd())
    parser.add_argument('--json-output',type=Path)
    args=parser.parse_args()
    result=run(args.repo_root)
    print('PASS: archived capacity -> SoH -> observation -> fold/config -> OOF chain')
    for k,v in result.items():
        if k not in ('limitations','nasa_reference_ah','status','scope'):print(f'  {k}: {v}')
    print('NOT VERIFIED: original NASA native-to-CSV conversion or full model refits')
    if args.json_output:
        args.json_output.parent.mkdir(parents=True,exist_ok=True)
        args.json_output.write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
        print('Report written:',args.json_output)
