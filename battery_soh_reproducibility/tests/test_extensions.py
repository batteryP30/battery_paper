"""Fast scientific guardrails for the new local workflows."""
import numpy as np
import pandas as pd
import pytest
from battery_soh_repro.censoring import censored_prefix, temporal_resample, centered_segment_filter
from battery_soh_repro.protocol_qc import audit_cycle,screen_battery
from battery_soh_repro.experiment_utils import compare_frozen_predictions,paired_battery_bootstrap


def synthetic_cycle(step4=True,complete_cv=True,end_voltage=2.7):
    # Three-step synthetic logger sequence: CC charge, CV taper and discharge.
    rows=[]
    for time,current,voltage,step in [(0,.55,3.9,2),(10,.55,4.2,2)]:
        rows.append([1,time,current,voltage,step])
    if step4:
        for time,current in [(20,.3),(30,.15),(40,.04 if complete_cv else .12)]:
            rows.append([1,time,current,4.2,4])
    for j,(time,v) in enumerate([(50,4.15),(60,3.8),(70,3.6),(80,end_voltage)]):
        rows.append([1,time,-1.1,v,7])
    return pd.DataFrame(rows,columns=['Cycle_Index','Test_Time(s)','Current(A)','Voltage(V)','Step_Index'])


def test_cycle_screen_passes_complete_cv_and_cutoff():
    row,_=audit_cycle(synthetic_cycle())
    assert row['Quality_Pass'] and row['CV_Pass'] and row['Discharge_Pass']
    assert row['CV_Interpretation']=='LOGGED_STEP4_CV_TAPER_CONSISTENT'


def test_missing_cv_and_cutoff_failure_are_different():
    a,_=audit_cycle(synthetic_cycle(step4=False))
    b,_=audit_cycle(synthetic_cycle(end_voltage=2.9))
    assert not a['Quality_Pass'] and a['CV_Interpretation']=='NO_STEP4_RUN'
    assert b['CV_Pass'] and not b['Discharge_Pass']


def test_censored_time_never_uses_future_sample():
    df=synthetic_cycle()
    sub,a=censored_prefix(df,'CALCE',seconds=15,min_samples=2,max_gap_seconds=100,
                          min_time_coverage=.1)
    assert sub is not None
    dis=sub['Test_Time(s)']
    assert dis.max()-dis.min()<=15
    assert a['Status']=='VALID_PREFIX'


def test_voltage_prefix_is_first_downward_crossing_only():
    df=synthetic_cycle()
    sub,a=censored_prefix(df,'CALCE',voltage=3.6,min_samples=2)
    assert a['Status']=='VALID_PREFIX'
    assert sub['Test_Time(s)'].max()==70
    seq=temporal_resample(sub,96)
    assert seq.shape==(96,3)
    assert np.isclose(seq[-1,0],3.6)


def test_filter_respects_step_boundary():
    z=pd.DataFrame({'Cycle_Index':[1]*6,'Step_Index':[2]*3+[7]*3,
                    'Voltage(V)':[4.,4.,4.,2.,2.,2.],
                    'Current(A)':[.5,.5,.5,-1.,-1.,-1.]})
    f=centered_segment_filter(z,window=5)
    assert np.allclose(f.iloc[:3]['Voltage(V)'],4.)
    assert np.allclose(f.iloc[3:]['Voltage(V)'],2.)


def test_paired_bootstrap_on_battery_metrics_not_individual_cycles():
    frame=pd.DataFrame({'battery_id':['A','B'],'RMSE_Frozen':[1,2],'RMSE_New':[.5,1.5]})
    stat=paired_battery_bootstrap(frame,'RMSE_Frozen','RMSE_New',reps=1000)
    assert stat['Batteries']==2 and abs(stat['Mean_Paired_RMSE_Difference']+.5)<1e-12
    assert abs(stat['CI95_Lower']+.5)<1e-12


def test_frozen_target_mismatch_aborts_pairing():
    f=pd.DataFrame({'battery_id':['A'],'Cycle_Index':[1],'Calculated_SOH(%)':[99.],
                    'Prediction_Raw':[98.]})
    n=f.copy();n['Calculated_SOH(%)']=90.
    with pytest.raises(ValueError,match='targets differ'):
        compare_frozen_predictions(f,n,['battery_id','Cycle_Index'],'Calculated_SOH(%)')


def test_sequence_model_shape_if_torch_installed():
    torch=pytest.importorskip('torch')
    from battery_soh_repro.sequence_models import CONFIGS,create_model
    for typ in CONFIGS:
        model=create_model(typ,list(CONFIGS[typ].values())[0],True)
        y=model(torch.zeros(2,96,3),torch.zeros(2))
        assert y.shape==(2,)
