"""Source-only LOBO and zero-shot age/mean baselines for NASA--CALCE transfer.

Targets from the destination source are NEVER used for source training or
hyperparameter selection. Training mean averages battery means, not cycles.
Unclipped linear-age extrapolations are reported even if the error is large.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from battery_soh_repro.baselines import age_model,tune_age_hgb
from battery_soh_repro.experiment_utils import metric_by_battery,record_inputs


def evaluate(train:pd.DataFrame,test:pd.DataFrame,model_name:str,age_features:list[str],target:str):
    if model_name=='TrainMean':
        value=train.groupby('battery_id')[target].mean().mean()
        return np.full(len(test),value), 'none'
    kind='hgb' if model_name=='AgeHGB' else 'linear'
    if kind=='linear': age_features = ['Cycle_Age']
    config=tune_age_hgb(train,age_features,target)[1] if kind=='hgb' else 'linear'
    mod=age_model(train,age_features,target,kind=kind,config=config if kind=='hgb' else 'H1')
    return mod.predict(test[age_features].to_numpy(float)),config


def run(data:Path,output_dir:Path):
    output_dir.mkdir(parents=True,exist_ok=True)
    d=pd.read_csv(data,low_memory=False)
    if set(d.source_dataset.unique())!={'NASA','CALCE'}:raise ValueError('Both sources required')
    rows=[];settings=[]
    for train_source,test_source in [('NASA','CALCE'),('CALCE','NASA')]:
        training=d[d.source_dataset==train_source]
        target=d[d.source_dataset==test_source]
        for model_name in ['TrainMean','AgeLinear','AgeHGB']:
            pred,conf=evaluate(training,target,model_name,['Cycle_Age','Log_Cycle_Age'], 'Calculated_SOH(%)')
            z=target[['battery_id','Cycle_Index','source_dataset','Calculated_SOH(%)']].copy()
            z['Prediction_Raw']=pred
            z['Evaluation']='ZeroShot';z['Train_Source']=train_source;z['Model']=model_name
            rows.append(z)
            settings.append({'Model':model_name,'Train_Source':train_source,'Test_Source':test_source,'Config':conf})
    for source,g in d.groupby('source_dataset'):
        for heldout in sorted(g.battery_id.unique()):
            train=g[g.battery_id!=heldout];test=g[g.battery_id==heldout]
            for model_name in ['TrainMean','AgeLinear','AgeHGB']:
                pred,conf=evaluate(train,test,model_name,['Cycle_Age','Log_Cycle_Age'],'Calculated_SOH(%)')
                z=test[['battery_id','Cycle_Index','source_dataset','Calculated_SOH(%)']].copy()
                z['Prediction_Raw']=pred
                z['Evaluation']='SourceOnlyLOBO';z['Train_Source']=source;z['Model']=model_name
                rows.append(z)
                settings.append({'Model':model_name,'Train_Source':source,'Test_Source':source,
                                 'Holdout':heldout,'Config':conf})
    pred=pd.concat(rows,ignore_index=True)
    records=[];macro=[]
    for (evaluation,source,model),sub in pred.groupby(['Evaluation','Train_Source','Model']):
        per=metric_by_battery(sub,'Calculated_SOH(%)','Prediction_Raw')
        per.insert(0,'Model',model);per.insert(0,'Train_Source',source);per.insert(0,'Evaluation',evaluation)
        records.append(per)
        macro.append({'Evaluation':evaluation,'Train_Source':source,'Model':model,
                      'Test_Source':','.join(sorted(sub.source_dataset.unique())),
                      'N':len(sub),'Batteries':len(per),'Equal_Battery_RMSE':per.RMSE.mean(),
                      'Equal_Battery_MAE':per.MAE.mean()})
    pred.to_csv(output_dir/'baseline_transfer_predictions.csv',index=False)
    pd.concat(records,ignore_index=True).to_csv(output_dir/'baseline_transfer_per_battery.csv',index=False)
    pd.DataFrame(macro).to_csv(output_dir/'baseline_transfer_macro.csv',index=False)
    pd.DataFrame(settings).to_csv(output_dir/'baseline_transfer_configuration.csv',index=False)
    record_inputs(output_dir,{'constant_observations':data},
                  {'no_target_source_labels_for_fitting':True,'unclipped_predictions':True})
    print(pd.DataFrame(macro).to_string(index=False))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',type=Path,default=Path('outputs/constant_dataset/constant_observations.csv'))
    p.add_argument('--output-dir',type=Path,default=Path('outputs/age_mean_transfer'))
    a=p.parse_args();run(a.data,a.output_dir)
if __name__=='__main__':main()
