"""Local PyTorch LSTM/Transformer comparison using true-3.6-V discharge prefixes.

Run `--stage prepare` with the raw ZIP and a frozen observations CSV to build
96-point tensors. Run `--stage train` to perform battery-held-out selection and
training. The trained results are research reproductions; verify numerical
agreement to the saved Step 11KC experimental record before citing any score.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import random
import numpy as np
import pandas as pd
from battery_soh_repro.censoring import censored_prefix, temporal_resample
from battery_soh_repro.data_pipeline import (prepare_raw_archive,process_files,
    load_nasa_constant_measurements_targetfree,CALCE_CONSTANT_BATTERIES,NASA_CONSTANT_BATTERIES)
from battery_soh_repro.experiment_utils import ensure_unique,record_inputs


def prepare(raw_zip:Path, observations:Path, output_dir:Path, work_dir:Path, length:int=96):
    out=output_dir;out.mkdir(parents=True,exist_ok=True)
    original=pd.read_csv(observations)
    man,_=prepare_raw_archive(raw_zip,work_dir)
    sequences=[];records=[];audits=[]
    for b,sheets in CALCE_CONSTANT_BATTERIES+[(b,None) for b in NASA_CONSTANT_BATTERIES]:
        src='CALCE' if sheets else 'NASA'
        raw=(process_files(str(man[b]['folder']),sheets) if sheets is not None else
             load_nasa_constant_measurements_targetfree(man[b]['csv_path'],b))
        for cycle,z in raw.groupby('Cycle_Index',sort=True):
            prefix,a=censored_prefix(z,src,voltage=3.6,min_samples=10,max_gap_seconds=float('inf'))
            audits.append({'battery_id':b,'Cycle_Index':int(cycle),**a})
            if prefix is not None:
                sequences.append(temporal_resample(prefix,length))
                records.append({'battery_id':b,'Cycle_Index':int(cycle)})
    joined=pd.DataFrame(records)
    ensure_unique(joined,['battery_id','Cycle_Index'])
    joined['Tensor_Row']=np.arange(len(joined))
    joined=joined.merge(original[['battery_id','Cycle_Index','Calculated_SOH(%)',
                                  'Cycle_Age','source_dataset']],
                        on=['battery_id','Cycle_Index'],how='inner',validate='one_to_one')
    z=np.asarray(sequences,dtype=np.float32)
    np.savez_compressed(out/'sequences.npz',inputs=z,
                        ids=np.array([f'{r["battery_id"]}|{r["Cycle_Index"]}' for r in records]))
    joined.to_csv(out/'sequence_index.csv',index=False)
    pd.DataFrame(audits).to_csv(out/'sequence_prefix_audit.csv',index=False)
    if len(joined)!=4589:raise ValueError(f'Expected 4589 in-range 3.6-V sequence observations, got {len(joined)}')
    record_inputs(out,{'raw_zip':raw_zip,'observations':observations},
                  {'sequence_length':length,'channels':['Voltage(V)','Current(A)','Elapsed_s'],
                   'first_downward_3p6V_crossing':True})
    return out


def run_training(data_dir:Path, output_dir:Path,models:list[str],seed:int=314159,
                 max_epochs:int=60,batch_size:int=256,device:str='cpu'):
    try:
        import torch
        from torch.utils.data import DataLoader,TensorDataset
        from battery_soh_repro.sequence_models import CONFIGS,create_model
    except ImportError as exc:
        raise RuntimeError('Install optional sequence requirements: pip install -r requirements-sequence.txt') from exc
    output_dir.mkdir(parents=True,exist_ok=True)
    index=pd.read_csv(data_dir/'sequence_index.csv')
    archive=np.load(data_dir/'sequences.npz')
    values=archive['inputs'][index.Tensor_Row.to_numpy(int)]
    ages=np.log1p(index.Cycle_Age.to_numpy(float))
    targets=index['Calculated_SOH(%)'].to_numpy(float)
    battery=index.battery_id.astype(str).to_numpy()
    source=index.source_dataset.astype(str).to_numpy()
    unique=sorted(set(battery))
    if len(unique)!=10 or len(index)!=4589: raise ValueError('Unexpected 3.6-V population')
    if not np.isfinite(values).all():raise ValueError('Nonfinite raw prefix data')

    def set_seed(s):
        random.seed(s);np.random.seed(s);torch.manual_seed(s)
        if torch.cuda.is_available():torch.cuda.manual_seed_all(s)

    def scaler(train_idx):
        mu=values[train_idx].mean(axis=(0,1),keepdims=True)
        sd=np.maximum(values[train_idx].std(axis=(0,1),keepdims=True),1e-8)
        ym=float(targets[train_idx].mean());ys=max(float(targets[train_idx].std()),1e-8)
        am=float(ages[train_idx].mean());as_=max(float(ages[train_idx].std()),1e-8)
        return mu,sd,ym,ys,am,as_

    def predict(model,ids,stats):
        mu,sd,ym,ys,am,as_=stats
        x=torch.as_tensor((values[ids]-mu)/sd,dtype=torch.float32,device=device)
        a=torch.as_tensor((ages[ids]-am)/as_,dtype=torch.float32,device=device)
        model.eval()
        outs=[]
        with torch.no_grad():
            for i in range(0,len(ids),batch_size):
                outs.extend(model(x[i:i+batch_size],a[i:i+batch_size]).detach().cpu().numpy().tolist())
        return np.asarray(outs)*ys+ym

    def fit(family,config,train_ids,val_ids,with_age,chosen_seed,epochs=None):
        set_seed(chosen_seed)
        stats=scaler(train_ids)
        mu,sd,ym,ys,am,as_=stats
        x=torch.tensor((values[train_ids]-mu)/sd,dtype=torch.float32)
        a=torch.tensor((ages[train_ids]-am)/as_,dtype=torch.float32)
        y=torch.tensor((targets[train_ids]-ym)/ys,dtype=torch.float32)
        counts=pd.Series(battery[train_ids]).value_counts()
        weights=np.array([1/counts[b] for b in battery[train_ids]],dtype=float)
        weights=weights/weights.mean()
        ds=TensorDataset(x,a,y,torch.tensor(weights,dtype=torch.float32))
        loader=DataLoader(ds,batch_size=batch_size,shuffle=True)
        mod=create_model(family,config,with_age,length=values.shape[1]).to(device)
        opt=torch.optim.AdamW(mod.parameters(),lr=config['lr'],weight_decay=1e-4)
        best_score=float('inf');best_epoch=max_epochs;best_state=None;since=0
        loops=epochs or max_epochs
        for ep in range(loops):
            mod.train()
            for xb,ab,yb,wb in loader:
                xb,ab,yb,wb=[t.to(device) for t in (xb,ab,yb,wb)]
                opt.zero_grad(set_to_none=True)
                p=mod(xb,ab)
                loss=(wb*(p-yb).square()).sum()/wb.sum()
                loss.backward();torch.nn.utils.clip_grad_norm_(mod.parameters(),1.)
                opt.step()
            if val_ids is not None:
                estimated=predict(mod,val_ids,stats)
                errors=[]
                for b in sorted(set(battery[val_ids])):
                    ii=battery[val_ids]==b
                    errors.append(np.sqrt(np.mean((targets[val_ids][ii]-estimated[ii])**2)))
                score=float(np.mean(errors))
                if score<best_score-1e-9:
                    best_score=score;best_epoch=ep+1;since=0
                    best_state={k:v.detach().cpu().clone() for k,v in mod.state_dict().items()}
                else:since+=1
                if ep>=9 and since>=8:break
        if best_state is not None:mod.load_state_dict(best_state)
        return mod,stats,best_score,best_epoch

    predictions=[];selection=[]
    for family in models:
        for with_age in (False,True):
            for heldout in unique:
                train_idx=np.flatnonzero(battery!=heldout)
                test_idx=np.flatnonzero(battery==heldout)
                # Choose a pair of inner-validation batteries from outer TRAINING only.
                calce=sorted({battery[i] for i in train_idx if source[i]=='CALCE'})
                nasa=sorted({battery[i] for i in train_idx if source[i]=='NASA'})
                if not calce or not nasa:raise ValueError('Each fold requires both source datasets')
                validation_batt=(calce[0],nasa[0])
                val_idx=np.flatnonzero(np.isin(battery,list(validation_batt)))
                inner_idx=np.flatnonzero(~np.isin(battery,[heldout,*validation_batt]))
                assert not set(battery[test_idx]).intersection(set(battery[inner_idx]))
                candidates=[]
                for key,conf in CONFIGS[family].items():
                    model,stats,score,epoch=fit(family,conf,inner_idx,val_idx,with_age,seed)
                    candidates.append((score,key,conf,epoch))
                _,bestkey,conf,bestep=min(candidates,key=lambda x:(x[0],x[1]))
                pred_seeds=[]
                for fseed in (42,2026):
                    fitted,scale,_,_=fit(family,conf,train_idx,None,with_age,fseed,epochs=bestep)
                    pred_seeds.append(predict(fitted,test_idx,scale))
                pred=np.mean(pred_seeds,axis=0)
                for ix,yh in zip(test_idx,pred):
                    predictions.append({'Model':family,'With_Age':with_age,
                        'Formulation':f'{family}_{"WITH_AGE" if with_age else "ELECTRICAL"}',
                        'battery_id':battery[ix],'source_dataset':source[ix],
                        'Cycle_Index':int(index.Cycle_Index.iloc[ix]),
                        'Calculated_SOH(%)':targets[ix],'Prediction_Raw':yh})
                selection.append({'Model':family,'With_Age':with_age,'Holdout_Battery':heldout,
                                  'Config':bestkey,'Best_Epoch':bestep,'Inner_Validation':','.join(validation_batt),
                                  'Inner_Mean_RMSE':min(candidates)[0]})
                print(f'{family}/{with_age} {heldout} {bestkey}: epoch {bestep}',flush=True)
    pred=pd.DataFrame(predictions)
    pred.to_csv(output_dir/'sequence_3p6_oof_predictions.csv',index=False)
    pd.DataFrame(selection).to_csv(output_dir/'sequence_3p6_selection.csv',index=False)
    summary=[]
    for name,g in pred.groupby('Formulation'):
        rmses=[np.sqrt(np.mean((h['Calculated_SOH(%)']-h.Prediction_Raw)**2))
               for _,h in g.groupby('battery_id')]
        summary.append({'Formulation':name,'N':len(g),'Batteries':len(rmses),
                        'Equal_Battery_RMSE':float(np.mean(rmses))})
    pd.DataFrame(summary).to_csv(output_dir/'sequence_3p6_macro.csv',index=False)
    (output_dir/'training_parameters.json').write_text(json.dumps({
        'architecture_search':'deterministic one NASA and one CALCE validation battery; training-only',
        'max_epochs':max_epochs,'batch_size':batch_size,'selection_seed':seed,
        'final_ensemble_seeds':[42,2026],'battery_sample_weights':True,
        'stopping_patience_epochs':8,'minimum_training_epochs':10,
        'important':'New source reconstruction; compare output against archived Step 11KC before claiming byte-identical original reproduction'
    },indent=2))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--stage',choices=['prepare','train'],required=True)
    p.add_argument('--raw-zip',type=Path)
    p.add_argument('--observations',type=Path,default=Path('outputs/constant_dataset/constant_observations.csv'))
    p.add_argument('--data-dir',type=Path,default=Path('outputs/sequence_3p6_input'))
    p.add_argument('--work-dir',type=Path,default=Path('work/sequence_3p6'))
    p.add_argument('--output-dir',type=Path,default=Path('outputs/sequence_3p6_results'))
    p.add_argument('--models',nargs='+',choices=['LSTM','Transformer'],default=['LSTM','Transformer'])
    p.add_argument('--device',default='cpu')
    a=p.parse_args()
    if a.stage=='prepare':
        if not a.raw_zip:p.error('--raw-zip is required for the prepare stage')
        prepare(a.raw_zip,a.observations,a.data_dir,a.work_dir)
    else:run_training(a.data_dir,a.output_dir,a.models,device=a.device)
if __name__=='__main__':main()
