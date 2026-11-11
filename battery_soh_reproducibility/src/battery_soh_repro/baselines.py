"""Battery-held-out age-only baselines and source-transfer controls."""
from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from battery_soh_repro.experiment_utils import metric_by_battery

HGB_CONFIGS = {
    'H0': dict(learning_rate=.05,max_iter=300,max_leaf_nodes=15,min_samples_leaf=5,l2_regularization=1.),
    'H1': dict(learning_rate=.04,max_iter=500,max_leaf_nodes=15,min_samples_leaf=10,l2_regularization=2.),
    'H2': dict(learning_rate=.03,max_iter=700,max_leaf_nodes=31,min_samples_leaf=10,l2_regularization=3.),
}


def battery_weights(battery):
    s=pd.Series(battery)
    c=s.value_counts()
    w=s.map(lambda b:1./c[b]).to_numpy(float)
    return w/w.mean()


def age_model(frame:pd.DataFrame, age_columns:list[str], target:str,
              kind:str='linear', config:str='H1'):
    x=frame[age_columns].to_numpy(float)
    if not np.isfinite(x).all():raise ValueError('Non-finite age features')
    y=frame[target].to_numpy(float)
    weights=battery_weights(frame.battery_id)
    if kind=='linear':
        model=LinearRegression().fit(x, y, sample_weight=weights)
    elif kind=='hgb':
        model=HistGradientBoostingRegressor(loss='squared_error',early_stopping=False,
                                               random_state=42,**HGB_CONFIGS[config])
        model.fit(x,y,sample_weight=weights)
    else:raise ValueError(kind)
    return model


def tune_age_hgb(frame:pd.DataFrame, age_columns:list[str], target:str):
    vals=[]
    for config in HGB_CONFIGS:
        err=[]
        for b in sorted(frame.battery_id.unique()):
            t=frame[frame.battery_id!=b]
            e=frame[frame.battery_id==b]
            fit=age_model(t,age_columns,target,kind='hgb',config=config)
            pred=fit.predict(e[age_columns].to_numpy(float))
            err.append(np.sqrt(np.mean((e[target].to_numpy(float)-pred)**2)))
        vals.append((float(np.mean(err)),config))
    return min(vals, key=lambda x:(x[0],x[1]))


def age_only_lobo(frame:pd.DataFrame, age_columns:list[str], target:str,
                  kind:str='hgb') -> tuple[pd.DataFrame,pd.DataFrame]:
    predictions=[]; folds=[]
    for b in sorted(frame.battery_id.unique()):
        train=frame[frame.battery_id!=b]
        test=frame[frame.battery_id==b]
        score,config=tune_age_hgb(train,age_columns,target) if kind=='hgb' else (np.nan,'linear')
        mod=age_model(train,age_columns,target,kind=kind,config=config if kind=='hgb' else 'H1')
        yh=mod.predict(test[age_columns].to_numpy(float))
        for (idx,row),p in zip(test.iterrows(),yh):
            predictions.append({'battery_id':b,'source_dataset':row['source_dataset'],
                                'Prediction_Raw':float(p),target:float(row[target]),
                                '_Original_Row':int(idx)})
        folds.append({'Holdout_Battery':b,'Config':config,
                      'Inner_Mean_Battery_RMSE':score})
    return pd.DataFrame(predictions),pd.DataFrame(folds)
