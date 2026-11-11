"""STEP 11M — STRICT-ELECTRICAL LINEAR/RIDGE REGRESSION BENCHMARK

Reviewer 2 requested a simple electrical-feature regression comparator to HGB.
This script runs fully independent, nested, group-held-out linear models on the
existing FROZEN Step 3/5 datasets and compares them against FROZEN Step 4/6
HGB out-of-fold predictions. It NEVER reads raw battery files or retrains HGB.

Run as a local Python command. Defaults to both
constant (10 batteries, 4590 cycles) and variable (6 batteries, 165 independent
reference observations) protocol datasets.

Key design:
* Same frozen strict candidate feature family and age definitions as HGB.
* Inner LOBO does all availability screening, imputation, Spearman ranking,
  correlation pruning, model selection and scaling from inner-training only.
* Outer LOBO gives battery-disjoint out-of-fold predictions.
* Primary metric = mean of per-battery RMSE (macro), on raw predictions.
* Equal-total-battery training weights; test battery is never used for fitting.
* Cross-source coverage >=95% inside EACH represented training source.
* Compares matched OOF predictions with saved Step 4/6 predictions.
* Paired battery bootstrap resamples fixed OOF battery errors; no retraining.

This is a LINEAR/RIDGE BASELINE, not an HGB replication. Feature-selection
hyperparameters and regularization are chosen independently inside inner LOBO.
The model-specific feature selections can differ from those of the frozen HGB.
"""
from __future__ import annotations

import os
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler

# ============================================================================
# 0. COLAB / SETTINGS
# ============================================================================
# Local command-line settings. No notebook or hosted runtime required.
PROJECT_ROOT = Path('.')
OUTPUT_DIR = Path('outputs/electrical_linear_baseline')
RUN_CONSTANT = True
RUN_VARIABLE = True
IN_COLAB = False
AUTO_DOWNLOAD_WORKBOOK = False

RANDOM_STATE = 42
N_BOOTSTRAP = 20_000
MIN_SOURCE_COVERAGE = 0.95
CLIP_LOW, CLIP_HIGH = 70.0, 100.0

# Exactly the Step 6 feature-selection hyperparameter families. The linear
# regularization grid is new and predeclared; includes alpha=0 (OLS limit).
CORR_THRESHOLDS = (0.90, 0.93, 0.95)
TOP_K_OPTIONS = (8, 12, 16)
RIDGE_ALPHAS = (0.0, 0.1, 1.0, 10.0, 100.0)

# HGB's explicit age fields were NOT part of its electrical-only candidate set.
EXPLICIT_AGE_STRICT_FEATURES = {
    'LEGACY__Cycle_Age', 'LEGACY__Cycle_Index', 'LEGACY__Log_Cycle_Age'
}

# For reproducible, safe paths: data, frozen candidate list, and OOF predictions.
EXPERIMENTS = {
    'constant': {
        'data_file': 'Step5_Revised_Constant_Cycle_Observations.csv',
        'features_file': 'Step5_Frozen_Strict_Constant_Feature_Names.csv',
        'reference_file': 'Step6_Constant_HGB_Predictions.csv',
        'data_dirs': ('frozen_data/step5_constant', 'step5_constant'),
        'reference_dirs': ('results/step6_constant_hgb', 'step6_constant_hgb'),
        'feature_column': 'Feature',
        'target': 'Calculated_SOH(%)',
        'key': 'Cycle_Index',
        'age': ('Cycle_Age', 'Log_Cycle_Age'),
        'n_batteries': 10,
        'n_rows': 4590,
        'feature_prefixes': ('',),
        'hgb_ref_formulations': {
            'ELECTRICAL_LINEAR': 'STRICT_ELECTRICAL_HGB',
            'ELECTRICAL_AGE_LINEAR': 'STRICT_ELECTRICAL_PLUS_AGE_HGB',
        },
    },
    'variable': {
        'data_file': 'Step3_Revised_Variable_Independent_Observations.csv',
        'features_file': 'Step3_Frozen_Strict_Variable_Feature_Names.csv',
        'reference_file': 'Step4_HGB_Predictions.csv',
        'data_dirs': ('frozen_data/step3', 'step3'),
        'reference_dirs': ('results/step4_variable_hgb', 'results/step4', 'step4'),
        'feature_column': 'Cycle_Level_Strict_Feature',
        'target': 'Target_SOH',
        'key': 'R5E_Reference_Epoch',
        'age': ('Age_Coord', 'Log_Age_Coord'),
        'n_batteries': 6,
        'n_rows': 165,
        # The Step 3 macro-cycle dataframe stores both aggregated observations.
        'feature_prefixes': ('MED__', 'LAST__'),
        'hgb_ref_formulations': {
            'ELECTRICAL_LINEAR': 'STRICT_ELECTRICAL_HGB',
            'ELECTRICAL_AGE_LINEAR': 'STRICT_ELECTRICAL_PLUS_AGE_HGB',
        },
    },
}

# ============================================================================
# 1. INPUT DISCOVERY, DATA VALIDATION, FEATURE POLICY
# ============================================================================
def find_input(filename, preferred_dirs):
    for relative in preferred_dirs:
        candidate = PROJECT_ROOT / relative / filename
        if candidate.is_file():
            return candidate
    # Equivalent filenames emitted by the preserved local raw-data pipeline.
    aliases = {
        'Step5_Revised_Constant_Cycle_Observations.csv': 'constant_dataset/constant_observations.csv',
        'Step5_Frozen_Strict_Constant_Feature_Names.csv': 'constant_dataset/constant_strict_feature_names.csv',
        'Step6_Constant_HGB_Predictions.csv': 'constant_hgb/constant_hgb_predictions.csv',
        'Step3_Revised_Variable_Independent_Observations.csv': 'variable_dataset/variable_independent_observations.csv',
        'Step3_Frozen_Strict_Variable_Feature_Names.csv': 'variable_dataset/variable_strict_feature_names.csv',
        'Step4_HGB_Predictions.csv': 'variable_hgb/variable_hgb_predictions.csv',
    }
    alternative = PROJECT_ROOT / aliases.get(filename, '__no_alias__')
    if alternative.is_file():
        return alternative
    matches = sorted(PROJECT_ROOT.rglob(filename))
    if not matches:
        raise FileNotFoundError(
            f'Cannot locate {filename} anywhere in {PROJECT_ROOT}.\n'
            'Provide the original Step 3/5 data and Step 4/6 model predictions in the local project-root folder.'
        )
    if len(matches) > 1:
        raise RuntimeError(
            f'Multiple copies of {filename} found; select one unambiguously:\n' +
            '\n'.join(map(str, matches))
        )
    return matches[0]


def cycle_key(val):
    z = pd.to_numeric(pd.Series([val]), errors='coerce').iloc[0]
    return format(float(z), '.12g') if np.isfinite(z) else str(val)


def validate_and_load(protocol, setting):
    data_path = find_input(setting['data_file'], setting['data_dirs'])
    feats_path = find_input(setting['features_file'], setting['data_dirs'])
    ref_path = find_input(setting['reference_file'], setting['reference_dirs'])

    frame = pd.read_csv(data_path, low_memory=False)
    fr = pd.read_csv(feats_path)
    reference = pd.read_csv(ref_path, low_memory=False)

    req = ['battery_id', 'source_dataset', setting['target'], setting['key'], *setting['age']]
    missing = [c for c in req if c not in frame.columns]
    if missing:
        raise ValueError(f'{protocol} frozen data missing required columns: {missing}')
    if setting['feature_column'] not in fr:
        raise ValueError(f'{protocol} feature list lacks {setting["feature_column"]}')

    frozen_features = fr[setting['feature_column']].dropna().astype(str).tolist()
    if len(frozen_features) != len(set(frozen_features)):
        raise AssertionError('Frozen strict feature list contains duplicates')
    if protocol == 'constant' and len(frozen_features) != 335:
        raise AssertionError(f'Expected 335 constant strict features, got {len(frozen_features)}')
    if protocol == 'variable' and len(frozen_features) != 314:
        raise AssertionError(f'Expected 314 variable strict base features, got {len(frozen_features)}')
    electrical_base = [f for f in frozen_features if f not in EXPLICIT_AGE_STRICT_FEATURES]
    candidates = [prefix + f for prefix in setting['feature_prefixes'] for f in electrical_base]

    missing_features = [f for f in candidates if f not in frame]
    if missing_features:
        raise AssertionError(
            f'{protocol}: missing frozen candidate features, first 15: {missing_features[:15]}'
        )

    if len(frame) != setting['n_rows'] or frame['battery_id'].nunique() != setting['n_batteries']:
        raise AssertionError(
            f'{protocol}: expected {setting["n_rows"]} observations across '
            f'{setting["n_batteries"]} batteries, observed {len(frame)} across '
            f'{frame["battery_id"].nunique()}. Do not silently change the population.'
        )

    frame = frame.copy()
    frame['_target'] = pd.to_numeric(frame[setting['target']], errors='coerce')
    if not np.isfinite(frame['_target']).all():
        raise ValueError(f'{protocol}: non-finite targets')
    if not frame['_target'].between(70 - 1e-8, 100 + 1e-8).all():
        raise ValueError(f'{protocol}: target outside the frozen 70–100% scope')
    frame['_obs_key'] = frame[setting['key']].map(cycle_key)
    if frame.duplicated(['battery_id', '_obs_key']).any():
        raise AssertionError(f'{protocol}: duplicate battery+observation keys')

    needed_ref = ['Formulation', 'battery_id', setting['key'], setting['target'], 'Prediction_Raw']
    if any(c not in reference.columns for c in needed_ref):
        raise ValueError(f'{protocol}: frozen HGB predictions missing columns: '
                         f'{[c for c in needed_ref if c not in reference.columns]}')
    reference['_obs_key'] = reference[setting['key']].map(cycle_key)

    # Require that BOTH frozen HGB formulations cover exactly the model population.
    for model_label, ref_name in setting['hgb_ref_formulations'].items():
        subset = reference.loc[reference['Formulation'].eq(ref_name)].copy()
        if len(subset) != len(frame) or subset.duplicated(['battery_id','_obs_key']).any():
            raise AssertionError(f'{protocol} {ref_name}: frozen HGB population mismatch')
        joined = frame[['battery_id','_obs_key','_target']].merge(
            subset[['battery_id','_obs_key',setting['target']]],
            on=['battery_id','_obs_key'], how='left', validate='one_to_one', indicator=True
        )
        if not joined['_merge'].eq('both').all():
            raise AssertionError(f'{protocol} {ref_name}: frozen HGB keys do not match')
        if not np.allclose(joined['_target'].to_numpy(float),
                           pd.to_numeric(joined[setting['target']],errors='coerce').to_numpy(float),
                           atol=1e-8,rtol=0):
            raise AssertionError(f'{protocol} {ref_name}: frozen targets differ')

    audit = {
        'Protocol': protocol,
        'Input_File': str(data_path),
        'Frozen_Feature_File': str(feats_path),
        'HGB_Reference_File': str(ref_path),
        'Observations': len(frame),
        'Independent_Batteries': frame['battery_id'].nunique(),
        'Strict_Base_Feature_Count': len(frozen_features),
        'Strict_Electrical_Candidates': len(candidates),
        'Age_Columns': ', '.join(setting['age']),
        'Min_SoH': float(frame['_target'].min()),
        'Max_SoH': float(frame['_target'].max()),
        'Outcome': 'PASS',
    }
    return frame, candidates, reference, audit

# ============================================================================
# 2. SOURCE-AWARE TRAINING-ONLY SELECTION (SAME POLICY AS STEP 6)
# ============================================================================
def finite_frame(frame, columns):
    return (frame.loc[:, list(columns)]
            .apply(pd.to_numeric, errors='coerce')
            .replace([np.inf, -np.inf], np.nan))


def training_source_coverage_filter(train, candidates, minimum=MIN_SOURCE_COVERAGE):
    """Exactly the source-wise Step 6 95% rule, calculated by column blocks."""
    values = finite_frame(train, candidates)
    counts = values.nunique(dropna=True)
    srcs = sorted(train['source_dataset'].dropna().unique())
    coverage = {
        str(src): values.loc[train['source_dataset'].eq(src)].notna().mean()
        for src in srcs
    }
    eligible = []
    audit = []
    for feature in candidates:
        covers = {src:float(table[feature]) for src,table in coverage.items()}
        unique_values = int(counts[feature])
        keep = unique_values > 1 and all(np.isfinite(v) and v >= minimum
                                         for v in covers.values())
        audit.append({'Feature':feature,'Eligible':bool(keep),
                      'Unique_Training_Values':unique_values,
                      **{f'Coverage_{k}':v for k,v in covers.items()}})
        if keep:
            eligible.append(feature)
    if not eligible:
        raise RuntimeError('No electrical features after training-only source coverage filter')
    return eligible, audit


def prepare_feature_selector(train, candidates):
    """Rank training features by the original absolute Spearman statistic.

    A full 300x300 Spearman matrix for every nested fold is unnecessary.
    Store normalized ranks once and compute *only* the pairwise correlations
    needed by greedy pruning. The resulting selection criterion is unchanged.
    """
    eligible, coverage_audit = training_source_coverage_filter(train, candidates)
    X_raw = finite_frame(train, eligible)
    imp = SimpleImputer(strategy='median')
    X = pd.DataFrame(imp.fit_transform(X_raw), index=train.index, columns=eligible)
    ranked_X = X.rank(method='average')
    ranked_y = pd.Series(train['_target'].to_numpy(float), index=train.index).rank(method='average')
    scores = ranked_X.corrwith(ranked_y).abs().replace(
        [np.inf,-np.inf],np.nan).fillna(0)
    ranked_order = scores.sort_values(ascending=False, kind='mergesort').index.tolist()

    ranked_matrix = ranked_X.to_numpy(dtype=np.float64, copy=False)
    mu = ranked_matrix.mean(axis=0)
    sd = ranked_matrix.std(axis=0)
    sd[sd == 0] = 1.0
    normalized_ranks = (ranked_matrix - mu) / sd
    return {'ranked_order':ranked_order,'scores':scores,
            'normalized_ranks':normalized_ranks,'rank_to_index':{f:i for i,f in enumerate(eligible)},
            'eligible':eligible,'coverage_audit':coverage_audit}


def choose_features(selector, top_k, corr_threshold):
    """Exact Spearman greedy pruning computed on demand (no full matrix)."""
    selected = []
    normalized=selector['normalized_ranks']
    pos=selector['rank_to_index']
    n=normalized.shape[0]
    for feature in selector['ranked_order']:
        if len(selected) >= int(top_k):
            break
        if selected:
            i=pos[feature]
            j=[pos[f] for f in selected]
            correlations=np.abs(normalized[:,i] @ normalized[:,j] / n)
            if bool((correlations >= float(corr_threshold)).any()):
                continue
        selected.append(feature)
    if not selected:
        raise RuntimeError('No features selected')
    return selected

# ============================================================================
# 3. BATTERY-BALANCED REGRESSION, INNER LOBO, OUTER LOBO
# ============================================================================
def equal_battery_weights(train):
    counts = train['battery_id'].value_counts()
    w = train['battery_id'].map(lambda x:1.0/counts[x]).to_numpy(float)
    return w / w.mean()


def rmse(y, yhat):
    return float(np.sqrt(mean_squared_error(y,yhat)))


def mae(y, yhat):
    return float(mean_absolute_error(y,yhat))


def fit_preprocessing(train, test, selected, age_cols):
    columns = list(selected) + list(age_cols)
    # Each split's training-only imputation and standardization.
    X_tr = finite_frame(train, columns)
    X_te = finite_frame(test, columns)
    imputer = SimpleImputer(strategy='median')
    xtr = imputer.fit_transform(X_tr)
    xte = imputer.transform(X_te)
    weights = equal_battery_weights(train)
    scaler = StandardScaler()
    xtr = scaler.fit_transform(xtr, sample_weight=weights)
    xte = scaler.transform(xte)
    return xtr, xte, weights, columns


def fit_predict_linear(train, test, selected, age_cols, alpha):
    xtr,xte,weights,columns = fit_preprocessing(train,test,selected,age_cols)
    model = Ridge(alpha=float(alpha), fit_intercept=True, solver='auto')
    model.fit(xtr, train['_target'].to_numpy(float), sample_weight=weights)
    raw = np.asarray(model.predict(xte),dtype=float)
    return raw, columns, model


def precompute_inner(outer_train, candidates):
    prepared=[]
    for holdout in sorted(outer_train['battery_id'].unique()):
        inn_train = outer_train.loc[~outer_train['battery_id'].eq(holdout)].copy()
        inn_test  = outer_train.loc[ outer_train['battery_id'].eq(holdout)].copy()
        selector = prepare_feature_selector(inn_train,candidates)  # inner train only
        prepared.append((holdout,inn_train,inn_test,selector))
    return prepared


def tune_linear_model(inner, age_cols):
    # Cache one source-safe selected matrix per (threshold, K, inner split),
    # then try each predeclared alpha without refitting the preprocessing.
    records=[]
    for threshold in CORR_THRESHOLDS:
        for top_k in TOP_K_OPTIONS:
            errors={a:[] for a in RIDGE_ALPHAS}
            abs_errors={a:[] for a in RIDGE_ALPHAS}
            for inner_holdout,train,test,selector in inner:
                selected=choose_features(selector,top_k,threshold)
                xtr,xte,w,_ = fit_preprocessing(train,test,selected,age_cols)
                yy=test['_target'].to_numpy(float)
                for alpha in RIDGE_ALPHAS:
                    model=Ridge(alpha=float(alpha),fit_intercept=True,solver='auto')
                    model.fit(xtr,train['_target'].to_numpy(float),sample_weight=w)
                    pred=model.predict(xte)
                    errors[alpha].append(rmse(yy,pred))
                    abs_errors[alpha].append(mae(yy,pred))
            for alpha in RIDGE_ALPHAS:
                records.append({'Corr_Threshold':threshold,'Top_K':top_k,
                                'Alpha':float(alpha),
                                'Inner_Macro_RMSE':float(np.mean(errors[alpha])),
                                'Inner_Macro_MAE':float(np.mean(abs_errors[alpha])),
                                'Inner_Batteries':len(inner)})
    ranked=pd.DataFrame(records).sort_values(
        ['Inner_Macro_RMSE','Top_K','Corr_Threshold','Alpha'],kind='mergesort'
    ).reset_index(drop=True)
    best=ranked.iloc[0]
    return float(best['Corr_Threshold']),int(best['Top_K']),float(best['Alpha']),ranked


def run_protocol(protocol, setting):
    frame,candidates,reference,audit=validate_and_load(protocol,setting)
    battery_ids=sorted(frame['battery_id'].unique())
    result={k:[] for k in ['predictions','folds','tuning','features','coverage']}
    print(f'\n{"="*78}\n{protocol.upper()} — {len(frame):,} observations, '
          f'{len(battery_ids)} batteries; {len(candidates)} strict electrical candidates\n{"="*78}')

    # Run both formulations. Each form independently tunes the selection and alpha.
    for fold_no,battery in enumerate(battery_ids,1):
        start=time.time()
        outer_train=frame.loc[~frame['battery_id'].eq(battery)].copy()
        outer_test =frame.loc[ frame['battery_id'].eq(battery)].copy()
        inner=precompute_inner(outer_train,candidates)
        selector=prepare_feature_selector(outer_train,candidates)
        for info in selector['coverage_audit']:
            result['coverage'].append({'Protocol':protocol,'Outer_Holdout':battery,**info})
        for model_label,include_age in [('ELECTRICAL_LINEAR',False),('ELECTRICAL_AGE_LINEAR',True)]:
            age_cols=setting['age'] if include_age else ()
            threshold,k,alpha,ranked=tune_linear_model(inner,age_cols)
            selected=choose_features(selector,k,threshold)
            raw,columns,model=fit_predict_linear(outer_train,outer_test,selected,age_cols,alpha)
            clipped=np.clip(raw,CLIP_LOW,CLIP_HIGH)
            y=outer_test['_target'].to_numpy(float)
            formulation=f'{protocol.upper()}_{model_label}'
            result['folds'].append({
                'Protocol':protocol,'Formulation':formulation,'Holdout_Battery':battery,
                'N_Test':len(outer_test),'Source':str(outer_test['source_dataset'].iloc[0]),
                'Eligible_Training_Features':len(selector['eligible']),
                'Selected_Electrical_Features':len(selected),'Added_Age_Features':len(age_cols),
                'Best_Corr_Threshold':threshold,'Best_Top_K':k,'Best_Alpha':alpha,
                'Fitted_Model':'OLS' if alpha==0 else 'Ridge',
                'RMSE_Raw':rmse(y,raw),'MAE_Raw':mae(y,raw),
                'RMSE_Clipped':rmse(y,clipped),'MAE_Clipped':mae(y,clipped),
                'Elapsed_Seconds':round(time.time()-start,2)
            })
            ranked.insert(0,'Protocol',protocol)
            ranked.insert(1,'Formulation',formulation)
            ranked.insert(2,'Outer_Holdout',battery)
            result['tuning'].append(ranked)
            for rank,name in enumerate(selected,1):
                result['features'].append({
                    'Protocol':protocol,'Formulation':formulation,'Outer_Holdout':battery,
                    'Rank':rank,'Feature':name,
                    'Training_Spearman_Abs':float(selector['scores'][name]),
                    'Corr_Threshold':threshold,'Top_K':k,'Alpha':alpha,
                })
            for (_,row),a,b in zip(outer_test.iterrows(),raw,clipped):
                result['predictions'].append({
                    'Protocol':protocol,'Formulation':formulation,
                    'Holdout_Battery':battery,'battery_id':row['battery_id'],
                    'source_dataset':row['source_dataset'],
                    'Observation_ID':row[setting['key']], '_obs_key':row['_obs_key'],
                    'Age':row[setting['age'][0]],
                    'Target_SOH':float(row['_target']),
                    'Prediction_Raw':float(a),'Prediction_Clipped70_100':float(b),
                    'Residual_Raw':float(row['_target']-a),
                    'Residual_Clipped70_100':float(row['_target']-b),
                })
            print(f' {fold_no:02d}/{len(battery_ids)} {battery:>7} | '
                  f'{model_label:22} | K={k:2d} thr={threshold:.2f} '
                  f'alpha={alpha:<6g} | RMSE={rmse(y,raw):.4f}')
        print(f'   fold total elapsed {time.time()-start:.1f} sec')

    pred=pd.DataFrame(result['predictions'])
    folds=pd.DataFrame(result['folds'])
    selected=pd.DataFrame(result['features'])
    tuning=pd.concat(result['tuning'],ignore_index=True)
    coverage=pd.DataFrame(result['coverage'])
    macro=(folds.groupby(['Protocol','Formulation'],as_index=False)
           .agg(Batteries=('Holdout_Battery','nunique'),
                Macro_RMSE_Raw=('RMSE_Raw','mean'),Macro_MAE_Raw=('MAE_Raw','mean'),
                Macro_RMSE_Clipped=('RMSE_Clipped','mean'),Macro_MAE_Clipped=('MAE_Clipped','mean'),
                Observations=('N_Test','sum')))
    freq=(selected.groupby(['Protocol','Formulation','Feature'],as_index=False)
          .agg(Outer_Folds_Selected=('Outer_Holdout','nunique'),
               Mean_Training_Spearman=('Training_Spearman_Abs','mean')))

    # Strict equality with frozen original HGB predictions, all batteries and cycles.
    joined=[]
    paired=[]
    for model_label,ref_name in setting['hgb_ref_formulations'].items():
        formulation=f'{protocol.upper()}_{model_label}'
        base=pred.loc[pred['Formulation'].eq(formulation)].copy()
        old=reference.loc[reference['Formulation'].eq(ref_name)].copy()
        old=old[['battery_id','_obs_key',setting['target'],'Prediction_Raw']].rename(
            columns={'Prediction_Raw':'HGB_Prediction_Raw',setting['target']:'HGB_Target_SOH'})
        compare=base.merge(old,on=['battery_id','_obs_key'],how='left',validate='one_to_one',indicator=True)
        if len(compare)!=len(frame) or not compare['_merge'].eq('both').all():
            raise AssertionError(f'{formulation}: regression/HGB unmatched observations')
        if not np.allclose(compare['Target_SOH'],compare['HGB_Target_SOH'],rtol=0,atol=1e-8):
            raise AssertionError(f'{formulation}: regression/HGB target mismatch')
        compare['HGB_Residual_Raw']=compare['HGB_Target_SOH']-compare['HGB_Prediction_Raw']
        compare['HGB_Formulation']=ref_name
        compare=compare.drop(columns=['_merge'])
        joined.append(compare)
        for b,group in compare.groupby('battery_id'):
            yy=group['Target_SOH'].to_numpy(float)
            ridge=group['Prediction_Raw'].to_numpy(float)
            hgb=group['HGB_Prediction_Raw'].to_numpy(float)
            paired.append({'Protocol':protocol,'Formulation':formulation,'Battery':b,
                           'N':len(group),'Linear_RMSE':rmse(yy,ridge),'HGB_RMSE':rmse(yy,hgb),
                           'Linear_MAE':mae(yy,ridge),'HGB_MAE':mae(yy,hgb)})
    joined=pd.concat(joined,ignore_index=True)
    paired=pd.DataFrame(paired)
    paired['Delta_RMSE_LinearMinusHGB']=paired['Linear_RMSE']-paired['HGB_RMSE']
    paired['Delta_MAE_LinearMinusHGB']=paired['Linear_MAE']-paired['HGB_MAE']

    boot=[]
    for formulation,g in paired.groupby('Formulation'):
        d=g['Delta_RMSE_LinearMinusHGB'].to_numpy(float)
        rng=np.random.default_rng(RANDOM_STATE)
        boot_idx=rng.integers(0,len(d),size=(N_BOOTSTRAP,len(d)))
        resampled=d[boot_idx].mean(axis=1)
        lo,hi=np.quantile(resampled,[0.025,0.975])
        boot.append({'Protocol':protocol,'Formulation':formulation,
                     'Batteries':len(d),'Paired_Macro_Delta_RMSE_LinearMinusHGB':float(d.mean()),
                     'CI95_Lower':float(lo),'CI95_Upper':float(hi),
                     'Resamples':N_BOOTSTRAP,
                     'Interval_Type':'Fixed out-of-fold errors; no model refit',
                     'Interpretation':'Negative favors linear; positive favors frozen HGB'})
    boot=pd.DataFrame(boot)
    comparison=(paired.groupby(['Protocol','Formulation'],as_index=False)
                .agg(Linear_Macro_RMSE=('Linear_RMSE','mean'),
                     Frozen_HGB_Macro_RMSE=('HGB_RMSE','mean'),
                     Linear_Macro_MAE=('Linear_MAE','mean'),
                     Frozen_HGB_Macro_MAE=('HGB_MAE','mean'),
                     Batteries=('Battery','nunique'), Observations=('N','sum')))
    comparison['Delta_RMSE_LinearMinusHGB']=comparison['Linear_Macro_RMSE']-comparison['Frozen_HGB_Macro_RMSE']

    return {'data_audit':pd.DataFrame([audit]),'macro':macro,'folds':folds,
            'predictions':pred,'tuning':tuning,'selected':selected,'frequency':freq,
            'coverage':coverage,'matched':joined,'per_battery_paired':paired,
            'comparison':comparison,'bootstrap':boot}

# ============================================================================
# 4. EXPORT WORKBOOK, MACHINE-READABLE TABLES, REPRODUCIBILITY NOTES
# ============================================================================
def methods_table():
    return pd.DataFrame([
        ('Objective','Reviewer 2 simple electrical-feature linear-regression control.'),
        ('Information setting','Same frozen strict electrical candidates as Step 4/6; optional explicit age is separate.'),
        ('Units','Constant=cycle, variable=independent reference observation (macro interval).'),
        ('Input timing','Original frozen full-observation features; NOT Step 11G 3.6V and NOT Step 11L 600-s.'),
        ('Variable candidates','MED__ and LAST__ expansions of Step 3 cycle-level strict feature list.'),
        ('Availability filter','At least 95% finite coverage in each training source; training-only.'),
        ('Feature ranking','Absolute Spearman rank correlation with training target (original Step 4/6 method).'),
        ('Correlation pruning','Greedy Spearman redundancy removal; thresholds: '+str(CORR_THRESHOLDS)),
        ('Top K','Candidate selected feature caps: '+str(TOP_K_OPTIONS)),
        ('Regression','OLS (alpha=0) or Ridge (alpha>0) with intercept; sklearn solver=svd.'),
        ('Regularization grid','Alpha values: '+str(RIDGE_ALPHAS)),
        ('Scaling','Median imputation and weighted StandardScaler fit inside current training split.'),
        ('Training weights','Each battery has equal total weight within its training split.'),
        ('Inner validation','Inner leave-one-battery-out: tuning of selection and alpha by battery macro RMSE.'),
        ('Outer validation','Outer leave-one-battery-out; all test batteries unseen in training.'),
        ('Primary prediction','Raw (unclipped), consistent with frozen Step 4/6 HGB primary evaluation.'),
        ('Clipped sensitivity','Predictions separately clipped to 70–100%; NOT the primary reported comparison.'),
        ('HGB reference','Read frozen Step 4/6 predictions; exact same observation keys and targets are asserted.'),
        ('Bootstrap','20,000 paired resamples of fixed battery errors; does NOT include model retraining.'),
        ('Selection caveat','Independently tuned linear pipeline is comparable as a pipeline; not same features as HGB.'),
        ('Scientific scope','70–100% capacity-retention SoH only; no claim about knee/EOL regimes.'),
        ('No additional raw signal preprocessing','Frozen CSV data reused without re-extraction.'),
    ],columns=['Method','Detail'])


def write_workbook(tables):
    outfile=OUTPUT_DIR/'Reviewer_Step11M_Linear_Regression_Baseline.xlsx'
    names={
        'methods':'Methods', 'data_audit':'Input_Audit', 'macro':'Macro_Summary',
        'comparison':'HGB_Comparison', 'bootstrap':'Paired_Bootstrap',
        'folds':'Per_Battery','per_battery_paired':'Matched_Battery',
        'predictions':'OOF_Predictions', 'matched':'Matched_OOF',
        'selected':'Selected_Features','frequency':'Selection_Frequency',
        'tuning':'Inner_Tuning','coverage':'Feature_Coverage'
    }
    # Export a single transparent and versionable workbook; all tables are
    # derived from frozen observations and out-of-fold predictions.
    with pd.ExcelWriter(outfile,engine='openpyxl') as writer:
        for key,sheet in names.items():
            tbl = methods_table() if key=='methods' else tables.get(key, pd.DataFrame())
            if tbl is None or tbl.empty: continue
            tbl.to_excel(writer,sheet_name=sheet,index=False)
            ws=writer.sheets[sheet]
            ws.freeze_panes='A2'
            ws.auto_filter.ref=ws.dimensions
            from openpyxl.styles import PatternFill,Font,Alignment
            fill=PatternFill('solid', fgColor='183153')
            for cell in ws[1]:
                cell.fill=fill
                cell.font=Font(bold=True,color='FFFFFF')
                cell.alignment=Alignment(wrap_text=True,vertical='center')
            ws.row_dimensions[1].height=30
            for col in ws.columns:
                letter=col[0].column_letter
                sample=list(col[:min(len(col),51)])
                lens=[len(str(cell.value)) for cell in sample if cell.value is not None]
                ws.column_dimensions[letter].width=min(43,max(13, max(lens,default=14)+2))
    for name,key in [('Step11M_Macro.csv','macro'),('Step11M_HGB_Comparison.csv','comparison'),
                     ('Step11M_OOF_Predictions.csv','predictions'),
                     ('Step11M_Paired_Bootstrap.csv','bootstrap')]:
        if key in tables and not tables[key].empty:
            tables[key].to_csv(OUTPUT_DIR/name,index=False)
    return outfile


def main():
    chosen=[p for p,flag in [('constant',RUN_CONSTANT),('variable',RUN_VARIABLE)] if flag]
    if not chosen:
        raise ValueError('Enable RUN_CONSTANT and/or RUN_VARIABLE')
    print('STEP 11M | frozen strict linear/Ridge LOBO comparator')
    print('Project:',PROJECT_ROOT)
    print('Protocols:',', '.join(chosen))
    print('This script NEVER retrains the frozen HGB references.\n')
    t0=time.time()
    outputs={p:run_protocol(p,EXPERIMENTS[p]) for p in chosen}
    keys=list(next(iter(outputs.values())).keys())
    tables={k:pd.concat([o[k] for o in outputs.values()],ignore_index=True) for k in keys}
    outfile=write_workbook(tables)
    print('\n'+'='*85)
    print('FINAL MATCHED HGB COMPARISON (battery-macro SOH percentage-point RMSE)')
    print(tables['comparison'].to_string(index=False))
    print('\nPAIRED BATTERY-BOOTSTRAP (linear minus HGB):')
    print(tables['bootstrap'].to_string(index=False))
    print(f'\nSaved: {outfile}\nElapsed: {(time.time()-t0)/60:.1f} minutes')
    return tables


if __name__ == '__main__':
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project-root',type=Path,default=Path('.'))
    parser.add_argument('--output-dir',type=Path,default=Path('outputs/electrical_linear_baseline'))
    parser.add_argument('--protocol',choices=['constant','variable','both'],default='both')
    cli=parser.parse_args()
    PROJECT_ROOT=cli.project_root.resolve()
    OUTPUT_DIR=cli.output_dir.resolve()
    OUTPUT_DIR.mkdir(parents=True,exist_ok=True)
    RUN_CONSTANT=cli.protocol in ('constant','both')
    RUN_VARIABLE=cli.protocol in ('variable','both')
    main()

