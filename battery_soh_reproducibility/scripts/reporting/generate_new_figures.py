"""Generate seven new manuscript figures from experimental prediction/metric CSVs.

No numeric RMSE is hardcoded. The metric input for comparative bar/point plots
uses the documented columns: Figure, Panel, Model, RMSE, Dataset (optional).
Rows for figures 3, 4 and 5 must be generated from the corresponding local
experiment outputs and verified against the final submitted manuscript.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

PLOTS={
  1:'Fig1_Constant_LOBO_Measured_vs_Predicted_SOH.png',
  2:'Fig2_Constant_LOBO_Residual_Diagnostics.png',
  3:'Fig3_Full_Discharge_vs_600s_vs_3p6V.png',
  4:'Fig4_SourceOnly_LOBO_vs_CrossDataset_Transfer.png',
  5:'Fig5_HGB_vs_Tuned_LSTM_Transformer_at_3p6V.png',
  6:'Fig6_Screened_01_Trajectories.png',
  7:'Fig7_Screened_02_Residual_Diagnostics.png',
}


def apply_style():
    plt.rcParams.update({'font.family':'serif','font.serif':['EB Garamond','DejaVu Serif'],
                         'axes.spines.top':False,'axes.spines.right':False,
                         'font.size':10,'axes.grid':True,'grid.alpha':.14})


def selected(predictions:pd.DataFrame,formulation:str='STRICT_ELECTRICAL_PLUS_AGE_HGB'):
    if 'Formulation' in predictions:
        predictions=predictions[predictions.Formulation.eq(formulation)].copy()
    if predictions.empty:raise ValueError(f'No predictions for {formulation}')
    required=['battery_id','Cycle_Index','Calculated_SOH(%)','Prediction_Raw']
    if not set(required).issubset(predictions.columns):raise ValueError('Expected prediction columns: '+str(required))
    return predictions


def trajectories(pred:pd.DataFrame,out:Path,screened:bool):
    pred=selected(pred)
    order=['B0005','B0006','B0007','B0018','CS2_33','CS2_34','CS2_35','CS2_36','CS2_37','CS2_38']
    fig,axes=plt.subplots(2,5,figsize=(16.8,7.7),sharey=True)
    for ax,b in zip(axes.flat,order):
        d=pred[pred.battery_id.eq(b)].sort_values('Cycle_Index')
        if d.empty:raise ValueError('Battery not present: '+b)
        y=d['Calculated_SOH(%)']; p=d.Prediction_Raw;x=d.Cycle_Index
        rmse=np.sqrt(np.mean((y-p)**2))
        win=9 if b.startswith('B') else 15
        ax.scatter(x,y,s=4,alpha=.18,color='#555555')
        ax.scatter(x,p,s=4,alpha=.18,color='#008a91')
        ax.plot(x,y.rolling(win,center=True,min_periods=1).median(),color='#555555',lw=1.7)
        ax.plot(x,p.rolling(win,center=True,min_periods=1).median(),color='#008a91',lw=1.7)
        ax.set_title(f'{b} · RMSE {rmse:.2f}',fontsize=11)
        ax.set_ylim(68,101)
        ax.set_xlabel('Cycle index')
    axes[0,0].set_ylabel('SoH (%)');axes[1,0].set_ylabel('SoH (%)')
    fig.suptitle(('Post hoc protocol-screened' if screened else 'Full-discharge')+
                 ' HGB: held-out battery trajectories',fontsize=15)
    fig.text(.5,.01,'Faint dots: unsmoothed targets and predictions. Bold lines: rolling-median visual guides; all errors use raw values.',ha='center',fontsize=9)
    fig.tight_layout(rect=(0,.04,1,.96));fig.savefig(out,dpi=300,bbox_inches='tight');plt.close(fig)


def diagnostics(pred:pd.DataFrame,out:Path,screened:bool):
    pred=selected(pred)
    y=pred['Calculated_SOH(%)'].to_numpy(float)
    p=pred.Prediction_Raw.to_numpy(float)
    e=p-y
    fig,ax=plt.subplots(2,2,figsize=(12,9));ax=ax.ravel()
    ax[0].scatter(y,p,s=6,c='#377cae',alpha=.14)
    ax[0].plot([69,101],[69,101],'--',c='#777777')
    ax[0].set(xlabel='Measured SoH (%)',ylabel='Predicted SoH (%)',title='A | Predictions versus measurements')
    ax[1].scatter(y,e,s=6,c='#189348',alpha=.18);ax[1].axhline(0,ls='--',color='#777777')
    ax[1].set(xlabel='Measured SoH (%)',ylabel='Prediction error (pp)',title='B | Error versus measured SoH')
    for source,c in [('CALCE','#008a91'),('NASA','#2b5681')]:
        sub=pred.source_dataset.eq(source) if 'source_dataset' in pred else pred.battery_id.str.startswith('CS2_').eq(source=='CALCE')
        ax[2].hist(e[sub],bins=np.linspace(-15,12,65),density=True,histtype='step',linewidth=2,label=f'{source} (n={sum(sub)})',color=c)
    ax[2].legend(frameon=False);ax[2].set(xlabel='Prediction error (pp)',ylabel='Density',title='C | Residual distribution by source')
    batteries=[]
    for b,z in pred.groupby('battery_id'):
        err=z.Prediction_Raw-z['Calculated_SOH(%)']
        batteries.append((b,np.sqrt(np.mean(err*err))))
    batteries.sort(key=lambda x:x[1])
    ax[3].barh([x[0] for x in batteries],[x[1] for x in batteries],color=['#008a91' if b.startswith('CS2') else '#345a8b' for b,_ in batteries])
    ax[3].set(xlabel='Battery RMSE (pp)',title='D | Heterogeneity across held-out batteries')
    fig.suptitle(('Post hoc quality-screened' if screened else 'Full-discharge')+' HGB: errors and failure cases',fontsize=15)
    fig.tight_layout(rect=(0,0,1,.96));fig.savefig(out,dpi=300,bbox_inches='tight');plt.close(fig)


def comparison(metrics:pd.DataFrame,figure:int,out:Path):
    sub=metrics[metrics.Figure.eq(figure)].copy()
    if sub.empty:raise ValueError(f'No metric rows for figure {figure}')
    panels=sub.Panel.astype(str).unique().tolist()
    fig,axes=plt.subplots(1,len(panels),figsize=(5.5*len(panels),5),squeeze=False)
    for ax,panel in zip(axes.flat,panels):
        g=sub[sub.Panel.astype(str).eq(panel)]
        x=np.arange(len(g));ax.bar(x,g.RMSE,color=['#008a91','#37618b','#8062a4']*(len(g)//3+1))
        ax.set_xticks(x,g.Model,rotation=10,ha='center')
        for ix,value in enumerate(g.RMSE):ax.text(ix,value+.035,f'{value:.3f}',ha='center',fontsize=9)
        ax.set_ylim(0,max(g.RMSE)*1.17);ax.set_ylabel('Equal-battery mean RMSE (pp)')
        ax.set_title(panel)
    title={3:'Prediction performance at different information boundaries',
           4:'Within-source versus zero-shot transfer',
           5:'True 3.6 V: engineered HGB versus tuned sequence models'}[figure]
    fig.suptitle(title,fontsize=15)
    fig.tight_layout(rect=(0,0,1,.94));fig.savefig(out,dpi=300,bbox_inches='tight');plt.close(fig)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--original-predictions',type=Path)
    p.add_argument('--screened-predictions',type=Path)
    p.add_argument('--comparison-metrics',type=Path)
    p.add_argument('--output-dir',type=Path,default=Path('outputs/manuscript_figures'))
    a=p.parse_args();a.output_dir.mkdir(parents=True,exist_ok=True)
    apply_style()
    if a.original_predictions:
        d=pd.read_csv(a.original_predictions)
        trajectories(d,a.output_dir/PLOTS[1],False)
        diagnostics(d,a.output_dir/PLOTS[2],False)
    if a.screened_predictions:
        d=pd.read_csv(a.screened_predictions)
        trajectories(d,a.output_dir/PLOTS[6],True)
        diagnostics(d,a.output_dir/PLOTS[7],True)
    if a.comparison_metrics:
        m=pd.read_csv(a.comparison_metrics)
        if not {'Figure','Panel','Model','RMSE'}.issubset(m.columns):
            raise ValueError('Comparison input must provide Figure,Panel,Model,RMSE')
        for f in [3,4,5]:comparison(m,f,a.output_dir/PLOTS[f])
    if not any([a.original_predictions,a.screened_predictions,a.comparison_metrics]):
        p.error('Provide at least one source CSV (no fabricated plotted values)')
if __name__=='__main__':main()
