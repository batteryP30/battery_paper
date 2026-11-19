# Selected functions are the original Step 11A CALCE target-reconstruction functions,
# reproduced verbatim from Reviewer_Step11A_Target_Audit_Colab.ipynb.
SOH_MIN, SOH_MAX = 70.0, 100.0
REF_I = 0.22
REF_I_TOL = 0.04
DISCHARGE_I_THRESHOLD = 0.02
MIN_BLOCK_SAMPLES = 5
CYCLE, BATTERY, SOURCE, TARGET = 'Cycle_Index', 'battery_id', 'source_dataset', 'Calculated_SOH(%)'

def norm(s):
    return ''.join(ch.lower() for ch in str(s) if ch.isalnum())

def find_col(df, names, required=True):
    lookup = {norm(c): c for c in df.columns}
    for name in names:
        if norm(name) in lookup:
            return lookup[norm(name)]
    if required:
        raise KeyError(f'None of these columns found: {names}')
    return None

def natural_key(v):
    return [int(x) if x.isdigit() else x.lower() for x in re.split(r'(\d+)', str(v))]

def parse_calce_date(path):
    parts = Path(path).stem.split('_')
    for start in range(len(parts)-3, -1, -1):
        m,d,y = parts[start:start+3]
        if not (m.isdigit() and d.isdigit() and y.isdigit()) or len(y) not in (2,4):
            continue
        yy = int(y) + (2000 if len(y)==2 else 0)
        try:
            return pd.Timestamp(year=yy, month=int(m), day=int(d))
        except ValueError:
            pass
    return None

def calce_files(folder):
    folder = Path(folder)
    fs = [p for p in folder.rglob('*.xlsx') if not p.name.startswith('~$')]
    if not fs:
        raise FileNotFoundError(f'No xlsx files in {folder}')
    if len(fs)>1:
        bad = [p for p in fs if parse_calce_date(p) is None]
        if bad:
            raise ValueError(f'Cannot parse CALCE workbook date(s): {[p.name for p in bad]}')
    fs.sort(key=lambda p: (parse_calce_date(p) is None, parse_calce_date(p) or pd.Timestamp.max, str(p).lower()))
    return fs

def read_calce(folder, sheets):
    frames, last = [], 0.0
    for path in calce_files(folder):
        with pd.ExcelFile(path) as xf:
            for sheet in sheets:
                if sheet not in xf.sheet_names:
                    continue
                d = pd.read_excel(xf, sheet_name=sheet)
                if CYCLE not in d.columns:
                    raise KeyError(f'{path.name}/{sheet}: missing {CYCLE}')
                d[CYCLE] = pd.to_numeric(d[CYCLE], errors='coerce')
                d = d[d[CYCLE].notna()].copy()
                if d.empty:
                    continue
                d[CYCLE] = d[CYCLE] + last
                last = float(d[CYCLE].max())
                frames.append(d)
    if not frames:
        raise RuntimeError(f'No usable CALCE sheets in {folder}')
    return pd.concat(frames, ignore_index=True)

def time_qc(values):
    t = pd.to_numeric(pd.Series(values), errors='coerce').to_numpy(float)
    t = np.sort(t[np.isfinite(t)])
    dup = int(pd.Series(t).duplicated().sum()) if len(t) else 0
    if len(t)<2:
        return dict(Finite_Time_Samples=len(t), Duplicate_Timestamps=dup,
                    Positive_DT_Intervals=0, Nonpositive_DT_Intervals=0,
                    Median_Positive_DT_s=np.nan, P95_Positive_DT_s=np.nan, Max_Positive_DT_s=np.nan)
    dt = np.diff(t); dt = dt[np.isfinite(dt)]; pos = dt[dt>0]
    return dict(Finite_Time_Samples=len(t), Duplicate_Timestamps=dup,
                Positive_DT_Intervals=int((dt>0).sum()), Nonpositive_DT_Intervals=int((dt<=0).sum()),
                Median_Positive_DT_s=float(np.median(pos)) if len(pos) else np.nan,
                P95_Positive_DT_s=float(np.quantile(pos,.95)) if len(pos) else np.nan,
                Max_Positive_DT_s=float(np.max(pos)) if len(pos) else np.nan)

def constant_targets(raw,b,src):
    d=raw[[CYCLE,'Test_Time(s)','Current(A)']].copy()
    for c in [CYCLE,'Test_Time(s)','Current(A)']: d[c]=pd.to_numeric(d[c],errors='coerce')
    rows=[]
    for cyc,g0 in d.groupby(CYCLE,sort=True,dropna=True):
        ns=len(g0); mt=g0['Test_Time(s)'].isna().sum(); mi=g0['Current(A)'].isna().sum()
        g=g0.dropna(subset=['Test_Time(s)','Current(A)']).sort_values('Test_Time(s)')
        t=g['Test_Time(s)'].to_numpy(float); i=g['Current(A)'].to_numpy(float)
        finite=np.isfinite(t)&np.isfinite(i); t=t[finite]; i=i[finite]; diag=time_qc(t)
        if len(t)<2:
            rows.append({BATTERY:b,SOURCE:src,'Protocol':'CONSTANT',CYCLE:float(cyc),
                         'N_Source_Rows':ns,'Missing_Time_Rows':mt,'Missing_Current_Rows':mi,
                         'N_Finite_Time_Current':len(t),'Duration_s':np.nan,
                         'Measured_Discharge_Capacity_Ah':np.nan,'Raw_QC_Pass':False,
                         'Raw_QC_Reason':'fewer_than_2_finite_samples',**diag}); continue
        order=np.argsort(t); t=t[order]; i=i[order]
        dt=np.diff(t,prepend=t[0]); dt=np.where(np.isfinite(dt)&(dt>0),dt,0.0)
        q=float(np.sum(np.maximum(-i,0.0)*dt)/3600.0); dur=float(t.max()-t.min())
        qc=bool(np.isfinite(q) and q>0 and np.isfinite(dur) and dur>0)
        rows.append({BATTERY:b,SOURCE:src,'Protocol':'CONSTANT',CYCLE:float(cyc),
                     'N_Source_Rows':ns,'Missing_Time_Rows':mt,'Missing_Current_Rows':mi,
                     'N_Finite_Time_Current':len(t),'Duration_s':dur,
                     'Measured_Discharge_Capacity_Ah':q,'Raw_QC_Pass':qc,
                     'Raw_QC_Reason':'' if qc else 'nonpositive_or_nonfinite_capacity_or_duration',**diag})
    a=pd.DataFrame(rows).sort_values(CYCLE).reset_index(drop=True)
    valid=a['Raw_QC_Pass']; qref=float(a.loc[valid,'Measured_Discharge_Capacity_Ah'].iloc[0])
    a['Reference_Capacity_Ah']=qref
    a[TARGET]=a['Measured_Discharge_Capacity_Ah']/qref*100.0
    a['Below_70']=valid&(a[TARGET]<SOH_MIN)
    a['In_Primary_70_100_Scope']=valid&a[TARGET].between(SOH_MIN,SOH_MAX,inclusive='both')
    a['Above_100']=valid&(a[TARGET]>SOH_MAX)
    a['Target_Construction']='RAW_DISCHARGE_CURRENT_TIME_INTEGRATION'
    a['Integration_Method']='Step5 validated rectangular/right-sample sum: sum(max(-I,0)*positive_dt)/3600'
    a['Current_Sign_Convention']='NASA discharge-only current -> -abs(source current)' if src=='NASA' else 'CALCE negative current = discharge'
    a['Gap_Treatment']='All finite positive dt retained; no gap-length exclusion; nonpositive dt contributes zero'
    return a

def trapz_block(block):
    b=block[['Test_Time(s)','Current(A)']].copy()
    b['Test_Time(s)']=pd.to_numeric(b['Test_Time(s)'],errors='coerce'); b['Current(A)']=pd.to_numeric(b['Current(A)'],errors='coerce')
    b=b.dropna().sort_values('Test_Time(s)')
    if len(b)<2: return np.nan,np.nan
    t=b['Test_Time(s)'].to_numpy(float); i=np.abs(b['Current(A)'].to_numpy(float))
    dt=np.diff(t); imid=.5*(i[:-1]+i[1:]); ok=np.isfinite(dt)&np.isfinite(imid)&(dt>0)
    if not ok.any(): return np.nan,np.nan
    return float(np.sum(imid[ok]*dt[ok])/3600), float(np.sum(dt[ok]))

def calce_variable_targets(raw,b):
    d=raw[[CYCLE,'Test_Time(s)','Current(A)']].copy(); d['_row']=np.arange(len(d))
    for c in [CYCLE,'Test_Time(s)','Current(A)']: d[c]=pd.to_numeric(d[c],errors='coerce')
    rows=[]
    for cyc,g0 in d.groupby(CYCLE,sort=True,dropna=True):
        ns=len(g0); mt=g0['Test_Time(s)'].isna().sum(); mi=g0['Current(A)'].isna().sum()
        g=g0.dropna(subset=['Test_Time(s)','Current(A)']).sort_values(['Test_Time(s)','_row']).copy()
        g['_dis']=g['Current(A)'] < -abs(DISCHARGE_I_THRESHOLD)
        g['_start']=g['_dis'] & ~g['_dis'].shift(fill_value=False); g['_block']=g['_start'].cumsum()
        candidates=[]
        for bid,blk in g.loc[g['_dis']].groupby('_block',sort=True):
            if len(blk)<MIN_BLOCK_SAMPLES: continue
            med=float(np.nanmedian(np.abs(blk['Current(A)'].to_numpy(float))))
            q,dur=trapz_block(blk); diag=time_qc(blk['Test_Time(s)'])
            candidates.append(dict(block=int(bid),samples=len(blk),median=med,error=abs(med-REF_I) if np.isfinite(med) else np.nan,capacity=q,duration=dur,**diag))
        valid=[x for x in candidates if np.isfinite(x['capacity']) and x['capacity']>0 and np.isfinite(x['median'])]
        chosen=min(valid,key=lambda x:x['error']) if valid else None
        if chosen is None:
            status='no_valid_discharge_block'; chosen=dict(block=np.nan,samples=np.nan,median=np.nan,error=np.nan,capacity=np.nan,duration=np.nan,**time_qc([]))
        elif chosen['error']>REF_I_TOL:
            status='closest_block_outside_current_tolerance'; chosen=dict(chosen); chosen['capacity']=np.nan
        else: status='valid_0.22A_reference'
        rows.append({BATTERY:b,SOURCE:'CALCE','Protocol':'VARIABLE',CYCLE:float(cyc),
                     'N_Source_Rows':ns,'Missing_Time_Rows':mt,'Missing_Current_Rows':mi,
                     'Candidate_Discharge_Blocks':len(candidates),'Selected_Block':chosen['block'],
                     'Selected_Samples':chosen['samples'],'Reference_Target_Current_A':REF_I,
                     'Reference_Median_Current_A':chosen['median'],'Reference_Current_Error_A':chosen['error'],
                     'Selected_Duration_s':chosen['duration'],'Reference_Capacity_Ah':chosen['capacity'],
                     'Selection_Status':status,'Raw_QC_Pass':status=='valid_0.22A_reference',
                     **{k:chosen[k] for k in time_qc([]).keys()}})
    a=pd.DataFrame(rows).sort_values(CYCLE).reset_index(drop=True)
    valid=a['Raw_QC_Pass']&np.isfinite(a['Reference_Capacity_Ah']); first=a.index[valid][0]; qref=float(a.loc[first,'Reference_Capacity_Ah'])
    a['Baseline_Reference_Capacity_Ah']=qref; a[TARGET]=np.nan
    a.loc[valid,TARGET]=a.loc[valid,'Reference_Capacity_Ah']/qref*100.0; a.loc[first,TARGET]=100.0
    a['Below_70']=valid&(a[TARGET]<SOH_MIN); a['In_Primary_70_100_Scope']=valid&a[TARGET].between(SOH_MIN,SOH_MAX,inclusive='both'); a['Above_100']=valid&(a[TARGET]>SOH_MAX)
    a['Eligible_Independent_Observation']=valid; a['Primary_Independent_Observation']=valid&a['In_Primary_70_100_Scope']
    a['Target_Construction']='RAW_0p22A_REFERENCE_CURRENT_INTEGRATION_NO_TARGET_CLEANING'
    a['Integration_Method']='Trapezoidal abs(current) integration over finite positive dt in selected 0.22 A block'
    a['Current_Sign_Convention']='CALCE negative current identifies discharge; capacity uses abs(current)'
    a['Gap_Treatment']='All finite positive dt retained; no gap-length exclusion; nonpositive dt excluded'
    return a
