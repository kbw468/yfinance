import pandas as pd, numpy as np, warnings; warnings.filterwarnings('ignore')
pd.set_option('display.width',300); pd.set_option('display.max_rows',400); pd.set_option('display.max_columns',40)
P=pd.read_pickle('P.pkl'); S=pd.read_pickle('S.pkl'); G=pd.read_pickle('G.pkl'); L=pd.read_pickle('L.pkl')
OFF=pd.read_pickle('OFF.pkl'); ON=pd.read_pickle('ON.pkl'); idx=P.index
EX0,EX1=pd.Timestamp('2020-02-01'),pd.Timestamp('2020-07-31')
OFF=OFF[~((OFF.trough>=EX0)&(OFF.trough<=EX1))].reset_index(drop=True)   # drop Jan-2020 event: its window is inside exclusion
OFF.to_pickle('OFF.pkl')
SER=['VIX','VIX9D','VIX3M','VIX6M','VXN','VVIX','MOVE','GVZ','OVX','SKEW','TNX','TYX','TS_VIX_VIX3M','TS_VIX9D_VIX','X_MOVE_VIX','X_VVIX_VIX']
def rocz(name,n):
    key=f'{name}_roc{n}_z252' if name not in ('TNX','TYX') else f'{name}_chg{n}_z252'
    return G[key]
OFFS=list(range(-10,11))
def signature(anchors,series,n,thr=1.0):
    """median z-ROC at each offset; cumulative fired-by-k fraction; baseline fired-by-k from all non-event days."""
    z=rocz(series,n); pos=[idx.get_loc(a) for a in anchors]
    med=[]; fired=[]
    M=np.array([[z.iloc[p+k] if 0<=p+k<len(z) else np.nan for k in OFFS] for p in pos])
    med=np.nanmedian(M,axis=0)
    fired=np.nanmean(np.maximum.accumulate(np.where(np.isnan(M),-9,M),axis=1)>thr,axis=0)
    # baseline: random anchor days (every 5th day outside exclusion), same cumulative fired-by-k over a window of same length
    zz=z.values; base=[]
    for k_i,k in enumerate(OFFS):
        w=k-OFFS[0]+1  # window length from -10 to k
        roll=pd.Series(zz).rolling(w).max().shift(-(k_i))  # max over [t-10 .. t+k] aligned at t
        base.append((roll>thr).mean())
    return med,fired,np.array(base)
def table(anchors,title,n):
    rows_med={}; rows_f={}; rows_b={}
    for s in SER:
        z=rocz(s,n)
        if z.notna().sum()<500: continue
        ok=[a for a in anchors if not np.isnan(z.loc[a]) ]
        if len(ok)<8: continue
        med,f,b=signature(ok,s,n)
        rows_med[f'{s} (n={len(ok)})']=med; rows_f[s]=f; rows_b[s]=b
    print(f'\n==== {title} | ROC{n} z-score (vs own trailing 252d) MEDIAN across events, offsets -10..+10 ====')
    print(pd.DataFrame(rows_med,index=OFFS).T.round(2).to_string())
    print(f'\n---- fraction of events where ROC{n} z>1.0 had fired by offset k (cumulative from -10), minus BASELINE (same stat on all days) ----')
    D=pd.DataFrame(rows_f,index=OFFS).T; B=pd.DataFrame(rows_b,index=OFFS).T
    out=(D-B).round(2); out.insert(0,'base@+0',B[0].round(2)); print(out.to_string())
for n in (1,3,5,10):
    table(OFF.peak.tolist(),f'RISK-OFF anchored at SPY PEAK ({len(OFF)} events)',n)
for n in (3,5):
    table(OFF.break2pct.tolist(),f'RISK-OFF anchored at -2% BREAK day ({len(OFF)} events)',n)
for n in (1,3,5,10):
    table(ON.trough.tolist(),f'RISK-ON anchored at SPY TROUGH ({len(ON)} events)',n)
