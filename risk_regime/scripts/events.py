import pandas as pd, numpy as np, warnings; warnings.filterwarnings('ignore')
pd.set_option('display.width',260); pd.set_option('display.max_rows',400); pd.set_option('display.max_columns',40)
P=pd.read_pickle('P.pkl'); S=pd.read_pickle('S.pkl'); G=pd.read_pickle('G.pkl'); L=pd.read_pickle('L.pkl')
spy=P['SPY']; lspy=np.log(spy); idx=spy.index
EX0,EX1=pd.Timestamp('2020-02-01'),pd.Timestamp('2020-07-31')
def in_ex(d): return (d>=EX0)&(d<=EX1)
# ---- RISK-OFF ONSETS: SPY peak followed by >=7% drop within 42d, peak is a 63d high; dedupe 42d
offs=[]; ons=[]
hi63=spy.rolling(63).max()
fwdmin42=spy[::-1].rolling(42).min()[::-1].shift(-1)
cand=(spy>=hi63)&(np.log(fwdmin42/spy)<=-0.07)
last=None
for d in idx[cand.fillna(False)]:
    if last is not None and (d-last).days<60: continue
    # refine: take the actual highest close in [d, d+10] that is still followed by -7%
    w=spy.loc[d:].iloc[:11]; pk=w.idxmax()
    if np.log(fwdmin42.loc[pk]/spy.loc[pk])>-0.07: pk=d
    if in_ex(pk): continue
    trough=spy.loc[pk:].iloc[1:43].idxmin(); depth=np.log(spy.loc[trough]/spy.loc[pk])
    # first close <=-2% and <=-5% from peak
    sub=np.log(spy.loc[pk:].iloc[:43]/spy.loc[pk]); b2=sub[sub<=-0.02].index[0]; b5=sub[sub<=-0.05].index[0]
    offs.append({'peak':pk,'trough':trough,'depth':depth,'days_to_trough':idx.get_loc(trough)-idx.get_loc(pk),'break2pct':b2,'d_to_2pct':idx.get_loc(b2)-idx.get_loc(pk),'d_to_5pct':idx.get_loc(b5)-idx.get_loc(pk)})
    last=pk
OFF=pd.DataFrame(offs)
# ---- RISK-ON ONSETS: trough after >=7% DD (from 63d high) that is followed by +7% in 42d with no lower low in 42d
lo=spy.rolling(63).max(); dd=np.log(spy/lo)
fwdmax42=spy[::-1].rolling(42).max()[::-1].shift(-1)
cand=(dd<=-0.07)&(np.log(fwdmax42/spy)>=0.07)&(spy<=fwdmin42)
last=None
for d in idx[cand.fillna(False)]:
    if last is not None and (d-last).days<60: continue
    if in_ex(d): continue
    rec=spy.loc[d:].iloc[1:43]; rebound=np.log(rec.max()/spy.loc[d])
    ons.append({'trough':d,'dd_at_trough':dd.loc[d],'rebound42':rebound,'days_to_peak42':idx.get_loc(rec.idxmax())-idx.get_loc(d)})
    last=d
ON=pd.DataFrame(ons)
OFF.to_pickle('OFF.pkl'); ON.to_pickle('ON.pkl')
print(f'RISK-OFF onsets: {len(OFF)}  (peak -> >=7% within 42d). median depth {OFF.depth.median():.3f}, median days to -2% {OFF.d_to_2pct.median()}, to -5% {OFF.d_to_5pct.median()}')
print(OFF.assign(peak=OFF.peak.dt.date,trough=OFF.trough.dt.date,break2pct=OFF.break2pct.dt.date).round(3).to_string())
print(f'\nRISK-ON onsets: {len(ON)} (trough after >=7% DD, +7% in 42d)')
print(ON.assign(trough=ON.trough.dt.date).round(3).to_string())
