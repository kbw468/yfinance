import pandas as pd, numpy as np, warnings; warnings.filterwarnings('ignore')
Sraw=pd.read_pickle('S_raw.pkl').drop(columns=['VIX1Y'],errors='ignore'); P0=pd.read_pickle('P.pkl').drop(columns=['VIX1Y'],errors='ignore')
idx=P0['SPY'].dropna().index   # SPY trading days only
S=Sraw.reindex(idx).ffill(limit=3); P=P0.reindex(idx)
S.to_pickle('S.pkl'); P.to_pickle('P.pkl'); np.log(P).diff().to_pickle('R.pkl')
spy=P['SPY']; EX0,EX1=pd.Timestamp('2020-02-01'),pd.Timestamp('2020-07-31')
L=pd.DataFrame(index=idx)
for h in (1,2,3,5,10,21,42,63):
    L[f'fwd{h}']=np.log(spy.shift(-h)/spy)
    L[f'fwdDD{h}']=np.log(spy[::-1].rolling(h).min()[::-1].shift(-1)/spy)
    L[f'fwdUP{h}']=np.log(spy[::-1].rolling(h).max()[::-1].shift(-1)/spy)
    tf=pd.Series(idx,index=idx).shift(-h)
    bad=((idx>=EX0)&(idx<=EX1))|((tf>=EX0)&(tf<=EX1)).values
    L[f'ok{h}']=~bad & L[f'fwd{h}'].notna()
L['riskoff21']=(L['fwdDD21']<=-0.05).astype(float); L['riskon21']=(L['fwdUP21']>=0.05).astype(float)
L.to_pickle('L.pkl')
print('rows',len(idx),'nan check fwd5 within ok21:',L.loc[L.ok21,'fwd5'].isna().sum(), L.loc[L.ok21,'fwdDD21'].isna().sum())
