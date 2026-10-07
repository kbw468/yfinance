import pandas as pd, numpy as np, glob, os
D='data'
def load(s): 
    d=pd.read_csv(f'{D}/{s}.csv',index_col=0,parse_dates=True); return d
sig={}
for s in ['VIX','MOVE','VXN','VVIX','TNX','TYX','GVZ','OVX','VIX1D','VIX9D','VIX3M','VIX6M','SKEW','FVX','IRX']:
    sig[s]=load(s)['Close']
S=pd.DataFrame(sig)
S.to_pickle('S_raw.pkl')
for c in S:
    x=S[c].dropna()
    print(f'{c:6s} n={len(x):5d} first={x.index[0].date()} zeros={(x==0).sum():3d} min={x.min():8.2f} p1={x.quantile(.01):8.2f} med={x.median():8.2f} p99={x.quantile(.99):8.2f} max={x.max():8.2f} maxabs_dlog={np.abs(np.log(x[x>0]).diff()).max():.2f}')
# gaps
print()
for c in S:
    x=S[c].dropna(); gaps=x.index.to_series().diff().dt.days; big=gaps[gaps>7]
    if len(big): print(c,'gaps>7d:',[(str(i.date()),int(g)) for i,g in big.items()][:8])
# tickers
T={}
for f in sorted(glob.glob(f'{D}/*.csv')):
    n=os.path.basename(f)[:-4]
    if n in sig or n in ('RVX',): continue
    T[n]=load(n)['Close']
P=pd.DataFrame(T); P.to_pickle('P.pkl')
print(P.shape, P.columns.tolist())
