import pandas as pd, numpy as np, warnings; warnings.filterwarnings('ignore')
S=pd.read_pickle('S.pkl'); P=pd.read_pickle('P.pkl'); r=pd.read_pickle('R.pkl')

cols={}
def pct(x,w): return x.rolling(w,min_periods=int(w*0.8)).rank(pct=True)
def z(x,w): return (x-x.rolling(w).mean())/x.rolling(w).std()
VOL=['VIX','MOVE','VXN','VVIX','GVZ','OVX','VIX1D','VIX9D','VIX3M','VIX6M','SKEW']
RATES=['TNX','TYX','FVX','IRX']
# ratios as series
R={'TS_VIX_VIX3M':S['VIX']/S['VIX3M'],'TS_VIX9D_VIX':S['VIX9D']/S['VIX'],'TS_VIX3M_VIX6M':S['VIX3M']/S['VIX6M'],
   'TS_VIX_VIX6M':S['VIX']/S['VIX6M'],'TS_VIX1D_VIX9D':S['VIX1D']/S['VIX9D'],
   'X_MOVE_VIX':S['MOVE']/S['VIX'],'X_VXN_VIX':S['VXN']/S['VIX'],'X_VVIX_VIX':S['VVIX']/S['VIX'],
   'X_GVZ_VIX':S['GVZ']/S['VIX'],'X_OVX_VIX':S['OVX']/S['VIX']}
for c in VOL:
    x=S[c]; lx=np.log(x)
    for n in (1,2,3,5,10,21):
        roc=lx-lx.shift(n); cols[f'{c}_roc{n}']=roc
        cols[f'{c}_roc{n}_z252']=z(roc,252)                 # ROC normalized to own trailing 1y distribution
    cols[f'{c}_accel5']=cols[f'{c}_roc5']-cols[f'{c}_roc5'].shift(5)       # acceleration
    cols[f'{c}_roc5_pct252']=pct(cols[f'{c}_roc5'],252)
    cols[f'{c}_roc10_pct252']=pct(cols[f'{c}_roc10'],252)
    cols[f'{c}_roc21_pct252']=pct(cols[f'{c}_roc21'],252)
    cols[f'{c}_lvl']=x; cols[f'{c}_pct252']=pct(x,252)
    # ROC sign persistence: consecutive up days
    up=(lx.diff()>0).astype(int); cols[f'{c}_updays5']=up.rolling(5).sum()
    # ROC vs 20d low (spike from base)
    cols[f'{c}_vs20dlow']=lx-lx.rolling(20).min()
for c in RATES:
    x=S[c]
    for n in (1,2,3,5,10,21):
        cols[f'{c}_chg{n}']=x-x.shift(n); cols[f'{c}_chg{n}_z252']=z(x-x.shift(n),252)
        cols[f'{c}_roc{n}']=np.log(x.clip(lower=0.05))-np.log(x.clip(lower=0.05)).shift(n)
    cols[f'{c}_accel5']=cols[f'{c}_chg5']-cols[f'{c}_chg5'].shift(5)
    cols[f'{c}_lvl']=x; cols[f'{c}_pct252']=pct(x,252)
for k,v in R.items():
    cols[k]=v
    for n in (1,2,3,5,10,21):
        cols[f'{k}_roc{n}']=np.log(v)-np.log(v).shift(n); cols[f'{k}_roc{n}_z252']=z(cols[f'{k}_roc{n}'],252)
    cols[f'{k}_pct252']=pct(v,252)
# curve
cv={'curve_10y3m':S['TNX']-S['IRX'],'curve_30y10y':S['TYX']-S['TNX'],'curve_10y5y':S['TNX']-S['FVX']}
for k,v in cv.items():
    cols[k]=v
    for n in (1,5,10,21): cols[f'{k}_chg{n}']=v-v.shift(n); cols[f'{k}_chg{n}_z252']=z(v-v.shift(n),252)
# realized vol & VRP & their ROCs
rv10=r['SPY'].rolling(10).std()*np.sqrt(252)*100; rv21=r['SPY'].rolling(21).std()*np.sqrt(252)*100
cols['RV10']=rv10; cols['RV21']=rv21; cols['RV10_roc5']=np.log(rv10)-np.log(rv10).shift(5); cols['RV21_roc5']=np.log(rv21)-np.log(rv21).shift(5)
cols['VRP21']=S['VIX']-rv21; cols['VRP21_chg5']=cols['VRP21']-cols['VRP21'].shift(5)
cols['VRP10']=S['VIX']-rv10
# vol ROC relative to SPY ROC (vol rising faster than SPY move justifies)
for n in (1,5,10):
    cols[f'VIX_SPY_beta_dev{n}']=cols[f'VIX_roc{n}']+ 4.5*(np.log(P['SPY'])-np.log(P['SPY']).shift(n))  # typical VIX beta to SPY ~ -4.5 log/log
# cross-ROC spreads: who is moving faster
cols['dROC5_MOVE_VIX']=cols['MOVE_roc5']-cols['VIX_roc5']
cols['dROC5_VVIX_VIX']=cols['VVIX_roc5']-cols['VIX_roc5']
cols['dROC5_VXN_VIX']=cols['VXN_roc5']-cols['VIX_roc5']
cols['dROC5_VIX9D_VIX']=cols['VIX9D_roc5']-cols['VIX_roc5']
cols['dROC5_VIX_VIX3M']=cols['VIX_roc5']-cols['VIX3M_roc5']
cols['dROC5_GVZ_VIX']=cols['GVZ_roc5']-cols['VIX_roc5']
cols['dROC5_OVX_VIX']=cols['OVX_roc5']-cols['VIX_roc5']
# SPY context
cols['SPY_ret5']=np.log(P['SPY']).diff(5); cols['SPY_ret21']=np.log(P['SPY']).diff(21); cols['SPY_dd63']=np.log(P['SPY']/P['SPY'].rolling(63).max())
G=pd.DataFrame(cols)
G.to_pickle('G.pkl')
print(G.shape); print(len([c for c in G if 'roc' in c or 'chg' in c or 'accel' in c or 'dROC' in c]),'ROC-type features')
