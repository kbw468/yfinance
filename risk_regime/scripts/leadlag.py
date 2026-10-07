import pandas as pd, numpy as np, warnings; warnings.filterwarnings('ignore')
pd.set_option('display.width',300); pd.set_option('display.max_rows',400); pd.set_option('display.max_columns',40)
S=pd.read_pickle('S.pkl'); P=pd.read_pickle('P.pkl'); G=pd.read_pickle('G.pkl'); L=pd.read_pickle('L.pkl')
OFF=pd.read_pickle('OFF.pkl'); ON=pd.read_pickle('ON.pkl'); idx=P.index
EX=(idx>=pd.Timestamp('2020-02-01'))&(idx<=pd.Timestamp('2020-07-31'))
d=np.log(S[['VIX','VIX9D','VIX3M','VIX6M','VXN','VVIX','MOVE','GVZ','OVX','SKEW','VIX1D']]).diff()
d['TNX']=S['TNX'].diff(); d['TYX']=S['TYX'].diff(); d['SPY']=np.log(P['SPY']).diff(); d['HYG']=np.log(P['HYG']).diff(); d['IWM']=np.log(P['IWM']).diff(); d['TLT']=np.log(P['TLT']).diff()
d=d[~EX]
print('=== LEAD/LAG: corr( dX_t , dSPY_{t+k} ). k>0 means X today predicts SPY k days later. k<0 means SPY leads X. ===')
K=range(-3,4)
rows={}
for c in d.columns:
    if c=='SPY': continue
    rows[c]=[d[c].corr(d['SPY'].shift(-k)) for k in K]
print(pd.DataFrame(rows,index=[f'k={k}' for k in K]).T.round(3).to_string())
print('\n=== LEAD/LAG vs VIX: corr( dX_t , dVIX_{t+k} ) ===')
rows={}
for c in d.columns:
    if c=='VIX': continue
    rows[c]=[d[c].corr(d['VIX'].shift(-k)) for k in K]
print(pd.DataFrame(rows,index=[f'k={k}' for k in K]).T.round(3).to_string())
# Conditional: in stress (VIX pct252>0.7) vs calm
for name,m in [('CALM (VIX pct<0.4)',G['VIX_pct252']<0.4),('STRESS (VIX pct>0.75)',G['VIX_pct252']>0.75)]:
    dm=d[m.reindex(d.index).fillna(False)]
    print(f'\n=== {name}: corr(dX_t, dSPY_t+1) and corr(dX_t, dVIX_t+1) ===')
    print(pd.DataFrame({'lead1_SPY':[dm[c].corr(dm['SPY'].shift(-1)) for c in d.columns if c!='SPY'],'lead1_VIX':[dm[c].corr(dm['VIX'].shift(-1)) for c in d.columns if c!='SPY'],'contemp_SPY':[dm[c].corr(dm['SPY']) for c in d.columns if c!='SPY']},index=[c for c in d.columns if c!='SPY']).round(3).to_string())
# Granger-style: does lagged dX add to AR(1) of dSPY? use t-stat from OLS with 5 lags
import statsmodels.api as sm
print('\n=== Granger-style t-stats of sum of lags 1..3 of dX in regressing dSPY on own lags + dX lags (full sample ex-2020 window) ===')
rows=[]
for c in d.columns:
    if c=='SPY': continue
    df=pd.DataFrame({'y':d['SPY']})
    for l in (1,2,3): df[f'y{l}']=d['SPY'].shift(l); df[f'x{l}']=d[c].shift(l)
    df=df.dropna()
    if len(df)<500: continue
    X=sm.add_constant(df.drop(columns='y')); res=sm.OLS(df['y'],X).fit(cov_type='HAC',cov_kwds={'maxlags':5})
    rows.append({'X':c,'n':len(df),'t_x1':res.tvalues['x1'],'t_x2':res.tvalues['x2'],'t_x3':res.tvalues['x3'],'F_pval':res.f_test('x1=0,x2=0,x3=0').pvalue})
print(pd.DataFrame(rows).set_index('X').round(3).to_string())
# ---- ORDERING AT EVENTS: first day in [-10,+10] where ROC3 z>1 (riskoff) ; for riskon: first day ROC3 z<-1 after trough-5
def ordering(anchors,cond,title):
    SER=['VIX','VIX9D','VIX3M','VIX6M','VXN','VVIX','MOVE','GVZ','OVX','SKEW','TNX','TYX']
    rows=[]
    for a in anchors:
        p=idx.get_loc(a); rec={'anchor':a.date()}
        for s in SER:
            key=f'{s}_roc3_z252' if s not in ('TNX','TYX') else f'{s}_chg3_z252'
            z=G[key].iloc[max(p-10,0):p+11]
            if z.isna().all(): rec[s]=np.nan; continue
            f=np.where(cond(z.values))[0]
            rec[s]=(f[0]-min(10,p)) if len(f) else np.nan
        rows.append(rec)
    T=pd.DataFrame(rows).set_index('anchor')
    print(f'\n=== {title}: first offset (days vs anchor) where condition met, per event ===')
    print(T.to_string())
    print('\nmedian first-fire offset (lower = earlier):'); print(T.median().sort_values().round(1).to_string())
    print('share of events firing at or before day 0:'); print((T<=0).sum().div(T.notna().sum()).sort_values(ascending=False).round(2).to_string())
ordering(OFF.peak.tolist(),lambda z: z>1.0,'RISK-OFF (anchor=peak): ROC3 z>1 (vol up / yields up)')
ordering(OFF.peak.tolist(),lambda z: z<-1.0,'RISK-OFF (anchor=peak): ROC3 z<-1 (vol DOWN / yields DOWN) -- compression tells')
ordering(ON.trough.tolist(),lambda z: z<-1.0,'RISK-ON (anchor=trough): ROC3 z<-1 (vol collapsing)')
