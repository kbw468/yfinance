import pandas as pd, numpy as np, json, warnings; warnings.filterwarnings('ignore')
pd.set_option('display.width',320); pd.set_option('display.max_rows',200)
G=pd.read_pickle('G.pkl'); S=pd.read_pickle('S.pkl'); P=pd.read_pickle('P.pkl'); RU=pd.read_pickle('RU.pkl'); idx=P.index
z=lambda k: G[k]
# dimension tags daily (only meaningful when VIX roc5z>1)
DIM=pd.DataFrame({'rateled':(z('MOVE_roc5_z252')>1)&(z('TNX_chg5_z252')>0.5),'ftq':z('TNX_chg5_z252')<-1,'oil':z('OVX_roc5_z252')>1.5,'goldvol':z('GVZ_roc5_z252')>1,'equityonly':(z('MOVE_roc5_z252')<0)&(z('TNX_chg5_z252').abs()<0.5)&(z('OVX_roc5_z252')<0.5)}).fillna(False)
last=idx[-1]; print('as of',last.date())
SER=['VIX','VIX1D','VIX9D','VIX3M','VIX6M','VXN','VVIX','MOVE','GVZ','OVX','SKEW','TNX','TYX']
rows=[]
for s in SER:
    lv=S[s].iloc[-1]; pre='chg' if s in ('TNX','TYX') else 'roc'
    rows.append({'series':s,'level':round(lv,2),'pct252':round(G[f'{s}_pct252'].iloc[-1],2),
        'roc1':round(G[f'{s}_{pre}1'].iloc[-1]*(100 if pre=='roc' else 100),2),'roc3':round(G[f'{s}_{pre}3'].iloc[-1]*100,2),'roc5':round(G[f'{s}_{pre}5'].iloc[-1]*100,2),'roc10':round(G[f'{s}_{pre}10'].iloc[-1]*100,2),'roc21':round(G[f'{s}_{pre}21'].iloc[-1]*100,2),
        'z3':round(G[f'{s}_{pre}3_z252'].iloc[-1],2),'z5':round(G[f'{s}_{pre}5_z252'].iloc[-1],2),'z10':round(G[f'{s}_{pre}10_z252'].iloc[-1],2),'z21':round(G[f'{s}_{pre}21_z252'].iloc[-1],2)})
T=pd.DataFrame(rows).set_index('series'); print(T.to_string())
ratios={'VIX/VIX3M':'TS_VIX_VIX3M','VIX9D/VIX':'TS_VIX9D_VIX','VIX1D/VIX9D':'TS_VIX1D_VIX9D','VVIX/VIX':'X_VVIX_VIX','MOVE/VIX':'X_MOVE_VIX','GVZ/VIX':'X_GVZ_VIX','OVX/VIX':'X_OVX_VIX'}
rr=[]
for n,k in ratios.items(): rr.append({'ratio':n,'level':round(G[k].iloc[-1],3),'pct252':round(G[f'{k}_pct252'].iloc[-1],2),'roc5z':round(G[f'{k}_roc5_z252'].iloc[-1],2),'roc3z':round(G[f'{k}_roc3_z252'].iloc[-1],2)})
RT=pd.DataFrame(rr).set_index('ratio'); print(RT.to_string())
print('\nSPY dd63:',round(G['SPY_dd63'].iloc[-1]*100,2),'% ret5:',round(G['SPY_ret5'].iloc[-1]*100,2),'% ret21:',round(G['SPY_ret21'].iloc[-1]*100,2),'%')
print('\nrules live today:',[k for k in RU.columns if RU[k].iloc[-1]])
print('rules fired in last 10 days:'); print(RU.iloc[-10:].astype(int).T.to_string())
print('dimension tags last 5 days:'); print(DIM.iloc[-5:].astype(int).T.to_string())
# ---- export dashboard JSON: last 504 days of SPY, VIX, z-scores for key series, rule fires; plus full-history rule fire dates for context
N=504
keys={'VIX':'VIX_roc5_z252','VVIX/VIX':'X_VVIX_VIX_roc5_z252','MOVE':'MOVE_roc5_z252','TNX':'TNX_chg5_z252','GVZ':'GVZ_roc5_z252','OVX':'OVX_roc5_z252','VIX/VIX3M':'TS_VIX_VIX3M_roc3_z252','VIX9D/VIX':'TS_VIX9D_VIX_roc5_z252','SKEW':'SKEW_roc10_z252','VXN':'VXN_roc5_z252','VIX1D':'VIX1D_roc3_z252','VIX21':'VIX_roc21_z252'}
sub=G.iloc[-N:]
out={'asof':str(last.date()),'dates':[d.strftime('%Y-%m-%d') for d in sub.index],
     'spy':[round(float(x),2) for x in P['SPY'].iloc[-N:]],'vix':[round(float(x),2) for x in S['VIX'].iloc[-N:]],
     'z':{k:[None if np.isnan(v) else round(float(v),2) for v in sub[c]] for k,c in keys.items()},
     'rules':{k:[int(v) for v in RU[k].iloc[-N:]] for k in RU.columns},
     'dims':{k:[int(v) for v in DIM[k].iloc[-N:]] for k in DIM.columns},
     'levels':T.reset_index().to_dict(orient='records'),'ratios':RT.reset_index().to_dict(orient='records'),
     'spy_dd63':round(float(G['SPY_dd63'].iloc[-1])*100,2),'spy_ret5':round(float(G['SPY_ret5'].iloc[-1])*100,2),'spy_ret21':round(float(G['SPY_ret21'].iloc[-1])*100,2)}
# rule fire history (first-fire dates) for full sample, for the "history" strip
first={k:[d.strftime('%Y-%m-%d') for d in RU.index[(RU[k]&~RU[k].shift(1,fill_value=False))]] for k in RU.columns}
out['rule_history']=first
# freshness: index series before the 3-day forward-fill, and tickers (never filled); anything ending before the as-of session is stale
Sraw=pd.read_pickle('S_raw.pkl').drop(columns=['VIX1Y'],errors='ignore')
stale=[{'series':c,'last':str(Sraw[c].dropna().index[-1].date())} for c in Sraw.columns if Sraw[c].dropna().index[-1]<last]
stale+=[{'series':c,'last':str(P[c].dropna().index[-1].date())} for c in P.columns if P[c].dropna().index[-1]<last]
out['stale']=stale; print('stale series:',stale if stale else 'none')
# playbook tables
PB=pd.read_pickle('PLAYBOOK.pkl')
pb={}
for k in PB.columns.levels[0]:
    d=PB[k].sort_values('rel10'); pb[k]={'tickers':d.index.tolist(),'rel10':[None if np.isnan(x) else round(float(x),2) for x in d['rel10']],'hit10':[None if np.isnan(x) else round(float(x),2) for x in d['hit10']],'rel21':[None if np.isnan(x) else round(float(x),2) for x in d['rel21']],'n':[int(x) for x in d['n']]}
out['playbook']=pb
json.dump(out,open('dash_data.json','w'))
import os; print('json KB',os.path.getsize('dash_data.json')//1024)
