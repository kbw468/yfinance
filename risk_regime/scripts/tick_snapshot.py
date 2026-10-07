import pandas as pd, numpy as np, json, warnings; warnings.filterwarnings('ignore')
G=pd.read_pickle('G.pkl'); L=pd.read_pickle('L.pkl'); P=pd.read_pickle('P.pkl'); idx=P.index
RR=pd.read_pickle('RR.pkl'); MAT=pd.read_pickle('MAT.pkl'); BET=pd.read_pickle('BET.pkl'); OFF=pd.read_pickle('OFF.pkl'); ON=pd.read_pickle('ON.pkl')
EX=(idx>=pd.Timestamp('2020-02-01'))&(idx<=pd.Timestamp('2020-07-31')); ok=L['ok21']&~EX
TICK=[c for c in P.columns if c not in ('LQD','SHY','SPY')]
lp=np.log(P); rel=lp[TICK].sub(lp['SPY'],axis=0)
z=lambda k: G[k]
# ---- ticker-pair rules (ticker rel ROC5 z x index ROC z) ----
TR={
 'HYG_credit_crack': ((RR[('HYG',5)]<-1)&(z('VIX_roc5_z252')>1), 'HYG rel ROC5 z < -1 and VIX 5d ROC z > 1', 'Credit breaking relative while vol spikes.'),
 'KRE_banks_vs_yields': ((RR[('KRE',5)]<-1)&(z('TNX_chg5_z252')>1), 'KRE rel ROC5 z < -1 and TNX 5d chg z > 1', 'Regional banks breaking while yields rip.'),
 'TLT_duration_bid_calm': ((RR[('TLT',5)]>1)&(z('VIX_roc5_z252')<-1), 'TLT rel ROC5 z > 1 and VIX 5d ROC z < -1', 'Duration ripping relative while vol collapses. Fwd10 rel TLT -3.4%.'),
 'XLU_defensive_bid': ((RR[('XLU',5)]>1)&(z('TNX_chg5_z252')>1), 'XLU rel ROC5 z > 1 and TNX 5d chg z > 1', 'Utilities bid into rising yields.'),
 'XLRE_bid_yields_up': ((RR[('XLRE',5)]>1)&(z('TNX_chg5_z252')>1), 'XLRE rel ROC5 z > 1 and TNX 5d chg z > 1', 'REITs bid into rising yields.'),
 'IWM_beta_chase': ((RR[('IWM',5)]>1)&(z('VIX_roc5_z252')>1), 'IWM rel ROC5 z > 1 and VIX 5d ROC z > 1', 'Small caps leading while vol rises.'),
 'XLU_month_lead': (RR[('XLU',21)]>1, 'XLU rel ROC21 z > 1', 'Utilities outperforming for a month. Top-quintile P 0.27 vs 0.12 bottom.'),
}
stats={}; TRdf={}
for k,(m,d,n) in TR.items():
    m=m.fillna(False); f=m&~m.shift(1,fill_value=False)&ok
    stats[k]={'def':d,'note':n,'n':int(f.sum()),'P_off':round(float(L.loc[f,'riskoff21'].mean()),3) if f.sum() else None,'fwd21':round(float(L.loc[f,'fwd21'].mean()*100),2) if f.sum() else None,'fwdDD21':round(float(L.loc[f,'fwdDD21'].mean()*100),2) if f.sum() else None,'last':str(f[f].index[-1].date()) if f.sum() else 'never','live':bool(m.iloc[-1]),'recent':bool(m.iloc[-10:].any())}
    TRdf[k]=m
TRdf=pd.DataFrame(TRdf)
print(pd.DataFrame(stats).T[['n','P_off','fwd21','fwdDD21','last','live','recent']].to_string())
# ---- current rel ROC tape ----
tape=[]
for t in TICK:
    tape.append({'t':t,'rel5':round(float(rel[t].diff(5).iloc[-1]*100),2),'rel10':round(float(rel[t].diff(10).iloc[-1]*100),2),'rel21':round(float(rel[t].diff(21).iloc[-1]*100),2),
                 'z5':None if np.isnan(RR[(t,5)].iloc[-1]) else round(float(RR[(t,5)].iloc[-1]),2),'z10':None if np.isnan(RR[(t,10)].iloc[-1]) else round(float(RR[(t,10)].iloc[-1]),2),'z21':None if np.isnan(RR[(t,21)].iloc[-1]) else round(float(RR[(t,21)].iloc[-1]),2)})
# ---- lead signatures: median rel ROC5 z at day -3 (peaks) and +3 (troughs) ----
def sigval(anchors,off):
    out={}
    for t in TICK:
        s=RR[(t,5)]; v=[s.iloc[idx.get_loc(a)+off] for a in anchors if 0<=idx.get_loc(a)+off<len(s) and not np.isnan(s.iloc[idx.get_loc(a)+off])]
        out[t]=round(float(np.median(v)),2) if len(v)>=8 else None
    return out
lead={'peak_m3':sigval(OFF.peak.tolist(),-3),'peak_0':sigval(OFF.peak.tolist(),0),'trough_0':sigval(ON.trough.tolist(),0),'trough_p3':sigval(ON.trough.tolist(),3),'trough_p5':sigval(ON.trough.tolist(),5)}
# ---- matrix ----
cols=[c for c in MAT.columns]
mat={'rows':MAT.index.tolist(),'cols':[f'{a} {b}' for a,b in cols],'vals':[[None if np.isnan(v) else round(float(v),2) for v in MAT.loc[r]] for r in MAT.index]}
beta={'rows':TICK,'cols':list(BET['stress'].columns),'vals':[[None if np.isnan(v) else round(float(v),2) for v in BET['stress'].loc[t]] for t in TICK]}
# predictor table (21d): P_off top vs bottom quintile per ticker
pred={}
for t in TICK:
    x=RR[(t,21)]; m=x.notna()&ok
    if m.sum()<600: continue
    q=pd.qcut(x[m].rank(method='first'),5,labels=False); y=L.loc[m,'riskoff21']
    pred[t]={'bottomQ':round(float(y[q==0].mean()),3),'topQ':round(float(y[q==4].mean()),3)}
out={'tick_rules':stats,'tick_rules_hist':{k:[int(v) for v in TRdf[k].iloc[-504:]] for k in TRdf.columns},'tape':tape,'lead':lead,'mat':mat,'beta':beta,'pred':pred}
json.dump(out,open('tick_data.json','w')); import os; print('KB',os.path.getsize('tick_data.json')//1024)
