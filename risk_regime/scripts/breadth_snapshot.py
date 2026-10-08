# Export today's breadth readouts (equal-weight / cap-weight ratio ROC) for the dashboard. Context layer; nothing here feeds the dial.
import pandas as pd, numpy as np, json, warnings; warnings.filterwarnings('ignore')
G=pd.read_pickle('G.pkl'); L=pd.read_pickle('L.pkl'); P=pd.read_pickle('P.pkl'); idx=P.index
B=pd.read_pickle('BREADTH.pkl'); F=B['F']; BR=B['BR']; RAT=B['RAT']
EX=(idx>=pd.Timestamp('2020-02-01'))&(idx<=pd.Timestamp('2020-07-31')); ok=L['ok21']&~EX
zz=lambda x:(x-x.rolling(252).mean())/x.rolling(252).std()
r63={k:zz(RAT[k].diff(63)) for k in RAT.columns}
tnx=G['TNX_chg21_z252']; spyz=G['SPY_ret21']/G['SPY_ret21'].rolling(252).std(); vix21=G['VIX_roc21_z252']
CELLS={
 'RSP_bleed_month_yields_up':   ((F[('RSP/SPY','roc21z')]<-1)&(tnx>1),'RSP/SPY 21d ratio ROC z < -1 and TNX 21d chg z > 1','Equal weight losing a month while yields rip. In the dial at -8 for 15 sessions it adds 0.3%/yr and three OUT spells with no drawdown gain. Not adopted.'),
 'RSP_bleed_quarter_yields_up': ((r63['RSP/SPY']<-1)&(tnx>1),'RSP/SPY 63d ratio ROC z < -1 and TNX 21d chg z > 1','A quarter of narrowing into a yield rip. The strongest breadth cell; neutral when added to the dial. Not adopted.'),
 'QQQE_bleed_quarter_yields_up':((r63['QQQE/QQQ']<-1)&(tnx>1),'QQQE/QQQ 63d ratio ROC z < -1 and TNX 21d chg z > 1','Nasdaq concentration tightening into a yield rip. Data from 2012.'),
 'EW_rotation_inside_decline':  ((F[('RSP/SPY','roc21z')]>0)&(spyz<-1),'RSP/SPY 21d ratio ROC z > 0 and SPY 21d return z < -1','Equal weight outperforming while the index falls: defensive rotation, not a low. Worse when added to the dial.'),
 'RSP_broadening_vol_compressing':((r63['RSP/SPY']>1)&(vix21<-1),'RSP/SPY 63d ratio ROC z > 1 and VIX 21d ROC z < -1','Broadening while vol compresses over the month: the safest breadth cell.'),
}
rnd=lambda v: None if (v is None or (isinstance(v,float) and np.isnan(v))) else round(float(v),2)
stats={}; H={}
for k,(m,d,n) in CELLS.items():
    m=m.fillna(False); f=m&~m.shift(1,fill_value=False)&ok; a=m&ok; a18=a&(idx>=pd.Timestamp('2018-01-01'))
    stats[k]={'def':d,'note':n,'n':int(f.sum()),'P_off':rnd(L.loc[f,'riskoff21'].mean()) if f.sum() else None,'fwd21':rnd(L.loc[f,'fwd21'].mean()*100) if f.sum() else None,'fwdDD21':rnd(L.loc[f,'fwdDD21'].mean()*100) if f.sum() else None,
              'days':int(a.sum()),'P_days':rnd(L.loc[a,'riskoff21'].mean()) if a.sum() else None,'P_days_2018':rnd(L.loc[a18,'riskoff21'].mean()) if a18.sum() else None,
              'last':str(f[f].index[-1].date()) if f.sum() else 'never','live':bool(m.iloc[-1]),'recent':bool(m.iloc[-10:].any()),
              'since':(lambda v: str(v.index[-1].date()) if bool(v.iloc[-1]) and len(v) else None)(m&~m.shift(1,fill_value=False)) if bool(m.iloc[-1]) else None}
    ff=m&~m.shift(1,fill_value=False); stats[k]['since']=str(ff[ff].index[-1].date()) if bool(m.iloc[-1]) and ff.any() else None
    H[k]=[int(v) for v in m.iloc[-504:]]
print(pd.DataFrame(stats).T[['n','P_off','fwd21','fwdDD21','days','P_days','P_days_2018','last','live','recent']].to_string())
def qP(x):
    m=x.notna()&ok
    if m.sum()<600: return None,None
    q=pd.qcut(x[m].rank(method='first'),5,labels=False); y=L.loc[m,'riskoff21']; return rnd(y[q==0].mean()),rnd(y[q==4].mean())
pairs=[]
for k in RAT.columns:
    b21,t21=qP(F[(k,'roc21z')]); b63,t63=qP(r63[k])
    pairs.append({'pair':k,'chg21':rnd(RAT[k].diff(21).iloc[-1]*100),'chg63':rnd(RAT[k].diff(63).iloc[-1]*100),'roc5z':rnd(F[(k,'roc5z')].iloc[-1]),'roc21z':rnd(F[(k,'roc21z')].iloc[-1]),'roc63z':rnd(r63[k].iloc[-1]),'accel5z':rnd(F[(k,'accel5z')].iloc[-1]),
                  'P_bot21':b21,'P_top21':t21,'P_bot63':b63,'P_top63':t63})
composite={'med5':rnd(BR['sector EW breadth roc5z (median of 9)'].iloc[-1]),'med21':rnd(BR['sector EW breadth roc21z (median of 9)'].iloc[-1]),'share':rnd(BR['share of sectors EW>CW over 21d'].iloc[-1]),'share_z':rnd(BR['share z'].iloc[-1])}
hist={'dates':[str(d.date()) for d in idx[-504:]],'RSP/SPY 21d':[rnd(v) for v in F[('RSP/SPY','roc21z')].iloc[-504:]],'RSP/SPY 63d':[rnd(v) for v in r63['RSP/SPY'].iloc[-504:]],'QQQE/QQQ 63d':[rnd(v) for v in r63['QQQE/QQQ'].iloc[-504:]],'TNX 21d':[rnd(v) for v in tnx.iloc[-504:]]}
out={'asof':str(idx[-1].date()),'pairs':pairs,'composite':composite,'tnx21z':rnd(tnx.iloc[-1]),'spy21z':rnd(spyz.iloc[-1]),'cells':stats,'cells_hist':H,'hist':hist}
json.dump(out,open('breadth_data.json','w')); import os; print('KB',os.path.getsize('breadth_data.json')//1024)
print(pd.DataFrame(pairs).set_index('pair').to_string()); print(composite, 'TNX21z',out['tnx21z'],'SPY21z',out['spy21z'])
