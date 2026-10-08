import pandas as pd, numpy as np, warnings; warnings.filterwarnings('ignore')
pd.set_option('display.width',320); pd.set_option('display.max_rows',400); pd.set_option('display.max_columns',40)
G=pd.read_pickle('G.pkl'); L=pd.read_pickle('L.pkl'); P=pd.read_pickle('P.pkl'); R=pd.read_pickle('R.pkl'); idx=P.index
OFF=pd.read_pickle('OFF.pkl'); ON=pd.read_pickle('ON.pkl')
EX=(idx>=pd.Timestamp('2020-02-01'))&(idx<=pd.Timestamp('2020-07-31')); ok=L['ok21']&~EX
def z(x,w=252): return (x-x.rolling(w).mean())/x.rolling(w).std()
ld=lambda s: np.log(pd.read_csv(f'data/{s}.csv',index_col=0,parse_dates=True)['Close'].reindex(idx))
PAIRS={'RSP/SPY':('RSP','SPY'),'QQQE/QQQ':('QQQE','QQQ'),'RSPT/XLK':('RSPT','XLK'),'RSPS/XLP':('RSPS','XLP'),'RSPH/XLV':('RSPH','XLV'),'RSPF/XLF':('RSPF','XLF'),'RSPD/XLY':('RSPD','XLY'),'RSPG/XLE':('RSPG','XLE'),'RSPU/XLU':('RSPU','XLU'),'RSPM/XLB':('RSPM','XLB'),'RSPN/XLI':('RSPN','XLI'),'RSPR/XLRE':('RSPR','XLRE'),'RSPC/XLC':('RSPC','XLC')}
RAT={}; F={}
for k,(a,b) in PAIRS.items():
    la=ld(a); lb=np.log(P[b]) if b in P else ld(b); RAT[k]=la-lb
    for n in (5,10,21): F[(k,f'roc{n}z')]=z(RAT[k].diff(n))
    F[(k,'accel5z')]=z(RAT[k].diff(5)-RAT[k].diff(5).shift(5))
RAT=pd.DataFrame(RAT); F=pd.DataFrame(F)
SEC=[k for k in PAIRS if k not in ('RSP/SPY','QQQE/QQQ','RSPR/XLRE','RSPC/XLC')]
BR={'sector EW breadth roc5z (median of 9)':F[[(k,'roc5z') for k in SEC]].median(axis=1),'sector EW breadth roc21z (median of 9)':F[[(k,'roc21z') for k in SEC]].median(axis=1),
    'share of sectors EW>CW over 21d':(pd.concat([RAT[k].diff(21) for k in SEC],axis=1)>0).mean(axis=1)}
BR=pd.DataFrame(BR); BR['share z']=z(BR['share of sectors EW>CW over 21d'])
pd.to_pickle({'RAT':RAT,'F':F,'BR':BR},'BREADTH.pkl')
OFFS=list(range(-10,11))
def sig(cols,anchors,title):
    rows={}
    for c in cols:
        s=F[c] if c in F else BR[c]; pos=[idx.get_loc(a) for a in anchors if not np.isnan(s.iloc[idx.get_loc(a)])]
        if len(pos)<8: continue
        M=np.array([[s.iloc[p+k] if 0<=p+k<len(s) else np.nan for k in OFFS] for p in pos]); rows[f'{c if isinstance(c,str) else " ".join(c)} (n={len(pos)})']=np.nanmedian(M,axis=0)
    print(f'\n=== {title} ==='); print(pd.DataFrame(rows,index=OFFS).T.round(2).to_string())
cols=[('RSP/SPY','roc5z'),('RSP/SPY','roc21z'),('RSP/SPY','accel5z'),('QQQE/QQQ','roc5z'),('QQQE/QQQ','roc21z'),'sector EW breadth roc5z (median of 9)','sector EW breadth roc21z (median of 9)','share z']+[(k,'roc21z') for k in SEC]
sig(cols,OFF.peak.tolist(),'EQUAL-WEIGHT / CAP-WEIGHT ratio ROC z around 36 RISK-OFF PEAKS (negative = equal weight losing = breadth deteriorating)')
sig(cols,ON.trough.tolist(),'around 40 RISK-ON TROUGHS')
print('\n=== predictive quintiles: P(5% DD/21d) and fwd21 by quintile (base 0.174 / 0.86) ===')
for c in cols:
    x=F[c] if c in F else BR[c]; m=x.notna()&ok
    if m.sum()<600: continue
    q=pd.qcut(x[m].rank(method='first'),5,labels=False); y=L.loc[m,'riskoff21']; f=L.loc[m,'fwd21']*100
    nm=c if isinstance(c,str) else ' '.join(c)
    print(f'{nm:42s} n={m.sum():5d} P: '+' '.join(f'{y[q==i].mean():.3f}' for i in range(5))+'  fwd21: '+' '.join(f'{f[q==i].mean():5.2f}' for i in range(5)))
def grid(x,y,lx,ly):
    m=x.notna()&y.notna()&ok
    A=pd.cut(x[m],[-9,-1,0,1,9],labels=['<-1','-1..0','0..1','>1']); B=pd.cut(y[m],[-9,-1,0,1,9],labels=['<-1','-1..0','0..1','>1'])
    d=pd.DataFrame({'A':A,'B':B,'s':L.loc[m,'riskoff21'],'f':L.loc[m,'fwd21']*100,'d':L.loc[m,'fwdDD21']*100})
    print(f'\n### {lx} (rows) x {ly} (cols): P(5% DD/21d) | fwd21 % | DD21 % | n')
    print(pd.concat({'P':d.pivot_table(index='A',columns='B',values='s',aggfunc='mean').round(2),'fwd21':d.pivot_table(index='A',columns='B',values='f',aggfunc='mean').round(2),'DD21':d.pivot_table(index='A',columns='B',values='d',aggfunc='mean').round(1),'n':d.pivot_table(index='A',columns='B',values='s',aggfunc='count')},axis=1).to_string())
grid(F[('RSP/SPY','roc5z')],G['VIX_roc5_z252'],'RSP/SPY roc5z','VIX roc5z')
grid(F[('RSP/SPY','roc21z')],G['VIX_roc5_z252'],'RSP/SPY roc21z','VIX roc5z')
grid(F[('RSP/SPY','roc21z')],G['VIX_roc21_z252'],'RSP/SPY roc21z','VIX roc21z')
grid(F[('RSP/SPY','roc5z')],G['X_VVIX_VIX_roc5_z252'],'RSP/SPY roc5z','VVIX/VIX roc5z')
grid(F[('RSP/SPY','roc21z')],G['TNX_chg21_z252'],'RSP/SPY roc21z','TNX chg21z')
grid(F[('RSP/SPY','roc5z')],G['MOVE_roc5_z252'],'RSP/SPY roc5z','MOVE roc5z')
grid(BR['sector EW breadth roc21z (median of 9)'],G['VIX_roc5_z252'],'sector breadth roc21z','VIX roc5z')
grid(BR['share z'],G['VIX_roc21_z252'],'share-of-sectors z','VIX roc21z')
grid(F[('QQQE/QQQ','roc21z')],G['VXN_roc5_z252'],'QQQE/QQQ roc21z','VXN roc5z')
grid(F[('RSP/SPY','roc21z')],G['SPY_ret21']/G['SPY_ret21'].rolling(252).std(),'RSP/SPY roc21z','SPY 21d return z')
print('\n=== TODAY ===')
t=pd.DataFrame({'ratio 21d chg %':[RAT[k].diff(21).iloc[-1]*100 for k in PAIRS],'roc5z':[F[(k,'roc5z')].iloc[-1] for k in PAIRS],'roc21z':[F[(k,'roc21z')].iloc[-1] for k in PAIRS],'accel5z':[F[(k,'accel5z')].iloc[-1] for k in PAIRS]},index=list(PAIRS)).round(2); print(t.to_string())
print(BR.iloc[-1].round(2).to_string())
