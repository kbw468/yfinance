import pandas as pd, numpy as np, warnings, json; warnings.filterwarnings('ignore')
pd.set_option('display.width',340); pd.set_option('display.max_rows',400); pd.set_option('display.max_columns',60)
G=pd.read_pickle('G.pkl'); L=pd.read_pickle('L.pkl'); P=pd.read_pickle('P.pkl'); S=pd.read_pickle('S.pkl'); R=pd.read_pickle('R.pkl'); idx=P.index
OFF=pd.read_pickle('OFF.pkl'); ON=pd.read_pickle('ON.pkl')
EX=(idx>=pd.Timestamp('2020-02-01'))&(idx<=pd.Timestamp('2020-07-31')); ok=L['ok21']&~EX
TICK=[c for c in P.columns if c not in ('LQD','SHY')]
def z(x,w=252): return (x-x.rolling(w).mean())/x.rolling(w).std()
RV10=R[TICK].rolling(10).std()*np.sqrt(252)*100; RV21=R[TICK].rolling(21).std()*np.sqrt(252)*100; RV63=R[TICK].rolling(63).std()*np.sqrt(252)*100
RVR=RV10/RV21                      # realized term structure (short/long)
RVR_z=z(np.log(RVR)); RVR63=RV21/RV63; RVR63_z=z(np.log(RVR63))
RELRV=RV21[TICK].div(RV21['SPY'],axis=0); RELRV_roc5z=z(np.log(RELRV).diff(5))   # relative realized vol vs SPY, ROC
RV10_roc5z=z(np.log(RV10).diff(5))
# native implied/realized pairs
IVRV={'SPY:VIX/RV':S['VIX']/RV21['SPY'],'QQQ:VXN/RV':S['VXN']/RV21['QQQ'],'TLT:MOVE/RV':S['MOVE']/RV21['TLT'],'IEF:MOVE/RV':S['MOVE']/RV21['IEF'],'GLD:GVZ/RV':S['GVZ']/RV21['GLD'],'XOP:OVX/RV':S['OVX']/RV21['XOP'],'BNO:OVX/RV':S['OVX']/RV21['BNO'],'XLE:OVX/RV':S['OVX']/RV21['XLE'],'IWM:VIX/RV':S['VIX']/RV21['IWM'],'HYG:VIX/RV':S['VIX']/RV21['HYG']}
IVRV=pd.DataFrame(IVRV); IVRV_z=z(np.log(IVRV)); IVRV_roc5z=z(np.log(IVRV).diff(5)); IVRV_pct=IVRV.rolling(252).rank(pct=True)
# breadth
BR=(RVR[[t for t in TICK if t!='SPY']]>1.2).mean(axis=1); BR_z=z(BR,252)
pd.to_pickle({'RV10':RV10,'RV21':RV21,'RVR':RVR,'RVR_z':RVR_z,'RELRV':RELRV,'RELRV_roc5z':RELRV_roc5z,'IVRV':IVRV,'IVRV_z':IVRV_z,'IVRV_roc5z':IVRV_roc5z,'IVRV_pct':IVRV_pct,'BR':BR,'BR_z':BR_z,'RV10_roc5z':RV10_roc5z},'RV.pkl')
OFFS=list(range(-10,11))
def sig(df,anchors,title):
    rows={}
    for c in df.columns:
        s=df[c]; pos=[idx.get_loc(a) for a in anchors if not np.isnan(s.iloc[idx.get_loc(a)])]
        if len(pos)<8: continue
        M=np.array([[s.iloc[p+k] if 0<=p+k<len(s) else np.nan for k in OFFS] for p in pos]); rows[f'{c} (n={len(pos)})']=np.nanmedian(M,axis=0)
    T=pd.DataFrame(rows,index=OFFS).T; print(f'\n=== {title} ==='); print(T.round(2).to_string()); return T
A=sig(RVR_z,OFF.peak.tolist(),'RV10/RV21 z (realized term structure) around 36 RISK-OFF PEAKS. + = short-term realized expanding vs 1m')
print('\n--- sorted by day -3 value (whose realized is already expanding before the top) ---'); print(A[-3].sort_values(ascending=False).round(2).head(15).to_string())
B=sig(RVR_z,ON.trough.tolist(),'RV10/RV21 z around 40 RISK-ON TROUGHS')
C=sig(IVRV_roc5z,OFF.peak.tolist(),'IV/RV ratio ROC5 z (implied vs ticker realized) around PEAKS. + = implied rising faster than realized')
D=sig(IVRV_roc5z,ON.trough.tolist(),'IV/RV ratio ROC5 z around TROUGHS')
E=sig(IVRV_z,OFF.peak.tolist(),'IV/RV ratio LEVEL z around PEAKS')
F=sig(IVRV_z,ON.trough.tolist(),'IV/RV ratio LEVEL z around TROUGHS')
Bs=sig(pd.DataFrame({'RV breadth (share RV10/RV21>1.2)':BR,'breadth z':BR_z}),OFF.peak.tolist(),'REALIZED VOL BREADTH around PEAKS'); sig(pd.DataFrame({'RV breadth':BR,'breadth z':BR_z}),ON.trough.tolist(),'REALIZED VOL BREADTH around TROUGHS')
# ---- predictive: IV/RV level pct and ROC vs SPY drawdown; breadth x VIX ROC grid
from sklearn.metrics import roc_auc_score
print('\n=== IV/RV native pairs as SPY drawdown predictors: P(5% DD/21d) by quintile of LEVEL (pct252) and of ROC5 z; base 0.174 ===')
for c in IVRV.columns:
    for lab,x in [('level',IVRV_z[c]),('roc5z',IVRV_roc5z[c])]:
        m=x.notna()&ok
        if m.sum()<600: continue
        q=pd.qcut(x[m].rank(method='first'),5,labels=False); y=L.loc[m,'riskoff21']; f=L.loc[m,'fwd21']*100
        print(f'{c:14s} {lab:6s} P(off) by Q: '+' '.join(f'{y[q==i].mean():.3f}' for i in range(5))+'   fwd21% by Q: '+' '.join(f'{f[q==i].mean():5.2f}' for i in range(5))+f'  n={m.sum()}')
def grid(x,y,lx,ly,target='spy'):
    m=x.notna()&y.notna()&ok
    A_=pd.cut(x[m],[-9,-1,0,1,9],labels=['<-1','-1..0','0..1','>1']); B_=pd.cut(y[m],[-9,-1,0,1,9],labels=['<-1','-1..0','0..1','>1'])
    d=pd.DataFrame({'A':A_,'B':B_,'s':L.loc[m,'riskoff21'],'f':L.loc[m,'fwd21']*100})
    print(f'\n### {lx} (rows) x {ly} (cols): P(SPY 5% DD/21d) | mean SPY fwd21 % | n')
    print(pd.concat({'P(DD)':d.pivot_table(index='A',columns='B',values='s',aggfunc='mean').round(2),'fwd21':d.pivot_table(index='A',columns='B',values='f',aggfunc='mean').round(2),'n':d.pivot_table(index='A',columns='B',values='s',aggfunc='count')},axis=1).to_string())
grid(BR_z,G['VIX_roc5_z252'],'RV breadth z','VIX ROC5 z')
grid(RVR_z['SPY'],G['VIX_roc5_z252'],'SPY RV10/RV21 z','VIX ROC5 z')
grid(IVRV_roc5z['SPY:VIX/RV'],G['VIX_roc5_z252'],'SPY IV/RV ROC5 z','VIX ROC5 z')
grid(IVRV_z['SPY:VIX/RV'],G['VIX_roc5_z252'],'SPY IV/RV level z','VIX ROC5 z')
grid(IVRV_roc5z['TLT:MOVE/RV'],G['MOVE_roc5_z252'],'TLT IV/RV ROC5 z','MOVE ROC5 z')
grid(IVRV_z['TLT:MOVE/RV'],G['TNX_chg5_z252'],'TLT IV/RV level z','TNX chg5 z')
grid(IVRV_roc5z['GLD:GVZ/RV'],G['GVZ_roc5_z252'],'GLD IV/RV ROC5 z','GVZ ROC5 z')
grid(IVRV_roc5z['XOP:OVX/RV'],G['OVX_roc5_z252'],'XOP IV/RV ROC5 z','OVX ROC5 z')
grid(IVRV_roc5z['QQQ:VXN/RV'],G['VXN_roc5_z252'],'QQQ IV/RV ROC5 z','VXN ROC5 z')
grid(RVR_z['HYG'],G['VIX_roc5_z252'],'HYG RV10/RV21 z','VIX ROC5 z')
grid(RVR_z['TLT'],G['MOVE_roc5_z252'],'TLT RV10/RV21 z','MOVE ROC5 z')
grid(RVR_z['KRE'],G['TNX_chg5_z252'],'KRE RV10/RV21 z','TNX chg5 z')
grid(RELRV_roc5z['IWM'],G['VIX_roc5_z252'],'IWM rel-RV ROC5 z','VIX ROC5 z')
grid(RELRV_roc5z['HYG'],G['VIX_roc5_z252'],'HYG rel-RV ROC5 z','VIX ROC5 z')
# ---- per-ticker: RVR z top/bottom quintile → SPY P(DD); and ticker own fwd10 rel by (RVR z>1) x (VIX roc5z>1 / <-1)
lp=np.log(P); rel=lp[TICK].sub(lp['SPY'],axis=0); fwd10=(rel.shift(-10)-rel)*100
rows=[]
for t in TICK:
    x=RVR_z[t]; m=x.notna()&ok
    if m.sum()<600: continue
    q=pd.qcut(x[m].rank(method='first'),5,labels=False); y=L.loc[m,'riskoff21']
    vu=(G['VIX_roc5_z252']>1); vd=(G['VIX_roc5_z252']<-1); ex=(x>1); cp=(x<-0.5)
    rows.append({'ticker':t,'P_off_RVRtopQ':y[q==4].mean(),'P_off_RVRbotQ':y[q==0].mean(),
      'fwd10rel_RVexp_VIXup':fwd10[t][m&ex&vu].median(),'n1':int((m&ex&vu).sum()),'fwd10rel_RVexp_VIXdown':fwd10[t][m&ex&vd].median(),'n2':int((m&ex&vd).sum()),
      'fwd10rel_RVcomp_VIXup':fwd10[t][m&cp&vu].median(),'n3':int((m&cp&vu).sum()),'fwd10rel_RVcomp_VIXdown':fwd10[t][m&cp&vd].median(),'n4':int((m&cp&vd).sum())})
T=pd.DataFrame(rows).set_index('ticker'); T.to_pickle('RVT.pkl')
print('\n=== per-ticker: P(SPY 5% DD) when ticker RV10/RV21 z in top vs bottom quintile; and ticker median 10d rel return by (own realized expanding z>1 / compressed z<-0.5) x (VIX ROC5 z >1 / <-1) ===')
print(T.round(3).to_string())
