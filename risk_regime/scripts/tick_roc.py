import pandas as pd, numpy as np, warnings, json; warnings.filterwarnings('ignore')
pd.set_option('display.width',340); pd.set_option('display.max_rows',400); pd.set_option('display.max_columns',60)
G=pd.read_pickle('G.pkl'); L=pd.read_pickle('L.pkl'); P=pd.read_pickle('P.pkl'); S=pd.read_pickle('S.pkl'); idx=P.index
OFF=pd.read_pickle('OFF.pkl'); ON=pd.read_pickle('ON.pkl')
EX=(idx>=pd.Timestamp('2020-02-01'))&(idx<=pd.Timestamp('2020-07-31'))
TICK=[c for c in P.columns if c not in ('LQD','SHY','SPY')]
lp=np.log(P); rel=lp[TICK].sub(lp['SPY'],axis=0)      # log ratio ticker/SPY
def z(x,w=252): return (x-x.rolling(w).mean())/x.rolling(w).std()
RR={}; RA={}
for t in TICK:
    for n in (5,10,21):
        RR[(t,n)]=z(rel[t].diff(n)); RA[(t,n)]=z(lp[t].diff(n))
RR=pd.DataFrame(RR); RA=pd.DataFrame(RA)
RR.to_pickle('RR.pkl'); RA.to_pickle('RA.pkl')
ok=L['ok21']&~EX
# ---------- 1. relative ROC z signature around peaks and troughs ----------
OFFS=list(range(-10,11))
def sig(anchors,n):
    rows={}
    for t in TICK:
        s=RR[(t,n)]; pos=[idx.get_loc(a) for a in anchors if not np.isnan(s.iloc[idx.get_loc(a)])]
        if len(pos)<8: continue
        M=np.array([[s.iloc[p+k] if 0<=p+k<len(s) else np.nan for k in OFFS] for p in pos]); rows[f'{t} (n={len(pos)})']=np.nanmedian(M,axis=0)
    return pd.DataFrame(rows,index=OFFS).T
print('=== RELATIVE-TO-SPY ROC5 z, median across 36 RISK-OFF PEAKS (negative = losing to SPY) ===')
A=sig(OFF.peak.tolist(),5); print(A.round(2).to_string())
print('\n--- sorted by value at day -3 (who is already breaking relative before the top) ---'); print(A[-3].sort_values().round(2).to_string())
print('\n=== RELATIVE ROC5 z around 40 RISK-ON TROUGHS ==='); B=sig(ON.trough.tolist(),5); print(B.round(2).to_string())
print('\n--- sorted by value at day +3 (who leads the first bounce) ---'); print(B[3].sort_values(ascending=False).round(2).to_string())
# ---------- 2. does a ticker's relative ROC predict SPY drawdowns? ----------
from sklearn.metrics import roc_auc_score
rows=[]
for t in TICK:
    for n in (5,10,21):
        x=RR[(t,n)]; m=x.notna()&ok
        if m.sum()<600: continue
        y=L.loc[m,'riskoff21']; auc=roc_auc_score(y,x[m]); q=pd.qcut(x[m].rank(method='first'),5,labels=False)
        lo=q==0; hi=q==4
        rows.append({'ticker':t,'n':n,'obs':int(m.sum()),'AUC':max(auc,1-auc),'sign':'weak->off' if auc<.5 else 'strong->off','P_off_bottomQ':y[lo].mean(),'P_off_topQ':y[hi].mean(),'spy_fwd21_bottomQ':L.loc[m,'fwd21'][lo].mean()*100,'spy_fwd21_topQ':L.loc[m,'fwd21'][hi].mean()*100})
E=pd.DataFrame(rows); E['gap']=(E.P_off_bottomQ-E.P_off_topQ).abs()
print('\n=== Ticker relative ROC z as SPY drawdown predictor (P(5% DD/21d) in bottom vs top quintile; base 0.174) top 25 by gap ===')
print(E.sort_values('gap',ascending=False).head(25).round(3).to_string())
# ---------- 3. ticker x index ROC matrix: median 10d fwd relative return when index ROC5 z > 1 and < -1 ----------
IDX={'VIX':'VIX_roc5_z252','VVIX/VIX':'X_VVIX_VIX_roc5_z252','VVIX':'VVIX_roc5_z252','MOVE':'MOVE_roc5_z252','TNX':'TNX_chg5_z252','TYX':'TYX_chg5_z252','GVZ':'GVZ_roc5_z252','OVX':'OVX_roc5_z252','VIX9D/VIX':'TS_VIX9D_VIX_roc5_z252','VIX/VIX3M':'TS_VIX_VIX3M_roc5_z252','VXN/VIX':'X_VXN_VIX_roc5_z252','SKEW':'SKEW_roc10_z252','VIX1D':'VIX1D_roc3_z252'}
fwd10=(rel.shift(-10)-rel)*100
MAT={}; MATN={}
for k,c in IDX.items():
    for lab,cond in [('up',G[c]>1),('down',G[c]<-1)]:
        m=cond.fillna(False)&ok
        MAT[(k,lab)]=fwd10[m].median(); MATN[(k,lab)]=fwd10[m].notna().sum()
MAT=pd.DataFrame(MAT); MATN=pd.DataFrame(MATN)
print('\n=== TICKER x INDEX ROC: median 10d fwd return RELATIVE to SPY (%) when index 5d ROC z > 1 ("up") or < -1 ("down") ===')
print(MAT.round(2).to_string())
MAT.to_pickle('MAT.pkl'); MATN.to_pickle('MATN.pkl')
# ---------- 4. contemporaneous ROC beta: ticker daily rel return on index daily dlog, calm vs stress ----------
d=pd.DataFrame({k:np.log(S[k]).diff() for k in ['VIX','VVIX','MOVE','GVZ','OVX','SKEW','VIX9D']}); d['TNX']=S['TNX'].diff(); d['TYX']=S['TYX'].diff()
dr=rel.diff()
BET={}
for regime,m in [('all',~EX),('stress',(G['VIX_pct252']>0.75)&~EX)]:
    for k in d.columns:
        x=d[k][m]; BET[(regime,k)]=[dr[t][m].cov(x)/x.var()*100 if dr[t][m].notna().sum()>300 else np.nan for t in TICK]
BET=pd.DataFrame(BET,index=TICK)
print('\n=== ROC BETA: % change in ticker/SPY ratio per 1.00 (100%) dlog move in index (per 1 pct-pt for TNX/TYX); all days vs stress ===')
print(BET.round(2).to_string()); BET.to_pickle('BET.pkl')
# ---------- 5. ticker ROC x index ROC two-way grids for the pairs that matter ----------
def grid(t,ic,lab,n=5):
    x=RR[(t,n)]; y=G[ic]; m=x.notna()&y.notna()&ok
    A_=pd.cut(x[m],[-9,-1,0,1,9],labels=['<-1','-1..0','0..1','>1']); B_=pd.cut(y[m],[-9,-1,0,1,9],labels=['<-1','-1..0','0..1','>1'])
    f=fwd10[t][m]; sp=L.loc[m,'riskoff21']
    print(f'\n### {t} rel ROC5 z (rows) x {lab} (cols): median 10d fwd rel return % | P(SPY 5% DD)')
    T1=pd.DataFrame({'A':A_,'B':B_,'f':f,'s':sp}).pivot_table(index='A',columns='B',values='f',aggfunc='median').round(2)
    T2=pd.DataFrame({'A':A_,'B':B_,'f':f,'s':sp}).pivot_table(index='A',columns='B',values='s',aggfunc='mean').round(2)
    print(pd.concat({'fwd10 rel %':T1,'P(SPY DD)':T2},axis=1).to_string())
grid('KRE','MOVE_roc5_z252','MOVE ROC5 z'); grid('KRE','TNX_chg5_z252','TNX chg5 z'); grid('XHB','TNX_chg5_z252','TNX chg5 z'); grid('XLU','TNX_chg5_z252','TNX chg5 z')
grid('TLT','MOVE_roc5_z252','MOVE ROC5 z'); grid('TLT','VIX_roc5_z252','VIX ROC5 z'); grid('HYG','VIX_roc5_z252','VIX ROC5 z'); grid('HYG','X_VVIX_VIX_roc5_z252','VVIX/VIX ROC5 z')
grid('XOP','OVX_roc5_z252','OVX ROC5 z'); grid('XES','OVX_roc5_z252','OVX ROC5 z'); grid('GLD','GVZ_roc5_z252','GVZ ROC5 z'); grid('XME','GVZ_roc5_z252','GVZ ROC5 z')
grid('XSD','X_VXN_VIX_roc5_z252','VXN/VIX ROC5 z'); grid('QQQ','X_VXN_VIX_roc5_z252','VXN/VIX ROC5 z'); grid('IWM','VIX_roc5_z252','VIX ROC5 z'); grid('XBI','VIX_roc5_z252','VIX ROC5 z')
grid('XLP','VIX_roc5_z252','VIX ROC5 z'); grid('XLF','TNX_chg5_z252','TNX chg5 z'); grid('XLRE','TNX_chg5_z252','TNX chg5 z'); grid('XAR','VIX_roc5_z252','VIX ROC5 z')
