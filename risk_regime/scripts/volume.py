import pandas as pd, numpy as np, warnings; warnings.filterwarnings('ignore')
pd.set_option('display.width',320); pd.set_option('display.max_rows',200); pd.set_option('display.max_columns',40)
G=pd.read_pickle('G.pkl'); L=pd.read_pickle('L.pkl'); P=pd.read_pickle('P.pkl'); idx=P.index
OFF=pd.read_pickle('OFF.pkl'); ON=pd.read_pickle('ON.pkl')
EX=(idx>=pd.Timestamp('2020-02-01'))&(idx<=pd.Timestamp('2020-07-31')); ok=L['ok21']&~EX
def z(x,w=252): return (x-x.rolling(w).mean())/x.rolling(w).std()
VOL={}
for t in ['SPY','QQQ','IWM','HYG','TLT','GLD','XLE','KRE']:
    d=pd.read_csv(f'data/{t}.csv',index_col=0,parse_dates=True)['Volume'].reindex(idx)
    lv=np.log(d.replace(0,np.nan))
    VOL[(t,'v_rel21')]=z(lv-lv.rolling(21).mean().shift(1))          # today's volume vs trailing month (z of log ratio)
    VOL[(t,'v_roc5z')]=z(lv.rolling(5).mean()-lv.rolling(5).mean().shift(5))   # 5d avg volume ROC
    VOL[(t,'v_roc21z')]=z(lv.rolling(21).mean()-lv.rolling(21).mean().shift(21))
    VOL[(t,'v_pct252')]=lv.rolling(252).rank(pct=True)
    # signed volume: volume ROC x sign of 5d price move (up-volume vs down-volume)
    r5=np.log(P[t]).diff(5)
    VOL[(t,'upvol')]=VOL[(t,'v_roc5z')]*np.sign(r5)                     # + = volume expanding on up move, - = on down move
VOL=pd.DataFrame(VOL); VOL.to_pickle('VOL.pkl')
OFFS=list(range(-10,11))
def sig(cols,anchors,title):
    rows={}
    for c in cols:
        s=VOL[c]; pos=[idx.get_loc(a) for a in anchors if not np.isnan(s.iloc[idx.get_loc(a)])]
        if len(pos)<8: continue
        M=np.array([[s.iloc[p+k] if 0<=p+k<len(s) else np.nan for k in OFFS] for p in pos]); rows[f'{c[0]} {c[1]} (n={len(pos)})']=np.nanmedian(M,axis=0)
    print(f'\n=== {title} ==='); print(pd.DataFrame(rows,index=OFFS).T.round(2).to_string())
cols=[(t,k) for t in ['SPY','QQQ','IWM','HYG'] for k in ['v_rel21','v_roc5z','v_roc21z','upvol']]
sig(cols,OFF.peak.tolist(),'VOLUME around 36 RISK-OFF PEAKS (median z)')
sig(cols,ON.trough.tolist(),'VOLUME around 40 RISK-ON TROUGHS (median z)')
# predictive: quintiles
print('\n=== volume features as SPY drawdown predictors: P(5% DD/21d) by quintile (base 0.174), fwd21 by quintile ===')
for c in cols+[('TLT','v_roc5z'),('KRE','v_roc5z'),('XLE','v_roc5z')]:
    x=VOL[c]; m=x.notna()&ok
    if m.sum()<600: continue
    q=pd.qcut(x[m].rank(method='first'),5,labels=False); y=L.loc[m,'riskoff21']; f=L.loc[m,'fwd21']*100
    print(f'{c[0]:4s} {c[1]:9s} P: '+' '.join(f'{y[q==i].mean():.3f}' for i in range(5))+'  fwd21: '+' '.join(f'{f[q==i].mean():5.2f}' for i in range(5)))
# grids: volume ROC x VIX ROC ; upvol x VIX ROC ; volume x price (distribution days)
def grid(x,y,lx,ly):
    m=x.notna()&y.notna()&ok
    A=pd.cut(x[m],[-9,-1,0,1,9],labels=['<-1','-1..0','0..1','>1']); B=pd.cut(y[m],[-9,-1,0,1,9],labels=['<-1','-1..0','0..1','>1'])
    d=pd.DataFrame({'A':A,'B':B,'s':L.loc[m,'riskoff21'],'f':L.loc[m,'fwd21']*100})
    print(f'\n### {lx} (rows) x {ly} (cols): P(SPY 5% DD/21d) | mean fwd21 % | n')
    print(pd.concat({'P':d.pivot_table(index='A',columns='B',values='s',aggfunc='mean').round(2),'fwd21':d.pivot_table(index='A',columns='B',values='f',aggfunc='mean').round(2),'n':d.pivot_table(index='A',columns='B',values='s',aggfunc='count')},axis=1).to_string())
grid(VOL[('SPY','v_roc5z')],G['VIX_roc5_z252'],'SPY vol ROC5 z','VIX ROC5 z')
grid(VOL[('SPY','upvol')],G['VIX_roc5_z252'],'SPY signed vol (+up/-down)','VIX ROC5 z')
grid(VOL[('SPY','v_roc5z')],G['SPY_ret5']/G['SPY_ret5'].rolling(252).std(),'SPY vol ROC5 z','SPY 5d return z')
grid(VOL[('SPY','v_roc5z')],G['X_VVIX_VIX_roc5_z252'],'SPY vol ROC5 z','VVIX/VIX ROC5 z')
grid(VOL[('HYG','v_roc5z')],G['VIX_roc5_z252'],'HYG vol ROC5 z','VIX ROC5 z')
grid(VOL[('IWM','v_roc5z')],G['VIX_roc5_z252'],'IWM vol ROC5 z','VIX ROC5 z')
grid(VOL[('SPY','v_roc21z')],G['VIX_roc21_z252'],'SPY vol ROC21 z','VIX ROC21 z')
grid(VOL[('SPY','v_rel21')],G['VIX_roc1_z252'],'SPY today vol vs month','VIX 1d ROC z')
