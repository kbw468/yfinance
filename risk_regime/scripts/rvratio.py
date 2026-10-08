import pandas as pd, numpy as np, warnings; warnings.filterwarnings('ignore')
pd.set_option('display.width',320); pd.set_option('display.max_rows',300); pd.set_option('display.max_columns',40)
G=pd.read_pickle('G.pkl'); L=pd.read_pickle('L.pkl'); P=pd.read_pickle('P.pkl'); R=pd.read_pickle('R.pkl'); RU=pd.read_pickle('RU.pkl'); idx=P.index
OFF=pd.read_pickle('OFF.pkl'); ON=pd.read_pickle('ON.pkl')
EX=(idx>=pd.Timestamp('2020-02-01'))&(idx<=pd.Timestamp('2020-07-31')); ok=L['ok21']&~EX
def z(x,w=252): return (x-x.rolling(w).mean())/x.rolling(w).std()
RV={}; RAT={}; RATz={}; RATroc={}
for t in ['SPY','QQQ']:
    for w in (5,10,21,63,126): RV[(t,w)]=R[t].rolling(w).std()*np.sqrt(252)*100
    for a,b in [(5,21),(10,21),(10,63),(21,63),(21,126)]:
        k=(t,f'{a}/{b}'); RAT[k]=RV[(t,a)]/RV[(t,b)]; RATz[k]=z(np.log(RAT[k])); RATroc[k]=z(np.log(RAT[k]).diff(5))
RAT=pd.DataFrame(RAT); RATz=pd.DataFrame(RATz); RATroc=pd.DataFrame(RATroc)
OFFS=list(range(-10,11))
def sig(df,anchors,title):
    rows={}
    for c in df.columns:
        s=df[c]; pos=[idx.get_loc(a) for a in anchors if not np.isnan(s.iloc[idx.get_loc(a)])]
        M=np.array([[s.iloc[p+k] if 0<=p+k<len(s) else np.nan for k in OFFS] for p in pos]); rows[f'{c[0]} {c[1]} (n={len(pos)})']=np.nanmedian(M,axis=0)
    print(f'\n=== {title} ==='); print(pd.DataFrame(rows,index=OFFS).T.round(2).to_string())
sig(RATz,OFF.peak.tolist(),'RV RATIO LEVEL z around 36 PEAKS')
sig(RATz,ON.trough.tolist(),'RV RATIO LEVEL z around 40 TROUGHS')
sig(RATroc,OFF.peak.tolist(),'RV RATIO 5d ROC z around PEAKS')
print('\n=== quintiles: P(5% DD/21d) and P(10% DD/63d) and fwd21 by quintile of ratio level z (base P5 0.174) ===')
p10=(L['fwdDD63']<=-0.10).astype(float); base10=p10[ok&L['ok63']].mean()
for c in RATz.columns:
    x=RATz[c]; m=x.notna()&ok&L['ok63']; q=pd.qcut(x[m].rank(method='first'),5,labels=False); y=L.loc[m,'riskoff21']; f=L.loc[m,'fwd21']*100; y10=p10[m]
    print(f'{c[0]} {c[1]:7s} P5: '+' '.join(f'{y[q==i].mean():.3f}' for i in range(5))+'  P10/63: '+' '.join(f'{y10[q==i].mean():.3f}' for i in range(5))+'  fwd21: '+' '.join(f'{f[q==i].mean():5.2f}' for i in range(5)))
print('base P10/63',round(base10,3))
def grid(x,y,lx,ly):
    m=x.notna()&y.notna()&ok
    A=pd.cut(x[m],[-9,-1,0,1,9],labels=['<-1','-1..0','0..1','>1']); B=pd.cut(y[m],[-9,-1,0,1,9],labels=['<-1','-1..0','0..1','>1'])
    d=pd.DataFrame({'A':A,'B':B,'s':L.loc[m,'riskoff21'],'f':L.loc[m,'fwd21']*100,'d':L.loc[m,'fwdDD63']*100})
    print(f'\n### {lx} (rows) x {ly} (cols): P(SPY 5% DD/21d) | mean fwd21 % | mean 63d maxDD % | n')
    print(pd.concat({'P':d.pivot_table(index='A',columns='B',values='s',aggfunc='mean').round(2),'fwd21':d.pivot_table(index='A',columns='B',values='f',aggfunc='mean').round(2),'DD63':d.pivot_table(index='A',columns='B',values='d',aggfunc='mean').round(1),'n':d.pivot_table(index='A',columns='B',values='s',aggfunc='count')},axis=1).to_string())
for t in ['SPY','QQQ']:
    for rr in ['10/21','21/63']:
        grid(RATz[(t,rr)],G['VIX_roc5_z252'],f'{t} RV{rr} z','VIX ROC5 z')
        grid(RATz[(t,rr)],G['X_VVIX_VIX_roc5_z252'],f'{t} RV{rr} z','VVIX/VIX ROC5 z')
    grid(RATz[(t,'21/63')],G['TNX_chg21_z252'],f'{t} RV21/63 z','TNX chg21 z')
    grid(RATz[(t,'21/63')],G['MOVE_roc21_z252'],f'{t} RV21/63 z','MOVE ROC21 z')
    grid(RATz[(t,'21/63')],G['VIX_roc21_z252'],f'{t} RV21/63 z','VIX ROC21 z')
# as qualifier on main rules: first-fires split by SPY RV21/63 z above/below 0 and RV10/21 z
print('\n=== main rules split by SPY realized ratio at the fire (first-fires, ex-2020): P(5% DD/21d), fwd21, n ===')
for rule in ['SETUP_rates','ONSET_impulse','ONSET_vvixlag','CONT_2ndleg','FAIL_yieldsup','CAP_alldims','CAP_vix_gvz','ONCONF_collapse']:
    f=RU[rule]&~RU[rule].shift(1,fill_value=False)&ok
    for rr in ['10/21','21/63']:
        x=RATz[('SPY',rr)]
        for lab,mm in [('ratio z<0 (realized compressed)',f&(x<0)),('ratio z>=0 (realized expanding)',f&(x>=0))]:
            if mm.sum()<5: continue
            print(f'{rule:16s} SPY RV{rr} {lab:32s} n={mm.sum():3d} P={L.loc[mm,"riskoff21"].mean():.2f} fwd21={L.loc[mm,"fwd21"].mean()*100:5.2f} DD63={L.loc[mm,"fwdDD63"].mean()*100:5.1f}')
print('\nTODAY:'); print(pd.DataFrame({'ratio':RAT.iloc[-1].round(2),'level z':RATz.iloc[-1].round(2),'5d ROC z':RATroc.iloc[-1].round(2)}).to_string())
