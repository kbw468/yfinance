import pandas as pd, numpy as np, warnings; warnings.filterwarnings('ignore')
pd.set_option('display.width',320); pd.set_option('display.max_rows',300)
G=pd.read_pickle('G.pkl'); L=pd.read_pickle('L.pkl'); P=pd.read_pickle('P.pkl'); R=pd.read_pickle('R.pkl'); idx=P.index
EX=(idx>=pd.Timestamp('2020-02-01'))&(idx<=pd.Timestamp('2020-07-31'))
ls=np.log(P['SPY']); r=R['SPY']
def z(x,w=252): return (x-x.rolling(w).mean())/x.rolling(w).std()
F={h:(ls.shift(-h)-ls)*100 for h in (3,5,10,21,42,63)}
D21=L['fwdDD21']*100; D42=L['fwdDD42']*100; D63=L['fwdDD63']*100
ERAS={'2008-17':(idx>=pd.Timestamp('2008-01-01'))&(idx<pd.Timestamp('2018-01-01')),'2018-26':(idx>=pd.Timestamp('2018-01-01')),'all':(idx>=pd.Timestamp('2008-01-01'))}
def ev(name,mask,show_dates=False):
    mask=mask.fillna(False)&~EX; rows=[]
    for era,em in ERAS.items():
        m=mask&em; n=int(m.sum())
        if n<5: rows.append({'era':era,'n':n}); continue
        rows.append({'era':era,'n':n,'fwd5':F[5][m].mean(),'fwd10':F[10][m].mean(),'fwd21':F[21][m].mean(),'fwd42':F[42][m].mean(),'fwd63':F[63][m].mean(),'P(21<0)':(F[21][m]<0).mean(),'DD21':D21[m].mean(),'DD63':D63[m].mean(),'P(DD63<=-10)':(D63[m]<=-10).mean(),'RV21 +21d/now':(RV21.shift(-21)[m]/RV21[m]).median()})
    print(f'\n=== {name} ==='); print(pd.DataFrame(rows).set_index('era').round(2).to_string())
    if show_dates:
        m=mask&ERAS['all']; print('   dates:',' '.join(f"{d.date()}({F[21][d]:+.1f}/{D63[d]:.1f})" for d in idx[m]))
for T in ['SPY','QQQ']:
    rr=R[T]; RV21=rr.rolling(21).std()*np.sqrt(252)*100; RV63=rr.rolling(63).std()*np.sqrt(252)*100; RAT=RV21/RV63
    roc=np.log(RAT).diff(5); rocz=z(roc); lvl_z=z(np.log(RAT))
    print(f'\n\n######## {T} 21/63 ########')
    for th in (0.9,1.0,1.1,1.2,1.3):
        up=(RAT>th)&(RAT.shift(1)<=th)
        ev(f'{T} 21/63 crosses above {th}',up)
    up1=(RAT>1.0)&(RAT.shift(1)<=1.0)
    ev(f'{T} 21/63 crosses 1.0 FAST (5d ROC z > 1)',up1&(rocz>1),show_dates=(T=='SPY'))
    ev(f'{T} 21/63 crosses 1.0 SLOW (5d ROC z <= 1)',up1&(rocz<=1))
    ev(f'{T} 21/63 crosses 1.0 from a compressed base (ratio was <0.8 within prior 21d)',up1&(RAT.rolling(21).min().shift(1)<0.8),show_dates=(T=='SPY'))
    ev(f'{T} 21/63 crosses 1.0 with SPY within 3% of 63d high',up1&(G['SPY_dd63']>-0.03),show_dates=(T=='SPY'))
    ev(f'{T} 21/63 crosses 1.0 with SPY >=5% off high',up1&(G['SPY_dd63']<=-0.05))
    ev(f'{T} 21/63 crosses 1.0 with VIX 21d ROC z > 1',up1&(G['VIX_roc21_z252']>1))
    ev(f'{T} 21/63 crosses 1.0 with VIX 21d ROC z <= 0 (realized up, implied not)',up1&(G['VIX_roc21_z252']<=0))
    ev(f'{T} 21/63 crosses 1.0 with TNX 21d chg z > 1',up1&(G['TNX_chg21_z252']>1))
    ev(f'{T} 21/63 crosses 1.0 with MOVE 21d ROC z > 1',up1&(G['MOVE_roc21_z252']>1))
    ev(f'{T} 21/63 crosses 1.0 with VVIX/VIX 5d ROC z < -1',up1&(G['X_VVIX_VIX_roc5_z252']<-1))
    # persistence: ratio holds >1 for 10 sessions after crossing
    hold=up1.shift(10)&(RAT.rolling(10).min()>1.0)
    ev(f'{T} 21/63 crossed 1.0 ten sessions ago and has HELD above 1 since',hold)
    dn=(RAT<1.0)&(RAT.shift(1)>=1.0)
    ev(f'{T} 21/63 crosses BELOW 1.0',dn)
    ev(f'{T} 21/63 crosses below 1.0 with SPY >=5% off high (post-stress normalization)',dn&(G['SPY_dd63']<=-0.05))
    print('today',T,'21/63 =',round(RAT.iloc[-1],3),'5d ROC z',round(rocz.iloc[-1],2),'level z',round(lvl_z.iloc[-1],2),'21d min of ratio',round(RAT.iloc[-21:].min(),3))
