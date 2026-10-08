import pandas as pd, numpy as np, warnings; warnings.filterwarnings('ignore')
pd.set_option('display.width',320); pd.set_option('display.max_rows',300)
G=pd.read_pickle('G.pkl'); L=pd.read_pickle('L.pkl'); P=pd.read_pickle('P.pkl'); R=pd.read_pickle('R.pkl'); VOL=pd.read_pickle('VOL.pkl'); idx=P.index
EX=(idx>=pd.Timestamp('2020-02-01'))&(idx<=pd.Timestamp('2020-07-31'))
ls=np.log(P['SPY']); r=R['SPY']
RV={w:r.rolling(w).std()*np.sqrt(252)*100 for w in (5,10,21,63)}
RAT={'10/21':RV[10]/RV[21],'21/63':RV[21]/RV[63],'5/21':RV[5]/RV[21]}
fwd=lambda h:(ls.shift(-h)-ls)*100
F={h:fwd(h) for h in (1,3,5,10,21)}
absF5=r.abs().rolling(5).mean().shift(-5)*100     # mean |daily move| over next 5 days
absB5=r.abs().rolling(5).mean()*100               # over prior 5 days
rv10f=RV[10].shift(-10)                            # realized 10d vol measured 10 days later
volf=VOL[('SPY','v_roc5z')].shift(-5)              # volume ROC z 5 days later
ERAS={'2008-2017':(idx>=pd.Timestamp('2008-01-01'))&(idx<pd.Timestamp('2018-01-01')),'2018-2026':(idx>=pd.Timestamp('2018-01-01')),'2021-2026':(idx>=pd.Timestamp('2021-01-01')),'all 2008+':(idx>=pd.Timestamp('2008-01-01'))}
def ev(name,mask):
    mask=mask.fillna(False)&~EX; rows=[]
    for era,em in ERAS.items():
        m=mask&em; n=int(m.sum())
        if n<8: rows.append({'era':era,'n':n}); continue
        rows.append({'era':era,'n':n,'fwd1':F[1][m].mean(),'fwd3':F[3][m].mean(),'fwd5':F[5][m].mean(),'fwd10':F[10][m].mean(),'fwd21':F[21][m].mean(),'P(fwd5<0)':(F[5][m]<0).mean(),
                     'abs5 after/before':(absF5[m]/absB5[m]).median(),'RV10 later/now':(rv10f[m]/RV[10][m]).median(),'vol ROC z +5d':volf[m].mean(),'DD21':L.loc[m,'fwdDD21'].mean()*100,'UP21':L.loc[m,'fwdUP21'].mean()*100})
    print(f'\n=== {name} ==='); print(pd.DataFrame(rows).set_index('era').round(2).to_string())
up5=ls.diff(5)<0; dn=ls.diff(5)<0; upp=ls.diff(5)>0
def xup(s,th): return (s>th)&(s.shift(1)<=th)
def xdn(s,th): return (s<th)&(s.shift(1)>=th)
# baseline
ev('BASELINE all days',pd.Series(True,index=idx))
for th in (1.0,1.2,1.5):
    ev(f'SPY RV10/21 crosses ABOVE {th}',xup(RAT['10/21'],th))
    ev(f'SPY RV10/21 crosses ABOVE {th} with SPY down over prior 5d (selling)',xup(RAT['10/21'],th)&dn)
    ev(f'SPY RV10/21 crosses ABOVE {th} with SPY up over prior 5d',xup(RAT['10/21'],th)&upp)
for th in (0.8,0.7):
    ev(f'SPY RV10/21 crosses BELOW {th}',xdn(RAT['10/21'],th))
for th in (1.0,1.2):
    ev(f'SPY RV21/63 crosses ABOVE {th}',xup(RAT['21/63'],th))
    ev(f'SPY RV21/63 crosses ABOVE {th} with SPY down prior 5d',xup(RAT['21/63'],th)&dn)
ev('SPY RV21/63 crosses BELOW 0.8',xdn(RAT['21/63'],0.8))
# vol-control style levels: 21d realized crossing 15/20/25%
for th in (15,20,25):
    ev(f'SPY 21d REALIZED crosses ABOVE {th}%',xup(RV[21],th))
    ev(f'SPY 21d REALIZED crosses ABOVE {th}% with SPY down prior 5d',xup(RV[21],th)&dn)
for th in (12,15):
    ev(f'SPY 21d REALIZED crosses BELOW {th}%',xdn(RV[21],th))
# ratio cross combined with VIX ROC: does implied confirm?
ev('RV10/21 crosses above 1.2 AND VIX 5d ROC z > 1',xup(RAT['10/21'],1.2)&(G['VIX_roc5_z252']>1))
ev('RV10/21 crosses above 1.2 AND VIX 5d ROC z < 0 (realized up, implied not)',xup(RAT['10/21'],1.2)&(G['VIX_roc5_z252']<0))
ev('RV10/21 crosses above 1.2 AND VVIX/VIX ROC5 z < -1',xup(RAT['10/21'],1.2)&(G['X_VVIX_VIX_roc5_z252']<-1))
ev('RV10/21 crosses above 1.2 AND VVIX/VIX ROC5 z > 0',xup(RAT['10/21'],1.2)&(G['X_VVIX_VIX_roc5_z252']>0))
