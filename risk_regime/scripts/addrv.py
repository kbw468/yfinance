import pandas as pd, numpy as np, warnings; warnings.filterwarnings('ignore')
pd.set_option('display.width',320)
exec(open('score2.py').read().split("V={}")[0])
r_=R['SPY']; RV5=r_.rolling(5).std(); RV10=r_.rolling(10).std(); RV21=r_.rolling(21).std(); RV63=r_.rolling(63).std()
def zz(x,w=252): return (x-x.rolling(w).mean())/x.rolling(w).std()
rv5=zz(np.log(RV21).diff(5)); rv10=zz(np.log(RV21).diff(10)); rv21=zz(np.log(RV21).diff(21)); rat5=zz(np.log(RV21/RV63).diff(5)); acc=zz(np.log(RV21).diff(5)-np.log(RV21).diff(5).shift(5))
S=pd.read_pickle('S.pkl'); vrp5=zz((S['VIX']-RV21*np.sqrt(252)*100).diff(5))
NEW={
 'RVX_premium_collapse':   ((vrp5<-1)&(z('VIX_roc21_z252')>1),15,-10),
 'RVX_ratio_into_yields':  ((rat5>1)&(z('TNX_chg21_z252')>1),15,-8),
 'RVX_triple_vvixlag':     ((rv5>1)&(z('VIX_roc5_z252')>1)&(z('X_VVIX_VIX_roc5_z252')<-1),15,-8),
 'RVX_accel_into_yields':  ((acc>1)&(z('TNX_chg21_z252')>1),15,-6),
 'RVX_month_and_week':     ((rv21>1)&(rv5>1),15,-6),
 'RVX_compress_MOVE_spike':((rv5<-1)&(z('MOVE_roc5_z252')>1),10,+8),
 'RVX_realized_up_frontend_down':((rv10>1)&(z('TS_VIX9D_VIX_roc5_z252')<-1),10,+8),
}
for k,(s_,w,p) in NEW.items(): SIG[k]=s_
C1n={k:v for k,v in C1.items() if k!='at_highs'}
r=R['SPY']; m=~EX&(idx>=pd.Timestamp('2008-01-01')); bh=r[m]
def stats(x):
    c=x.cumsum(); dd=c-c.cummax(); return f"ann {x.mean()*252*100:5.2f} vol {x.std()*np.sqrt(252)*100:5.2f} sh {x.mean()/x.std()*np.sqrt(252):.2f} maxDD {dd.min()*100:6.1f} ulcer {np.sqrt((dd**2).mean())*100:5.2f}"
def states2(S,oi=20,oe=35):
    v=S.values; st=np.ones(len(v),dtype=int); s=1
    for i in range(len(v)):
        x=v[i]
        if np.isnan(x): st[i]=s; continue
        if s==1 and x<=oi: s=0
        elif s==0 and x>oe: s=1
        st[i]=s
    return pd.Series(st,index=S.index)
EP=[('2008-05-19','2009-03-09'),('2010-04-23','2010-07-02'),('2011-04-29','2011-10-03'),('2015-07-20','2015-08-25'),('2015-11-03','2016-02-11'),('2018-01-26','2018-02-08'),('2018-09-20','2018-12-24'),('2022-01-03','2022-10-12'),('2023-07-31','2023-10-27'),('2024-07-16','2024-08-05'),('2025-02-19','2025-04-08'),('2026-01-27','2026-03-30')]
def report(name,W):
    S_,_=build(W,C1n); ST=states2(S_); w=ST.shift(1).map({0:0,1:1}); x=(w*r)[m]
    o=(ST==0)&m; spells=0; last=None
    for dt in idx[o]:
        if last is None or (idx.get_loc(dt)-idx.get_loc(last))>1: spells+=1
        last=dt
    caps=[]
    for a,b in EP:
        a=pd.Timestamp(a); b=pd.Timestamp(b); seg=r.loc[a:b].iloc[1:]; sp=seg.sum(); stt=(w.loc[seg.index]*seg).sum()
        q=idx.get_loc(b); rec=r.iloc[q+1:q+43]; sp2=rec.sum(); st2=(w.loc[rec.index]*rec).sum(); caps.append(f"{str(a.date())[2:7]}:{stt/sp:.2f}/{st2/sp2:.2f}")
    ok=L['ok21']&m; out=(ST==0)[ok]
    print(f'{name:36s} {stats(x)} | in {w[m].mean():.2f} spells {spells:2d} | OUT P5 {L.loc[ok,"riskoff21"][out].mean():.2f} fwd21 {L.loc[ok,"fwd21"][out].mean()*100:5.2f} | today {int(S_.iloc[-1])}')
    print('   ',' '.join(caps))
print('B&H'.ljust(36),stats(bh))
report('CURRENT stay-in',W1)
for k,(s_,w,p) in NEW.items():
    W=dict(W1); W[k]=(w,p); report(f'+ {k}',W)
W=dict(W1)
for k,(s_,w,p) in NEW.items(): W[k]=(w,p)
report('+ ALL seven',W)
W=dict(W1)
for k in ['RVX_premium_collapse','RVX_ratio_into_yields','RVX_triple_vvixlag']: W[k]=NEW[k][1:]
report('+ three risk-off only',W)
W=dict(W1)
for k in ['RVX_compress_MOVE_spike','RVX_realized_up_frontend_down']: W[k]=NEW[k][1:]
report('+ two risk-on only',W)
W=dict(W1); W['RVX_premium_collapse']=(15,-10); W['RVX_month_and_week']=(15,-6); report('+ premium_collapse + month_and_week',W)
W=dict(W1); W['RVX_premium_collapse']=(15,-8); report('+ premium_collapse at -8',W)
W=dict(W1); W['RVX_premium_collapse']=(21,-10); report('+ premium_collapse, 21d window',W)
W=dict(W1); W['RVX_premium_collapse']=(15,-12); report('+ premium_collapse at -12',W)
