# Breadth rules tested as additions to the stay-in dial (first pass). Research script, not in the run_all chain; run from risk_regime/work with scripts/score2.py copied alongside or cwd set so score2.py resolves.
import pandas as pd, numpy as np, warnings; warnings.filterwarnings('ignore')
pd.set_option('display.width',320)
exec(open('score2.py').read().split("V={}")[0])
S=pd.read_pickle('S.pkl'); B=pd.read_pickle('BREADTH.pkl'); F=B['F']; BR=B['BR']
_rv21=R['SPY'].rolling(21).std(); _zz=lambda x:(x-x.rolling(252).mean())/x.rolling(252).std(); _vrp5=_zz((S['VIX']-_rv21*np.sqrt(252)*100).diff(5))
SIG['RVX_premium_collapse']=(_vrp5<-1)&(z('VIX_roc21_z252')>1)
W0=dict(W1); W0['RVX_premium_collapse']=(15,-8)
C1n={k:v for k,v in C1.items() if k!='at_highs'}
NEW={
 'BR_rsp_bleed_vixspike':   ((F[('RSP/SPY','roc5z')]<-1)&(z('VIX_roc5_z252')>1),15,-8),
 'BR_rsp_bleed_month_vixspike':((F[('RSP/SPY','roc21z')]<-1)&(z('VIX_roc5_z252')>1),15,-8),
 'BR_qqqe_bleed_month':     (F[('QQQE/QQQ','roc21z')]>1,10,-6),   # note sign checked below: QQQE/QQQ roc21z top quintile P 0.242
 'BR_qqqe_bleed_month_neg': (F[('QQQE/QQQ','roc21z')]<-1,10,-6),
 'BR_utilities_ew_bid':     (F[('RSPU/XLU','roc21z')]>1,10,-5),
 'BR_sector_share_up':      (BR['share z']>1,10,-5),
 'BR_rsp_accel_after_dd':   ((F[('RSP/SPY','accel5z')]>1)&(dd63<=-0.05),10,+8),
 'BR_rsp_strong_vixcollapse':((F[('RSP/SPY','roc5z')]>1)&(z('VIX_roc5_z252')<-1),10,+6),
}
for k,(s_,w,p) in NEW.items(): SIG[k]=s_
r=R['SPY']; m=~EX&(idx>=pd.Timestamp('2008-01-01')); bh=r[m]
def stats(x):
    c=x.cumsum(); dd=c-c.cummax(); return f"ann {x.mean()*252*100:5.2f} vol {x.std()*np.sqrt(252)*100:5.2f} sh {x.mean()/x.std()*np.sqrt(252):.2f} maxDD {dd.min()*100:6.1f} ulcer {np.sqrt((dd**2).mean())*100:5.2f}"
def states2(S_,oi=20,oe=35):
    v=S_.values; st=np.ones(len(v),dtype=int); s=1
    for i in range(len(v)):
        x=v[i]
        if np.isnan(x): st[i]=s; continue
        if s==1 and x<=oi: s=0
        elif s==0 and x>oe: s=1
        st[i]=s
    return pd.Series(st,index=S_.index)
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
    print(f'{name:34s} {stats(x)} | in {w[m].mean():.2f} spells {spells:2d} | today {int(S_.iloc[-1])}'); print('   ',' '.join(caps))
# first-fire stats for each breadth rule
ok=L['ok21']&m
print('first-fire stats:')
for k,(s_,w,p) in NEW.items():
    f=s_.fillna(False)&~s_.fillna(False).shift(1,fill_value=False)&ok
    print(f'  {k:30s} n={f.sum():4d} P5/21 {L.loc[f,"riskoff21"].mean():.2f} fwd21 {L.loc[f,"fwd21"].mean()*100:5.2f} DD21 {L.loc[f,"fwdDD21"].mean()*100:5.2f}')
print('B&H'.ljust(34),stats(bh))
report('CURRENT (with premium collapse)',W0)
for k,(s_,w,p) in NEW.items():
    W=dict(W0); W[k]=(w,p); report(f'+ {k}',W)
