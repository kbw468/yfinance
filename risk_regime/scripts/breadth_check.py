import pandas as pd, numpy as np, warnings; warnings.filterwarnings('ignore')
pd.set_option('display.width',320)
exec(open('score2.py').read().split("V={}")[0])
S=pd.read_pickle('S.pkl'); B=pd.read_pickle('BREADTH.pkl'); F=B['F']; BR=B['BR']; RAT=B['RAT']
_rv21=R['SPY'].rolling(21).std(); _zz=lambda x:(x-x.rolling(252).mean())/x.rolling(252).std(); _vrp5=_zz((S['VIX']-_rv21*np.sqrt(252)*100).diff(5))
SIG['RVX_premium_collapse']=(_vrp5<-1)&(z('VIX_roc21_z252')>1)
W0=dict(W1); W0['RVX_premium_collapse']=(15,-8); C1n={k:v for k,v in C1.items() if k!='at_highs'}
ok=L['ok21']&~EX; r63=_zz(RAT['RSP/SPY'].diff(63)); tnx=z('TNX_chg21_z252'); r21=F[('RSP/SPY','roc21z')]
print('TODAY: RSP/SPY roc21z %.2f roc63z %.2f | TNX chg21z %.2f TYX chg21z %.2f | SPY ret21 z %.2f'%(r21.iloc[-1],r63.iloc[-1],tnx.iloc[-1],z('TYX_chg21_z252').iloc[-1],(G['SPY_ret21']/G['SPY_ret21'].rolling(252).std()).iloc[-1]))
for nm,cond in (('21d bleed & yields up',(r21<-1)&(tnx>1)),('63d bleed & yields up',(r63<-1)&(tnx>1))):
    c=cond.fillna(False)
    print(f'\n{nm}: live today={bool(c.iloc[-1])}; last true day={c[c].index[-1].date() if c.any() else None}')
    f=c&~c.shift(1,fill_value=False)&ok
    for era,(a,b) in {'2008-17':('2008-01-01','2017-12-31'),'2018-26':('2018-01-01','2026-12-31'),'all':('2006-01-01','2026-12-31')}.items():
        mm=f&(idx>=pd.Timestamp(a))&(idx<=pd.Timestamp(b)); d=c&ok&(idx>=pd.Timestamp(a))&(idx<=pd.Timestamp(b))
        print(f'  {era}: first-fires n={mm.sum():3d} P {L.loc[mm,"riskoff21"].mean():.2f} fwd21 {L.loc[mm,"fwd21"].mean()*100:5.2f} DD21 {L.loc[mm,"fwdDD21"].mean()*100:5.2f} | all days n={d.sum():3d} P {L.loc[d,"riskoff21"].mean():.2f} fwd21 {L.loc[d,"fwd21"].mean()*100:5.2f}')
    print('  first-fire dates:',' '.join(str(d.date()) for d in idx[f]))
    print('  by year, days true:',c[ok].groupby(c[ok].index.year).sum().to_dict())
# yearly diff of the best candidate vs current, and its extra OUT spells
def states2(S_,oi=20,oe=35):
    v=S_.values; st=np.ones(len(v),dtype=int); s=1
    for i in range(len(v)):
        x=v[i]
        if np.isnan(x): st[i]=s; continue
        if s==1 and x<=oi: s=0
        elif s==0 and x>oe: s=1
        st[i]=s
    return pd.Series(st,index=S_.index)
r=R['SPY']; m=~EX&(idx>=pd.Timestamp('2008-01-01'))
SIG['BR_rsp_bleed_yields']=(r21<-1)&(tnx>1); W=dict(W0); W['BR_rsp_bleed_yields']=(15,-8)
S0,_=build(W0,C1n); S1,_=build(W,C1n); ST0=states2(S0); ST1=states2(S1)
b0=(ST0.shift(1)*r)[m]; b1=(ST1.shift(1)*r)[m]
y=pd.DataFrame({'current':b0.groupby(b0.index.year).sum()*100,'with_breadth_rule':b1.groupby(b1.index.year).sum()*100}); y['diff']=y.with_breadth_rule-y.current
print('\nyearly diff (pts):'); print(y.round(1).to_string())
def spells(ST):
    o=(ST==0)&m; out=[]; start=None; prev=None
    for dt in idx[o]:
        if start is None or (idx.get_loc(dt)-idx.get_loc(prev))>1:
            if start is not None: out.append((start,prev))
            start=dt
        prev=dt
    if start is not None: out.append((start,prev))
    return out
sp0=spells(ST0); sp1=spells(ST1)
print('\nOUT spells current:',len(sp0),' with rule:',len(sp1))
s0={a for a,b in sp0}
for a,b in sp1:
    seg=r.loc[a:b]; tag='' if a in s0 else '  <-- NEW/CHANGED'
    print(f'  {a.date()} -> {b.date()} ({len(seg)}d) SPY {seg.sum()*100:6.1f}%{tag}')
print('\nspells in current not in rule version:')
s1={a for a,b in sp1}
for a,b in sp0:
    if a not in s1: seg=r.loc[a:b]; print(f'  {a.date()} -> {b.date()} ({len(seg)}d) SPY {seg.sum()*100:6.1f}%')
