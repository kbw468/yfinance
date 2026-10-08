import pandas as pd, numpy as np, warnings; warnings.filterwarnings('ignore')
pd.set_option('display.width',320)
exec(open('score2.py').read().split("V={}")[0])
R_=R['SPY']; RV10=R_.rolling(10).std(); RV21=R_.rolling(21).std(); RV63=R_.rolling(63).std()
def z(x,w=252): return (x-x.rolling(w).mean())/x.rolling(w).std()
z1021=z(np.log(RV10/RV21)); z2163=z(np.log(RV21/RV63))
C1n={k:v for k,v in C1.items() if k!='at_highs'}
# qualified signal set
SIGQ=dict(SIG)
SIGQ['ONCONF_collapse']=SIG['ONCONF_collapse']&(z1021>=0)                       # (1) flip confirmation only counts if short realized had expanded
SIGQ['FAIL_yieldsup_q']=SIG['FAIL_yieldsup']&(z2163<0)                          # (2) heavier when quarterly realized compressed
SIGQ['ONSET_impulse_q']=SIG['ONSET_impulse']&(z2163<0)
SIGQ['CAP_alldims_q']=SIG['CAP_alldims']&(z2163<0)                              # (3) heavier capitulation when quarterly realized compressed
SIGQ['CAP_vix_gvz_q']=SIG['CAP_vix_gvz']&(z2163<0)
WQ=dict(W1); WQ['FAIL_yieldsup_q']=(15,-10); WQ['ONSET_impulse_q']=(15,-8); WQ['CAP_alldims_q']=(42,10); WQ['CAP_vix_gvz_q']=(42,8)
def buildq(W,ctxw,base=60):
    parts={}
    for k,(w,pts) in W.items(): parts[k]=recent(SIGQ[k],w).astype(float)*pts
    for k,pts in ctxw.items(): parts[k]=CTXS[k].fillna(False).astype(float)*pts
    CAPANYq=SIGQ['CAP_alldims']|SIGQ['CAP_vix_gvz']|SIGQ['CAP_vvixout']
    parts['drawdown_no_capitulation']=((dd63<=-0.07)&~recent(CAPANYq,42)).fillna(False).astype(float)*-10
    PARTS=pd.DataFrame(parts); return (base+PARTS.sum(axis=1)).clip(0,100),PARTS
def states2(S,oi=20,oe=35):
    v=S.values; st=np.ones(len(v),dtype=int); s=1
    for i in range(len(v)):
        x=v[i]
        if np.isnan(x): st[i]=s; continue
        if s==1 and x<=oi: s=0
        elif s==0 and x>oe: s=1
        st[i]=s
    return pd.Series(st,index=S.index)
r=R['SPY']; m=~EX&(idx>=pd.Timestamp('2008-01-01')); bh=r[m]
def stats(x):
    c=x.cumsum(); dd=c-c.cummax(); return f"ann {x.mean()*252*100:5.2f} vol {x.std()*np.sqrt(252)*100:5.2f} sh {x.mean()/x.std()*np.sqrt(252):.2f} maxDD {dd.min()*100:6.1f} ulcer {np.sqrt((dd**2).mean())*100:5.2f}"
EP=[('2008-05-19','2009-03-09'),('2010-04-23','2010-07-02'),('2011-04-29','2011-10-03'),('2015-07-20','2015-08-25'),('2015-11-03','2016-02-11'),('2018-01-26','2018-02-08'),('2018-09-20','2018-12-24'),('2022-01-03','2022-10-12'),('2023-07-31','2023-10-27'),('2024-07-16','2024-08-05'),('2025-02-19','2025-04-08'),('2026-01-27','2026-03-30')]
def report(name,S):
    ST=states2(S); w=ST.shift(1).map({0:0,1:1}); x=(w*r)[m]
    o=(ST==0)&m; spells=0; last=None
    for dt in idx[o]:
        if last is None or (idx.get_loc(dt)-idx.get_loc(last))>1: spells+=1
        last=dt
    caps=[]
    for a,b in EP:
        a=pd.Timestamp(a); b=pd.Timestamp(b); seg=r.loc[a:b].iloc[1:]; sp=seg.sum(); stt=(w.loc[seg.index]*seg).sum()
        q=idx.get_loc(b); rec=r.iloc[q+1:q+43]; sp2=rec.sum(); st2=(w.loc[rec.index]*rec).sum(); caps.append(f"{str(a.date())[:7]}:{stt/sp:.2f}/{st2/sp2:.2f}")
    ok=L['ok21']&m; out=(ST==0)[ok]
    print(f'{name:34s} {stats(x)} | in {w[m].mean():.2f} spells {spells} | OUT: P5 {L.loc[ok,"riskoff21"][out].mean():.2f} P10/63 {(L.loc[ok,"fwdDD63"]<=-0.10)[out].mean():.2f} fwd63 {L.loc[ok,"fwd63"][out].mean()*100:5.2f} | today {int(S.iloc[-1])}')
    print('   ',' '.join(caps))
    yr=pd.DataFrame({'SPY':bh,'book':x}); Y=(yr.groupby(yr.index.year).sum()*100).round(1); return Y
print('B&H'.ljust(34),stats(bh))
S0,_=build(W1,C1n); Y0=report('CURRENT stay-in (no qualifiers)',S0)
# apply each qualifier alone and together
W_a=dict(W1); Sa,_=buildq(W_a,C1n); report('(1) flip-confirm needs RV10/21>=0',Sa)
WQ2=dict(W1); WQ2.update({'FAIL_yieldsup_q':(15,-10),'ONSET_impulse_q':(15,-8)}); SIGQ_b=dict(SIGQ); SIGQ_b['ONCONF_collapse']=SIG['ONCONF_collapse']
SIGQ_save=dict(SIGQ); SIGQ.update({'ONCONF_collapse':SIG['ONCONF_collapse']}); Sb,_=buildq(WQ2,C1n); report('(2) fail/impulse heavier if RV21/63<0',Sb); SIGQ.update(SIGQ_save)
WQ3=dict(W1); WQ3.update({'CAP_alldims_q':(42,10),'CAP_vix_gvz_q':(42,8)}); SIGQ.update({'ONCONF_collapse':SIG['ONCONF_collapse']}); Sc,_=buildq(WQ3,C1n); report('(3) capitulation heavier if RV21/63<0',Sc); SIGQ.update(SIGQ_save)
Sq,PQ=buildq(WQ,C1n); Yq=report('(1)+(2)+(3) together',Sq)
print('\nyearly book, current vs qualified:'); print(pd.concat({'SPY':Y0.SPY,'current':Y0.book,'qualified':Yq.book},axis=1).T.to_string())
pd.to_pickle({'Sq':Sq,'PQ':PQ},'SCORE_Q.pkl')
