import pandas as pd, numpy as np, warnings; warnings.filterwarnings('ignore')
pd.set_option('display.width',320); pd.set_option('display.max_rows',200)
exec(open('score2.py').read().split("V={}")[0])
r=R['SPY']; m=~EX&(idx>=pd.Timestamp('2008-01-01')); bh=r[m]
def stats(x):
    c=x.cumsum(); dd=c-c.cummax(); return dict(ann=round(float(x.mean()*252*100),2),vol=round(float(x.std()*np.sqrt(252)*100),2),sharpe=round(float(x.mean()/x.std()*np.sqrt(252)),2),maxDD=round(float(dd.min()*100),1),ulcer=round(float(np.sqrt((dd**2).mean())*100),2))
EP=[('2008-05-19','2009-03-09'),('2010-04-23','2010-07-02'),('2011-04-29','2011-10-03'),('2015-07-20','2015-08-25'),('2015-11-03','2016-02-11'),('2018-01-26','2018-02-08'),('2018-09-20','2018-12-24'),('2022-01-03','2022-10-12'),('2023-07-31','2023-10-27'),('2024-07-16','2024-08-05'),('2025-02-19','2025-04-08'),('2026-01-27','2026-03-30')]
def report(name,out):   # out: boolean series, True = out of market that day (applied next day)
    w=(~out).shift(1).fillna(True).astype(float); x=(w*r)[m]
    st=stats(x); exp=round(float(w[m].mean()),2); spells=int(((out&~out.shift(1,fill_value=False))[m]).sum()); per_yr=round(spells/(m.sum()/252),1)
    rows=[]
    for a,b in EP:
        a=pd.Timestamp(a); b=pd.Timestamp(b); seg=r.loc[a:b].iloc[1:]; sp=seg.sum()*100; stt=(w.loc[seg.index]*seg).sum()*100
        q=idx.get_loc(b); rec=r.iloc[q+1:q+43]; sp2=rec.sum()*100; st2=(w.loc[rec.index]*rec).sum()*100
        rows.append(f"{str(a.date())[:7]}:{stt/sp:.2f}/{st2/sp2:.2f}")
    print(f"{name:52s} ann {st['ann']:5.2f} vol {st['vol']:5.2f} sh {st['sharpe']:.2f} maxDD {st['maxDD']:6.1f} ulcer {st['ulcer']:5.2f} | in {exp:.2f} spells/yr {per_yr:4.1f}")
    print('    loss/rebound captured: '+' '.join(rows))
    return w
print('B&H'.ljust(52),stats(bh))
def recent(mm,n): return mm.fillna(False).astype(int).rolling(n,min_periods=1).max().astype(bool)
ONP=SIG['CAP_alldims']|SIG['CAP_vix_gvz']|SIG['CAP_vvixout']|SIG['ONCONF_collapse']|SIG['VVIX_confirms_spike']
# --- A: current state book for reference (dd-first weights, no softener)
C1n={k:v for k,v in C1.items() if k!='at_highs'}; W10={k:(int(w*1.5),p) if p<0 else (max(10,w//2),p) for k,(w,p) in W1.items()}
S10,_=build(W10,C1n)
def states_h(S,oi,oe,ni,ne):
    v=S.values; st=np.zeros(len(v),dtype=int); s=2
    for i in range(len(v)):
        x=v[i]
        if np.isnan(x): st[i]=s; continue
        if s==0: s=2 if x>ni else (1 if x>oe else 0)
        elif s==1: s=0 if x<=oi else (2 if x>ni else 1)
        else: s=0 if x<=oi else (1 if x<=ne else 2)
        st[i]=s
    return pd.Series(st,index=S.index)
report('CURRENT state book (OUT<=30 exit>45)',states_h(S10,30,45,60,50)==0)
# --- B: balanced score, OUT only at a deep reading
S1,_=build(W1,C1n)
for oi,oe in [(20,35),(15,30),(10,25)]:
    report(f'balanced dial, OUT<={oi} exit>{oe}',states_h(S1,oi,oe,60,50)==0)
# --- C: event-driven OUT. fires: high-conviction onset configs. out for N sessions, early re-entry on any ON print
HI={'ONSET_vvixlag':SIG['ONSET_vvixlag'],'ONSET_movelag':SIG['ONSET_movelag'],'CONT_2ndleg':SIG['CONT_2ndleg'],'FAIL_yieldsup':SIG['FAIL_yieldsup'],'HYG_credit_crack':SIG['HYG_credit_crack'],
    'SPY_realized_outruns_implied':SIG['SPY_realized_outruns_implied'],'QQQ_realized_outruns_VXN':SIG['QQQ_realized_outruns_VXN'],'KRE_banks_vs_yields':SIG['KRE_banks_vs_yields']}
def event_out(fires,N,reentry=True,price_conf=None,need2=False):
    f=fires.fillna(False)
    if price_conf is not None: f=f&(G['SPY_dd63']<=price_conf)
    if need2: f=f&(recent(fires,10).astype(int).rolling(10).sum()>=2)   # crude: fire count in window
    out=pd.Series(False,index=idx); cnt=0
    fv=f.values; onv=ONP.fillna(False).values
    for i in range(len(idx)):
        if fv[i]: cnt=N
        if reentry and onv[i] and not fv[i]: cnt=0
        out.iloc[i]=cnt>0
        if cnt>0: cnt-=1
    return out
ALL=pd.concat(HI.values(),axis=1).any(axis=1)
CORE=SIG['ONSET_vvixlag']|SIG['CONT_2ndleg']|SIG['FAIL_yieldsup']|SIG['HYG_credit_crack']
for N in (5,10,15,21):
    report(f'EVENT: any of 8 high-conviction fires, out {N}d, re-entry on ON print',event_out(ALL,N))
for N in (10,15,21):
    report(f'EVENT: core 4 (vvixlag, 2ndleg, fail, credit crack), out {N}d',event_out(CORE,N))
for N in (10,15):
    report(f'EVENT: any of 8, out {N}d, only if SPY already <=-2% off high',event_out(ALL,N,price_conf=-0.02))
    report(f'EVENT: any of 8, out {N}d, only if SPY already <=-3% off high',event_out(ALL,N,price_conf=-0.03))
for N in (10,15):
    report(f'EVENT: core 4, out {N}d, no early re-entry',event_out(CORE,N,reentry=False))
# --- D: price-break only baseline for comparison: out 10d when SPY makes a -3% break from 63d high (no vol info)
brk=(G['SPY_dd63']<=-0.03)&(G['SPY_dd63'].shift(1)>-0.03)
report('PRICE ONLY: out 10d after a -3% break from 63d high',event_out(brk,10,reentry=False))
# state-level stats for the best few will be printed after choice
