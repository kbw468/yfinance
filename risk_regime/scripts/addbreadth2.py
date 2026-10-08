# second pass: the three interaction cells breadth.py found that were not yet tried in the dial, as rules and as contexts,
# plus a 63-day (quarter) ratio ROC for the slow-narrowing story
import pandas as pd, numpy as np, warnings; warnings.filterwarnings('ignore')
pd.set_option('display.width',320)
exec(open('score2.py').read().split("V={}")[0])
S=pd.read_pickle('S.pkl'); B=pd.read_pickle('BREADTH.pkl'); F=B['F']; BR=B['BR']; RAT=B['RAT']
_rv21=R['SPY'].rolling(21).std(); _zz=lambda x:(x-x.rolling(252).mean())/x.rolling(252).std(); _vrp5=_zz((S['VIX']-_rv21*np.sqrt(252)*100).diff(5))
SIG['RVX_premium_collapse']=(_vrp5<-1)&(z('VIX_roc21_z252')>1)
W0=dict(W1); W0['RVX_premium_collapse']=(15,-8)
C1n={k:v for k,v in C1.items() if k!='at_highs'}
spyz=G['SPY_ret21']/G['SPY_ret21'].rolling(252).std()
sect=BR['sector EW breadth roc21z (median of 9)']
r63={k:_zz(RAT[k].diff(63)) for k in ('RSP/SPY','QQQE/QQQ')}
ok=L['ok21']&~EX
# --- 63d ratio ROC: signature at peaks and quintiles
OFFS=list(range(-10,11))
for k,s_ in r63.items():
    for nm,anch in (('PEAKS',OFF.peak.tolist()),('TROUGHS',ON.trough.tolist())):
        pos=[idx.get_loc(a) for a in anch if not np.isnan(s_.iloc[idx.get_loc(a)])]
        M=np.array([[s_.iloc[p+q] if 0<=p+q<len(s_) else np.nan for q in OFFS] for p in pos])
        print(f'{k} roc63z at {nm} (n={len(pos)}):',' '.join(f'{v:5.2f}' for v in np.nanmedian(M,axis=0)))
    m_=s_.notna()&ok; q=pd.qcut(s_[m_].rank(method='first'),5,labels=False); y=L.loc[m_,'riskoff21']; f=L.loc[m_,'fwd21']*100
    print(f'{k} roc63z quintiles n={m_.sum()} P: '+' '.join(f'{y[q==i].mean():.3f}' for i in range(5))+'  fwd21: '+' '.join(f'{f[q==i].mean():5.2f}' for i in range(5)))
    # 63d bleed x yields / VIX
    for gk,gl in (('TNX_chg21_z252','TNX chg21z'),('VIX_roc21_z252','VIX roc21z'),('VIX_roc5_z252','VIX roc5z')):
        g=G[gk]; mm=m_&g.notna()
        for lo,hi,lab in ((-9,-1,'<-1'),(1,9,'>1')):
            c=mm&(s_<-1)&(g>lo)&(g<=hi); c2=mm&(s_>1)&(g>lo)&(g<=hi)
            print(f'   roc63z<-1 & {gl} {lab:4s}: n={c.sum():4d} P {L.loc[c,"riskoff21"].mean():.2f} fwd21 {L.loc[c,"fwd21"].mean()*100:5.2f} | roc63z>1 & {gl} {lab:4s}: n={c2.sum():4d} P {L.loc[c2,"riskoff21"].mean():.2f} fwd21 {L.loc[c2,"fwd21"].mean()*100:5.2f}')
print('today roc63z:',{k:round(float(s_.iloc[-1]),2) for k,s_ in r63.items()})
NEW={
 'BR_rsp_bleed_yields_15':   ((F[('RSP/SPY','roc21z')]<-1)&(z('TNX_chg21_z252')>1),15,-8),
 'BR_rsp_bleed_yields_21':   ((F[('RSP/SPY','roc21z')]<-1)&(z('TNX_chg21_z252')>1),21,-10),
 'BR_rsp_bleed63_yields':    ((r63['RSP/SPY']<-1)&(z('TNX_chg21_z252')>1),21,-8),
 'BR_ew_rotation_in_decline':((F[('RSP/SPY','roc21z')]>0)&(spyz<-1),15,-8),
 'BR_ew_rotation_in_decline_strong':((F[('RSP/SPY','roc21z')]>1)&(spyz<-1),15,-10),
 'BR_sector_bleed_vixflat':  ((sect<-1)&(z('VIX_roc5_z252')>-1)&(z('VIX_roc5_z252')<=0),10,-5),
}
for k,(s_,w,p) in NEW.items(): SIG[k]=s_
CTXS['ctx_ew_rotation_in_decline']=(F[('RSP/SPY','roc21z')]>0)&(spyz<-1)
CTXS['ctx_rsp_bleed_yields']=(F[('RSP/SPY','roc21z')]<-1)&(z('TNX_chg21_z252')>1)
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
def report(name,W,C):
    S_,_=build(W,C); ST=states2(S_); w=ST.shift(1).map({0:0,1:1}); x=(w*r)[m]
    o=(ST==0)&m; spells=0; last=None
    for dt in idx[o]:
        if last is None or (idx.get_loc(dt)-idx.get_loc(last))>1: spells+=1
        last=dt
    caps=[]
    for a,b in EP:
        a=pd.Timestamp(a); b=pd.Timestamp(b); seg=r.loc[a:b].iloc[1:]; sp=seg.sum(); stt=(w.loc[seg.index]*seg).sum()
        q=idx.get_loc(b); rec=r.iloc[q+1:q+43]; sp2=rec.sum(); st2=(w.loc[rec.index]*rec).sum(); caps.append(f"{str(a.date())[2:7]}:{stt/sp:.2f}/{st2/sp2:.2f}")
    print(f'{name:36s} {stats(x)} | in {w[m].mean():.2f} spells {spells:2d} | today {int(S_.iloc[-1])}'); print('   ',' '.join(caps))
print('\nfirst-fire stats (ok21, ex-2020):')
for k,(s_,w,p) in NEW.items():
    f=s_.fillna(False)&~s_.fillna(False).shift(1,fill_value=False)&ok
    print(f'  {k:34s} n={f.sum():4d} P5/21 {L.loc[f,"riskoff21"].mean():.2f} fwd10 {L.loc[f,"fwd10"].mean()*100:5.2f} fwd21 {L.loc[f,"fwd21"].mean()*100:5.2f} DD21 {L.loc[f,"fwdDD21"].mean()*100:5.2f} DD63 {L.loc[f,"fwdDD63"].mean()*100:5.2f}')
for k in ('ctx_ew_rotation_in_decline','ctx_rsp_bleed_yields'):
    c=CTXS[k].fillna(False)&ok; print(f'  {k:34s} days={c.sum():4d} P5/21 {L.loc[c,"riskoff21"].mean():.2f} fwd21 {L.loc[c,"fwd21"].mean()*100:5.2f} DD21 {L.loc[c,"fwdDD21"].mean()*100:5.2f}')
print('\nB&H'.ljust(37),stats(bh))
report('CURRENT (with premium collapse)',W0,C1n)
for k,(s_,w,p) in NEW.items():
    W=dict(W0); W[k]=(w,p); report(f'+ rule {k}',W,C1n)
for k,p in (('ctx_ew_rotation_in_decline',-5),('ctx_ew_rotation_in_decline',-8),('ctx_rsp_bleed_yields',-5),('ctx_rsp_bleed_yields',-8)):
    C=dict(C1n); C[k]=p; report(f'+ ctx {k} {p}',W0,C)
# both interaction rules together
W=dict(W0); W['BR_rsp_bleed_yields_15']=(15,-8); W['BR_ew_rotation_in_decline']=(15,-8); report('+ both interaction rules',W,C1n)
