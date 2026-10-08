import pandas as pd, numpy as np, warnings; warnings.filterwarnings('ignore')
pd.set_option('display.width',320); pd.set_option('display.max_rows',200)
exec(open('score2.py').read().split("V={}")[0])
V1,_=build(W1,C1)                                   # balanced component score
W10={k:(int(w*1.5),p) if p<0 else (max(10,w//2),p) for k,(w,p) in W1.items()}
V10,_=build(W10,C1,dd_cap=30,reentry=4)              # drawdown-first (current)
V10b,_=build(W10,C1)                                 # drawdown-first weights, no cap, no slow re-entry
r=R['SPY']; m=~EX&(idx>=pd.Timestamp('2008-01-01')); bh=r[m]
def stats(x):
    c=x.cumsum(); dd=c-c.cummax(); return dict(ann=round(x.mean()*252*100,2),vol=round(x.std()*np.sqrt(252)*100,2),sharpe=round(x.mean()/x.std()*np.sqrt(252),2),maxDD=round(dd.min()*100,1),ulcer=round(np.sqrt((dd**2).mean())*100,2))
EP=[('2008-05-19','2009-03-09'),('2010-04-23','2010-07-02'),('2011-04-29','2011-10-03'),('2015-07-20','2015-08-25'),('2015-11-03','2016-02-11'),('2018-01-26','2018-02-08'),('2018-09-20','2018-12-24'),('2022-01-03','2022-10-12'),('2023-07-31','2023-10-27'),('2024-07-16','2024-08-05'),('2025-02-19','2025-04-08'),('2026-01-27','2026-03-30')]
print('B&H',stats(bh))
def states(S,out,red):   # 0 = GET OUT (<=out), 1 = REDUCE (out<s<=red), 2 = RISK ON (>red)
    return pd.Series(np.where(S<=out,0,np.where(S<=red,1,2)),index=S.index)
def evaluate(name,S,out,red,red_exp=0.5):
    st=states(S,out,red); ok=L['ok21']&m
    print(f'\n==== {name}: GET OUT <= {out}, REDUCE {out+1}-{red}, RISK ON > {red} ====')
    d=pd.DataFrame({'st':st,'off':L['riskoff21'],'f21':L['fwd21']*100,'f63':L['fwd63']*100,'dd21':L['fwdDD21']*100,'dd63':L['fwdDD63']*100,'dd10_63':(L['fwdDD63']<=-0.10).astype(float)})[ok]
    T=d.groupby('st').agg(share=('off','count'),P_off21=('off','mean'),fwd21=('f21','mean'),fwd63=('f63','mean'),DD21=('dd21','mean'),DD63=('dd63','mean'),P_10pct_63=('dd10_63','mean'),f63_p5=('f63',lambda x: x.quantile(.05)))
    T['share']=(T['share']/len(d)).round(3); T.index=['GET OUT','REDUCE','RISK ON']; print(T.round(2).to_string())
    stl=st.shift(1)[m]
    for lab,expo in [('binary: 100% unless GET OUT',{0:0,1:1,2:1}),(f'3-state: 100/{int(red_exp*100)}/0',{0:0,1:red_exp,2:1})]:
        w=stl.map(expo); x=w*bh; print(f'{lab:32s}',stats(x),'avg exp',round(w.mean(),2))
    # episodes for binary
    w=st.shift(1).map({0:0,1:1,2:1}); rows=[]
    for a,b in EP:
        a=pd.Timestamp(a); b=pd.Timestamp(b); seg=r.loc[a:b].iloc[1:]; sp=seg.sum()*100; stt=(w.loc[seg.index]*seg).sum()*100
        q=idx.get_loc(b); rec=r.iloc[q+1:q+43]; sp2=rec.sum()*100; st2=(w.loc[rec.index]*rec).sum()*100
        # days from peak until state first hit GET OUT
        seq=st.iloc[idx.get_loc(a):q+1].values; k=np.where(seq==0)[0]; dout=int(k[0]) if len(k) else None
        rows.append({'peak':a.date(),'SPY':round(sp,1),'binary':round(stt,1),'capt':round(stt/sp,2),'reb_SPY':round(sp2,1),'reb_bin':round(st2,1),'capt_g':round(st2/sp2,2),'days_to_OUT':dout,'state@pk':['OUT','RED','ON'][st.loc[a]],'state@tr':['OUT','RED','ON'][st.loc[b]]})
    E=pd.DataFrame(rows); print(E.to_string()); print('median capt',E.capt.median(),' median capt_g',E.capt_g.median())
    # transitions: fwd after entering each state
    ent=(st!=st.shift(1))&m&L['ok21']
    for s_,nm in [(0,'into GET OUT'),(2,'into RISK ON')]:
        e=ent&(st==s_); print(f'{nm}: n={e.sum()} fwd21 {L.loc[e,"fwd21"].mean()*100:.2f}% fwd63 {L.loc[e,"fwd63"].mean()*100:.2f}% P(5%DD21) {L.loc[e,"riskoff21"].mean():.2f} medDD21 {L.loc[e,"fwdDD21"].median()*100:.2f}%')
    # churn: number of state changes per year, avg days in GET OUT spells
    ch=(st!=st.shift(1))[m].sum()/ (m.sum()/252); print('state changes per year',round(ch,1))
for nm,S in [('balanced V1',V1),('drawdown-first V10',V10),('dd-first weights, no cap/slow (V10b)',V10b)]:
    for out,red in [(30,55),(35,55),(30,60),(40,60)]:
        evaluate(f'{nm}',S,out,red)
