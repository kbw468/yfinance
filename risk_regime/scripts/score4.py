exec(open("score2.py").read().split("V={}")[0])
def stats(x):
    c=x.cumsum(); dd=(c-c.cummax()); ulcer=np.sqrt((dd**2).mean())
    return dict(ann=round(x.mean()*252*100,2),vol=round(x.std()*np.sqrt(252)*100,2),sharpe=round(x.mean()/x.std()*np.sqrt(252),2),maxDD=round(dd.min()*100,1),ulcer=round(ulcer*100,2),calmar=round((x.mean()*252)/(-dd.min()),2))
EP=[('2008-05-19','2009-03-09'),('2010-04-23','2010-07-02'),('2011-04-29','2011-10-03'),('2015-07-20','2015-08-25'),('2015-11-03','2016-02-11'),('2018-01-26','2018-02-08'),('2018-09-20','2018-12-24'),('2022-01-03','2022-10-12'),('2023-07-31','2023-10-27'),('2024-07-16','2024-08-05'),('2025-02-19','2025-04-08'),('2026-01-27','2026-03-30')]
V={}
W6={k:(int(w*1.5) if p<0 else w, p) for k,(w,p) in W1.items()}
W10={k:(w, p) if p<0 else (max(10,w//2), p) for k,(w,p) in W6.items()}   # risk-on adds persist half as long
V['V8']=build(W6,C1,dd_cap=30,reentry=4)
V['V10 V8 + shorter risk-on windows']=build(W10,C1,dd_cap=30,reentry=4)
V['V11 V10 + cap25']=build(W10,C1,dd_cap=25,reentry=4)
r=R['SPY']; m=~EX&(idx>=pd.Timestamp('2008-01-01')); bh=r[m]
rows=[]
for k,(s,_) in V.items():
    w=(s.shift(1)/100)[m]; rows.append({'variant':k,**stats(w*bh),'avg_exp':round(w.mean(),2)})
print(pd.DataFrame(rows).set_index('variant').to_string())
for k,(s,_) in V.items():
    w=s.shift(1)/100; rows=[]
    for a,b in EP:
        a=pd.Timestamp(a); b=pd.Timestamp(b); seg=r.loc[a:b].iloc[1:]; st=(w.loc[seg.index]*seg).sum()*100; sp=seg.sum()*100
        q=idx.get_loc(b); rec=r.iloc[q+1:q+43]; st2=(w.loc[rec.index]*rec).sum()*100; sp2=rec.sum()*100
        rows.append({'peak':a.date(),'SPY%':round(sp,1),'strat%':round(st,1),'capt_loss':round(st/sp,2),'capt_gain':round(st2/sp2,2),'d@pk':int(s.loc[a]),'d@tr':int(s.loc[b]),'d+21':int(s.iloc[q+21])})
    T=pd.DataFrame(rows); print(f'\n--- {k} --- median capt_loss {T.capt_loss.median():.2f} median capt_gain {T.capt_gain.median():.2f}'); print(T.to_string())
print('\nTODAY:',{k:int(s.iloc[-1]) for k,(s,_) in V.items()})
