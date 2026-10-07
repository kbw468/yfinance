exec(open('score2.py').read().split("V={}")[0])
def stats(x):
    c=x.cumsum(); dd=(c-c.cummax()); ulcer=np.sqrt((dd**2).mean())
    return dict(ann=round(x.mean()*252*100,2),vol=round(x.std()*np.sqrt(252)*100,2),sharpe=round(x.mean()/x.std()*np.sqrt(252),2),maxDD=round(dd.min()*100,1),ulcer=round(ulcer*100,2),calmar=round((x.mean()*252)/(-dd.min()),2))
EP=[('2008-05-19','2009-03-09'),('2010-04-23','2010-07-02'),('2011-04-29','2011-10-03'),('2015-07-20','2015-08-25'),('2015-11-03','2016-02-11'),('2018-01-26','2018-02-08'),('2018-09-20','2018-12-24'),('2022-01-03','2022-10-12'),('2023-07-31','2023-10-27'),('2024-07-16','2024-08-05'),('2025-02-19','2025-04-08'),('2026-01-27','2026-03-30')]
V={}
V['V1 current']=build(W1,C1)
W6={k:(int(w*1.5) if p<0 else w, p) for k,(w,p) in W1.items()}   # same points, risk-off fires persist 50% longer
V['V6 W1 + cap30 in DD + reentry4']=build(W1,C1,dd_cap=30,reentry=4)
V['V7 W1 + cap20 in DD + reentry4']=build(W1,C1,dd_cap=20,reentry=4)
V['V8 longer windows + cap30 + reentry4']=build(W6,C1,dd_cap=30,reentry=4)
V['V9 longer windows + cap20 + reentry3']=build(W6,C1,dd_cap=20,reentry=3)
r=R['SPY']; m=~EX&(idx>=pd.Timestamp('2008-01-01')); bh=r[m]
rows=[{'variant':'SPY buy & hold',**stats(bh),'avg_exp':1.0}]
for k,(s,_) in V.items():
    w=(s.shift(1)/100)[m]; rows.append({'variant':k,**stats(w*bh),'avg_exp':round(w.mean(),2)})
print(pd.DataFrame(rows).set_index('variant').to_string())
for k,(s,_) in V.items():
    w=s.shift(1)/100; rows=[]
    for a,b in EP:
        a=pd.Timestamp(a); b=pd.Timestamp(b); seg=r.loc[a:b].iloc[1:]; st=(w.loc[seg.index]*seg).sum()*100; sp=seg.sum()*100
        q=idx.get_loc(b); rec=r.iloc[q+1:q+43]; st2=(w.loc[rec.index]*rec).sum()*100; sp2=rec.sum()*100
        rows.append({'peak':a.date(),'SPY%':round(sp,1),'strat%':round(st,1),'capt_loss':round(st/sp,2),'reb42 SPY%':round(sp2,1),'strat_r%':round(st2,1),'capt_gain':round(st2/sp2,2),'d@pk':int(s.loc[a]),'d@tr':int(s.loc[b]),'d+10':int(s.iloc[q+10]),'d+21':int(s.iloc[q+21])})
    T=pd.DataFrame(rows); print(f'\n--- {k} --- median capt_loss {T.capt_loss.median():.2f}  median capt_gain {T.capt_gain.median():.2f}'); print(T.to_string())
# yearly returns for V1 vs V8 vs B&H
print('\nyearly: ')
yr=pd.DataFrame({'SPY':bh,'V1':(V['V1 current'][0].shift(1)/100)[m]*bh,'V6':(V['V6 W1 + cap30 in DD + reentry4'][0].shift(1)/100)[m]*bh,'V8':(V['V8 longer windows + cap30 + reentry4'][0].shift(1)/100)[m]*bh})
print((yr.groupby(yr.index.year).sum()*100).round(1).to_string())
print('\nTODAY:',{k:int(s.iloc[-1]) for k,(s,_) in V.items()})
pd.to_pickle({k:s for k,(s,_) in V.items()},'SCORE_VARIANTS2.pkl')
