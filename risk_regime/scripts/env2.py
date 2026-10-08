import pandas as pd, numpy as np, warnings; warnings.filterwarnings('ignore')
pd.set_option('display.width',320); pd.set_option('display.max_rows',200)
exec(open('score2.py').read().split("V={}")[0])
VOL=pd.read_pickle('VOL.pkl')
# two volume qualifiers
SIG['HYG_volume_dry_in_spike']=(VOL[('HYG','v_roc5z')]<-1)&(z('VIX_roc5_z252')>1)
SIG['SPY_volume_climax']=(VOL[('SPY','v_rel21')]>1)&(z('VIX_roc5_z252')>1)&((G['SPY_ret5']/G['SPY_ret5'].rolling(252).std())<-1)
W10={k:(int(w*1.5),p) if p<0 else (max(10,w//2),p) for k,(w,p) in W1.items()}
W10v=dict(W10); W10v['HYG_volume_dry_in_spike']=(22,-8); W10v['SPY_volume_climax']=(10,5)
Sbase,_=build(W10,C1); Svol,_=build(W10v,C1); S1,_=build(W1,C1)
r=R['SPY']; m=~EX&(idx>=pd.Timestamp('2008-01-01')); bh=r[m]
def stats(x):
    c=x.cumsum(); dd=c-c.cummax(); return dict(ann=round(x.mean()*252*100,2),vol=round(x.std()*np.sqrt(252)*100,2),sharpe=round(x.mean()/x.std()*np.sqrt(252),2),maxDD=round(dd.min()*100,1),ulcer=round(np.sqrt((dd**2).mean())*100,2))
EP=[('2008-05-19','2009-03-09'),('2010-04-23','2010-07-02'),('2011-04-29','2011-10-03'),('2015-07-20','2015-08-25'),('2015-11-03','2016-02-11'),('2018-01-26','2018-02-08'),('2018-09-20','2018-12-24'),('2022-01-03','2022-10-12'),('2023-07-31','2023-10-27'),('2024-07-16','2024-08-05'),('2025-02-19','2025-04-08'),('2026-01-27','2026-03-30')]
def states_h(S,out_in,out_exit,on_in,on_exit):
    """hysteresis: enter OUT at <=out_in, leave OUT when >out_exit; enter ON at >on_in, leave ON when <=on_exit"""
    v=S.values; st=np.zeros(len(v),dtype=int); s=2
    for i in range(len(v)):
        x=v[i]
        if np.isnan(x): st[i]=s; continue
        if s==0:
            if x>on_in: s=2
            elif x>out_exit: s=1
        elif s==1:
            if x<=out_in: s=0
            elif x>on_in: s=2
        else:
            if x<=out_in: s=0
            elif x<=on_exit: s=1
        st[i]=s
    return pd.Series(st,index=S.index)
def evaluate(name,st,S):
    ok=L['ok21']&m
    d=pd.DataFrame({'st':st,'off':L['riskoff21'],'f21':L['fwd21']*100,'f63':L['fwd63']*100,'dd63':L['fwdDD63']*100,'dd10':(L['fwdDD63']<=-0.10).astype(float)})[ok]
    T=d.groupby('st').agg(share=('off','count'),P_off21=('off','mean'),fwd21=('f21','mean'),fwd63=('f63','mean'),DD63=('dd63','mean'),P10_63=('dd10','mean')); T['share']=(T['share']/len(d)).round(2); T.index=['OUT','REDUCE','ON'][:len(T)]
    stl=st.shift(1)[m]; wb=stl.map({0:0,1:1,2:1}); w3=stl.map({0:0,1:0.5,2:1})
    ch=(st!=st.shift(1))[m].sum()/(m.sum()/252)
    rows=[]
    for a,b in EP:
        a=pd.Timestamp(a); b=pd.Timestamp(b); seg=r.loc[a:b].iloc[1:]; sp=seg.sum()*100; w=st.shift(1).map({0:0,1:1,2:1}); stt=(w.loc[seg.index]*seg).sum()*100
        q=idx.get_loc(b); rec=r.iloc[q+1:q+43]; sp2=rec.sum()*100; st2=(w.loc[rec.index]*rec).sum()*100
        seq=st.iloc[idx.get_loc(a):q+1].values; k=np.where(seq==0)[0]
        rows.append({'peak':str(a.date())[:7],'capt':round(stt/sp,2),'capt_g':round(st2/sp2,2),'d_out':int(k[0]) if len(k) else None})
    E=pd.DataFrame(rows)
    print(f'\n==== {name} ====  changes/yr {ch:.1f}'); print(T.round(2).to_string())
    print('binary 100/100/0 :',stats(wb*bh),'exp',round(wb.mean(),2)); print('3-state 100/50/0 :',stats(w3*bh),'exp',round(w3.mean(),2))
    print('episodes capt:',' '.join(f"{x.peak}:{x.capt}" for x in E.itertuples()),'| median',E.capt.median(),' rebound capt median',E.capt_g.median())
    return T
print('B&H',stats(bh))
for nm,S in [('dd-first weights (V10b)',Sbase),('V10b + volume qualifiers',Svol),('balanced V1',S1)]:
    evaluate(f'{nm} | no hysteresis 30/55',pd.Series(np.where(S<=30,0,np.where(S<=55,1,2)),index=S.index),S)
    evaluate(f'{nm} | hysteresis OUT in<=30 exit>40, ON in>55 exit<=45',states_h(S,30,40,55,45),S)
    evaluate(f'{nm} | hysteresis OUT in<=30 exit>45, ON in>60 exit<=50',states_h(S,30,45,60,50),S)
    evaluate(f'{nm} | hysteresis OUT in<=35 exit>45, ON in>55 exit<=45',states_h(S,35,45,55,45),S)
pd.to_pickle({'Sbase':Sbase,'Svol':Svol,'S1':S1},'ENV_SCORES.pkl')
