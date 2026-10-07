import pandas as pd, numpy as np, json, warnings; warnings.filterwarnings('ignore')
pd.set_option('display.width',320); pd.set_option('display.max_rows',200)
G=pd.read_pickle('G.pkl'); L=pd.read_pickle('L.pkl'); P=pd.read_pickle('P.pkl'); R=pd.read_pickle('R.pkl'); RU=pd.read_pickle('RU.pkl'); RR=pd.read_pickle('RR.pkl'); RV=pd.read_pickle('RV.pkl'); idx=P.index
OFF=pd.read_pickle('OFF.pkl'); ON=pd.read_pickle('ON.pkl')
EX=(idx>=pd.Timestamp('2020-02-01'))&(idx<=pd.Timestamp('2020-07-31'))
z=lambda k: G[k]; RVR_z=RV['RVR_z']; IVr=RV['IVRV_roc5z']; IVp=RV['IVRV_pct']
def recent(m,n): return m.fillna(False).astype(int).rolling(n,min_periods=1).max().astype(bool)
spy=P['SPY']; lspy=np.log(spy); dd63=G['SPY_dd63']
SIG={
 'SETUP_rates':RU['SETUP_rates'],'SETUP_complac':RU['SETUP_complac'],'ONSET_impulse':RU['ONSET_impulse'],'ONSET_vvixlag':RU['ONSET_vvixlag'],'ONSET_movelag':RU['ONSET_movelag'],'ONSET_ovxdiv':RU['ONSET_ovxdiv'],
 'CONT_2ndleg':RU['CONT_2ndleg'],'FAIL_yieldsup':RU['FAIL_yieldsup'],'CAP_alldims':RU['CAP_alldims'],'CAP_vix_gvz':RU['CAP_vix_gvz'],'CAP_vvixout':RU['CAP_vvixout'],'ONCONF_collapse':RU['ONCONF_collapse'],
 'VVIX_confirms_spike':(z('VIX_roc5_z252')>1)&(z('X_VVIX_VIX_roc5_z252')>-0.3),
 'HYG_credit_crack':(RR[('HYG',5)]<-1)&(z('VIX_roc5_z252')>1),'KRE_banks_vs_yields':(RR[('KRE',5)]<-1)&(z('TNX_chg5_z252')>1),'TLT_duration_bid_calm':(RR[('TLT',5)]>1)&(z('VIX_roc5_z252')<-1),
 'XLU_defensive_bid':(RR[('XLU',5)]>1)&(z('TNX_chg5_z252')>1),'IWM_beta_chase':(RR[('IWM',5)]>1)&(z('VIX_roc5_z252')>1),'XLU_month_lead':RR[('XLU',21)]>1,
 'SPY_realized_outruns_implied':(IVr['SPY:VIX/RV']<-1)&(z('VIX_roc5_z252')>1),'SPY_implied_outruns_realized':(IVr['SPY:VIX/RV']>1)&(z('VIX_roc5_z252')>1),'QQQ_realized_outruns_VXN':(IVr['QQQ:VXN/RV']<-1)&(z('VXN_roc5_z252')>1),
 'HYG_credit_vol_cheap_VIX':IVp['HYG:VIX/RV']<0.2,'XLE_energy_vol_cheap_OVX':IVp['XLE:OVX/RV']<0.2,'TLT_MOVE_leads_realized':(RVR_z['TLT']<-1)&(z('MOVE_roc5_z252')>1),'HYG_realized_expanding_spike':(RVR_z['HYG']>1)&(z('VIX_roc5_z252')>1),
}
CTXS={'rates_pressure_vix_asleep':((z('TNX_chg21_z252')>1.5)|(z('TYX_chg21_z252')>1.5))&(z('VIX_pct252')<0.3),'complacency_both_compressed':(z('VIX_roc21_z252')<-1)&(z('VIX_roc5_z252')<-1),
 'vol_collapsing_from_high':(z('VIX_pct252')>0.75)&(z('VIX_roc5_z252')<-1),'at_highs':dd63>-0.01,'vix_floor_vvix_floor':(z('VIX_pct252')<0.15)&(z('VVIX_pct252')<0.15)}
CAPANY=SIG['CAP_alldims']|SIG['CAP_vix_gvz']|SIG['CAP_vvixout']
def build(W,ctxw,base=60,dd_cap=None,reentry=None):
    parts={}
    for k,(w,pts) in W.items(): parts[k]=recent(SIG[k],w).astype(float)*pts
    for k,pts in ctxw.items(): parts[k]=CTXS[k].fillna(False).astype(float)*pts
    parts['drawdown_no_capitulation']=((dd63<=-0.07)&~recent(CAPANY,42)).fillna(False).astype(float)*(-10 if dd_cap is None else -15)
    PARTS=pd.DataFrame(parts); s=(base+PARTS.sum(axis=1))
    if dd_cap is not None:  # while SPY >=5% off its 63d high and no confirmation/capitulation in last 10 sessions, cap the dial
        cap=(dd63<=-0.05)&~recent(SIG['ONCONF_collapse']|CAPANY,10); s=s.where(~cap.fillna(False),np.minimum(s,dd_cap))
    if reentry is not None:  # after any day <=35, the dial may rise only `reentry` points per session (slow re-entry)
        v=s.values.copy(); out=v.copy()
        for i in range(1,len(v)):
            if out[i-1]<=35 or (v[i]>out[i-1] and out[i-1]<60): out[i]=min(v[i],out[i-1]+reentry)
            else: out[i]=v[i]
        s=pd.Series(out,index=s.index)
    return s.clip(0,100),PARTS
W1={'SETUP_rates':(10,-8),'SETUP_complac':(10,-5),'ONSET_impulse':(15,-10),'ONSET_vvixlag':(15,-15),'ONSET_movelag':(15,-10),'ONSET_ovxdiv':(15,-8),'CONT_2ndleg':(15,-15),'FAIL_yieldsup':(15,-20),'CAP_alldims':(42,25),'CAP_vix_gvz':(42,12),'CAP_vvixout':(42,12),'ONCONF_collapse':(21,10),'VVIX_confirms_spike':(15,8),
    'HYG_credit_crack':(21,-20),'KRE_banks_vs_yields':(15,-10),'TLT_duration_bid_calm':(15,-8),'XLU_defensive_bid':(15,-8),'IWM_beta_chase':(10,-5),'XLU_month_lead':(1,-5),
    'SPY_realized_outruns_implied':(15,-10),'SPY_implied_outruns_realized':(15,8),'QQQ_realized_outruns_VXN':(15,-10),'HYG_credit_vol_cheap_VIX':(1,-8),'XLE_energy_vol_cheap_OVX':(1,-4),'TLT_MOVE_leads_realized':(15,-6),'HYG_realized_expanding_spike':(15,-6)}
C1={'rates_pressure_vix_asleep':-5,'complacency_both_compressed':-5,'vol_collapsing_from_high':10,'at_highs':5,'vix_floor_vvix_floor':-3}
# drawdown-first: longer windows on risk-off, bigger points, smaller/shorter risk-on adds, cap while in drawdown, slow re-entry
W2={k:(int(w*1.5) if p<0 else w, int(p*1.4) if p<0 else int(p*0.7)) for k,(w,p) in W1.items()}
C2={'rates_pressure_vix_asleep':-8,'complacency_both_compressed':-8,'vol_collapsing_from_high':6,'at_highs':3,'vix_floor_vvix_floor':-5}
V={}
V['V1 current']=build(W1,C1)
V['V2 heavier/longer']=build(W2,C2)
V['V3 V2 + cap30 in drawdown']=build(W2,C2,dd_cap=30)
V['V4 V3 + slow re-entry 4/day']=build(W2,C2,dd_cap=30,reentry=4)
V['V5 V3 + slow re-entry 2/day']=build(W2,C2,dd_cap=30,reentry=2)
r=R['SPY']; m=~EX&(idx>=pd.Timestamp('2008-01-01'))
def stats(x):
    c=x.cumsum(); dd=(c-c.cummax()); ulcer=np.sqrt((dd**2).mean())
    return dict(ann=round(x.mean()*252*100,2),vol=round(x.std()*np.sqrt(252)*100,2),sharpe=round(x.mean()/x.std()*np.sqrt(252),2),maxDD=round(dd.min()*100,1),ulcer=round(ulcer*100,2),calmar=round((x.mean()*252)/(-dd.min()),2))
rows=[]; bh=r[m]
rows.append({'variant':'SPY buy & hold',**stats(bh),'avg_exp':1.0})
for k,(s,_) in V.items():
    w=(s.shift(1)/100)[m]; rows.append({'variant':k,**stats(w*bh),'avg_exp':round(w.mean(),2)})
print('=== 2008+ ex-2020 window, exposure = dial/100 in SPY lagged a day, rest cash ==='); print(pd.DataFrame(rows).set_index('variant').to_string())
# ---- episodes: major drawdowns since 2008 (peak -> trough) and the 42d recovery capture after the trough
EP=[('2008-05-19','2009-03-09'),('2010-04-23','2010-07-02'),('2011-04-29','2011-10-03'),('2015-07-20','2015-08-25'),('2015-11-03','2016-02-11'),('2018-01-26','2018-02-08'),('2018-09-20','2018-12-24'),('2022-01-03','2022-10-12'),('2023-07-31','2023-10-27'),('2024-07-16','2024-08-05'),('2025-02-19','2025-04-08'),('2026-01-27','2026-03-30')]
print('\n=== EPISODES: strategy loss peak->trough vs SPY, and 42d recovery capture after trough (strategy gain / SPY gain), plus dial at peak, at trough, 10d after ===')
for k,(s,_) in V.items():
    w=s.shift(1)/100; rows=[]
    for a,b in EP:
        a=pd.Timestamp(a); b=pd.Timestamp(b); seg=r.loc[a:b].iloc[1:]; st=(w.loc[seg.index]*seg).sum()*100; sp=seg.sum()*100
        q=idx.get_loc(b); rec=r.iloc[q+1:q+43]; st2=(w.loc[rec.index]*rec).sum()*100; sp2=rec.sum()*100
        rows.append({'peak':a.date(),'trough':b.date(),'SPY%':round(sp,1),'strat%':round(st,1),'captured_loss':round(st/sp,2) if sp else np.nan,'rebound42 SPY%':round(sp2,1),'strat%_r':round(st2,1),'captured_gain':round(st2/sp2,2) if sp2 else np.nan,'dial@peak':int(s.loc[a]),'dial@trough':int(s.loc[b]),'dial+10':int(s.iloc[q+10])})
    T=pd.DataFrame(rows); print(f'\n--- {k} ---'); print(T.to_string()); print('median captured loss',T.captured_loss.median(),' median captured gain',T.captured_gain.median())
# today's values
print('\nTODAY by variant:',{k:int(s.iloc[-1]) for k,(s,_) in V.items()})
pd.to_pickle({k:s for k,(s,_) in V.items()},'SCORE_VARIANTS.pkl')
