import pandas as pd, numpy as np, json, warnings; warnings.filterwarnings('ignore')
pd.set_option('display.width',300); pd.set_option('display.max_rows',200)
G=pd.read_pickle('G.pkl'); L=pd.read_pickle('L.pkl'); P=pd.read_pickle('P.pkl'); R=pd.read_pickle('R.pkl'); RU=pd.read_pickle('RU.pkl'); RR=pd.read_pickle('RR.pkl'); RV=pd.read_pickle('RV.pkl'); idx=P.index
EX=(idx>=pd.Timestamp('2020-02-01'))&(idx<=pd.Timestamp('2020-07-31'))
z=lambda k: G[k]; RVR_z=RV['RVR_z']; IVr=RV['IVRV_roc5z']; IVp=RV['IVRV_pct']
def recent(m,n): return m.fillna(False).astype(int).rolling(n,min_periods=1).max().astype(bool)
# ---- all rule series (same definitions as the three boards)
RULES={
 # main board (series, window days the fire stays in force, points)
 'SETUP_rates':      (RU['SETUP_rates'],10,-8),
 'SETUP_complac':    (RU['SETUP_complac'],10,-5),
 'ONSET_impulse':    (RU['ONSET_impulse'],15,-10),
 'ONSET_vvixlag':    (RU['ONSET_vvixlag'],15,-15),
 'ONSET_movelag':    (RU['ONSET_movelag'],15,-10),
 'ONSET_ovxdiv':     (RU['ONSET_ovxdiv'],15,-8),
 'CONT_2ndleg':      (RU['CONT_2ndleg'],15,-15),
 'FAIL_yieldsup':    (RU['FAIL_yieldsup'],15,-20),
 'CAP_alldims':      (RU['CAP_alldims'],42,+25),
 'CAP_vix_gvz':      (RU['CAP_vix_gvz'],42,+12),
 'CAP_vvixout':      (RU['CAP_vvixout'],42,+12),
 'ONCONF_collapse':  (RU['ONCONF_collapse'],21,+10),
 'VVIX_confirms_spike': ((z('VIX_roc5_z252')>1)&(z('X_VVIX_VIX_roc5_z252')>-0.3),15,+8),
 # ticker pair board
 'HYG_credit_crack': ((RR[('HYG',5)]<-1)&(z('VIX_roc5_z252')>1),21,-20),
 'KRE_banks_vs_yields': ((RR[('KRE',5)]<-1)&(z('TNX_chg5_z252')>1),15,-10),
 'TLT_duration_bid_calm': ((RR[('TLT',5)]>1)&(z('VIX_roc5_z252')<-1),15,-8),
 'XLU_defensive_bid': ((RR[('XLU',5)]>1)&(z('TNX_chg5_z252')>1),15,-8),
 'IWM_beta_chase':   ((RR[('IWM',5)]>1)&(z('VIX_roc5_z252')>1),10,-5),
 'XLU_month_lead':   (RR[('XLU',21)]>1,1,-5),
 # realized-vol board
 'SPY_realized_outruns_implied': ((IVr['SPY:VIX/RV']<-1)&(z('VIX_roc5_z252')>1),15,-10),
 'SPY_implied_outruns_realized': ((IVr['SPY:VIX/RV']>1)&(z('VIX_roc5_z252')>1),15,+8),
 'QQQ_realized_outruns_VXN': ((IVr['QQQ:VXN/RV']<-1)&(z('VXN_roc5_z252')>1),15,-10),
 'HYG_credit_vol_cheap_VIX': (IVp['HYG:VIX/RV']<0.2,1,-8),
 'XLE_energy_vol_cheap_OVX': (IVp['XLE:OVX/RV']<0.2,1,-4),
 'TLT_MOVE_leads_realized': ((RVR_z['TLT']<-1)&(z('MOVE_roc5_z252')>1),15,-6),
 'HYG_realized_expanding_spike': ((RVR_z['HYG']>1)&(z('VIX_roc5_z252')>1),15,-6),
}
# ---- slow context (continuous, always on)
CTX={
 'rates_pressure_vix_asleep': (((z('TNX_chg21_z252')>1.5)|(z('TYX_chg21_z252')>1.5))&(z('VIX_pct252')<0.3),-5),
 'complacency_both_compressed': ((z('VIX_roc21_z252')<-1)&(z('VIX_roc5_z252')<-1),-5),
 'drawdown_no_capitulation': ((G['SPY_dd63']<=-0.07)&~recent(RU['CAP_alldims']|RU['CAP_vix_gvz'],42),-10),
 'vol_collapsing_from_high': ((z('VIX_pct252')>0.75)&(z('VIX_roc5_z252')<-1),+10),
 'at_highs': (G['SPY_dd63']>-0.01,+5),
 'vix_floor_vvix_floor': ((z('VIX_pct252')<0.15)&(z('VVIX_pct252')<0.15),-3),
}
BASE=60
parts={}
for k,(m,w,pts) in RULES.items(): parts[k]=recent(m,w).astype(float)*pts
for k,(m,pts) in CTX.items(): parts[k]=m.fillna(False).astype(float)*pts
PARTS=pd.DataFrame(parts)
raw=BASE+PARTS.sum(axis=1)
SCORE=raw.clip(0,100)
pd.to_pickle({'score':SCORE,'parts':PARTS},'SCORE.pkl')
# ---- today
print('TODAY',idx[-1].date(),'score',int(SCORE.iloc[-1]),'raw',raw.iloc[-1])
today=PARTS.iloc[-1]; print('active components:'); print(today[today!=0].to_string())
print('\nlast 30 days:'); print(SCORE.iloc[-30:].astype(int).to_string())
# ---- backtest: buckets
ok=L['ok21']&~EX&(idx>=pd.Timestamp('2008-01-01'))
b=pd.cut(SCORE[ok],[-1,30,45,55,65,75,101],labels=['0-30','31-45','46-55','56-65','66-75','76-100'])
d=pd.DataFrame({'b':b,'off':L.loc[ok,'riskoff21'],'f21':L.loc[ok,'fwd21']*100,'f63':L.loc[ok,'fwd63']*100,'dd21':L.loc[ok,'fwdDD21']*100,'dd63':L.loc[ok,'fwdDD63']*100,'up21':L.loc[ok,'fwdUP21']*100})
print('\n=== SCORE buckets (2008+, ex-2020 window): n, P(5% DD/21d), mean fwd21, fwd63, mean maxDD21, maxDD63 ===')
print(d.groupby('b').agg(n=('off','count'),P_off=('off','mean'),fwd21=('f21','mean'),fwd63=('f63','mean'),DD21=('dd21','mean'),DD63=('dd63','mean'),UP21=('up21','mean')).round(2).to_string())
print('time share by bucket:'); print((d.b.value_counts(normalize=True).sort_index()*100).round(1).to_string())
# ---- allocation backtest: exposure = score/100 in SPY (lagged 1d), rest cash; vs buy & hold; also a 'beta tilt' version: score>65 -> IWM/QQQ mix? keep simple
r=R['SPY']; m=~EX&(idx>=pd.Timestamp('2008-01-01'))
w=(SCORE.shift(1)/100)[m]; rs=r[m]
strat=w*rs; bh=rs
def stats(x): 
    c=x.cumsum(); dd=(c-c.cummax()).min()
    return f'ann {x.mean()*252*100:5.2f}%  vol {x.std()*np.sqrt(252)*100:5.2f}%  sharpe {x.mean()/x.std()*np.sqrt(252):4.2f}  maxDD(log) {dd*100:6.1f}%'
print('\n=== allocation = score/100 in SPY, lagged a day, 2008+ ex-2020 ===')
print('score-weighted :',stats(strat)); print('buy & hold     :',stats(bh)); print('avg exposure',round(w.mean(),2))
# high-beta variant: when score>=70 hold IWM+QQQ 50/50 instead of SPY; when <=40 hold TLT with the unexposed part
hb=(R['IWM']+R['QQQ'])/2; tlt=R['TLT'].fillna(0)
s2=np.where(SCORE.shift(1)[m]>=70,hb[m],rs)*w + np.where(SCORE.shift(1)[m]<=40,tlt[m],0)*(1-w)
s2=pd.Series(s2,index=rs.index)
print('beta-tilt      :',stats(s2),' (>=70: IWM/QQQ, <=40: rest in TLT)')
# sub-periods
for a,bb in [('2008','2012'),('2013','2017'),('2018','2022'),('2023','2026')]:
    mm=(strat.index>=a)&(strat.index<=bb+'-12-31'); print(f'{a}-{bb}: score {stats(strat[mm])} | B&H {stats(bh[mm])}')
# score at the 36 peaks and 40 troughs
OFF=pd.read_pickle('OFF.pkl'); ON=pd.read_pickle('ON.pkl')
pk=[(a.date(),int(SCORE.loc[a]),int(SCORE.iloc[idx.get_loc(a)+5]),int(SCORE.iloc[idx.get_loc(a)+10])) for a in OFF.peak if a>=pd.Timestamp('2008-01-01')]
tr=[(a.date(),int(SCORE.loc[a]),int(SCORE.iloc[idx.get_loc(a)+5]),int(SCORE.iloc[idx.get_loc(a)+10])) for a in ON.trough if a>=pd.Timestamp('2008-01-01')]
print('\nscore at risk-off PEAKS (day 0, +5, +10):',pk); print('median',np.median([x[1] for x in pk]),np.median([x[2] for x in pk]),np.median([x[3] for x in pk]))
print('score at risk-on TROUGHS (day 0, +5, +10):',tr); print('median',np.median([x[1] for x in tr]),np.median([x[2] for x in tr]),np.median([x[3] for x in tr]))
print('unconditional median score 2008+:',SCORE[m].median())
# export for dashboard
json.dump({'asof':str(idx[-1].date()),'score':int(SCORE.iloc[-1]),'base':BASE,'components':{k:float(v) for k,v in today[today!=0].items()},'hist':[int(v) for v in SCORE.iloc[-504:]],
           'buckets':d.groupby('b').agg(n=('off','count'),P_off=('off','mean'),fwd21=('f21','mean'),fwd63=('f63','mean'),DD21=('dd21','mean')).round(3).reset_index().astype({'b':str}).to_dict(orient='records')},open('score_data.json','w'))
