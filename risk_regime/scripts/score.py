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
 'HYG_credit_crack':(RR[('HYG',5)]<-1)&(z('VIX_roc5_z252')>1),'TLT_duration_bid_calm':(RR[('TLT',5)]>1)&(z('VIX_roc5_z252')<-1),
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
    'HYG_credit_crack':(21,-20),'TLT_duration_bid_calm':(15,-8),'XLU_defensive_bid':(15,-8),'IWM_beta_chase':(10,-5),'XLU_month_lead':(1,-5),
    'SPY_realized_outruns_implied':(15,-10),'SPY_implied_outruns_realized':(15,8),'QQQ_realized_outruns_VXN':(15,-10),'HYG_credit_vol_cheap_VIX':(1,-8),'XLE_energy_vol_cheap_OVX':(1,-4),'TLT_MOVE_leads_realized':(15,-6),'HYG_realized_expanding_spike':(15,-6)}
C1={'rates_pressure_vix_asleep':-5,'complacency_both_compressed':-5,'vol_collapsing_from_high':10,'at_highs':5,'vix_floor_vvix_floor':-3}
# drawdown-first: longer windows on risk-off, bigger points, smaller/shorter risk-on adds, cap while in drawdown, slow re-entry
W2={k:(int(w*1.5) if p<0 else w, int(p*1.4) if p<0 else int(p*0.7)) for k,(w,p) in W1.items()}
C2={'rates_pressure_vix_asleep':-8,'complacency_both_compressed':-8,'vol_collapsing_from_high':6,'at_highs':3,'vix_floor_vvix_floor':-5}

# ===== STAY-IN ENVIRONMENT: balanced weights, no at-the-highs softener, two states, OUT only at a deep reading =====
C1={k:v for k,v in C1.items() if k!='at_highs'}
S=pd.read_pickle('S.pkl'); _rv21=R['SPY'].rolling(21).std(); _vrp5=(lambda x:(x-x.rolling(252).mean())/x.rolling(252).std())((S['VIX']-_rv21*np.sqrt(252)*100).diff(5))
SIG['RVX_premium_collapse']=(_vrp5<-1)&(z('VIX_roc21_z252')>1)      # VIX-realized premium collapsing while VIX has risen for a month: realized catching up to implied
W1=dict(W1); W1['RVX_premium_collapse']=(15,-8)
# 2026-10-08: the regional-banks-vs-yields pair rule (KRE rel ROC5 z < -1 with TNX 5d chg z > 1, -10 for 15 sessions) was removed at the user's direction.
# Its edge was 2007-09 (14 of 35 fires; 2018-26 fires: P 0.27, fwd21 +1.45%). Removing it: 15.03% -> 14.70% a year, same -20.9% max DD, 23 -> 21 OUT spells.
W1.pop('KRE_banks_vs_yields',None)
SCORE,PARTS=build(W1,C1)
OUT_IN,OUT_EXIT=20,35
def states2(S):
    v=S.values; st=np.ones(len(v),dtype=int); s=1   # 1 = IN, 0 = OUT
    for i in range(len(v)):
        x=v[i]
        if np.isnan(x): st[i]=s; continue
        if s==1 and x<=OUT_IN: s=0
        elif s==0 and x>OUT_EXIT: s=1
        st[i]=s
    return pd.Series(st,index=S.index)
ST=states2(SCORE); NAMES=['OUT','IN']
pd.to_pickle({'score':SCORE,'parts':PARTS,'state':ST},'SCORE.pkl')
r=R['SPY']; m=~EX&(idx>=pd.Timestamp('2008-01-01')); bh=r[m]
def stats(x):
    c=x.cumsum(); dd=(c-c.cummax()); return dict(ann=round(float(x.mean()*252*100),2),vol=round(float(x.std()*np.sqrt(252)*100),2),sharpe=round(float(x.mean()/x.std()*np.sqrt(252)),2),maxDD=round(float(dd.min()*100),1),ulcer=round(float(np.sqrt((dd**2).mean())*100),2))
today=PARTS.iloc[-1]; print('TODAY',idx[-1].date(),'score',int(SCORE.iloc[-1]),'state',NAMES[ST.iloc[-1]]); print(today[today!=0].to_string())
ok=L['ok21']&m
d=pd.DataFrame({'st':ST,'off':L['riskoff21'],'f21':L['fwd21']*100,'f63':L['fwd63']*100,'dd21':L['fwdDD21']*100,'dd63':L['fwdDD63']*100,'dd10':(L['fwdDD63']<=-0.10).astype(float),'dd5_63':(L['fwdDD63']<=-0.05).astype(float)})[ok]
T=d.groupby('st').agg(share=('off','count'),P_off21=('off','mean'),fwd21=('f21','mean'),fwd63=('f63','mean'),DD21=('dd21','mean'),DD63=('dd63','mean'),P5_63=('dd5_63','mean'),P10_63=('dd10','mean'),f63_p5=('f63',lambda x: x.quantile(.05))); T['share']=T['share']/len(d); T.index=NAMES; print(T.round(3).to_string())
stl=ST.shift(1)[m]; wb=stl.map({0:0,1:1})
SB=stats(wb*bh); BH=stats(bh); print('book',SB,'exp',round(float(wb.mean()),2)); print('B&H',BH)
ch=float((ST!=ST.shift(1))[m].sum()/(m.sum()/252)); print('changes/yr',round(ch,1))
EP=[('2008-05-19','2009-03-09'),('2010-04-23','2010-07-02'),('2011-04-29','2011-10-03'),('2015-07-20','2015-08-25'),('2015-11-03','2016-02-11'),('2018-01-26','2018-02-08'),('2018-09-20','2018-12-24'),('2022-01-03','2022-10-12'),('2023-07-31','2023-10-27'),('2024-07-16','2024-08-05'),('2025-02-19','2025-04-08'),('2026-01-27','2026-03-30')]
w=ST.shift(1).map({0:0,1:1}); eps=[]
for a,b in EP:
    a=pd.Timestamp(a); b=pd.Timestamp(b); seg=r.loc[a:b].iloc[1:]; sp=seg.sum()*100; stt=(w.loc[seg.index]*seg).sum()*100
    q=idx.get_loc(b); rec=r.iloc[q+1:q+43]; sp2=rec.sum()*100; st2=(w.loc[rec.index]*rec).sum()*100
    seq=ST.iloc[idx.get_loc(a):q+1].values; k=np.where(seq==0)[0]
    eps.append({'peak':str(a.date()),'trough':str(b.date()),'spy':round(sp,1),'book':round(stt,1),'capt_loss':round(stt/sp,2),'reb_spy':round(sp2,1),'reb_book':round(st2,1),'capt_gain':round(st2/sp2,2),'days_to_out':int(k[0]) if len(k) else None,'state_pk':NAMES[ST.loc[a]],'state_tr':NAMES[ST.loc[b]]})
E=pd.DataFrame(eps); print(E.to_string()); print('median capt_loss',E.capt_loss.median(),'capt_gain',E.capt_gain.median())
# OUT spells list
o=(ST==0)&m; spells=[]; 
for dt in idx[o]:
    if not spells or (dt-spells[-1][1]).days>1 and (idx.get_loc(dt)-idx.get_loc(spells[-1][1]))>1: spells.append([dt,dt])
    else: spells[-1][1]=dt
SP=[]
for a,b in spells:
    q0=idx.get_loc(a); q1=idx.get_loc(b); seg=r.iloc[q0:q1+2].sum()*100   # SPY move while out (incl next-day lag)
    SP.append({'from':str(a.date()),'to':str(b.date()),'sessions':q1-q0+1,'spy_while_out':round(float(seg),1)})
print(pd.DataFrame(SP).to_string())
ent=(ST!=ST.shift(1))&m&L['ok21']; trans={}
for s_,nm in enumerate(NAMES):
    e=ent&(ST==s_); trans[nm]={'n':int(e.sum()),'fwd21':round(float(L.loc[e,'fwd21'].mean()*100),2),'fwd63':round(float(L.loc[e,'fwd63'].mean()*100),2),'P_off21':round(float(L.loc[e,'riskoff21'].mean()),2)}
print(trans)
yr=pd.DataFrame({'SPY':bh,'book':wb*bh}); Y=(yr.groupby(yr.index.year).sum()*100).round(1); print(Y.T.to_string())
cur=ST.iloc[-1]; k=0
for v in ST.values[::-1]:
    if v==cur: k+=1
    else: break
json.dump({'asof':str(idx[-1].date()),'score':int(SCORE.iloc[-1]),'state':NAMES[cur],'state_idx':int(cur),'state_names':NAMES,'days_in_state':k,'thresholds':{'out_in':OUT_IN,'out_exit':OUT_EXIT},
  'base':60,'components':{k_:float(v) for k_,v in today[today!=0].items()},'hist':[int(v) for v in SCORE.iloc[-504:]],'state_hist':[int(v) for v in ST.iloc[-504:]],
  'states':{nm:{c:round(float(T.loc[nm,c]),3) for c in T.columns} for nm in NAMES},'binary':SB,'buyhold':BH,'avg_exposure':round(float(wb.mean()),2),'changes_per_year':round(ch,1),
  'episodes':eps,'spells':SP,'transitions':trans,'yearly':{str(k_):{'spy':float(v.SPY),'book':float(v.book)} for k_,v in Y.iterrows()}},open('score_data.json','w'))
