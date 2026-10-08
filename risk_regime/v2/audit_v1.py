# Stage 0: era-robustness audit of every v1 rule. Run from risk_regime/work.
import pandas as pd, numpy as np, warnings; warnings.filterwarnings('ignore')
pd.set_option('display.width',320)
G=pd.read_pickle('G.pkl'); L=pd.read_pickle('L.pkl'); P=pd.read_pickle('P.pkl'); R=pd.read_pickle('R.pkl'); S=pd.read_pickle('S.pkl'); RU=pd.read_pickle('RU.pkl'); RR=pd.read_pickle('RR.pkl'); RV=pd.read_pickle('RV.pkl'); idx=P.index
EX=(idx>=pd.Timestamp('2020-02-01'))&(idx<=pd.Timestamp('2020-07-31')); ok=L['ok21']&~EX
z=lambda k: G[k]; RVR_z=RV['RVR_z']; IVr=RV['IVRV_roc5z']; IVp=RV['IVRV_pct']
_rv21=R['SPY'].rolling(21).std(); _zz=lambda x:(x-x.rolling(252).mean())/x.rolling(252).std(); _vrp5=_zz((S['VIX']-_rv21*np.sqrt(252)*100).diff(5))
RULES={}
for c in RU.columns: RULES[('main',c)]=RU[c]
PAIR={'HYG_credit_crack':(RR[('HYG',5)]<-1)&(z('VIX_roc5_z252')>1),'KRE_banks_vs_yields (removed)':(RR[('KRE',5)]<-1)&(z('TNX_chg5_z252')>1),'TLT_duration_bid_calm':(RR[('TLT',5)]>1)&(z('VIX_roc5_z252')<-1),
 'XLU_defensive_bid':(RR[('XLU',5)]>1)&(z('TNX_chg5_z252')>1),'XLRE_bid_yields_up':(RR[('XLRE',5)]>1)&(z('TNX_chg5_z252')>1),'IWM_beta_chase':(RR[('IWM',5)]>1)&(z('VIX_roc5_z252')>1),'XLU_month_lead':RR[('XLU',21)]>1,'VVIX_confirms_spike':(z('VIX_roc5_z252')>1)&(z('X_VVIX_VIX_roc5_z252')>-0.3)}
for k,v in PAIR.items(): RULES[('pair',k)]=v
RVR={'RVX_premium_collapse':(_vrp5<-1)&(z('VIX_roc21_z252')>1),'SPY_realized_outruns_implied':(IVr['SPY:VIX/RV']<-1)&(z('VIX_roc5_z252')>1),'SPY_implied_outruns_realized':(IVr['SPY:VIX/RV']>1)&(z('VIX_roc5_z252')>1),
 'QQQ_realized_outruns_VXN':(IVr['QQQ:VXN/RV']<-1)&(z('VXN_roc5_z252')>1),'HYG_credit_vol_cheap_VIX':IVp['HYG:VIX/RV']<0.2,'XLE_energy_vol_cheap_OVX':IVp['XLE:OVX/RV']<0.2,'TLT_MOVE_leads_realized':(RVR_z['TLT']<-1)&(z('MOVE_roc5_z252')>1),'HYG_realized_expanding_spike':(RVR_z['HYG']>1)&(z('VIX_roc5_z252')>1)}
for k,v in RVR.items(): RULES[('realized',k)]=v
CTX={'rates_pressure_vix_asleep':((z('TNX_chg21_z252')>1.5)|(z('TYX_chg21_z252')>1.5))&(z('VIX_pct252')<0.3),'complacency_both_compressed':(z('VIX_roc21_z252')<-1)&(z('VIX_roc5_z252')<-1),'vol_collapsing_from_high':(z('VIX_pct252')>0.75)&(z('VIX_roc5_z252')<-1),'vix_floor_vvix_floor':(z('VIX_pct252')<0.15)&(z('VVIX_pct252')<0.15),'drawdown_no_capitulation':(G['SPY_dd63']<=-0.07)}
for k,v in CTX.items(): RULES[('context',k)]=v
RISKON={'CAP_alldims','CAP_vix_gvz','CAP_vvixout','ONCONF_collapse','VVIX_confirms_spike','SPY_implied_outruns_realized','vol_collapsing_from_high'}
ERAS={'to2012':('1990-01-01','2012-12-31'),'2013-19':('2013-01-01','2019-12-31'),'2020-26':('2020-01-01','2026-12-31')}
HALF={'2008-17':('2008-01-01','2017-12-31'),'2018-26':('2018-01-01','2026-12-31')}
def wilson(k,n,zc=1.96):
    if n==0: return (np.nan,np.nan)
    p=k/n; d=1+zc*zc/n; c=(p+zc*zc/(2*n))/d; h=zc*np.sqrt(p*(1-p)/n+zc*zc/(4*n*n))/d; return (c-h,c+h)
rows=[]
print('base rates by era: ',{e:round(float(L.loc[ok&(idx>=pd.Timestamp(a))&(idx<=pd.Timestamp(b)),'riskoff21'].mean()),3) for e,(a,b) in {**ERAS,**HALF}.items()})
print('base fwd21 by era:',{e:round(float(L.loc[ok&(idx>=pd.Timestamp(a))&(idx<=pd.Timestamp(b)),'fwd21'].mean()*100),2) for e,(a,b) in {**ERAS,**HALF}.items()})
for (grp,k),m in RULES.items():
    m=m.fillna(False).astype(bool); ctx=(grp=='context')
    f=(m&ok) if ctx else (m&~m.shift(1,fill_value=False)&ok)
    rec={'group':grp,'rule':k,'kind':'on' if k in RISKON else 'off','n_all':int(f.sum()),'P_all':round(float(L.loc[f,'riskoff21'].mean()),2) if f.sum() else np.nan,'fwd21_all':round(float(L.loc[f,'fwd21'].mean()*100),2) if f.sum() else np.nan}
    verdict=[]
    for e,(a,b) in {**ERAS,**HALF}.items():
        mm=f&(idx>=pd.Timestamp(a))&(idx<=pd.Timestamp(b)); base=L.loc[ok&(idx>=pd.Timestamp(a))&(idx<=pd.Timestamp(b)),'riskoff21'].mean(); basef=L.loc[ok&(idx>=pd.Timestamp(a))&(idx<=pd.Timestamp(b)),'fwd21'].mean()*100
        n=int(mm.sum()); kk=int(L.loc[mm,'riskoff21'].sum()) if n else 0; p=kk/n if n else np.nan; lo,hi=wilson(kk,n); fw=L.loc[mm,'fwd21'].mean()*100 if n else np.nan
        rec[f'{e} n']=n; rec[f'{e} P']=round(p,2) if n else np.nan; rec[f'{e} CI']=f'{lo:.2f}-{hi:.2f}' if n else ''; rec[f'{e} fwd21']=round(fw,2) if n else np.nan
        if e in HALF and n>=6:
            if k in RISKON: verdict.append('ok' if fw>basef else 'FAIL')
            else: verdict.append('ok' if (p>base and lo>base-0.05) else 'FAIL')
        elif e in HALF: verdict.append('n<6')
    rec['halves']=' / '.join(verdict); rows.append(rec)
T=pd.DataFrame(rows)
cols=['group','rule','kind','n_all','P_all','fwd21_all']+[f'{e} {x}' for e in ['to2012','2013-19','2020-26'] for x in ('n','P','fwd21')]+[f'{e} {x}' for e in ['2008-17','2018-26'] for x in ('n','P','CI','fwd21')]+['halves']
print(T[cols].to_string(index=False))
print('\nVERDICT: a risk-off rule passes a half when P beats that half\'s base rate and the Wilson lower bound is within 0.05 of it; a risk-on rule passes when fwd21 beats the half\'s base. n<6 = too few fires to judge.')
print('\nFAILING OR UNJUDGEABLE IN 2018-26:'); print(T[~T['halves'].str.endswith('ok')][['group','rule','n_all','P_all','2008-17 P','2018-26 n','2018-26 P','2018-26 fwd21','halves']].to_string(index=False))
T.to_pickle('AUDIT_V1.pkl')
