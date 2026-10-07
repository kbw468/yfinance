import pandas as pd, numpy as np, warnings; warnings.filterwarnings('ignore')
pd.set_option('display.width',320); pd.set_option('display.max_rows',400); pd.set_option('display.max_columns',50)
G=pd.read_pickle('G.pkl'); L=pd.read_pickle('L.pkl'); P=pd.read_pickle('P.pkl'); idx=P.index
OFF=pd.read_pickle('OFF.pkl'); ON=pd.read_pickle('ON.pkl'); z=lambda k: G[k]
RULES={
 'SETUP_rates':   (z('TNX_chg5_z252')>1)&(z('VIX_roc5_z252')<-0.3),
 'SETUP_complac': (z('VIX_roc21_z252')<-1)&(z('VIX_roc5_z252')<-1)&(z('SKEW_roc10_z252')<-0.5),
 'ONSET_impulse': (z('VIX_roc21_z252')<-0.3)&(z('VIX_roc5_z252')>1),
 'ONSET_vvixlag': (z('VIX_roc5_z252')>1)&(z('X_VVIX_VIX_roc5_z252')<-1),
 'ONSET_movelag': (z('VIX_roc21_z252')<-0.3)&(z('VIX_roc5_z252')>1)&(z('MOVE_roc5_z252')<0),
 'ONSET_ovxdiv':  (z('VIX_roc5_z252')>0.3)&(z('OVX_roc5_z252')<-1),
 'CONT_2ndleg':   (z('VIX_roc3_z252')>1)&(z('TS_VIX_VIX3M_roc3_z252')>1)&(G['SPY_dd63']<=-0.07),
 'CAP_alldims':   (z('VIX_roc5_z252')>1)&(z('GVZ_roc5_z252')>1)&(z('MOVE_roc5_z252')>0.5)&(z('TNX_chg5_z252')<-0.5),
 'CAP_vix_gvz':   (z('VIX_roc5_z252')>1)&(z('GVZ_roc5_z252')>1),
 'CAP_vvixout':   (z('VIX_roc5_z252')>1.5)&(z('X_VVIX_VIX_roc5_z252')>0),
 'ONCONF_collapse':(z('VIX_roc5_z252')<-1)&(z('X_VVIX_VIX_roc5_z252')>0.3)&(z('TS_VIX_VIX3M_roc5_z252')<-0.5),
 'FAIL_yieldsup': (z('VIX_roc10_z252')<-1)&(z('TNX_chg5_z252')>0.5)&(G['SPY_dd63']<=-0.05),
}
RU=pd.DataFrame({k:v.fillna(False) for k,v in RULES.items()}); RU.to_pickle('RU.pkl')
def firstfire(rule,p0,p1):
    w=RU[rule].iloc[max(p0,0):p1+1].values; return (max(p0,0)+np.argmax(w)-0) if w.any() else None
print('=== RISK-OFF SCORECARD: day (vs peak=0) of first fire of each rule in window [peak-25, break-5%]; "." = no fire; "x" = series not yet available ===')
rows=[]
for _,e in OFF.iterrows():
    p=idx.get_loc(e.peak); b5=p+int(e.d_to_5pct); rec={'peak':e.peak.date(),'depth%':round(e.depth*100,1),'to-5%':int(e.d_to_5pct),'dim':e.dim}
    for k in ['SETUP_rates','SETUP_complac','ONSET_impulse','ONSET_vvixlag','ONSET_movelag','ONSET_ovxdiv','CONT_2ndleg']:
        avail=G[{'SETUP_rates':'TNX_chg5_z252','SETUP_complac':'SKEW_roc10_z252','ONSET_impulse':'VIX_roc5_z252','ONSET_vvixlag':'X_VVIX_VIX_roc5_z252','ONSET_movelag':'MOVE_roc5_z252','ONSET_ovxdiv':'OVX_roc5_z252','CONT_2ndleg':'TS_VIX_VIX3M_roc3_z252'}[k]].iloc[p]
        if np.isnan(avail): rec[k]='x'; continue
        f=firstfire(k,p-25,b5); rec[k]='.' if f is None else str(f-p)
    rows.append(rec)
SC=pd.DataFrame(rows); print(SC.to_string())
anyfire=SC[['SETUP_rates','SETUP_complac','ONSET_impulse','ONSET_vvixlag','ONSET_movelag','ONSET_ovxdiv']].apply(lambda r: any(v not in ('.','x') for v in r),axis=1)
print('\nevents with at least one SETUP/ONSET fire before the -5% break:',anyfire.mean().round(2))
pre=SC[['SETUP_rates','SETUP_complac','ONSET_impulse','ONSET_vvixlag','ONSET_movelag','ONSET_ovxdiv']].apply(lambda r: any((v not in ('.','x')) and int(v)<=0 for v in r),axis=1)
print('events with a fire at or BEFORE the peak day:',pre.mean().round(2))
print('\n=== RISK-ON SCORECARD: day (vs trough=0) of first fire in [trough-10, trough+10] ===')
rows=[]
for _,e in ON.iterrows():
    q=idx.get_loc(e.trough); rec={'trough':e.trough.date(),'dd%':round(e.dd_at_trough*100,1),'rebound%':round(e.rebound42*100,1)}
    for k in ['CAP_alldims','CAP_vix_gvz','CAP_vvixout','ONCONF_collapse','FAIL_yieldsup']:
        avail=G[{'CAP_alldims':'GVZ_roc5_z252','CAP_vix_gvz':'GVZ_roc5_z252','CAP_vvixout':'X_VVIX_VIX_roc5_z252','ONCONF_collapse':'X_VVIX_VIX_roc5_z252','FAIL_yieldsup':'TNX_chg5_z252'}[k]].iloc[q]
        if np.isnan(avail): rec[k]='x'; continue
        f=firstfire(k,q-10,q+10); rec[k]='.' if f is None else str(f-q)
    rows.append(rec)
SO=pd.DataFrame(rows); print(SO.to_string())
# false alarm accounting per rule: of first-fires, share followed by >=5% DD within 21d / within 42d; and median fwd DD
print('\n=== RULE first-fire accounting (ex-2020): share followed by >=5% DD in 21d / 42d, median fwdDD21, count ===')
EX=(idx>=pd.Timestamp('2020-02-01'))&(idx<=pd.Timestamp('2020-07-31'))
for k in RULES:
    f=RU[k]&~RU[k].shift(1,fill_value=False)&~EX&L['ok42']
    print(f'{k:16s} n={f.sum():4d}  P(5%DD/21d)={(L.loc[f,"fwdDD21"]<=-0.05).mean():.2f}  P(5%DD/42d)={(L.loc[f,"fwdDD42"]<=-0.05).mean():.2f}  P(7%DD/42d)={(L.loc[f,"fwdDD42"]<=-0.07).mean():.2f}  medDD21={L.loc[f,"fwdDD21"].median()*100:5.2f}%  fwd21={L.loc[f,"fwd21"].mean()*100:5.2f}%  fwd42={L.loc[f,"fwd42"].mean()*100:5.2f}%')
print(f'{"BASE":16s} n={L.ok42.sum():4d}  P(5%DD/21d)={(L.loc[L.ok42,"fwdDD21"]<=-0.05).mean():.2f}  P(5%DD/42d)={(L.loc[L.ok42,"fwdDD42"]<=-0.05).mean():.2f}  P(7%DD/42d)={(L.loc[L.ok42,"fwdDD42"]<=-0.07).mean():.2f}  medDD21={L.loc[L.ok42,"fwdDD21"].median()*100:5.2f}%  fwd21={L.loc[L.ok42,"fwd21"].mean()*100:5.2f}%  fwd42={L.loc[L.ok42,"fwd42"].mean()*100:5.2f}%')
