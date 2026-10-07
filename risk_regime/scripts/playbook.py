import pandas as pd, numpy as np, warnings; warnings.filterwarnings('ignore')
pd.set_option('display.width',340); pd.set_option('display.max_rows',400); pd.set_option('display.max_columns',60)
G=pd.read_pickle('G.pkl'); L=pd.read_pickle('L.pkl'); P=pd.read_pickle('P.pkl'); idx=P.index; lp=np.log(P)
RU=pd.read_pickle('RU.pkl') if __import__('os').path.exists('RU.pkl') else None
EX=(idx>=pd.Timestamp('2020-02-01'))&(idx<=pd.Timestamp('2020-07-31'))
TICK=[c for c in P.columns if c not in ('LQD','SHY')]
z=lambda k: G[k]
RULES={
 'SETUP_rates':   (z('TNX_chg5_z252')>1)&(z('VIX_roc5_z252')<-0.3),
 'ONSET_impulse': (z('VIX_roc21_z252')<-0.3)&(z('VIX_roc5_z252')>1),
 'ONSET_vvixlag': (z('VIX_roc5_z252')>1)&(z('X_VVIX_VIX_roc5_z252')<-1),
 'CONT_2ndleg':   (z('VIX_roc3_z252')>1)&(z('TS_VIX_VIX3M_roc3_z252')>1)&(G['SPY_dd63']<=-0.07),
 'CAP_alldims':   (z('VIX_roc5_z252')>1)&(z('GVZ_roc5_z252')>1)&(z('MOVE_roc5_z252')>0.5)&(z('TNX_chg5_z252')<-0.5),
 'CAP_vix_gvz':   (z('VIX_roc5_z252')>1)&(z('GVZ_roc5_z252')>1),
 'ONCONF_collapse':(z('VIX_roc5_z252')<-1)&(z('X_VVIX_VIX_roc5_z252')>0.3)&(z('TS_VIX_VIX3M_roc5_z252')<-0.5),
 'FAIL_yieldsup': (z('VIX_roc10_z252')<-1)&(z('TNX_chg5_z252')>0.5)&(G['SPY_dd63']<=-0.05),
 # dimension tags at an onset (VIX roc5z>1) 
 'DIM_rateled':   (z('VIX_roc5_z252')>1)&(z('MOVE_roc5_z252')>1)&(z('TNX_chg5_z252')>0.5),
 'DIM_ftq':       (z('VIX_roc5_z252')>1)&(z('TNX_chg5_z252')<-1),
 'DIM_oil':       (z('VIX_roc5_z252')>1)&(z('OVX_roc5_z252')>1.5),
 'DIM_equityonly':(z('VIX_roc5_z252')>1)&(z('MOVE_roc5_z252')<0)&(z('TNX_chg5_z252').abs()<0.5)&(z('OVX_roc5_z252')<0.5),
}
fwd=lambda h: (lp.shift(-h)-lp)[TICK]
out={}
for k,m in RULES.items():
    f=(m.fillna(False)&~m.fillna(False).shift(1,fill_value=False)&~EX&L['ok21'])
    for h in (10,21):
        F=fwd(h)[f]; rel=F.sub(F['SPY'],axis=0)
        out[(k,f'rel{h}')]=rel.median()*100; out[(k,f'hit{h}')]=(rel>0).mean()
    out[(k,'n')]=F.notna().sum()
T=pd.DataFrame(out)
T.to_pickle('PLAYBOOK.pkl')
for k in RULES:
    sub=T[k].sort_values('rel10')
    print(f'\n=== after {k} first-fire: median ticker return RELATIVE to SPY (%), hit = share of fires beating SPY. n fires with data ===')
    print(sub.round(2).T.to_string())
