import pandas as pd, numpy as np, warnings; warnings.filterwarnings('ignore')
pd.set_option('display.width',320)
G=pd.read_pickle('G.pkl'); L=pd.read_pickle('L.pkl'); P=pd.read_pickle('P.pkl'); idx=P.index; z=lambda k: G[k]
EX=(idx>=pd.Timestamp('2020-02-01'))&(idx<=pd.Timestamp('2020-07-31'))
RULES={
 'SETUP_rates':   (z('TNX_chg5_z252')>1)&(z('VIX_roc5_z252')<-0.3),
 'ONSET_impulse': (z('VIX_roc21_z252')<-0.3)&(z('VIX_roc5_z252')>1),
 'ONSET_vvixlag': (z('VIX_roc5_z252')>1)&(z('X_VVIX_VIX_roc5_z252')<-1),
 'VVIX_confirm(benign)':(z('VIX_roc5_z252')>1)&(z('X_VVIX_VIX_roc5_z252')>-0.3),
 'CONT_2ndleg':   (z('VIX_roc3_z252')>1)&(z('TS_VIX_VIX3M_roc3_z252')>1)&(G['SPY_dd63']<=-0.07),
 'CAP_alldims':   (z('VIX_roc5_z252')>1)&(z('GVZ_roc5_z252')>1)&(z('MOVE_roc5_z252')>0.5)&(z('TNX_chg5_z252')<-0.5),
 'CAP_vix_gvz':   (z('VIX_roc5_z252')>1)&(z('GVZ_roc5_z252')>1),
 'FAIL_yieldsup': (z('VIX_roc10_z252')<-1)&(z('TNX_chg5_z252')>0.5)&(G['SPY_dd63']<=-0.05),
 'MOVE_alone(benign)':(z('MOVE_roc5_z252')>1)&(z('VIX_roc5_z252')<0.3),
}
periods={'1990-2007':(idx<pd.Timestamp('2008-01-01')),'2008-2015':(idx>=pd.Timestamp('2008-01-01'))&(idx<pd.Timestamp('2016-01-01')),'2016-2026':(idx>=pd.Timestamp('2016-01-01'))}
print('=== SUB-PERIOD check: first-fires, P(>=5% DD in 21d), mean fwd21 % (base in brackets) ===')
for k,m in RULES.items():
    f=m.fillna(False)&~m.fillna(False).shift(1,fill_value=False)&~EX&L['ok21']
    s=f'{k:22s}'
    for pn,pm in periods.items():
        ff=f&pm; ok=L['ok21']&pm
        if ff.sum()<5: s+=f' | {pn}: n={ff.sum():3d}  --  '; continue
        s+=f' | {pn}: n={ff.sum():3d} P={L.loc[ff,"riskoff21"].mean():.2f}({L.loc[ok,"riskoff21"].mean():.2f}) f21={L.loc[ff,"fwd21"].mean()*100:5.2f}({L.loc[ok,"fwd21"].mean()*100:4.2f})'
    print(s)
# threshold sensitivity for the two main onset rules
print('\n=== threshold sensitivity (P(5%DD21), fwd21%, n) ===')
for a in (0.7,1.0,1.3):
    for b in (-0.3,0.0):
        m=(z('TNX_chg5_z252')>a)&(z('VIX_roc5_z252')<b); f=m.fillna(False)&~m.fillna(False).shift(1,fill_value=False)&~EX&L['ok21']
        print(f'SETUP_rates TNX>{a} VIX<{b}: n={f.sum()} P={L.loc[f,"riskoff21"].mean():.3f} fwd21={L.loc[f,"fwd21"].mean()*100:.2f}')
for a in (0.7,1.0,1.3):
    for b in (-0.7,-1.0,-1.3):
        m=(z('VIX_roc5_z252')>a)&(z('X_VVIX_VIX_roc5_z252')<b); f=m.fillna(False)&~m.fillna(False).shift(1,fill_value=False)&~EX&L['ok21']
        print(f'ONSET_vvixlag VIX>{a} VVIX/VIX<{b}: n={f.sum()} P={L.loc[f,"riskoff21"].mean():.3f} fwd21={L.loc[f,"fwd21"].mean()*100:.2f}')
for a in (0.7,1.0,1.3):
    m=(z('VIX_roc5_z252')>a)&(z('GVZ_roc5_z252')>a); f=m.fillna(False)&~m.fillna(False).shift(1,fill_value=False)&~EX&L['ok63']
    print(f'CAP_vix_gvz both>{a}: n={f.sum()} fwd21={L.loc[f,"fwd21"].mean()*100:.2f} fwd63={L.loc[f,"fwd63"].mean()*100:.2f} P(fwd21>0)={(L.loc[f,"fwd21"]>0).mean():.2f}')
