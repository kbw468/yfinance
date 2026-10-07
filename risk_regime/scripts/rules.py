import pandas as pd, numpy as np, warnings; warnings.filterwarnings('ignore')
pd.set_option('display.width',320); pd.set_option('display.max_rows',400); pd.set_option('display.max_columns',50)
G=pd.read_pickle('G.pkl'); L=pd.read_pickle('L.pkl'); S=pd.read_pickle('S.pkl'); P=pd.read_pickle('P.pkl'); R=pd.read_pickle('R.pkl'); idx=P.index
OFF=pd.read_pickle('OFF.pkl'); ON=pd.read_pickle('ON.pkl')
EX=(idx>=pd.Timestamp('2020-02-01'))&(idx<=pd.Timestamp('2020-07-31'))
z=lambda k: G[k]; ok=L['ok21']; base=L.loc[ok,'riskoff21'].mean(); basef=L.loc[ok,'fwd21'].mean()*100
def first(m): m=m.fillna(False); return m&~m.shift(1,fill_value=False)&~EX
def evalrule(name,m,extra=''):
    f=first(m)&ok; t=L[f]
    # event coverage: for each risk-off event did rule fire in [peak-5, break5pct]? and lead vs -5% break
    cov=[]; lead=[]
    for _,e in OFF.iterrows():
        p=idx.get_loc(e.peak); b5=p+int(e.d_to_5pct); w=m.fillna(False).iloc[max(p-5,0):b5+1].values
        hit=w.any(); cov.append(hit); lead.append((b5-(max(p-5,0)+np.argmax(w))) if hit else np.nan)
    cov=np.array(cov); 
    # false alarms: fires not followed by >=5% DD in 21d
    print(f'{name:58s} fires={f.sum():4d} P(5%DD21)={t.riskoff21.mean():.3f} (base {base:.3f}) fwd5={t.fwd5.mean()*100:5.2f} fwd10={t.fwd10.mean()*100:5.2f} fwd21={t.fwd21.mean()*100:5.2f} (base {basef:.2f}) DD21={t.fwdDD21.mean()*100:5.2f} | events covered {cov.mean():.2f} ({cov.sum()}/{len(cov)}), median lead to -5% break {np.nanmedian(lead):.0f}d {extra}')
    return f
print('--- OFF-ONSET candidates (ROC configurations) ---')
compressed=z('VIX_roc21_z252')<-0.3
impulse3=z('VIX_roc3_z252')>1.0; impulse5=z('VIX_roc5_z252')>1.0
r1=evalrule('A  first impulse after compression: VIX roc21z<-.3 & roc5z>1',compressed&impulse5)
r1b=evalrule('A2 same with roc3z>1',compressed&impulse3)
r2=evalrule('B  A + MOVE not confirming (MOVE roc5z<0)',compressed&impulse5&(z('MOVE_roc5_z252')<0))
r3=evalrule('C  A + VVIX lagging (VVIX/VIX roc5z<-1)',compressed&impulse5&(z('X_VVIX_VIX_roc5_z252')<-1))
r3b=evalrule('C2 VIX roc5z>1 & VVIX/VIX roc5z<-1 (any trend)',impulse5&(z('X_VVIX_VIX_roc5_z252')<-1))
r3c=evalrule('C3 VIX roc5z>1 & VVIX/VIX roc5z>-0.3 (VVIX confirming) [expect benign]',impulse5&(z('X_VVIX_VIX_roc5_z252')>-0.3))
r4=evalrule('D  A + front-end leading (VIX9D/VIX roc3z>0.5)',compressed&impulse5&(z('TS_VIX9D_VIX_roc3_z252')>0.5))
r5=evalrule('E  A + yields had been rising (TNX chg10z>0.5)',compressed&impulse5&(z('TNX_chg10_z252')>0.5))
r5b=evalrule('E2 A + yields falling (TNX chg10z<-0.5) [FTQ flavour]',compressed&impulse5&(z('TNX_chg10_z252')<-0.5))
r6=evalrule('F  A + term structure flattening fast (VIX/VIX3M roc3z>1)',compressed&impulse5&(z('TS_VIX_VIX3M_roc3_z252')>1))
r7=evalrule('G  rate-shock setup: TNX chg5z>1 & VIX roc5z<-0.3 & MOVE roc5z<0',(z('TNX_chg5_z252')>1)&(z('VIX_roc5_z252')<-0.3)&(z('MOVE_roc5_z252')<0))
r7b=evalrule('G2 TNX chg5z>1 & VIX roc5z<-0.3 (no MOVE cond)',(z('TNX_chg5_z252')>1)&(z('VIX_roc5_z252')<-0.3))
r7c=evalrule('G3 TNX chg10z>1 & VIX roc10z<0',(z('TNX_chg10_z252')>1)&(z('VIX_roc10_z252')<0))
r8=evalrule('H  complacency: VIX roc21z<-1 & roc5z<-1 & SKEW roc10z<-0.5',(z('VIX_roc21_z252')<-1)&(z('VIX_roc5_z252')<-1)&(z('SKEW_roc10_z252')<-0.5))
r9=evalrule('I  commodity divergence: VIX roc5z>0.3 & OVX roc5z<-1',(z('VIX_roc5_z252')>0.3)&(z('OVX_roc5_z252')<-1))
r10=evalrule('J  continuation: VIX roc3z>1 & VIX/VIX3M roc3z>1 & SPY dd63<=-7%',impulse3&(z('TS_VIX_VIX3M_roc3_z252')>1)&(G['SPY_dd63']<=-0.07))
r11=evalrule('K  MOVE leads: MOVE roc5z>1 & VIX roc5z<0.3',(z('MOVE_roc5_z252')>1)&(z('VIX_roc5_z252')<0.3))
r12=evalrule('L  GVZ leads: GVZ roc5z>1 & VIX roc5z<0.3',(z('GVZ_roc5_z252')>1)&(z('VIX_roc5_z252')<0.3))
r13=evalrule('M  OVX leads: OVX roc5z>1 & VIX roc5z<0.3',(z('OVX_roc5_z252')>1)&(z('VIX_roc5_z252')<0.3))
r14=evalrule('N  VVIX leads: VVIX roc5z>1 & VIX roc5z<0.3',(z('VVIX_roc5_z252')>1)&(z('VIX_roc5_z252')<0.3))
r15=evalrule('O  SKEW leads: SKEW roc10z>1 & VIX roc10z<0',(z('SKEW_roc10_z252')>1)&(z('VIX_roc10_z252')<0))
r16=evalrule('P  VIX9D leads: VIX9D/VIX roc5z>1 & VIX roc5z<0.3',(z('TS_VIX9D_VIX_roc5_z252')>1)&(z('VIX_roc5_z252')<0.3))
r17=evalrule('Q  VIX1D leads: VIX1D roc3z>1.5 & VIX roc3z<0.5',(z('VIX1D_roc3_z252')>1.5)&(z('VIX_roc3_z252')<0.5))
print('\n--- ON / CAPITULATION candidates ---')
def evalon(name,m):
    f=first(m)&L['ok63']; t=L[f]
    lag=[]
    for _,e in ON.iterrows():
        q=idx.get_loc(e.trough); w=m.fillna(False).iloc[max(q-5,0):q+11].values; lag.append((np.argmax(w)-min(5,q)) if w.any() else np.nan)
    lag=np.array(lag)
    print(f'{name:58s} fires={f.sum():4d} fwd5={t.fwd5.mean()*100:5.2f} fwd10={t.fwd10.mean()*100:5.2f} fwd21={t.fwd21.mean()*100:5.2f} fwd63={t.fwd63.mean()*100:5.2f} (base21 {basef:.2f}, base63 {L.loc[L.ok63,"fwd63"].mean()*100:.2f}) DD21={t.fwdDD21.mean()*100:5.2f} P(fwd21>0)={(t.fwd21>0).mean():.2f} | troughs covered within [-5,+10]: {np.mean(~np.isnan(lag)):.2f}, median day {np.nanmedian(lag):.0f}')
    return f
c1=evalon('CAP1 VIX,GVZ roc5z>1 & MOVE roc5z>.5 & TNX chg5z<-.5',(z('VIX_roc5_z252')>1)&(z('GVZ_roc5_z252')>1)&(z('MOVE_roc5_z252')>0.5)&(z('TNX_chg5_z252')<-0.5))
c2=evalon('CAP2 VIX roc5z>1 & MOVE roc5z>1 & TNX chg5z<-1',(z('VIX_roc5_z252')>1)&(z('MOVE_roc5_z252')>1)&(z('TNX_chg5_z252')<-1))
c3=evalon('CAP3 VIX roc5z>1 & GVZ roc5z>1',(z('VIX_roc5_z252')>1)&(z('GVZ_roc5_z252')>1))
c4=evalon('CAP4 VIX roc5z>1.5 & VVIX/VIX roc5z>0 (VVIX outrunning)',(z('VIX_roc5_z252')>1.5)&(z('X_VVIX_VIX_roc5_z252')>0))
c5=evalon('CAP5 VIX roc5z>1 & VIX9D/VIX roc5z>1 (front-end panic)',(z('VIX_roc5_z252')>1)&(z('TS_VIX9D_VIX_roc5_z252')>1))
c6=evalon('CAP6 VIX1D roc3z>2 & VIX roc3z>1 (0DTE panic)',(z('VIX1D_roc3_z252')>2)&(z('VIX_roc3_z252')>1))
o1=evalon('ON1 VIX roc5z<-1 & VVIX/VIX roc5z>.3 & VIX/VIX3M roc5z<-.5',(z('VIX_roc5_z252')<-1)&(z('X_VVIX_VIX_roc5_z252')>0.3)&(z('TS_VIX_VIX3M_roc5_z252')<-0.5))
o2=evalon('ON2 ON1 & SPY dd63<=-5%',(z('VIX_roc5_z252')<-1)&(z('X_VVIX_VIX_roc5_z252')>0.3)&(z('TS_VIX_VIX3M_roc5_z252')<-0.5)&(G['SPY_dd63']<=-0.05))
o3=evalon('ON3 VIX roc5z<-1 & MOVE/VIX roc5z>1 (MOVE sticky, VIX collapsing)',(z('VIX_roc5_z252')<-1)&(z('X_MOVE_VIX_roc5_z252')>1))
o4=evalon('ON4 VIX roc5z<-1 & SKEW roc5z>0.5 & dd63<=-5%',(z('VIX_roc5_z252')<-1)&(z('SKEW_roc5_z252')>0.5)&(G['SPY_dd63']<=-0.05))
o5=evalon('ON5 VIX roc5z<-1.5 & dd63<=-7%',(z('VIX_roc5_z252')<-1.5)&(G['SPY_dd63']<=-0.07))
o6=evalon('ON6 VIX roc10z<-1 & TNX chg5z>0.5 & dd63<=-5% (yields back up)',(z('VIX_roc10_z252')<-1)&(z('TNX_chg5_z252')>0.5)&(G['SPY_dd63']<=-0.05))
pd.DataFrame({'A':r1,'B':r2,'C':r3,'C2':r3b,'E':r5,'G':r7,'J':r10,'CAP1':c1,'CAP3':c3,'ON2':o2,'ON3':o3}).to_pickle('RULES.pkl')
