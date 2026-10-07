import pandas as pd, numpy as np, json, warnings; warnings.filterwarnings('ignore')
G=pd.read_pickle('G.pkl'); L=pd.read_pickle('L.pkl'); P=pd.read_pickle('P.pkl'); S=pd.read_pickle('S.pkl'); idx=P.index
RV=pd.read_pickle('RV.pkl'); RVT=pd.read_pickle('RVT.pkl')
EX=(idx>=pd.Timestamp('2020-02-01'))&(idx<=pd.Timestamp('2020-07-31')); ok=L['ok21']&~EX
TICK=[c for c in P.columns if c not in ('LQD','SHY')]
z=lambda k: G[k]; RVR_z=RV['RVR_z']; IVz=RV['IVRV_z']; IVr=RV['IVRV_roc5z']; IVp=RV['IVRV_pct']
RULES={
 'SPY_realized_outruns_implied': ((IVr['SPY:VIX/RV']<-1)&(z('VIX_roc5_z252')>1),'SPY VIX/RV ROC5 z < -1 and VIX 5d ROC z > 1','Realized vol rising faster than VIX during the spike. Continuation.'),
 'SPY_implied_outruns_realized': ((IVr['SPY:VIX/RV']>1)&(z('VIX_roc5_z252')>1),'SPY VIX/RV ROC5 z > 1 and VIX 5d ROC z > 1','Fear premium: VIX up more than realized. Dip.'),
 'QQQ_realized_outruns_VXN': ((IVr['QQQ:VXN/RV']<-1)&(z('VXN_roc5_z252')>1),'QQQ VXN/RV ROC5 z < -1 and VXN 5d ROC z > 1','Nasdaq realized leading implied.'),
 'HYG_credit_vol_cheap_VIX': (IVp['HYG:VIX/RV']<0.2,'HYG VIX/RV pct252 < 20','VIX low relative to credit realized vol.'),
 'XLE_energy_vol_cheap_OVX': (IVp['XLE:OVX/RV']<0.2,'XLE OVX/RV pct252 < 20','OVX low relative to energy-equity realized vol.'),
 'TLT_MOVE_leads_realized': ((RVR_z['TLT']<-1)&(z('MOVE_roc5_z252')>1),'TLT RV10/RV21 z < -1 and MOVE 5d ROC z > 1','Bond implied spiking while bond realized is compressed.'),
 'HYG_realized_expanding_spike': ((RVR_z['HYG']>1)&(z('VIX_roc5_z252')>1),'HYG RV10/RV21 z > 1 and VIX 5d ROC z > 1','Credit realized expanding into the vol spike.'),
 'GLD_realized_collapse_GVZdown': ((IVr['GLD:GVZ/RV']>1)&(z('GVZ_roc5_z252')<-1),'GLD GVZ/RV ROC5 z > 1 and GVZ 5d ROC z < -1','Gold realized collapsing faster than GVZ. Small n.'),
 'XOP_realized_collapse_OVXdown': ((IVr['XOP:OVX/RV']>1)&(z('OVX_roc5_z252')<-1),'XOP OVX/RV ROC5 z > 1 and OVX 5d ROC z < -1','Energy realized collapsing faster than OVX. Small n.'),
}
stats={}; H={}
for k,(m,d,n) in RULES.items():
    m=m.fillna(False); f=m&~m.shift(1,fill_value=False)&ok
    stats[k]={'def':d,'note':n,'n':int(f.sum()),'P_off':round(float(L.loc[f,'riskoff21'].mean()),3) if f.sum() else None,'fwd21':round(float(L.loc[f,'fwd21'].mean()*100),2) if f.sum() else None,'fwdDD21':round(float(L.loc[f,'fwdDD21'].mean()*100),2) if f.sum() else None,'last':str(f[f].index[-1].date()) if f.sum() else 'never','live':bool(m.iloc[-1]),'recent':bool(m.iloc[-10:].any())}
    H[k]=[int(v) for v in m.iloc[-504:]]
print(pd.DataFrame(stats).T[['n','P_off','fwd21','fwdDD21','last','live','recent']].to_string())
r=lambda v: None if (v is None or (isinstance(v,float) and np.isnan(v))) else round(float(v),2)
tape=[]
for t in TICK:
    tape.append({'t':t,'rv10':r(RV['RV10'][t].iloc[-1]),'rv21':r(RV['RV21'][t].iloc[-1]),'rvr':r(RV['RVR'][t].iloc[-1]),'rvr_z':r(RVR_z[t].iloc[-1]),
                 'relrv':r(RV['RELRV'][t].iloc[-1]) if t!='SPY' else 1.0,'relrv_roc5z':r(RV['RELRV_roc5z'][t].iloc[-1]) if t!='SPY' else None,'rv10_roc5z':r(RV['RV10_roc5z'][t].iloc[-1]),
                 'P_top':r(RVT.loc[t,'P_off_RVRtopQ']) if t in RVT.index else None,'P_bot':r(RVT.loc[t,'P_off_RVRbotQ']) if t in RVT.index else None,
                 'exp_vixup':r(RVT.loc[t,'fwd10rel_RVexp_VIXup']) if t in RVT.index else None,'exp_vixdn':r(RVT.loc[t,'fwd10rel_RVexp_VIXdown']) if t in RVT.index else None,
                 'cmp_vixup':r(RVT.loc[t,'fwd10rel_RVcomp_VIXup']) if t in RVT.index else None,'cmp_vixdn':r(RVT.loc[t,'fwd10rel_RVcomp_VIXdown']) if t in RVT.index else None})
pairs=[]
for c in RV['IVRV'].columns:
    pairs.append({'pair':c,'level':r(RV['IVRV'][c].iloc[-1]),'pct252':r(IVp[c].iloc[-1]),'lvl_z':r(IVz[c].iloc[-1]),'roc5z':r(IVr[c].iloc[-1])})
breadth={'now':r(RV['BR'].iloc[-1]),'z':r(RV['BR_z'].iloc[-1]),'hist':[r(v) for v in RV['BR'].iloc[-504:]]}
out={'rv_rules':stats,'rv_rules_hist':H,'rv_tape':tape,'ivrv':pairs,'breadth':breadth}
json.dump(out,open('rv_data.json','w')); import os; print('KB',os.path.getsize('rv_data.json')//1024)
print(pd.DataFrame(pairs).to_string()); print('breadth',breadth['now'],breadth['z'])
