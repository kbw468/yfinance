import pandas as pd, numpy as np, warnings; warnings.filterwarnings('ignore')
pd.set_option('display.width',300); pd.set_option('display.max_rows',400)
G=pd.read_pickle('G.pkl'); L=pd.read_pickle('L.pkl'); S=pd.read_pickle('S.pkl'); P=pd.read_pickle('P.pkl'); idx=P.index
OFF=pd.read_pickle('OFF.pkl'); ON=pd.read_pickle('ON.pkl')
m=G['VIX1D_lvl'].notna()
print('VIX1D sample:',m.sum(),'days from',G.index[m][0].date())
# events in VIX1D era
pk=[d for d in OFF.peak if d>=pd.Timestamp('2023-06-01')]; tr=[d for d in ON.trough if d>=pd.Timestamp('2023-06-01')]
cols=['VIX1D_roc1_z252','VIX1D_roc3_z252','VIX1D_roc5_z252','TS_VIX1D_VIX9D_roc3_z252','TS_VIX1D_VIX9D','VIX9D_roc3_z252','VIX_roc3_z252','X_VVIX_VIX_roc5_z252','SKEW_roc10_z252','TNX_chg5_z252','MOVE_roc5_z252']
for name,anchors in [('PEAKS',pk),('TROUGHS',tr)]:
    for a in anchors:
        p=idx.get_loc(a); w=G.iloc[p-7:p+8][cols].copy(); w.index=range(-7,8)
        print(f'\n--- {name} anchor {a.date()} ---'); print(w.round(2).T.to_string())
# unconditional in-era: VIX1D ROC vs other ROCs as leader - corr of dlog VIX1D_t vs dSPY_{t+1}, dVIX_{t+1}
d=np.log(S[['VIX1D','VIX9D','VIX']]).diff(); d['SPY']=np.log(P['SPY']).diff(); d=d[m]
print('\ncorr dVIX1D_t vs dSPY_t+1:',round(d['VIX1D'].corr(d['SPY'].shift(-1)),3),' vs dVIX_t+1:',round(d['VIX1D'].corr(d['VIX'].shift(-1)),3),' vs dVIX9D_t+1:',round(d['VIX1D'].corr(d['VIX9D'].shift(-1)),3))
# VIX1D/VIX9D ratio: event-day premium. P(5% DD) by ratio bucket & by its ROC3
ok=L['ok21']&m
for c in ['TS_VIX1D_VIX9D','VIX1D_roc3_z252','TS_VIX1D_VIX9D_roc3_z252','VIX1D_roc5_z252']:
    q=pd.qcut(G.loc[ok,c],5,labels=False,duplicates='drop')
    t=pd.DataFrame({'q':q,'off':L.loc[ok,'riskoff21'],'fwd5':L.loc[ok,'fwd5']*100,'fwd21':L.loc[ok,'fwd21']*100,'dd21':L.loc[ok,'fwdDD21']*100}).groupby('q').agg(['mean','count']).round(2)
    print(f'\n{c} quintiles:'); print(t.to_string())
