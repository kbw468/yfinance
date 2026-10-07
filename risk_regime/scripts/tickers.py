import pandas as pd, numpy as np, warnings; warnings.filterwarnings('ignore')
pd.set_option('display.width',320); pd.set_option('display.max_rows',400); pd.set_option('display.max_columns',50)
P=pd.read_pickle('P.pkl'); S=pd.read_pickle('S.pkl'); G=pd.read_pickle('G.pkl'); L=pd.read_pickle('L.pkl')
OFF=pd.read_pickle('OFF.pkl'); ON=pd.read_pickle('ON.pkl'); idx=P.index; lp=np.log(P)
TICK=[c for c in P.columns if c not in ('LQD','SHY')]
# ---- classify each risk-off event by DIMENSION using ROC10 z over [peak, peak+10]
def dim(pk):
    p=idx.get_loc(pk); w=slice(p,p+11)
    g=lambda k: G[k].iloc[w].max() if G[k].iloc[w].notna().any() else np.nan
    gm=lambda k: G[k].iloc[w].min() if G[k].iloc[w].notna().any() else np.nan
    move=g('MOVE_roc10_z252'); tnx_up=g('TNX_chg10_z252'); tnx_dn=gm('TNX_chg10_z252'); ovx=g('OVX_roc10_z252'); gvz=g('GVZ_roc10_z252'); vix=g('VIX_roc10_z252')
    tags=[]
    if not np.isnan(move) and move>1.0: tags.append('RATEVOL')
    if not np.isnan(tnx_up) and tnx_up>1.0: tags.append('YIELDS_UP')
    if not np.isnan(tnx_dn) and tnx_dn<-1.0: tags.append('FTQ')
    if not np.isnan(ovx) and ovx>1.0: tags.append('OIL')
    if not np.isnan(gvz) and gvz>1.0: tags.append('GOLDVOL')
    return '+'.join(tags) if tags else 'VOL_ONLY', dict(MOVE=move,TNXup=tnx_up,TNXdn=tnx_dn,OVX=ovx,GVZ=gvz,VIX=vix)
rows=[]
for _,e in OFF.iterrows():
    t,dd=dim(e.peak); rows.append({'peak':e.peak.date(),'depth':round(e.depth,3),'dim':t,**{k:round(v,2) if not np.isnan(v) else np.nan for k,v in dd.items()}})
D=pd.DataFrame(rows); print('=== RISK-OFF EVENT DIMENSIONS (max ROC10 z in [peak, peak+10]) ==='); print(D.to_string())
OFF['dim']=D['dim'].values; OFF.to_pickle('OFF.pkl')
# ---- ticker returns around risk-off onsets: from peak to +5,+10,+21 and to trough ; relative to SPY
def rets(anchor,h):
    p=idx.get_loc(anchor); 
    if p+h>=len(idx): return pd.Series(np.nan,index=TICK)
    return (lp.iloc[p+h]-lp.iloc[p])[TICK]
def table(anchors,title,hs=(5,10,21)):
    print(f'\n=== {title} ===')
    out={}
    for h in hs:
        R=pd.DataFrame([rets(a,h) for a in anchors],index=anchors)
        rel=R.sub(R['SPY'],axis=0)
        out[f'abs{h}']=R.median()*100; out[f'rel{h}']=rel.median()*100; out[f'hit_rel{h}']=(rel>0).mean(); out[f'n{h}']=R.notna().sum()
    T=pd.DataFrame(out).sort_values('rel10')
    print(T.round(2).to_string())
    return T
Toff=table(OFF.peak.tolist(),'TICKER move from RISK-OFF PEAK (median %, rel = vs SPY, hit = share of events beating SPY)')
Ton=table(ON.trough.tolist(),'TICKER move from RISK-ON TROUGH')
Toff.to_pickle('Toff.pkl'); Ton.to_pickle('Ton.pkl')
# ---- by dimension
for dname,mask in [('contains RATEVOL/YIELDS_UP',OFF.dim.str.contains('RATEVOL|YIELDS_UP')),('contains FTQ (yields down)',OFF.dim.str.contains('FTQ')),('VOL_ONLY',OFF.dim=='VOL_ONLY'),('contains OIL',OFF.dim.str.contains('OIL'))]:
    a=OFF.loc[mask,'peak'].tolist()
    if len(a)>=4:
        R=pd.DataFrame([rets(x,10) for x in a],index=a); rel=R.sub(R['SPY'],axis=0)
        print(f'\n--- RISK-OFF dim={dname} (n={len(a)}): median 10d rel-to-SPY % (n tickers with data) ---')
        print(pd.DataFrame({'rel10':rel.median()*100,'abs10':R.median()*100,'n':R.notna().sum()}).sort_values('rel10').round(2).T.to_string())
# ---- do any tickers LEAD SPY at tops? ROC5 of ticker/SPY ratio in [-10,-1] before peak vs baseline
print('\n=== LEADING TICKERS: median ROC10 of ticker/SPY ratio measured AT the peak day (relative weakness into the top) ===')
rel10=(lp.sub(lp['SPY'],axis=0)).diff(10)[TICK]
at=pd.DataFrame([rel10.loc[a] for a in OFF.peak],index=OFF.peak)
base=rel10.median()
print(pd.DataFrame({'at_peak_med%':at.median()*100,'base_med%':base*100,'n':at.notna().sum(),'share_neg':(at<0).mean()}).sort_values('at_peak_med%').round(2).to_string())
print('\n=== LEADING TICKERS at TROUGH: median ROC10 of ticker/SPY ratio at trough day (relative strength into the low) ===')
at=pd.DataFrame([rel10.loc[a] for a in ON.trough],index=ON.trough)
print(pd.DataFrame({'at_trough_med%':at.median()*100,'n':at.notna().sum(),'share_pos':(at>0).mean()}).sort_values('at_trough_med%',ascending=False).round(2).to_string())
