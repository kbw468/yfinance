import pandas as pd, numpy as np, warnings; warnings.filterwarnings('ignore')
pd.set_option('display.width',300); pd.set_option('display.max_rows',200); pd.set_option('display.max_columns',40)
G=pd.read_pickle('G.pkl'); L=pd.read_pickle('L.pkl'); P=pd.read_pickle('P.pkl'); S=pd.read_pickle('S.pkl'); RV=pd.read_pickle('RV.pkl'); idx=P.index
spy=P['SPY']; lspy=np.log(spy)
EX=(idx>=pd.Timestamp('2020-02-01'))&(idx<=pd.Timestamp('2020-07-31'))
def fwd(h): return (lspy.shift(-h)-lspy)*100
def fdd(h): return np.log(spy[::-1].rolling(h).min()[::-1].shift(-1)/spy)*100
def fup(h): return np.log(spy[::-1].rolling(h).max()[::-1].shift(-1)/spy)*100
F21,F42,F63,F126=fwd(21),fwd(42),fwd(63),fwd(126); D21,D42,D63,D126=fdd(21),fdd(42),fdd(63),fdd(126); U42=fup(42)
# days to a 5% and 10% drawdown from the anchor (within 126d)
def days_to(th,h=126):
    out=pd.Series(np.nan,index=idx); a=lspy.values
    for i in range(len(a)-1):
        w=a[i+1:i+1+h]-a[i]; k=np.where(w<=np.log(1+th))[0]
        if len(k): out.iloc[i]=k[0]+1
    return out
DT5=days_to(-0.05); DT10=days_to(-0.10)
FULL={'VIX pct':G['VIX_pct252'],'VVIX pct':G['VVIX_pct252'],'MOVE pct':G['MOVE_pct252'],'TNX pct':G['TNX_pct252'],'TYX pct':G['TYX_pct252'],'SKEW pct':G['SKEW_pct252'],'GVZ pct':G['GVZ_pct252'],'OVX pct':G['OVX_pct252'],
 'VIX roc21z':G['VIX_roc21_z252'],'VVIX roc21z':G['VVIX_roc21_z252'],'MOVE roc21z':G['MOVE_roc21_z252'],'TNX chg21z':G['TNX_chg21_z252'],'TYX chg21z':G['TYX_chg21_z252'],'GVZ roc21z':G['GVZ_roc21_z252'],'OVX roc21z':G['OVX_roc21_z252'],
 'VIX roc5z':G['VIX_roc5_z252'],'MOVE roc5z':G['MOVE_roc5_z252'],'TNX chg5z':G['TNX_chg5_z252'],'VVIX/VIX roc5z':G['X_VVIX_VIX_roc5_z252'],
 'VIX/VIX3M pct':G['TS_VIX_VIX3M_pct252'],'MOVE/VIX pct':G['X_MOVE_VIX_pct252'],'VVIX/VIX pct':G['X_VVIX_VIX_pct252'],'curve 10y3m':G['curve_10y3m'],
 'SPY dd63':G['SPY_dd63']*100,'SPY ret21':G['SPY_ret21']*100,'SPY RV10/RV21 z':RV['RVR_z']['SPY'],'VIX/HYG RV z':RV['IVRV_z']['HYG:VIX/RV'],'MOVE/TLT RV z':RV['IVRV_z']['TLT:MOVE/RV'],'RV breadth z':RV['BR_z']}
RED={'VIX pct':G['VIX_pct252'],'TNX pct':G['TNX_pct252'],'TYX pct':G['TYX_pct252'],'SKEW pct':G['SKEW_pct252'],'VIX roc21z':G['VIX_roc21_z252'],'TNX chg21z':G['TNX_chg21_z252'],'TYX chg21z':G['TYX_chg21_z252'],'VIX roc5z':G['VIX_roc5_z252'],'TNX chg5z':G['TNX_chg5_z252'],'curve 10y3m':G['curve_10y3m'],'SPY dd63':G['SPY_dd63']*100,'SPY ret21':G['SPY_ret21']*100,'SPY RV10/RV21 z':RV['RVR_z']['SPY'],'VIX/SPY RV z':RV['IVRV_z']['SPY:VIX/RV']}
def search(feats,title,weights=None,k=12,sep=40):
    X=pd.DataFrame(feats); today=X.iloc[-1]; print(f'\n==== {title} ====\ntoday:'); print(today.round(2).to_string())
    m=X.notna().all(axis=1)&~EX&(idx<idx[-1]-pd.Timedelta(days=130))
    Z=(X-X[m].mean())/X[m].std(); w=pd.Series(1.0,index=X.columns) if weights is None else pd.Series(weights).reindex(X.columns).fillna(1.0)
    d=np.sqrt((((Z[m]-Z.iloc[-1])**2)*w).sum(axis=1)/w.sum())
    d=d.sort_values(); picks=[]
    for dt,dist in d.items():
        if all(abs((dt-p).days)>sep for p,_ in picks): picks.append((dt,dist))
        if len(picks)>=k: break
    rows=[]
    for dt,dist in picks:
        rows.append({'date':dt.date(),'dist':round(dist,2),'SPY dd63':round(X.loc[dt,'SPY dd63'],1),'fwd21':round(F21[dt],1),'fwd42':round(F42[dt],1),'fwd63':round(F63[dt],1),'fwd126':round(F126[dt],1),'maxDD21':round(D21[dt],1),'maxDD42':round(D42[dt],1),'maxDD63':round(D63[dt],1),'maxDD126':round(D126[dt],1),'maxUP42':round(U42[dt],1),'days_to_-5%':DT5[dt],'days_to_-10%':DT10[dt]})
    T=pd.DataFrame(rows); print(T.to_string())
    print('\nmedian of analogs: ',T[['fwd21','fwd42','fwd63','fwd126','maxDD21','maxDD42','maxDD63','maxDD126']].median().round(1).to_dict())
    print('share with >=5% DD within 42d:',round((T['maxDD42']<=-5).mean(),2),' within 63d:',round((T['maxDD63']<=-5).mean(),2),' >=10% within 126d:',round((T['maxDD126']<=-10).mean(),2))
    ok=m&F63.notna()
    print('baseline (all days): fwd63 median',round(F63[ok].median(),1),' maxDD42 median',round(D42[ok].median(),1),' maxDD63 median',round(D63[ok].median(),1),' P(5% DD in 42d)',round((D42[ok]<=-5).mean(),2),' P(5% DD in 63d)',round((D63[ok]<=-5).mean(),2),' P(10% DD in 126d)',round((D126[ok]<=-10).mean(),2))
    return T,X,picks
T1,X1,p1=search(FULL,'FULL VECTOR (29 features, 2008+), equal weight')
W={'MOVE pct':3,'TNX pct':3,'TYX pct':2,'TNX chg21z':3,'TYX chg21z':2,'MOVE roc21z':2,'VIX pct':3,'VVIX pct':3,'VIX roc21z':2,'SPY dd63':2,'VIX/HYG RV z':2,'RV breadth z':1.5}
T2,X2,p2=search(FULL,'FULL VECTOR, weighted toward today\'s extremes (rates, bond vol, equity vol, VVIX, SPY near high, credit IV/RV)',weights=W)
T3,X3,p3=search(RED,'REDUCED VECTOR (14 features, 1991+, no VVIX/MOVE)')
# show the feature profile of the top weighted analogs vs today
print('\n==== feature profile: today vs top 6 weighted analogs ====')
cols=['VIX pct','VVIX pct','MOVE pct','TNX pct','TYX pct','VIX roc21z','MOVE roc21z','TNX chg21z','TYX chg21z','VIX roc5z','TNX chg5z','MOVE/VIX pct','VVIX/VIX pct','SPY dd63','SPY ret21','SPY RV10/RV21 z','VIX/HYG RV z','RV breadth z']
prof=pd.DataFrame({str(dt.date()):X2.loc[dt,cols] for dt,_ in p2[:6]}); prof.insert(0,'TODAY',X2.iloc[-1][cols]); print(prof.round(2).to_string())
# coarse regime match: MOVE pct>=0.85, TNX pct>=0.85, TNX chg21z>=1.5, VIX pct<=0.3, SPY dd63>=-3%
cm=(G['MOVE_pct252']>=0.85)&(G['TNX_pct252']>=0.85)&(G['TNX_chg21_z252']>=1.5)&(G['VIX_pct252']<=0.30)&(G['SPY_dd63']>=-0.03)&~EX
cm2=cm&(G['VVIX_pct252']<=0.3)
for name,c in [('COARSE: MOVE pct>=85, TNX pct>=85, TNX 21d chg z>=1.5, VIX pct<=30, SPY within 3% of high',cm),('COARSE + VVIX pct<=30',cm2)]:
    c=c.fillna(False)&(idx<idx[-1]-pd.Timedelta(days=130)); ds=idx[c]; eps=[]
    for dt in ds:
        if not eps or (dt-eps[-1][-1]).days>40: eps.append([dt])
        else: eps[-1].append(dt)
    print(f'\n==== {name}: {len(ds)} days, {len(eps)} episodes ====')
    rows=[]
    for e in eps:
        a=e[0]; rows.append({'episode_start':a.date(),'days':len(e),'fwd21':round(F21[a],1),'fwd42':round(F42[a],1),'fwd63':round(F63[a],1),'fwd126':round(F126[a],1),'maxDD42':round(D42[a],1),'maxDD63':round(D63[a],1),'maxDD126':round(D126[a],1),'days_to_-5%':DT5[a],'days_to_-10%':DT10[a]})
    T=pd.DataFrame(rows); print(T.to_string())
    if len(T): print('median:',T[['fwd21','fwd42','fwd63','fwd126','maxDD42','maxDD63','maxDD126']].median().round(1).to_dict(),' P(5% DD/63d)',round((T.maxDD63<=-5).mean(),2),' P(10% DD/126d)',round((T.maxDD126<=-10).mean(),2))
