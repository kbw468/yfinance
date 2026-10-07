import pandas as pd, numpy as np, warnings; warnings.filterwarnings('ignore')
pd.set_option('display.width',300); pd.set_option('display.max_rows',200)
G=pd.read_pickle('G.pkl'); P=pd.read_pickle('P.pkl'); idx=P.index; spy=P['SPY']; lspy=np.log(spy)
EX=(idx>=pd.Timestamp('2020-02-01'))&(idx<=pd.Timestamp('2020-07-31'))
fwd=lambda h:(lspy.shift(-h)-lspy)*100; fdd=lambda h: np.log(spy[::-1].rolling(h).min()[::-1].shift(-1)/spy)*100
F21,F42,F63,F126=fwd(21),fwd(42),fwd(63),fwd(126); D42,D63,D126=fdd(42),fdd(63),fdd(126)
def dt(th,h=126):
    out=pd.Series(np.nan,index=idx); a=lspy.values
    for i in range(len(a)-1):
        w=a[i+1:i+1+h]-a[i]; k=np.where(w<=np.log(1+th))[0]
        if len(k): out.iloc[i]=k[0]+1
    return out
DT5=dt(-0.05); DT10=dt(-0.10)
recent=idx<idx[-1]-pd.Timedelta(days=130)
def run(name,c):
    c=c.fillna(False)&~EX&recent; ds=idx[c]; eps=[]
    for d in ds:
        if not eps or (d-eps[-1][-1]).days>40: eps.append([d])
        else: eps[-1].append(d)
    print(f'\n==== {name}: {len(ds)} days, {len(eps)} episodes ====')
    rows=[{'start':e[0].date(),'days':len(e),'fwd21':round(F21[e[0]],1),'fwd42':round(F42[e[0]],1),'fwd63':round(F63[e[0]],1),'fwd126':round(F126[e[0]],1),'maxDD42':round(D42[e[0]],1),'maxDD63':round(D63[e[0]],1),'maxDD126':round(D126[e[0]],1),'d_to_-5%':DT5[e[0]],'d_to_-10%':DT10[e[0]]} for e in eps]
    T=pd.DataFrame(rows); print(T.to_string())
    if len(T): print('median:',T[['fwd21','fwd42','fwd63','fwd126','maxDD42','maxDD63','maxDD126']].median().round(1).to_dict(),' P(5% DD/63d)',round((T.maxDD63<=-5).mean(),2),' P(10% DD/126d)',round((T.maxDD126<=-10).mean(),2))
base=(idx>=pd.Timestamp('2003-01-01'))&~EX&recent&F126.notna()
print('baseline 2003+: median fwd63',round(F63[base].median(),1),' fwd126',round(F126[base].median(),1),' maxDD63',round(D63[base].median(),1),' maxDD126',round(D126[base].median(),1),' P(5%/63d)',round((D63[base]<=-5).mean(),2),' P(10%/126d)',round((D126[base]<=-10).mean(),2))
run('A: TNX pct>=90 & TNX 21d chg z>=1.5 & VIX pct<=30 & SPY within 3% of 63d high (1991+)',(G['TNX_pct252']>=0.9)&(G['TNX_chg21_z252']>=1.5)&(G['VIX_pct252']<=0.3)&(G['SPY_dd63']>=-0.03))
run('B: A + MOVE pct>=80 (2003+)',(G['TNX_pct252']>=0.9)&(G['TNX_chg21_z252']>=1.5)&(G['VIX_pct252']<=0.3)&(G['SPY_dd63']>=-0.03)&(G['MOVE_pct252']>=0.8))
run('C: B + VVIX pct<=25 (2008+)',(G['TNX_pct252']>=0.9)&(G['TNX_chg21_z252']>=1.5)&(G['VIX_pct252']<=0.3)&(G['SPY_dd63']>=-0.03)&(G['MOVE_pct252']>=0.8)&(G['VVIX_pct252']<=0.25))
run('D: TYX pct>=95 & TYX 21d chg z>=2 & VIX pct<=20 (1991+)',(G['TYX_pct252']>=0.95)&(G['TYX_chg21_z252']>=2)&(G['VIX_pct252']<=0.2))
run('E: MOVE/VIX ratio pct>=95 & SPY within 3% of high (2003+)',(G['X_MOVE_VIX_pct252']>=0.95)&(G['SPY_dd63']>=-0.03))
