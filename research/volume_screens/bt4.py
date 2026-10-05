import pandas as pd, numpy as np, warnings
warnings.filterwarnings('ignore')
P = pd.read_pickle('panel.pkl').replace([np.inf,-np.inf],np.nan)
sec = pd.read_pickle('sector.pkl'); P['sector']=P.index.get_level_values('ticker').map(sec)
R_sn = pd.read_pickle('R_sn.pkl')          # sector-neutral pct ranks, dates with fwd21 realised
feats = list(R_sn.columns)
dates = R_sn.index.get_level_values('date').unique().sort_values()
Pb = P.loc[R_sn.index]
Tr = Pb.groupby(level='date')[['fwd21_ra','fwd63_ra']].rank(pct=True)
def nw_t(x, lag):
    x=np.asarray(x,float); n=len(x); m=x.mean(); e=x-m; v=(e@e)/n
    for l in range(1,lag+1): v+=2*(1-l/(lag+1))*(e[l:]@e[:-l])/n
    return m/np.sqrt(v/n) if v>0 else np.nan
# per-date IC matrix for each feature vs fwd63_ra (sector-neutral ranks)
IC = {}
for f in feats:
    g = pd.concat([R_sn[f], Tr['fwd63_ra']],axis=1).dropna()
    IC[f] = g.groupby(level='date').apply(lambda x: x.iloc[:,0].corr(x.iloc[:,1]) if len(x)>50 else np.nan)
IC = pd.DataFrame(IC).reindex(dates)
# walk-forward: at date d use IC rows whose fwd63 window ended before d (date <= d - 63 trading days ~ 13 weeks)
comp = pd.Series(np.nan, index=R_sn.index)
sel_hist = {}
for i,d in enumerate(dates):
    if i < 70: continue
    hist = IC.iloc[:i-13].dropna(how='all')
    if len(hist)<52: continue
    t = hist.apply(lambda c: nw_t(c.dropna(),12))
    chosen = t[t.abs()>1.5]
    if len(chosen)==0: continue
    sel_hist[d]=chosen
    sl = R_sn.loc[d, chosen.index]
    comp.loc[d] = ((sl-0.5)*np.sign(chosen)).mean(axis=1).values
Pb = Pb.assign(comp=comp)
oos = Pb[Pb.comp.notna()]
print('OOS dates', oos.index.get_level_values('date').nunique(), 'from', oos.index.get_level_values('date').min().date())
for t,H in (('fwd21_ra',21),('fwd63_ra',63),('fwd21_spy',21),('fwd63_spy',63)):
    g = oos[['comp',t]].dropna()
    ic = g.groupby(level='date').apply(lambda x: x['comp'].corr(x[t],method='spearman'))
    def qs(x):
        q = pd.qcut(x['comp'].rank(method='first'),5,labels=False); m=x[t].groupby(q).mean(); return pd.Series({'q5':m.iloc[-1],'q1':m.iloc[0],'uni':x[t].mean()})
    Q = g.groupby(level='date').apply(qs)
    print(f'{t}: OOS IC {ic.mean():.3f} NW-t {nw_t(ic,H//5):.2f} hit {(ic>0).mean():.2f} | Q5 {Q.q5.mean():.3f} Q1 {Q.q1.mean():.3f} univ {Q.uni.mean():.3f} | Q5-univ t {nw_t(Q.q5-Q.uni,H//5):.2f}')
# how often each feature selected, with sign
cnt = pd.DataFrame(sel_hist).T
print('\nselection frequency (share of OOS dates) and mean sign:')
print(pd.DataFrame({'freq':cnt.notna().mean(), 'sign':np.sign(cnt).mean()}).sort_values('freq',ascending=False).head(15).round(2).to_string())
# full-sample selection for live scoring
tfull = IC.apply(lambda c: nw_t(c.dropna(),12))
live = tfull[tfull.abs()>1.5].sort_values()
print('\nfull-sample selected (|t|>1.5) vs fwd63_ra sector-neutral:'); print(live.round(2).to_string())
live.to_pickle('live_sel.pkl')
