import pandas as pd, numpy as np, warnings, time, json, sys
import lightgbm as lgb
warnings.filterwarnings('ignore')
t0=time.time()
def load(panel, extra, tag):
    P = pd.read_pickle(panel).replace([np.inf,-np.inf],np.nan)
    E = pd.read_pickle(extra).replace([np.inf,-np.inf],np.nan)
    X = P.join(E); X['univ']=tag; return X
A = load('../panel.pkl','extra_lc.pkl','LC'); B = load('../u6/panel.pkl','extra_sm.pkl','SM')
B = B[~B.index.get_level_values('ticker').isin(A.index.get_level_values('ticker').unique())]
D = pd.concat([A,B]).sort_index()
sec = pd.concat([pd.read_pickle('../sector.pkl'), pd.read_pickle('../u6/sector.pkl')]); sec = sec[~sec.index.duplicated()]
D['sector'] = D.index.get_level_values('ticker').map(sec).fillna('Other')
# tradability filter at the date
D = D[(np.exp(D.e_logprice)>=3) & (np.exp(D.e_logadv)>=1e6)]
drop = [c for c in D.columns if c.startswith('fwd') or c.startswith('rel_')] + ['univ','sector']
FEATS = [c for c in D.columns if c not in drop]
print('rows', len(D), 'features', len(FEATS), 'names', D.index.get_level_values('ticker').nunique(), flush=True)
g = D.groupby([D.index.get_level_values('date'), D['sector']])
R = g[FEATS].rank(pct=True).sub(0.5).fillna(0).astype('float32')
TGT = {21:'fwd21_ra', 63:'fwd63_ra'}
Y = {h: g[t].rank(pct=True).sub(0.5) for h,t in TGT.items()}
dates = D.index.get_level_values('date').unique().sort_values()
didx = {d:i for i,d in enumerate(dates)}
rowdate = D.index.get_level_values('date').map(didx).values
def nw_t(x, lag):
    x=np.asarray(x,float); x=x[~np.isnan(x)]; n=len(x); m=x.mean(); e=x-m; v=(e@e)/n
    for l in range(1,lag+1): v+=2*(1-l/(lag+1))*(e[l:]@e[:-l])/n
    return m/np.sqrt(v/n)
MIN_TRAIN, REFIT = 78, 6
results = {}
Xall = R.values
for h in (21,63):
    y = Y[h].values; emb = h//5 + 1
    preds = {k: np.full(len(D), np.nan) for k in ('ridge','lgbm','sign')}
    for i in range(MIN_TRAIN+emb, len(dates), REFIT):
        tr = (rowdate <= i-emb) & ~np.isnan(y)
        te = (rowdate >= i) & (rowdate < i+REFIT)
        Xt, yt = Xall[tr], y[tr]
        # ridge
        lam = 0.05*len(yt)
        w = np.linalg.solve(Xt.T@Xt + lam*np.eye(Xt.shape[1]), Xt.T@yt)
        preds['ridge'][te] = Xall[te]@w
        # sign composite: per-feature IC t-stat on training dates, keep |t|>2, equal weight by sign
        trd = rowdate[tr]
        df = pd.DataFrame(Xt, columns=FEATS); df['y']=yt; df['d']=trd
        # fast per-date IC: corr of centered ranks
        ics = df.groupby('d').apply(lambda z: z[FEATS].corrwith(z['y']))
        tt = ics.apply(lambda c: nw_t(c.values, h//5))
        sel = tt[tt.abs()>2]
        preds['sign'][te] = Xall[te][:, [FEATS.index(f) for f in sel.index]] @ np.sign(sel.values) / max(len(sel),1) if len(sel) else 0
        # lgbm
        m = lgb.LGBMRegressor(n_estimators=250, learning_rate=0.03, num_leaves=15, min_child_samples=3000,
                              subsample=0.7, subsample_freq=1, colsample_bytree=0.6, reg_lambda=10, verbose=-1, n_jobs=4)
        m.fit(Xt, yt)
        preds['lgbm'][te] = m.predict(Xall[te])
        print(f'h{h} refit @ {dates[i].date()} train_rows {tr.sum()} sel {len(sel)} t={time.time()-t0:.0f}s', flush=True)
    out = pd.DataFrame(preds, index=D.index)
    gs = out.groupby([out.index.get_level_values('date'), D['sector']]); out['blend'] = gs['ridge'].rank(pct=True) + gs['lgbm'].rank(pct=True)
    out['univ'] = D['univ']; out['sector'] = D['sector']
    for c in ['fwd21_ra','fwd63_ra','fwd21_spy','fwd63_spy']: out[c]=D[c]
    out.to_pickle(f'oos_sn_h{h}.pkl')
print('done', time.time()-t0)
