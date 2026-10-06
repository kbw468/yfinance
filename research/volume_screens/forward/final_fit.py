import pandas as pd, numpy as np, warnings, pickle
import lightgbm as lgb
warnings.filterwarnings('ignore')
def load(panel, extra, tag):
    P = pd.read_pickle(panel).replace([np.inf,-np.inf],np.nan); E = pd.read_pickle(extra).replace([np.inf,-np.inf],np.nan)
    X = P.join(E); X['univ']=tag; return X
A = load('../panel.pkl','extra_lc.pkl','LC'); B = load('../u6/panel.pkl','extra_sm.pkl','SM')
B = B[~B.index.get_level_values('ticker').isin(A.index.get_level_values('ticker').unique())]
D = pd.concat([A,B]).sort_index()
D = D[(np.exp(D.e_logprice)>=3) & (np.exp(D.e_logadv)>=1e6)]
FEATS = [c for c in D.columns if not (c.startswith('fwd') or c.startswith('rel_') or c=='univ')]
g = D.groupby(level='date')
R = g[FEATS].rank(pct=True).sub(0.5).fillna(0)
dates = D.index.get_level_values('date').unique().sort_values()
model = {'feats':FEATS}
for h in (21,63):
    y = g[f'fwd{h}_ra'].rank(pct=True).sub(0.5)
    ok = y.notna().values
    Xt, yt = R.values[ok], y.values[ok]
    w = np.linalg.solve(Xt.T@Xt + 0.05*len(yt)*np.eye(Xt.shape[1]), Xt.T@yt)
    model[f'ridge{h}'] = w
    m = lgb.LGBMRegressor(n_estimators=250, learning_rate=0.03, num_leaves=15, min_child_samples=3000, subsample=0.7,
                          subsample_freq=1, colsample_bytree=0.6, reg_lambda=10, verbose=-1, n_jobs=4).fit(Xt, yt)
    m.booster_.save_model(f'lgbm{h}.txt')
    print(f'--- h{h} top ridge weights (rank-space; + = higher rank -> better fwd risk-adj return)')
    s = pd.Series(w, index=FEATS).sort_values()
    print(pd.concat([s.head(12), s.tail(12)]).round(4).to_string())
    imp = pd.Series(m.booster_.feature_importance('gain'), index=FEATS).sort_values(ascending=False)
    print('lgbm top gain:', (imp/imp.sum()).head(12).round(3).to_dict())
# reference cross-section = last panel date, raw values
last = dates[-1]
ref = D.xs(last, level='date')[FEATS]
model['ref_date'] = str(last.date()); model['ref'] = ref
Rl = ref.rank(pct=True).sub(0.5).fillna(0).values
for h in (21,63):
    p_r = Rl@model[f'ridge{h}']; p_l = lgb.Booster(model_file=f'lgbm{h}.txt').predict(Rl)
    model[f'ref_pred_ridge{h}'] = np.sort(p_r); model[f'ref_pred_lgbm{h}'] = np.sort(p_l)
    model[f'ref_pred_blend{h}'] = np.sort(pd.Series(p_r).rank(pct=True).values + pd.Series(p_l).rank(pct=True).values)
pickle.dump(model, open('fwd_model.pkl','wb'))
print('ref date', last.date(), 'ref names', len(ref))
