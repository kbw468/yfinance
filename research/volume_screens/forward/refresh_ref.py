"""Rebuild the reference cross-section from Nasdaq data so live scores rank against a current universe.
Replaces the reference only if at least 90% of names refresh. Run occasionally (weekly is plenty)."""
import sys, os, pickle, time, numpy as np, pandas as pd, lightgbm as lgb
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE); sys.path.insert(0, os.path.dirname(HERE))
from fwdfeat import features
from rate import history
from fwd_rate import shares_out
path = os.path.join(HERE,'fwd_model.pkl'); M = pickle.load(open(path,'rb')); FEATS = M['feats']
names = list(M['ref'].index); spy = history('SPY', years=4); rows = {}
for i,t in enumerate(names):
    try:
        x = history(t, years=4)
        d = pd.concat({t:x,'SPY':spy}, axis=1).swaplevel(0,1,axis=1).sort_index(axis=1).dropna(subset=[('Close','SPY')])
        if d['Volume'][t].iloc[-1] < 0.6*d['Volume'][t].iloc[-22:-1].mean(): d = d.iloc[:-1]
        O,H,L,C,V = [d[k] for k in ['Open','High','Low','Close','Volume']]
        f = features(O,H,L,C,V, {t: shares_out(t, C[t].iloc[-1])}).loc[t]
        if C[t].iloc[-1] >= 3 and np.exp(f['e_logadv']) >= 1e6: rows[t] = f[FEATS]
    except Exception: pass
    time.sleep(0.3)
    if i % 200 == 0: print(i, len(rows), flush=True)
if len(rows) < 0.9*len(names): sys.exit(f'only {len(rows)}/{len(names)} refreshed; reference left unchanged')
ref = pd.DataFrame(rows).T; Rl = ref.rank(pct=True).sub(0.5).fillna(0).values
for h in (21,63):
    pr = Rl@M[f'ridge{h}']; pl = lgb.Booster(model_file=os.path.join(HERE,f'lgbm{h}.txt')).predict(Rl)
    M[f'ref_pred_ridge{h}'] = np.sort(pr); M[f'ref_pred_lgbm{h}'] = np.sort(pl)
    M[f'ref_pred_blend{h}'] = np.sort(pd.Series(pr).rank(pct=True).values + pd.Series(pl).rank(pct=True).values)
M['ref'] = ref; M['ref_date'] = str(pd.Timestamp.today().date())
pickle.dump(M, open(path,'wb')); print('reference refreshed:', len(ref), 'names')
