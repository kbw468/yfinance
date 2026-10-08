"""Shared walk-forward machinery for v2. Import from risk_regime/work.

- Loads the feature store (F + MF), labels, SPY sessions.
- Candidate features exclude anything built on KRE (regional banks), by the user's direction.
- Global percentile ranks are computed once; the per-window screen is a Pearson correlation of those
  ranks with the target inside each half of the training window (a monotone transform of the within-
  window Spearman), which makes selection cheap enough to repeat for every year and every null draw.
- fit_predict_walk(): yearly refits on data whose labels end before the test year (purge of 100 calendar
  days by default, longer for longer label windows), monotone depth-2 gradient-boosted trees on the K selected features, no parameter tuning.
"""
import pandas as pd, numpy as np, time, warnings; warnings.filterwarnings('ignore')
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
_t0 = time.time()
F = pd.read_pickle('F.pkl'); MF = pd.read_pickle('MF.pkl'); L = pd.read_pickle('LAB.pkl'); O = pd.read_pickle('OHLCV.pkl'); idx = F.index
X = pd.concat([F, MF], axis=1); X = X.loc[:, [c for c in X.columns if not c.startswith('KRE:')]]
COLS = np.array(X.columns); XV = X.to_numpy(dtype='float32')
XR = X.rank(pct=True).to_numpy(dtype='float32'); NANSHARE = np.isnan(XV).mean(axis=0); XR = np.where(np.isnan(XR), 0.5, XR)
EX = np.asarray((idx >= pd.Timestamp('2020-02-01')) & (idx <= pd.Timestamp('2020-07-31')))
YEARS = np.array(idx.year); DATES = idx
ERAS = {'2005-12': (2005, 2012), '2013-19': (2013, 2019), '2020-26': (2020, 2026), 'all': (2005, 2026)}
PAR = dict(learning_rate=0.05, max_iter=200, max_bins=64, random_state=0, max_depth=2, min_samples_leaf=300, l2_regularization=2.0)
TARGETS = {'P_vol': 'volexp21', 'P_on': 'riskon21', 'P_off': 'riskoff21', 'P_off63': 'riskoff63'}
def label(name):
    lab = TARGETS.get(name, name); okc = 'ok63' if lab.endswith('63') else 'ok21'
    y = L[lab].to_numpy(dtype='float64'); ok = L[okc].to_numpy(dtype=bool) & ~EX & ~np.isnan(y)
    return y, ok
def screen(rows, y, k):
    """rows: training row indices (in time order). Returns (feature indices, signs, scores)."""
    h = len(rows) // 2; ics = []
    for part in (rows[:h], rows[h:]):
        A = XR[part]; yy = y[part]; Ac = A - A.mean(axis=0); yc = yy - yy.mean()
        ic = (Ac * yc[:, None]).sum(axis=0) / (np.sqrt((Ac ** 2).sum(axis=0)) * np.sqrt((yc ** 2).sum()) + 1e-9); ics.append(ic)
    ic1, ic2 = ics; valid = (np.sign(ic1) == np.sign(ic2)) & (NANSHARE < 0.5) & np.isfinite(ic1) & np.isfinite(ic2)
    score = np.where(valid, np.minimum(np.abs(ic1), np.abs(ic2)), 0.0); top = np.argsort(-score)[:k]
    return top, np.sign(ic1 + ic2)[top].astype(int), score[top]
def fit_predict_walk(y, ok, k=15, first=2005, mono=True, depth=2, leaf=300, iters=200, keep_last=False, years=None, purge=100):
    pred = np.full(len(y), np.nan); last = None; selcount = {}
    for year in range(first, int(YEARS.max()) + 1):
        ts = pd.Timestamp(f'{year}-01-01'); te = pd.Timestamp(f'{year}-12-31')
        tr = ok & np.asarray(DATES < ts - pd.Timedelta(days=purge)) & (YEARS >= 1993); test = np.asarray((DATES >= ts) & (DATES <= te))
        if tr.sum() < 1000 or test.sum() == 0: continue
        rows = np.where(tr)[0]; top, signs, score = screen(rows, y, k)
        for f in top: selcount[f] = selcount.get(f, 0) + 1
        p = dict(PAR); p.update(max_depth=depth, min_samples_leaf=leaf, max_iter=iters)
        clf = HistGradientBoostingClassifier(monotonic_cst=(signs if mono else None), **p).fit(XV[rows][:, top], y[rows])
        pred[test] = clf.predict_proba(XV[test][:, top])[:, 1]
        if keep_last: last = (list(COLS[top]), signs, score, clf)
    return pred, selcount, last
def auc_by_era(pred, y, ok):
    out = {}
    for e, (a, b) in ERAS.items():
        m = ok & ~np.isnan(pred) & (YEARS >= a) & (YEARS <= b)
        if m.sum() > 100 and len(np.unique(y[m])) == 2: out[e] = round(float(roc_auc_score(y[m], pred[m])), 3)
    return out
def quintiles(pred, y, ok, a=2005, b=2026):
    m = ok & ~np.isnan(pred) & (YEARS >= a) & (YEARS <= b); pp = pd.Series(pred[m]); q = pd.qcut(pp.rank(method='first'), 5, labels=False)
    g = pd.Series(y[m]).groupby(q.values).mean(); return [round(float(v), 3) for v in g]
def year_bootstrap_auc(pred, y, ok, reps=500, seed=0):
    """block bootstrap over calendar years of the pooled out-of-sample AUC"""
    rng = np.random.default_rng(seed); m = ok & ~np.isnan(pred) & (YEARS >= 2005); yrs = np.unique(YEARS[m]); vals = []
    for _ in range(reps):
        pick = rng.choice(yrs, size=len(yrs), replace=True); sel = np.concatenate([np.where(m & (YEARS == yy))[0] for yy in pick])
        if len(np.unique(y[sel])) == 2: vals.append(roc_auc_score(y[sel], pred[sel]))
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 50)), float(np.percentile(vals, 97.5))
print(f'wf: X {XV.shape}, candidates exclude KRE, ranks ready in {time.time()-_t0:.0f}s')
