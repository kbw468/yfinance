#!/usr/bin/env python3
"""v2 Stage 12b: recency-weighted fits. Run from risk_regime/work after recent.py.

Rolling windows (recent.py) throw away the older selloffs and land at chance on the drawdown target. The middle
path keeps every session since 1993 but weights each training row by 0.5 ** (age in years / half-life), so the
last few years dominate both the feature screen (weighted rank correlations, same sign required in both halves)
and the tree fit, while the older selloffs still count. Two half-lives, fixed before the run: 3 and 5 years.
Same pipeline otherwise (K15, depth-2 monotone trees, yearly refits, 100-day purge). Scored out of sample by year,
pooled 2015-26 / 2020-26 / 2023-26, and through the reference rule. Writes RECENCY.pkl.
"""
import sys, numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore'); sys.path.insert(0, '../v2'); import wf
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
pd.set_option('display.width', 300)
idx = wf.idx; YEARS = wf.YEARS; DATES = wf.DATES; PRED = pd.read_pickle('PRED2.pkl'); spy = wf.O['T']['SPY']['Close'].reindex(idx); r = np.log(spy).diff()
AGE0 = np.asarray((DATES - DATES[0]).days, dtype='float64') / 365.25
def wscreen(rows, y, w, k):
    h = len(rows) // 2; ics = []
    for part in (rows[:h], rows[h:]):
        A = wf.XR[part].astype('float64'); yy = y[part]; ww = w[part] / w[part].sum()
        Ac = A - (ww[:, None] * A).sum(axis=0); yc = yy - (ww * yy).sum()
        cov = (ww[:, None] * Ac * yc[:, None]).sum(axis=0); ic = cov / (np.sqrt((ww[:, None] * Ac ** 2).sum(axis=0)) * np.sqrt((ww * yc ** 2).sum()) + 1e-12); ics.append(ic)
    ic1, ic2 = ics; valid = (np.sign(ic1) == np.sign(ic2)) & (wf.NANSHARE < 0.5) & np.isfinite(ic1) & np.isfinite(ic2)
    score = np.where(valid, np.minimum(np.abs(ic1), np.abs(ic2)), 0.0); top = np.argsort(-score)[:k]
    return top, np.sign(ic1 + ic2)[top].astype(int)
def fit_weighted(y, ok, half, first=2015, k=15):
    pred = np.full(len(y), np.nan); last = None
    for year in range(first, int(YEARS.max()) + 1):
        ts = pd.Timestamp(f'{year}-01-01'); te = pd.Timestamp(f'{year}-12-31')
        tr = ok & np.asarray(DATES < ts - pd.Timedelta(days=100)) & (YEARS >= 1993); test = np.asarray((DATES >= ts) & (DATES <= te))
        rows = np.where(tr)[0]; age = (ts - DATES[rows]).days.to_numpy() / 365.25; w = np.zeros(len(y)); w[rows] = 0.5 ** (age / half)
        top, signs = wscreen(rows, y, w, k)
        clf = HistGradientBoostingClassifier(monotonic_cst=signs, **wf.PAR).fit(wf.XV[rows][:, top], y[rows], sample_weight=w[rows] / w[rows].mean())
        pred[test] = clf.predict_proba(wf.XV[test][:, top])[:, 1]; last = list(wf.COLS[top])
    return pred, last
def auc_y(p, y, ok, yy):
    m = ok & ~np.isnan(p) & (YEARS == yy); return round(float(roc_auc_score(y[m], p[m])), 3) if m.sum() > 50 and len(np.unique(y[m])) == 2 else None
def pooled(p, y, ok, a, b):
    m = ok & ~np.isnan(p) & (YEARS >= a) & (YEARS <= b); return round(float(roc_auc_score(y[m], p[m])), 3)
YRS = list(range(2015, 2027)); AUC = {}; POOL = {}; FEATS = {}; PW = {}
for t in ('P_off', 'P_on', 'P_vol'):
    y, ok = wf.label(t); fits = {'expanding, equal weight (production)': PRED[t].to_numpy()}
    for h in (3, 5):
        p, last = fit_weighted(y, ok, h); fits[f'all history, half-life {h} years'] = p; FEATS[(t, h)] = last; PW[(t, h)] = p
    for nm, p in fits.items():
        AUC[(t, nm)] = {yy: auc_y(p, y, ok, yy) for yy in YRS}; POOL[(t, nm)] = {'2015-26': pooled(p, y, ok, 2015, 2026), '2020-26': pooled(p, y, ok, 2020, 2026), '2023-26': pooled(p, y, ok, 2023, 2026)}
    print(f'{t} done', flush=True)
A = pd.DataFrame(AUC).T; print('=== out-of-sample AUC by year ==='); print(A.to_string())
P = pd.DataFrame(POOL).T; print('\n=== pooled out-of-sample AUC ==='); print(P.to_string())
print('\n=== features in the 2026 fit ===')
for (t, h), f in FEATS.items(): print(f'{t} half-life {h}y: {", ".join(f)}')
def tq(s, q, w=756): return s.rolling(w, min_periods=252).quantile(q).shift(1)
def states(o_, i_):
    o = o_.fillna(False).values; i = i_.fillna(False).values; st = np.ones(len(o), dtype=int); s = 1
    for k in range(len(o)):
        if s == 1 and o[k]: s = 0
        elif s == 0 and i[k]: s = 1
        st[k] = s
    return pd.Series(st, index=idx)
rows = []; SP = {}
for nm, PF in (('production', PRED['P_off']), ('half-life 3y', pd.Series(PW[('P_off', 3)], index=idx)), ('half-life 5y', pd.Series(PW[('P_off', 5)], index=idx))):
    S = states(PF >= tq(PF, 0.95), PF < tq(PF, 0.8)); w = S.shift(1); SP[nm] = S
    for a in ('2016-01-01', '2023-01-01'):
        m = (idx >= pd.Timestamp(a)) & ~np.isnan(PF.to_numpy()); x = (w * r)[m]; c = x.cumsum(); dd = c - c.cummax()
        o = pd.Series(((S == 0) & m).to_numpy(dtype=bool), index=idx); rows.append({'rule on': nm, 'from': a[:4], 'ann %': round(x.mean() * 252 * 100, 1), 'max DD %': round((np.exp(dd.min()) - 1) * 100, 1), 'times OUT': int((o & ~o.shift(1, fill_value=False)).sum())})
for a in ('2016-01-01', '2023-01-01'):
    x = r[idx >= pd.Timestamp(a)]; c = x.cumsum(); dd = c - c.cummax(); rows.append({'rule on': 'SPY buy and hold', 'from': a[:4], 'ann %': round(x.mean() * 252 * 100, 1), 'max DD %': round((np.exp(dd.min()) - 1) * 100, 1), 'times OUT': 0})
R = pd.DataFrame(rows).sort_values(['from', 'rule on']); print('\n=== reference rule (OUT at trailing 95th, IN below 80th) on each fit, 2020 included ==='); print(R.to_string(index=False))
print('\n=== OUT spells since 2023 ===')
for nm, S in SP.items():
    o = (S == 0) & (idx >= pd.Timestamp('2023-01-01')); sp = []; cur = None
    for d in idx[o]:
        if cur is None or idx.get_loc(d) - idx.get_loc(cur[1]) > 1:
            if cur: sp.append(cur)
            cur = [d, d]
        else: cur[1] = d
    if cur: sp.append(cur)
    print(f'{nm}: ' + ('; '.join(f'{a.date()} to {b.date()} (SPY {float(np.log(spy.loc[b] / spy.loc[a]) * 100):+.1f}%)' for a, b in sp) or 'none'))
today = {f'{t} half-life {h}y': (round(float(PW[(t, h)][-1]), 3), round(float((pd.Series(PW[(t, h)]).iloc[-757:-1] <= PW[(t, h)][-1]).mean()), 2)) for t in ('P_off', 'P_on', 'P_vol') for h in (3, 5)}
print('\ntoday (value, trailing percentile):', today)
pd.to_pickle({'auc': A, 'pooled': P, 'rules': R, 'today': today, 'pred': {f'{t} hl{h}': PW[(t, h)] for t in ('P_off', 'P_on', 'P_vol') for h in (3, 5)}}, 'RECENCY.pkl')
