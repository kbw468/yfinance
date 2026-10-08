#!/usr/bin/env python3
"""v2 Stage 12: is the model anchored on old markets? Run from risk_regime/work after model2.py and decision.py.

The production fit is an expanding window: each year's model learns from every session since 1993, so the
2026 model is about 95% pre-2024 data. Its inputs (trailing ranks) and the decision thresholds (trailing
756-session quantiles) are local; the learned mapping is not. This script asks whether that costs anything:
 (1) out-of-sample AUC of the production probabilities by calendar year, 2015 to date;
 (2) the same pipeline (same in-window screen, K15 depth-2 monotone trees, 100-day purge) refit yearly on a
     ROLLING window of only the last 5 or the last 8 years before each test year, scored on the same years;
 (3) the reference rule run on the rolling-window drawdown probability, 2023 to date and 2015 to date;
 (4) which features the last-window screen picks under each window.
Writes RECENT.pkl.
"""
import sys, numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore'); sys.path.insert(0, '../v2'); import wf
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
pd.set_option('display.width', 300)
idx = wf.idx; YEARS = wf.YEARS; DATES = wf.DATES; PRED = pd.read_pickle('PRED2.pkl')
spy = wf.O['T']['SPY']['Close'].reindex(idx); r = np.log(spy).diff()
def fit_rolling(y, ok, span, first=2015, k=15):
    pred = np.full(len(y), np.nan); last = None
    for year in range(first, int(YEARS.max()) + 1):
        ts = pd.Timestamp(f'{year}-01-01'); te = pd.Timestamp(f'{year}-12-31'); start = pd.Timestamp(f'{year - span}-01-01') - pd.Timedelta(days=100)
        tr = ok & np.asarray((DATES < ts - pd.Timedelta(days=100)) & (DATES >= start)); test = np.asarray((DATES >= ts) & (DATES <= te))
        if tr.sum() < 500 or test.sum() == 0 or len(np.unique(y[tr])) < 2: continue
        rows = np.where(tr)[0]; top, signs, _ = wf.screen(rows, y, k)
        clf = HistGradientBoostingClassifier(monotonic_cst=signs, **wf.PAR).fit(wf.XV[rows][:, top], y[rows])
        pred[test] = clf.predict_proba(wf.XV[test][:, top])[:, 1]; last = list(wf.COLS[top])
    return pred, last
def auc_years(pred, y, ok, years):
    out = {}
    for yy in years:
        m = ok & ~np.isnan(pred) & (YEARS == yy)
        out[yy] = round(float(roc_auc_score(y[m], pred[m])), 3) if m.sum() > 50 and len(np.unique(y[m])) == 2 else None
    return out
def pooled(pred, y, ok, a, b):
    m = ok & ~np.isnan(pred) & (YEARS >= a) & (YEARS <= b); return round(float(roc_auc_score(y[m], pred[m])), 3) if len(np.unique(y[m])) == 2 else None
YRS = list(range(2015, 2027)); OUT = {'auc': {}, 'pooled': {}, 'features': {}, 'rules': None}; ROLL = {}
for t in ('P_off', 'P_on', 'P_vol'):
    y, ok = wf.label(t); prod = PRED[t].to_numpy()
    OUT['auc'][(t, 'expanding since 1993 (production)')] = auc_years(prod, y, ok, YRS)
    OUT['pooled'][(t, 'expanding since 1993 (production)')] = {'2015-26': pooled(prod, y, ok, 2015, 2026), '2020-26': pooled(prod, y, ok, 2020, 2026), '2023-26': pooled(prod, y, ok, 2023, 2026), 'positives 2023-26': int(y[ok & (YEARS >= 2023)].sum())}
    for span in (5, 8):
        p, last = fit_rolling(y, ok, span); ROLL[(t, span)] = p; nm = f'rolling last {span} years'
        OUT['auc'][(t, nm)] = auc_years(p, y, ok, YRS); OUT['pooled'][(t, nm)] = {'2015-26': pooled(p, y, ok, 2015, 2026), '2020-26': pooled(p, y, ok, 2020, 2026), '2023-26': pooled(p, y, ok, 2023, 2026), 'positives 2023-26': int(y[ok & (YEARS >= 2023)].sum())}
        OUT['features'][(t, nm)] = last
A = pd.DataFrame(OUT['auc']).T; A.index.names = ['target', 'fit']; print('=== out-of-sample AUC by calendar year (None = no positives that year) ==='); print(A.to_string())
Pz = pd.DataFrame(OUT['pooled']).T; Pz.index.names = ['target', 'fit']; print('\n=== pooled out-of-sample AUC ==='); print(Pz.to_string())
print('\n=== features the 2026 fit uses ===')
LF = pd.read_pickle('LASTFIT2.pkl')
for t in ('P_off', 'P_on', 'P_vol'):
    print(f'{t} expanding: {", ".join(LF[t][0])}')
    for span in (5, 8): print(f'{t} rolling {span}y: {", ".join(OUT["features"][(t, f"rolling last {span} years")] or [])}')
def tq(s, q, w=756): return s.rolling(w, min_periods=252).quantile(q).shift(1)
def states(o_, i_):
    o = o_.fillna(False).values; i = i_.fillna(False).values; st = np.ones(len(o), dtype=int); s = 1
    for k in range(len(o)):
        if s == 1 and o[k]: s = 0
        elif s == 0 and i[k]: s = 1
        st[k] = s
    return pd.Series(st, index=idx)
rows = []; SPELLS = {}
for name, PF in (('reference rule, expanding fit (production)', PRED['P_off']), ('reference rule, rolling 5-year fit', pd.Series(ROLL[('P_off', 5)], index=idx)), ('reference rule, rolling 8-year fit', pd.Series(ROLL[('P_off', 8)], index=idx))):
    S = states(PF >= tq(PF, 0.95), PF < tq(PF, 0.8)); w = S.shift(1); SPELLS[name] = S
    for a in ('2016-01-01', '2023-01-01'):
        m = (idx >= pd.Timestamp(a)) & ~np.isnan(PF.to_numpy()); x = (w * r)[m]; c = x.cumsum(); dd = c - c.cummax()
        o = pd.Series(((S == 0) & m).to_numpy(dtype=bool), index=idx); sp = int((o & ~o.shift(1, fill_value=False)).sum())
        rows.append({'rule': name, 'from': a[:4], 'ann %': round(x.mean() * 252 * 100, 1), 'max DD %': round((np.exp(dd.min()) - 1) * 100, 1), 'times OUT': sp})
for a in ('2016-01-01', '2023-01-01'):
    m = idx >= pd.Timestamp(a); x = r[m]; c = x.cumsum(); dd = c - c.cummax(); rows.append({'rule': 'SPY buy and hold', 'from': a[:4], 'ann %': round(x.mean() * 252 * 100, 1), 'max DD %': round((np.exp(dd.min()) - 1) * 100, 1), 'times OUT': 0})
R = pd.DataFrame(rows).sort_values(['from', 'rule']); OUT['rules'] = R; print('\n=== the reference rule on each fit, 2020 included, log returns ==='); print(R.to_string(index=False))
today = {f'{t} rolling {s}y': (round(float(ROLL[(t, s)][-1]), 3), round(float((pd.Series(ROLL[(t, s)]).iloc[-757:-1] <= ROLL[(t, s)][-1]).mean()), 2)) for t in ('P_off', 'P_on', 'P_vol') for s in (5, 8)}
print('\ntoday (value, trailing percentile):', today, '| production P_off', round(float(PRED['P_off'].iloc[-1]), 3)); OUT['today'] = today
print('\n=== OUT spells since 2023 on each fit (SPY move while out, log %) ===')
for name, S in SPELLS.items():
    o = (S == 0) & (idx >= pd.Timestamp('2023-01-01')); sp = []; cur = None
    for d in idx[o]:
        if cur is None or idx.get_loc(d) - idx.get_loc(cur[1]) > 1:
            if cur: sp.append(cur)
            cur = [d, d]
        else: cur[1] = d
    if cur: sp.append(cur)
    print(f'{name}: ' + '; '.join(f'{a.date()} to {b.date()} ({float(np.log(spy.loc[b] / spy.loc[a]) * 100):+.1f}%)' for a, b in sp))
    OUT.setdefault('spells', {})[name] = [(str(a.date()), str(b.date())) for a, b in sp]
pd.to_pickle(OUT, 'RECENT.pkl')
