#!/usr/bin/env python3
"""v2 Stage 8c: the record before 2006. Run from risk_regime/work after model2.py and decision.py.

The main grid starts in 2006 because the walk-forward starts in 2005. Refitting yearly from 1998 instead, on the
1993+ history (which in those years holds little beyond SPY, the VIX family and rates), extends the out-of-sample
record of the reference rule, of pure price triggers and of vol-filtered price triggers across the 1999 whipsaw
year and the 2000-02 bear market. Predictions from 2005 on are identical to the main run by construction (each
year's fit uses the same training rows), so only the 2006-07 thresholds differ (a full three-year window instead of
a warm-up one). OUT conditions use >= as in decision.py. Writes EARLY.pkl.
"""
import sys, numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore'); sys.path.insert(0, '../v2'); import wf
from sklearn.metrics import roc_auc_score
pd.set_option('display.width', 320)
idx = wf.idx; spy = wf.O['T']['SPY']['Close'].reindex(idx); r = np.log(spy).diff(); YEARS = wf.YEARS
P = {}
for t in ('P_off', 'P_vol'):
    y, ok = wf.label(t); pred, _, _ = wf.fit_predict_walk(y, ok, k=15, first=1998); P[t] = pd.Series(pred, index=idx)
    e = ok & (YEARS >= 1998) & (YEARS <= 2004); print(f'{t}: AUC 1998-2004 {roc_auc_score(y[e], pred[e]):.3f}; by era {wf.auc_by_era(pred, y, ok)}')
PF, PV = P['P_off'], P['P_vol']
def tq(s, q, w=756): return s.rolling(w, min_periods=252).quantile(q).shift(1)
def states(out_cond, in_cond):
    o = out_cond.fillna(False).values; i = in_cond.fillna(False).values; st = np.ones(len(o), dtype=int); s = 1
    for k in range(len(o)):
        if s == 1 and o[k]: s = 0
        elif s == 0 and i[k]: s = 1
        st[k] = s
    return pd.Series(st, index=idx)
hi63 = spy.rolling(63, min_periods=63).max(); dd63 = spy / hi63 - 1; newhigh42 = spy >= spy.rolling(42, min_periods=42).max()
off95 = PF >= tq(PF, 0.95); calm = (PF < tq(PF, 0.8)) & (PV < tq(PV, 0.7)); alert = (PV >= tq(PV, 0.9)).astype(float).rolling(42, min_periods=1).max() > 0
ST = {'reference C: OUT P_off>=q.95, IN P_off<q.8': states(off95, PF < tq(PF, 0.8)), 'C strict: OUT P_off>q.95, IN P_off<q.8': states(PF > tq(PF, 0.95), PF < tq(PF, 0.8))}
for x in (0.05, 0.07, 0.10):
    D = dd63 <= -x
    ST[f'price {int(x*100)}% below 63d high | IN new 42d high'] = states(D, newhigh42)
    ST[f'price {int(x*100)}% below 63d high | IN calm or new 42d high'] = states(D, calm | newhigh42)
    ST[f'vol alert & price {int(x*100)}% | IN new 42d high'] = states(alert & D, newhigh42)
    ST[f'price {int(x*100)}% or P_off>=q.95 | IN new 42d high'] = states(D | off95, newhigh42)
EV = [(pd.Timestamp(a), pd.Timestamp(b), d) for a, b, d in [('2000-03-24', '2001-04-03', -27.3), ('2001-05-21', '2001-09-21', -25.6), ('2002-03-19', '2002-07-23', -31.7), ('2002-08-22', '2002-10-09', -18.9), ('2007-10-09', '2008-03-10', -17.8), ('2008-05-19', '2008-10-10', -37.5), ('2008-10-13', '2008-10-27', -17.2), ('2008-11-04', '2008-11-20', -24.9), ('2009-01-06', '2009-03-09', -27.1), ('2010-04-23', '2010-07-02', -15.7), ('2011-04-29', '2011-10-03', -18.6), ('2018-09-20', '2018-12-24', -19.3), ('2022-01-03', '2022-06-16', -23.0), ('2022-08-16', '2022-10-12', -16.7), ('2025-02-19', '2025-04-08', -18.8)]]
EXM = (idx >= pd.Timestamp('2020-02-01')) & (idx <= pd.Timestamp('2020-07-31'))
def stats(x):
    c = x.cumsum(); dd = c - c.cummax(); return dict(ann=round(x.mean() * 252 * 100, 2), sharpe=round(x.mean() / x.std() * np.sqrt(252), 2), maxDD=round(dd.min() * 100, 1))
def score(S, name, a0, b0):
    m = ~EXM & (idx >= pd.Timestamp(a0)) & (idx <= pd.Timestamp(b0)); w = S.shift(1); x = (w * r)[m]; row = {'rule': name, **stats(x), 'exposure': round(float(w[m].mean()), 2)}
    o = (S == 0) & m; spells = 0; last = None
    for d in idx[o]:
        if last is None or idx.get_loc(d) - idx.get_loc(last) > 1: spells += 1
        last = d
    row['spells'] = spells; caps = []
    for a, b, d in EV:
        if a < pd.Timestamp(a0) or b > pd.Timestamp(b0): continue
        seg = r.loc[a:b].iloc[1:]; caps.append(round(float((w.loc[seg.index] * seg).sum() / seg.sum()), 2))
    row['taken per 15% event'] = caps; row['mean taken'] = round(float(np.mean(caps)), 2) if caps else None; return row
TABLES = {}; S = ST['reference C: OUT P_off>=q.95, IN P_off<q.8']; w = S.shift(1)
print('\nreference rule, 1999-2005, yearly book vs SPY (log %):'); YR = []
for yr in range(1999, 2006):
    mm = (YEARS == yr); YR.append({'year': yr, 'book %': round(float((w * r)[mm].sum() * 100), 1), 'SPY %': round(float(r[mm].sum() * 100), 1), 'exposure': round(float(w[mm].mean()), 2)}); print(f'  {yr}: book {YR[-1]["book %"]:6.1f}   SPY {YR[-1]["SPY %"]:6.1f}   exposure {YR[-1]["exposure"]:.2f}')
for a0, b0, title in (('1999-01-01', '2005-12-31', '1999-2005: the 2000-02 bear (models trained on 5 to 11 years, mostly SPY, VIX and rates)'), ('2006-01-01', '2026-12-31', '2006-2026 with full three-year warm-up windows'), ('1999-01-01', '2026-12-31', '1999-2026, the whole out-of-sample record, 2020 excluded')):
    m = ~EXM & (idx >= pd.Timestamp(a0)) & (idx <= pd.Timestamp(b0)); bh = stats(r[m]); T = pd.DataFrame([{'rule': 'SPY buy and hold', **bh, 'exposure': 1.0, 'spells': 0, 'taken per 15% event': [], 'mean taken': 1.0}] + [score(S_, n, a0, b0) for n, S_ in ST.items()]); TABLES[title] = T
    print(f'\n=== {title} ===  SPY buy and hold: ann {bh["ann"]} sharpe {bh["sharpe"]} maxDD {bh["maxDD"]}'); print(T.to_string(index=False))
ties = (PF == tq(PF, 0.95)) & (idx >= '2006-01-01'); print(f'\nexact ties P_off == trailing q95 since 2006: {int(ties.sum())} sessions (tree outputs are discrete); the strict rule differs from the reference on these days')
pd.to_pickle({'tables': TABLES, 'yearly': YR, 'ties': int(ties.sum()), 'auc_early': {t: round(float(roc_auc_score(wf.label(t)[0][wf.label(t)[1] & (YEARS >= 1998) & (YEARS <= 2004)], P[t].to_numpy()[wf.label(t)[1] & (YEARS >= 1998) & (YEARS <= 2004)])), 3) for t in P}}, 'EARLY.pkl'); print('EARLY.pkl written')
