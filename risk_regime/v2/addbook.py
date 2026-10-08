#!/usr/bin/env python3
"""v2 Stage 11: ADD as its own book. Run from risk_regime/work after decision.py.

The action page says ADD when the rule is IN and the out-of-sample rally probability is at or above its trailing
80th percentile. Until now that was a reading of the quintile table. Here it is scored as a book, 2006 to date,
with the same lagged daily rebalance as the rules grid, everything out of sample and the thresholds trailing:

  sleeve   long SPY only on ADD sessions, flat otherwise: what the ADD condition itself earns, with its own
           drawdown, against the same sleeve defined on all IN sessions (the unconditional comparison)
  book     no leverage: a base weight when the rule is IN (75% or 50%), 100% when ADD fires, 0 when OUT, against
           the reference book that is 100% whenever IN. If the ADD book keeps up with the always-full book while
           holding a reserve most of the time, the ADD condition is selecting the sessions that pay.
Variants: trigger at the 80th or 90th percentile; instantaneous, or held until the rally probability falls below
the 60th (hysteresis). Both paths (Feb-Jul 2020 excluded and included). Writes ADDBOOK.pkl.
"""
import numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore'); pd.set_option('display.width', 330); pd.set_option('display.max_rows', 100)
F = pd.read_pickle('F.pkl'); idx = F.index; O = pd.read_pickle('OHLCV.pkl'); spy = O['T']['SPY']['Close'].reindex(idx); r = np.log(spy).diff()
PRED = pd.read_pickle('PRED2.pkl'); PO = PRED['P_on']; DEC = pd.read_pickle('DECISION.pkl'); IN = DEC['states']['C: OUT P_off>q0.95, IN P_off<q0.8']
EXM = (idx >= pd.Timestamp('2020-02-01')) & (idx <= pd.Timestamp('2020-07-31')); m_all = idx >= pd.Timestamp('2006-01-01'); m_ex = m_all & ~EXM
def tq(s, q, w=756): return s.rolling(w, min_periods=252).quantile(q).shift(1)
def hyst(on, off):
    o = on.fillna(False).values; f = off.fillna(False).values; st = np.zeros(len(o), dtype=int); s = 0
    for k in range(len(o)):
        if s == 0 and o[k]: s = 1
        elif s == 1 and f[k]: s = 0
        st[k] = s
    return pd.Series(st, index=idx)
ADD = {'ADD: P_on >= q.80': (PO >= tq(PO, 0.8)).astype(int), 'ADD: P_on >= q.90': (PO >= tq(PO, 0.9)).astype(int), 'ADD: P_on >= q.80, held until < q.60': hyst(PO >= tq(PO, 0.8), PO < tq(PO, 0.6)), 'ADD: P_on >= q.90, held until < q.60': hyst(PO >= tq(PO, 0.9), PO < tq(PO, 0.6))}
def stats(x):
    c = x.cumsum(); dd = c - c.cummax(); return dict(ann=round(x.mean() * 252 * 100, 2), sharpe=round(x.mean() / x.std() * np.sqrt(252), 2) if x.std() > 0 else 0.0, maxDD=round(dd.min() * 100, 1))
def spells(S, m):
    o = (S == 1) & m; n = 0; last = None
    for d in idx[o]:
        if last is None or idx.get_loc(d) - idx.get_loc(last) > 1: n += 1
        last = d
    return n
rows = []
for name, A in ADD.items():
    on = ((A == 1) & (IN == 1)).astype(int); w = on.shift(1).fillna(0)
    for tag, m in (('ex-2020', m_ex), ('incl-2020', m_all)):
        x = (w * r)[m]; s = stats(x); days = int(w[m].sum()); per = r[m][w[m] == 1]
        rows.append({'book': f'sleeve {name}', 'path': tag, **s, 'sessions long': days, 'share of IN sessions': round(days / max(1, int(IN.shift(1)[m].sum())), 2), 'mean daily bp while long': round(float(per.mean() * 1e4), 1), 'spells': spells(on, m)})
for tag, m in (('ex-2020', m_ex), ('incl-2020', m_all)):
    w = IN.shift(1).fillna(1); x = (w * r)[m]; per = r[m][w[m] == 1]
    rows.append({'book': 'sleeve: all IN sessions (unconditional)', 'path': tag, **stats(x), 'sessions long': int(w[m].sum()), 'share of IN sessions': 1.0, 'mean daily bp while long': round(float(per.mean() * 1e4), 1), 'spells': spells(IN, m)})
SL = pd.DataFrame(rows)
rows = []
for tag, m in (('ex-2020', m_ex), ('incl-2020', m_all)):
    rows.append({'book': 'reference: 100% when IN, 0 when OUT', 'path': tag, **stats((IN.shift(1).fillna(1) * r)[m]), 'mean weight': round(float(IN.shift(1).fillna(1)[m].mean()), 2)})
    rows.append({'book': 'SPY buy and hold', 'path': tag, **stats(r[m]), 'mean weight': 1.0})
    for base in (0.75, 0.5):
        rows.append({'book': f'{int(base*100)}% when IN, no ADD (the reserve never deployed)', 'path': tag, **stats((base * IN.shift(1).fillna(1) * r)[m]), 'mean weight': round(float((base * IN.shift(1).fillna(1))[m].mean()), 2)})
        for name, A in ADD.items():
            wt = IN * np.where(A == 1, 1.0, base); w = pd.Series(wt, index=idx).shift(1).fillna(base)
            rows.append({'book': f'{int(base*100)}% when IN, 100% on {name}', 'path': tag, **stats((w * r)[m]), 'mean weight': round(float(w[m].mean()), 2)})
BK = pd.DataFrame(rows); pd.to_pickle({'sleeve': SL, 'book': BK}, 'ADDBOOK.pkl')
print('=== the ADD condition as a sleeve: long SPY only on ADD sessions (rule IN), flat otherwise; against the same sleeve on all IN sessions ==='); print(SL.to_string(index=False))
print('\n=== the ADD condition as a no-leverage book: a base weight when IN, 100% when ADD fires, 0 when OUT ==='); print(BK.to_string(index=False))
yr = pd.DataFrame({'ADD q.80 held': ((IN * np.where(ADD['ADD: P_on >= q.80, held until < q.60'] == 1, 1.0, 0.75)).shift(1).fillna(0.75) * r)[m_all].groupby(idx[m_all].year).sum() * 100, '75% when IN': (0.75 * IN.shift(1).fillna(1) * r)[m_all].groupby(idx[m_all].year).sum() * 100, '100% when IN': (IN.shift(1).fillna(1) * r)[m_all].groupby(idx[m_all].year).sum() * 100, 'SPY': r[m_all].groupby(idx[m_all].year).sum() * 100}).round(1)
print('\nyearly, 2020 included (log %):'); print(yr.T.to_string())
