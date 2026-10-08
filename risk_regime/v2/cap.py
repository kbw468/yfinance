#!/usr/bin/env python3
"""v2 Stage 9: a hard drawdown cap. Run from risk_regime/work after model2.py.

The mandate is a cap, not a cost: a 15% peak-to-trough on the book is not allowed. No forecast in this data
enforces that (sections 8 and 12 of the README), so this script tests a mechanical control on the book's own
path, 1999 to 2026 with 2020 INCLUDED (a cap not tested on March 2020 is not tested):

  exposure(t+1) = min( M * cushion(t),  vol_target / realized_vol(t),  1 )
  cushion(t)    = cap + drawdown(t), drawdown measured against the book's high-water mark (all-time, or a
                  trailing window so the control re-levers on its own after a lost year)

plus a re-lever rule for the cash-lock that every such control has after a full drawdown: reset the high-water
mark to the current book after N sessions near zero exposure, when SPY makes a new 63-session high, or when the
out-of-sample rally probability is above its trailing 80th percentile (the one place the vol complex can earn its
keep here). Scored on the book's own maximum peak-to-trough (gaps included), CAGR, Sharpe, exposure, and the
book drawdown inside each 15% SPY decline and in Feb-Mar 2020. The whole grid is printed. Writes CAP.pkl.
"""
import sys, time, itertools, numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore'); sys.path.insert(0, '../v2'); import wf
pd.set_option('display.width', 360); pd.set_option('display.max_rows', 400); pd.set_option('display.max_columns', 40)
t0 = time.time(); idx = wf.idx; O = wf.O['T']['SPY']; spy = O['Close'].reindex(idx); hi = O['High'].reindex(idx); lo = O['Low'].reindex(idx)
r = np.log(spy).diff().fillna(0.0).to_numpy(); n = len(r); YEARS = wf.YEARS
cc21 = pd.Series(r, index=idx).rolling(21).std() * np.sqrt(252) * 100; cc10 = pd.Series(r, index=idx).rolling(10).std() * np.sqrt(252) * 100
pk10 = np.sqrt((np.log(hi / lo) ** 2).rolling(10).mean() / (4 * np.log(2))) * np.sqrt(252) * 100
RV = pd.concat([cc21, cc10, pk10], axis=1).max(axis=1).fillna(20.0).to_numpy()       # the highest of three realized-vol estimates, percent annualized
y, ok = wf.label('P_on'); pon, _, _ = wf.fit_predict_walk(y, ok, k=15, first=1998); PON = pd.Series(pon, index=idx)
pon_hi = (PON >= PON.rolling(756, min_periods=252).quantile(0.8).shift(1)).fillna(False).to_numpy()
newhigh63 = (spy >= spy.rolling(63, min_periods=63).max()).fillna(False).to_numpy()
START = int(np.where(idx >= pd.Timestamp('1999-01-01'))[0][0])
def sim(cap, M, vt, hwm, reset, N=63):
    """returns exposure series (applied to the next session) and the book (log)"""
    B = 0.0; logb = np.zeros(n); e = np.zeros(n); H = 0.0; hist = []; lock = 0
    for t in range(START, n):
        B += e[t - 1] * r[t] if t > START else 0.0; logb[t] = B
        if hwm == 'all': H = max(H, B)
        else:
            hist.append(B); hist = hist[-hwm:]; H = max(hist)
        d = B - H; cushion = cap + d
        ec = min(1.0, max(0.0, M * cushion)); ev = min(1.0, vt / RV[t]) if vt else 1.0; ex = min(ec, ev)
        lock = lock + 1 if ec < 0.1 else 0
        if ec < 0.1 and ((reset == 'time' and lock >= N) or (reset == 'newhigh' and newhigh63[t]) or (reset == 'P_on' and pon_hi[t]) or (reset == 'P_on_or_newhigh' and (pon_hi[t] or newhigh63[t]))):
            H = B; hist = [B]; lock = 0; cushion = cap; ec = min(1.0, M * cap); ex = min(ec, ev)
        e[t] = ex
    return e, logb
EV = [(pd.Timestamp(a), pd.Timestamp(b)) for a, b in [('2000-03-24', '2001-04-03'), ('2001-05-21', '2001-09-21'), ('2002-03-19', '2002-07-23'), ('2002-08-22', '2002-10-09'), ('2007-10-09', '2008-03-10'), ('2008-05-19', '2008-10-10'), ('2008-10-13', '2008-10-27'), ('2008-11-04', '2008-11-20'), ('2009-01-06', '2009-03-09'), ('2010-04-23', '2010-07-02'), ('2011-04-29', '2011-10-03'), ('2018-09-20', '2018-12-24'), ('2020-02-19', '2020-03-23'), ('2022-01-03', '2022-06-16'), ('2022-08-16', '2022-10-12'), ('2025-02-19', '2025-04-08')]]
def score(name, e, logb):
    x = pd.Series(np.r_[0.0, e[:-1] * r[1:]], index=idx).iloc[START:]; c = x.cumsum(); dd = c - c.cummax(); yrs = len(x) / 252
    row = {'rule': name, 'CAGR %': round((np.exp(x.sum() / yrs) - 1) * 100, 2), 'ann log %': round(x.mean() * 252 * 100, 2), 'sharpe': round(x.mean() / x.std() * np.sqrt(252), 2), 'max DD price %': round((np.exp(dd.min()) - 1) * 100, 1), 'mean exposure': round(float(np.mean(e[START:])), 2), 'sessions < 50% exposure': int((e[START:] < 0.5).sum())}
    dds = []
    for a, b in EV:
        seg = c.loc[a:b]; dds.append(round((np.exp(seg.min() - c.loc[:a].max()) - 1) * 100, 1))
    row['book DD inside each 15% SPY decline (2000-02 x4, 2007-09 x5, 2010, 2011, 2018, 2020, 2022 x2, 2025)'] = dds; row['worst inside'] = min(dds)
    row['worst year %'] = round(float((x.groupby(idx[START:].year).sum() * 100).min()), 1); return row
rows = [score('SPY buy and hold', np.ones(n), None)]
for cap, Mf, vt, hwm, reset in itertools.product((0.08, 0.10, 0.12), (1.0, 1.5), (None, 12, 16, 20), ('all', 252), ('time', 'newhigh', 'P_on', 'P_on_or_newhigh')):
    M = Mf / cap; e, lb = sim(cap, M, vt, hwm, reset); rows.append(score(f'cap {int(cap*100)}% M {Mf:.1f}/cap vol {vt or "-"} hwm {hwm} relever {reset}', e, lb))
for vt in (12, 16, 20):   # vol targeting alone, no drawdown control
    e = np.minimum(1.0, vt / RV); rows.append(score(f'vol target {vt} only', e, None))
T = pd.DataFrame(rows); pd.to_pickle({'table': T}, 'CAP.pkl')
cols = ['rule', 'CAGR %', 'sharpe', 'max DD price %', 'mean exposure', 'sessions < 50% exposure', 'worst inside', 'worst year %']
print(f'=== drawdown-cap overlays, 1999-2026, 2020 INCLUDED, daily rebalance at the close, cash at zero ({time.time()-t0:.0f}s) ===')
print('\n--- every rule whose book never fell more than 10% peak to trough, sorted by CAGR ---'); print(T[T['max DD price %'] >= -10].sort_values('CAGR %', ascending=False)[cols].head(25).to_string(index=False))
print('\n--- every rule whose book never fell more than 15%, sorted by CAGR ---'); print(T[T['max DD price %'] >= -15].sort_values('CAGR %', ascending=False)[cols].head(25).to_string(index=False))
print('\n--- the whole grid sorted by CAGR ---'); print(T.sort_values('CAGR %', ascending=False)[cols].to_string(index=False))
print('\n--- book drawdown inside each 15% SPY decline, for the best capped rules and the benchmarks ---')
best = list(T[T['max DD price %'] >= -10].sort_values('CAGR %', ascending=False).head(6).rule) + ['SPY buy and hold', 'vol target 16 only']
for _, rw in T[T.rule.isin(best)].iterrows(): print(f"  {rw['rule'][:62]:62s} {rw[[c for c in T.columns if c.startswith('book DD')][0]]}")
print(f'\ndone {time.time()-t0:.0f}s')
