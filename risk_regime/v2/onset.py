#!/usr/bin/env python3
"""v2 Stage 8b: onset rules. Run from risk_regime/work after big.py.

The event table in big.py splits the 15% declines in two. Those that start from an already stressed tape
(the 2008 legs, 2018 Q4, Aug 2022) have the drawdown probabilities at or near their extremes at the peak.
Those that start from a quiet tape (2007, 2010, 2011, Jan 2022, 2025) have the drawdown probabilities
asleep at the peak; only the vol-expansion probability moves, within 4 to 15 sessions and 1 to 9 percent
off the high. P_vol alone failed as a stay-in rule because it fires 40 to 116 times and its OUT state ends
as soon as vol has expanded. This script tests the obvious repair: a vol-expansion alert that becomes OUT
only when price confirms it (SPY a given percent below its 63-session closing high while the alert is
live), with re-entry on calmed probabilities or on a new 42-session closing high. OUT conditions use >= as in decision.py. Pure price rules with no
vol filter are in the grid as controls: if the vol complex does not beat a plain drawdown trigger, it adds
nothing. Scored on the 15% events exactly as in big.py. The whole grid is printed. Writes ONSET.pkl.
"""
import time, numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore')
pd.set_option('display.width', 340); pd.set_option('display.max_rows', 500); pd.set_option('display.max_columns', 60)
t0 = time.time(); F = pd.read_pickle('F.pkl'); idx = F.index; O = pd.read_pickle('OHLCV.pkl'); spy = O['T']['SPY']['Close'].reindex(idx); r = np.log(spy).diff()
PRED = pd.read_pickle('PRED2.pkl'); P15 = pd.read_pickle('PRED15.pkl'); BIG = pd.read_pickle('BIG.pkl'); DEC = pd.read_pickle('DECISION.pkl')
EX0, EX1 = pd.Timestamp('2020-02-01'), pd.Timestamp('2020-07-31'); EXM = (idx >= EX0) & (idx <= EX1); m = ~EXM & (idx >= pd.Timestamp('2006-01-01'))
EV15 = BIG['EV15']; EV10 = BIG['EV10']; EVS = EV15[(EV15.peak >= '2006-01-01') & ~EV15.covid]; EV10S = EV10[(EV10.peak >= '2005-06-01')]
in15 = pd.Series(False, index=idx)
for _, e in EVS.iterrows(): in15.loc[e.peak:e.trough] = True
def tq(s, q, w=756): return s.rolling(w, min_periods=252).quantile(q).shift(1)
def states(out_cond, in_cond):
    o = out_cond.fillna(False).values; i = in_cond.fillna(False).values; st = np.ones(len(o), dtype=int); s = 1
    for k in range(len(o)):
        if s == 1 and o[k]: s = 0
        elif s == 0 and i[k]: s = 1
        st[k] = s
    return pd.Series(st, index=idx)
def stats(x):
    c = x.cumsum(); dd = c - c.cummax(); return dict(ann=round(x.mean() * 252 * 100, 2), vol=round(x.std() * np.sqrt(252) * 100, 2), sharpe=round(x.mean() / x.std() * np.sqrt(252), 2), maxDD=round(dd.min() * 100, 1))
def score(S, name):
    w = S.shift(1); x = (w * r)[m]; row = {'rule': name, **stats(x), 'exposure': round(float(w[m].mean()), 2)}
    caps = []; firsts = []
    for _, e in EVS.iterrows():
        seg = r.loc[e.peak:e.trough].iloc[1:]; sp = seg.sum(); caps.append(round(float((w.loc[seg.index] * seg).sum() / sp), 2) if sp else np.nan)
        so = S.loc[e.peak:e.trough]; o = so[so == 0]
        firsts.append(f'{o.index[0].date()} ({idx.get_loc(o.index[0]) - idx.get_loc(e.peak)}s, {spy.loc[o.index[0]] / spy.loc[e.peak] * 100 - 100:.1f}%)' if len(o) else ('already OUT' if S.loc[e.peak] == 0 else 'never'))
    row['taken per 15% event'] = caps; row['first OUT per event'] = firsts; row['mean taken'] = round(float(np.nanmean(caps)), 2); row['worst taken'] = round(float(np.nanmax(caps)), 2)
    o = (S == 0) & m; spells = []; cur = None
    for d in idx[o]:
        if cur is None or idx.get_loc(d) - idx.get_loc(cur[1]) > 1:
            if cur: spells.append(cur)
            cur = [d, d]
        else: cur[1] = d
    if cur: spells.append(cur)
    row['spells'] = len(spells); just = sum(1 for a, b in spells if any((e.peak <= b + pd.Timedelta(days=45)) and (e.trough >= a) for _, e in EV10S.iterrows()))
    row['spells w/o any 10% decline'] = len(spells) - just
    row['return avoided inside 15% events, total %'] = round(float(-r[(w == 0) & m & in15].sum() * 100), 1)
    row['return given up outside 15% events, total %'] = round(float(r[(w == 0) & m & ~in15].sum() * 100), 1)
    row['today'] = 'IN' if S.iloc[-1] == 1 else 'OUT'; return row
PV, PF, P63 = PRED['P_vol'], PRED['P_off'], P15['dd15_63']
hi63 = spy.rolling(63, min_periods=63).max(); dd63 = spy / hi63 - 1; newhigh42 = spy >= spy.rolling(42, min_periods=42).max()
off95 = PF >= tq(PF, 0.95); off15_95 = P63 >= tq(P63, 0.95); calm = (PF < tq(PF, 0.8)) & (PV < tq(PV, 0.7))
INS = {'IN calm (P_off<q.8 & P_vol<q.7)': calm, 'IN new 42d high': newhigh42, 'IN calm or new 42d high': calm | newhigh42}
ST = {}
for x in (0.03, 0.05, 0.07, 0.10):
    D = dd63 <= -x
    for iname, ic in INS.items():
        ST[f'control: SPY {int(x*100)}% below 63d high | {iname}'] = states(D, ic)
        ST[f'control: SPY {int(x*100)}% below 63d high or P_off>q.95 | {iname}'] = states(D | off95, ic)
    for qa in (0.85, 0.90):
        for W in (21, 42):
            alert = (PV >= tq(PV, qa)).astype(float).rolling(W, min_periods=1).max() > 0
            for iname, ic in INS.items():
                ST[f'vol alert q{qa}/{W}s & SPY {int(x*100)}% below 63d high | {iname}'] = states(alert & D, ic)
                ST[f'vol alert q{qa}/{W}s & SPY {int(x*100)}% below 63d high, or P_off>q.95 | {iname}'] = states((alert & D) | off95, ic)
                if qa == 0.90 and W == 42: ST[f'vol alert q{qa}/{W}s & SPY {int(x*100)}% below 63d high, or P_off>q.95 or P15/63>q.95 | {iname}'] = states((alert & D) | off95 | off15_95, ic)
ST['reference C: OUT P_off>q0.95, IN P_off<q0.8'] = DEC['states']['C: OUT P_off>q0.95, IN P_off<q0.8']
print(f'{len(ST)} rules; 15% events scored:', [f'{a.date()}->{b.date()} {d}%' for a, b, d in zip(EVS.peak, EVS.trough, EVS['depth%'])])
rows = [{'rule': 'SPY buy and hold', **stats(r[m]), 'exposure': 1.0, 'taken per 15% event': [1.0] * len(EVS), 'first OUT per event': ['never'] * len(EVS), 'mean taken': 1.0, 'worst taken': 1.0, 'spells': 0, 'spells w/o any 10% decline': 0, 'return avoided inside 15% events, total %': 0.0, 'return given up outside 15% events, total %': 0.0, 'today': 'IN'}]
for name, S in ST.items(): rows.append(score(S, name))
TAB = pd.DataFrame(rows); TAB['net total %'] = TAB['return avoided inside 15% events, total %'] - TAB['return given up outside 15% events, total %']
cols = ['rule', 'ann', 'sharpe', 'maxDD', 'exposure', 'spells', 'spells w/o any 10% decline', 'mean taken', 'worst taken', 'return avoided inside 15% events, total %', 'return given up outside 15% events, total %', 'net total %', 'today']
print('\n=== whole grid, sorted by annualized return (2006-2026, Feb-Jul 2020 excluded; log returns; "taken" = share of each 15% peak-to-trough decline the book took) ===')
print(TAB.sort_values('ann', ascending=False)[cols].to_string(index=False))
print('\n=== controls vs vol-filtered, same price trigger and re-entry: does the vol complex add anything to a plain drawdown trigger? ===')
for x in (3, 5, 7, 10):
    for iname in INS:
        c = TAB[TAB.rule == f'control: SPY {x}% below 63d high | {iname}'].iloc[0]; v = TAB[TAB.rule == f'vol alert q0.9/42s & SPY {x}% below 63d high | {iname}'].iloc[0]
        print(f'  {x}% trigger, {iname:32s}  control: ann {c.ann:5.2f} sharpe {c.sharpe:.2f} maxDD {c.maxDD:6.1f} spells {int(c.spells):3d} false {int(c["spells w/o any 10% decline"]):3d} taken {c["mean taken"]:.2f}   |   vol alert q.9/42: ann {v.ann:5.2f} sharpe {v.sharpe:.2f} maxDD {v.maxDD:6.1f} spells {int(v.spells):3d} false {int(v["spells w/o any 10% decline"]):3d} taken {v["mean taken"]:.2f}')
print('\n=== first OUT session inside each 15% event (sessions after the peak, SPY already down), for the ten best rules by annualized return, the controls at 5%, and the reference ===')
show = list(TAB.sort_values('ann', ascending=False).head(10).rule) + [r_ for r_ in TAB.rule if r_.startswith('control: SPY 5% below 63d high |')] + ['reference C: OUT P_off>q0.95, IN P_off<q0.8']
for name in dict.fromkeys(show):
    rw = TAB[TAB.rule == name].iloc[0]; print(f'\n{name}  (ann {rw.ann}, maxDD {rw.maxDD}, spells {int(rw.spells)}, false {int(rw["spells w/o any 10% decline"])})')
    for (_, e), f, c in zip(EVS.iterrows(), rw['first OUT per event'], rw['taken per 15% event']): print(f'   {e.peak.date()} -> {e.trough.date()} {e["depth%"]:6.1f}%   first OUT {f:34s} taken {c}')
pd.to_pickle({'table': TAB, 'events': EVS}, 'ONSET.pkl'); pd.to_pickle(ST, 'STATES_ONSET.pkl'); print(f'\ndone {time.time()-t0:.0f}s')
