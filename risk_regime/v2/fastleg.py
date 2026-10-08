#!/usr/bin/env python3
"""Production fast leg (daily chain, after lhll.py). Run from risk_regime/work.
Inputs: rates of change over 1 to 5 sessions only, every series (price, against SPY, volume, vol indices and their ratios,
short realized-vol rates of change). Target: SPY falls 3% from the close within 10 sessions. Same pipeline as fast.py
(in-window same-sign screen, K15 depth-2 monotone trees, yearly refits from 2005, 30-day purge).
Leg: REDUCE when the probability reaches its trailing-756 95th percentile, clears below the 80th.
Instruction: REDUCE when any of the three legs is live (lower highs and lower lows, VIX outrunning VVIX, fast leg).
Writes FASTLEG.pkl."""
import sys, re, numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore'); sys.path.insert(0, '../v2'); import wf
cols = wf.COLS
short = re.compile(r':(roc|rel)(1|2|3|5)(_rank)?$|:vroc5(_rank)?$|:accel5_rank$|:cc(5|10)_roc5_rank$|:cc21_roc5(_rank)?$|prem_chg5_rank$|:disp(1|5)_roc5_rank$|:volvol21_roc5_rank$|:amihud_roc5_rank$')
keep = np.array([bool(short.search(c)) and not c.startswith('KRE:') and not c.startswith('MF:') for c in cols]); wf.NANSHARE = np.where(keep, wf.NANSHARE, 1.0)
idx = wf.idx; spy = wf.O['T']['SPY']['Close'].reindex(idx); r = np.log(spy).diff(); okb = wf.L['ok21'].to_numpy(bool) & ~wf.EX
fl = spy[::-1].rolling(10, min_periods=10).min()[::-1].shift(-1); y = (fl / spy - 1 <= -0.03).astype(float).where(fl.notna()); yv = y.to_numpy(float); ok = okb & ~np.isnan(yv)
pred, _, last = wf.fit_predict_walk(yv, ok, k=15, keep_last=True, purge=30); P = pd.Series(pred, index=idx); LAST = (last[0], [int(s) for s in last[1]])
def tq(s, q, w=756): return s.rolling(w, min_periods=252).quantile(q).shift(1)
def states(o_, i_):
    o = o_.fillna(False).values; i = i_.fillna(False).values; st = np.ones(len(o), dtype=int); s = 1
    for k in range(len(o)):
        if s == 1 and o[k]: s = 0
        elif s == 0 and i[k]: s = 1
        st[k] = s
    return pd.Series(st, index=idx)
LEG = states(P >= tq(P, 0.95), P < tq(P, 0.8))
LH = pd.read_pickle('LHLL.pkl'); PREV = LH['instruction']; INS = (PREV.astype(bool) & LEG.astype(bool)).astype(int)
def stats(S, a):
    m = idx >= pd.Timestamp(a); w = S.shift(1).fillna(1); x = (w * r)[m]; c = x.cumsum(); dd = c - c.cummax(); o = pd.Series(((S == 0) & m).to_numpy(bool), index=idx)
    return {'ann %': round(float(x.mean() * 252 * 100), 1), 'worst DD %': round(float((np.exp(dd.min()) - 1) * 100), 1), 'exits/yr': round(float((o & ~o.shift(1, fill_value=False)).sum()) / (m.sum() / 252), 1), 'exposure': round(float(w[m].mean()), 2)}
B = pd.DataFrame([{'from': a[:4], 'book': k, **stats(S, a)} for a in ('2010-01-01', '2016-01-01', '2023-01-01') for k, S in (('SPY buy and hold', pd.Series(1, index=idx)), ('two legs (lower highs/lows OR VIX outrunning VVIX)', PREV), ('fast leg alone', LEG), ('instruction: three legs', INS))])
sp = []; o = (LEG == 0); cur = None
for d in idx[o.to_numpy()]:
    if cur is None or idx.get_loc(d) - idx.get_loc(cur[1]) > 1:
        if cur: sp.append(cur)
        cur = [d, d]
    else: cur[1] = d
if cur: sp.append(cur)
SPELLS = [{'from': str(a.date()), 'to': str(b.date()), 'spy_while_out': round(float(np.log(spy.loc[b] / spy.loc[a]) * 100), 1)} for a, b in sp]
pct = float((P.iloc[-757:-1] <= P.iloc[-1]).mean()); chg = INS != INS.shift(1)
TODAY = {'P_fast': round(float(P.iloc[-1]), 4), 'P_fast_pct': round(pct, 3), 'q95': round(float(tq(P, 0.95).iloc[-1]), 4), 'q80': round(float(tq(P, 0.8).iloc[-1]), 4), 'live': bool(LEG.iloc[-1] == 0),
         'since': SPELLS[-1]['from'] if LEG.iloc[-1] == 0 and SPELLS else None, 'instruction': 'REDUCE' if INS.iloc[-1] == 0 else 'HOLD', 'state_since': str(idx[chg.to_numpy()][-1].date())}
print(B.to_string(index=False)); print('last fit:', ', '.join(f'{f} ({"+" if s > 0 else "-"})' for f, s in zip(*LAST))); print('today:', TODAY); print('last spells:', SPELLS[-6:])
pd.to_pickle({'pred': P, 'last': LAST, 'leg': LEG, 'instruction': INS, 'books': B, 'spells': SPELLS, 'today': TODAY}, 'FASTLEG.pkl')
