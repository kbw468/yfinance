#!/usr/bin/env python3
"""Market structure on SPY (and XLK): lower highs and lower lows, from swing points on closes. Run from risk_regime/work.
No moving averages, no ATR. Swing points, without look-ahead:
  zigzag p%: a swing high is confirmed once the close falls p% below the running high since the last swing low,
             a swing low once the close rises p% above the running low since the last swing high;
  fractal N: the close N sessions ago is a swing high (low) if it is the highest (lowest) close of the 2N+1 sessions
             around it, confirmed N sessions later.
State: LOWER HIGHS AND LOWER LOWS (exit) when the last confirmed swing high is below the one before it and the close
breaks below the last confirmed swing low; HIGHER HIGHS restored (back in) when the close breaks above the last
confirmed swing high. Grid fixed before scoring: zigzag 2, 3, 5, 7%; fractal 5, 10, 21. Scored as an exit on SPY and on
XLK, alone and OR'd with the current SPY instruction, 2006-26, 2016-26 and 2023-26, 2020 included, lagged one session.
Writes STRUCTURE.pkl."""
import numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore'); pd.set_option('display.width', 300); pd.set_option('display.max_rows', 300)
F = pd.read_pickle('F.pkl'); O = pd.read_pickle('OHLCV.pkl'); idx = F.index
def pivots_zigzag(c, p):
    """list of (confirm_index, pivot_index, kind, level) confirmed without look-ahead"""
    out = []; v = c.to_numpy(); mode = 0; hi_i = lo_i = 0
    for t in range(1, len(v)):
        if np.isnan(v[t]): continue
        if v[t] > v[hi_i] or np.isnan(v[hi_i]): hi_i = t
        if v[t] < v[lo_i] or np.isnan(v[lo_i]): lo_i = t
        if mode >= 0 and v[t] <= v[hi_i] * (1 - p): out.append((t, hi_i, 'H', v[hi_i])); mode = -1; lo_i = t
        elif mode <= 0 and v[t] >= v[lo_i] * (1 + p): out.append((t, lo_i, 'L', v[lo_i])); mode = 1; hi_i = t
    return out
def pivots_fractal(c, N):
    out = []; v = c.to_numpy()
    for t in range(2 * N, len(v)):
        m = t - N; w = v[t - 2 * N:t + 1]
        if np.isnan(w).any(): continue
        if v[m] == w.max(): out.append((t, m, 'H', v[m]))
        elif v[m] == w.min(): out.append((t, m, 'L', v[m]))
    return out
def structure_state(c, piv):
    v = c.to_numpy(); st = np.ones(len(v), int); s = 1; H = []; L = []; j = 0
    for t in range(len(v)):
        while j < len(piv) and piv[j][0] <= t:
            (H if piv[j][2] == 'H' else L).append(piv[j][3]); j += 1
        if np.isnan(v[t]): st[t] = s; continue
        if s == 1 and len(H) >= 2 and len(L) >= 1 and H[-1] < H[-2] and v[t] < L[-1]: s = 0
        elif s == 0 and len(H) >= 1 and v[t] > H[-1]: s = 1
        st[t] = s
    return pd.Series(st, index=c.index)
INS = pd.read_pickle('VVSIG.pkl')['instruction']
def stats(S, r, a, b):
    m = (idx >= pd.Timestamp(a)) & (idx <= pd.Timestamp(b)); w = S.shift(1).fillna(1); x = (w * r)[m]; c = x.cumsum(); dd = c - c.cummax()
    o = pd.Series(((S == 0) & m).to_numpy(bool), index=idx); yrs = m.sum() / 252
    return {'ann %': round(float(x.mean() * 252 * 100), 1), 'worst DD %': round(float((np.exp(dd.min()) - 1) * 100), 1), 'exits/yr': round(float((o & ~o.shift(1, fill_value=False)).sum()) / yrs, 1), 'exposure': round(float(w[m].mean()), 2)}
GRID = [('zigzag 2%', 'z', 0.02), ('zigzag 3%', 'z', 0.03), ('zigzag 5%', 'z', 0.05), ('zigzag 7%', 'z', 0.07), ('fractal 5', 'f', 5), ('fractal 10', 'f', 10), ('fractal 21', 'f', 21)]
OUT = {}; rows = []
for tk in ('SPY', 'XLK'):
    c = O['T'][tk]['Close'].reindex(idx); r = np.log(c).diff()
    for name, kind, par in GRID:
        piv = pivots_zigzag(c, par) if kind == 'z' else pivots_fractal(c, par); S = structure_state(c, piv); OUT[(tk, name)] = (S, piv)
    for a, b in (('2006-01-01', '2026-12-31'), ('2016-01-01', '2026-12-31'), ('2023-01-01', '2026-12-31')):
        lab = f'{a[:4]}-{b[:4]}'; rows.append({'ticker': tk, 'window': lab, 'book': 'buy and hold', **stats(pd.Series(1, index=idx), r, a, b)})
        if tk == 'SPY' and a >= '2010': rows.append({'ticker': tk, 'window': lab, 'book': 'current instruction', **stats(INS, r, a, b)})
        for name, _, _ in GRID:
            S = OUT[(tk, name)][0]; rows.append({'ticker': tk, 'window': lab, 'book': f'structure {name}', **stats(S, r, a, b)})
            if tk == 'SPY' and a >= '2010': rows.append({'ticker': tk, 'window': lab, 'book': f'current instruction OR structure {name}', **stats((INS.astype(bool) & S.astype(bool)).astype(int), r, a, b)})
R = pd.DataFrame(rows)
for tk in ('SPY', 'XLK'):
    for lab in ('2006-2026', '2016-2026', '2023-2026'):
        print(f'\n=== {tk} {lab} (2020 included) ==='); print(R[(R.ticker == tk) & (R.window == lab)].drop(columns=['ticker', 'window']).to_string(index=False))
c = O['T']['XLK']['Close'].reindex(idx)
print('\nXLK May-Sep 2026, state changes by grid point (OUT = lower highs and lower lows confirmed):')
for name, _, _ in GRID:
    S = OUT[('XLK', name)][0]; seg = S.loc['2026-05-15':'2026-10-07']; ch = seg[seg != seg.shift(1)].iloc[1:]
    print(f'  {name:11s} ' + (', '.join(f"{d.date()} {'IN' if v else 'OUT'} @ {c[d]:.2f}" for d, v in ch.items()) or 'no change'))
cs = O['T']['SPY']['Close'].reindex(idx)
print('\nSPY today by grid point:')
for name, _, _ in GRID:
    S, piv = OUT[('SPY', name)]; Hs = [p for p in piv if p[2] == 'H']; Ls = [p for p in piv if p[2] == 'L']
    print(f"  {name:11s} {'IN (no lower-high/lower-low break)' if S.iloc[-1] == 1 else 'OUT'} | last swing highs {[round(h[3], 2) for h in Hs[-2:]]} ({', '.join(str(idx[h[1]].date()) for h in Hs[-2:])}) | last swing low {round(Ls[-1][3], 2)} ({idx[Ls[-1][1]].date()}) | SPY {cs.iloc[-1]:.2f}")
pd.to_pickle({'table': R, 'states': {k: v[0] for k, v in OUT.items()}, 'pivots': {k: v[1] for k, v in OUT.items()}}, 'STRUCTURE.pkl')
