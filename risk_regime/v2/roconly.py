#!/usr/bin/env python3
"""The signal rebuilt on rates of change only: price, volume and volatility. Run from risk_regime/work.
Candidates: every feature whose quantity is a rate of change of a price (1 to 63 days, alone or against SPY), of volume,
or of a volatility (realized-vol ROC, implied-vol index ROC including VIX, VIX9D, VIX1D, VIX3M, VVIX, VXN, MOVE, OVX,
GVZ, SKEW and their ratios' ROC, the VIX premium's change, dispersion ROC). Excluded: every level, every level rank,
every vol level, correlation, multifractal, illiquidity level. Same pipeline as production (in-window same-sign screen,
K15 depth-2 monotone trees, yearly refits from 2005, 100-day purge).
Targets: 5% drawdown within 21 sessions (as production) and LOWER HIGHS AND LOWER LOWS: the highest close of the next 21
sessions below the highest close of the last 21, and the lowest close of the next 21 below the lowest of the last 21.
Decision: REDUCE when the probability reaches its trailing-756 95th percentile, HOLD again below the 80th (as the
reference rule), alone and OR'd with VIX outrunning VVIX (already rate-of-change only). Writes ROCONLY.pkl."""
import sys, time, numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore'); sys.path.insert(0, '../v2'); import wf
from sklearn.metrics import roc_auc_score
pd.set_option('display.width', 300); t0 = time.time()
cols = wf.COLS; KEEP_TOK = ('roc', ':rel', 'accel', 'vroc', 'prem_chg'); DROP_TOK = ('MF:', 'avgcorr', 'amihud21_rank', 'lvl_rank', 'breadth')
keep = np.array([any(t in c for t in KEEP_TOK) and not any(t in c for t in DROP_TOK) and 'KRE:' not in c for c in cols])
print(f'rate-of-change candidates: {keep.sum()} of {len(cols)}'); print('sample:', list(cols[keep][::max(1, keep.sum() // 25)])[:25])
wf.NANSHARE = np.where(keep, wf.NANSHARE, 1.0)          # the screen drops anything with NANSHARE >= 0.5
idx = wf.idx; spy = wf.O['T']['SPY']['Close'].reindex(idx); r = np.log(spy).diff()
past_hi = spy.rolling(21).max(); past_lo = spy.rolling(21).min()
fut_hi = spy[::-1].rolling(21, min_periods=21).max()[::-1].shift(-1); fut_lo = spy[::-1].rolling(21, min_periods=21).min()[::-1].shift(-1)
lhll = ((fut_hi < past_hi) & (fut_lo < past_lo)).astype(float).where(fut_hi.notna() & past_hi.notna())
EX = wf.EX; Y = wf.YEARS
ok_lhll = lhll.notna().to_numpy() & ~EX & wf.L['ok21'].to_numpy(bool)
TARGETS = {'P_off ROC-only (5% drawdown / 21)': wf.label('P_off'), 'P_lhll ROC-only (lower highs and lower lows / 21)': (lhll.to_numpy(float), ok_lhll),
           'P_on ROC-only (5% rally / 21)': wf.label('P_on'), 'P_vol ROC-only (vol x1.5 / 21)': wf.label('P_vol')}
PR = {}; rows = []
for name, (y, ok) in TARGETS.items():
    pred, sel, last = wf.fit_predict_walk(y, ok, k=15, keep_last=True); PR[name] = pd.Series(pred, index=idx)
    a = wf.auc_by_era(pred, y, ok); m23 = ok & ~np.isnan(pred) & (Y >= 2023); a23 = round(float(roc_auc_score(y[m23], pred[m23])), 3)
    rows.append({'target': name, 'base rate': round(float(np.nanmean(y[ok & (Y >= 2005)])), 3), **{f'AUC {k}': v for k, v in a.items()}, 'AUC 2023-26': a23, 'quintile hit rates': wf.quintiles(pred, y, ok)})
    print(f'{name}: {a} 2023-26 {a23}  ({time.time() - t0:.0f}s)'); print('   last-window features:', ', '.join(f'{f} ({"+" if s > 0 else "-"})' for f, s in zip(last[0], last[1])))
print('\nproduction (levels allowed), for comparison: P_off AUC all 0.636, 2023-26 0.657')
y, ok = wf.label('P_off')
for name in ('P_lhll ROC-only (lower highs and lower lows / 21)',):
    p = PR[name].to_numpy(); m = ok & ~np.isnan(p) & (Y >= 2005); print(f'{name} scored against the 5% drawdown target: AUC {roc_auc_score(y[m], p[m]):.3f}')
T = pd.DataFrame(rows); print(T.to_string(index=False))
def tq(s, q, w=756): return s.rolling(w, min_periods=252).quantile(q).shift(1)
def states(o_, i_):
    o = o_.fillna(False).values; i = i_.fillna(False).values; st = np.ones(len(o), dtype=int); s = 1
    for k in range(len(o)):
        if s == 1 and o[k]: s = 0
        elif s == 0 and i[k]: s = 1
        st[k] = s
    return pd.Series(st, index=idx)
VV = pd.read_pickle('VVSIG.pkl'); VS = VV['state']; INS = VV['instruction']; REF = pd.read_pickle('DECISION.pkl')['states']['C: OUT P_off>q0.95, IN P_off<q0.8']
BK = {'SPY buy and hold': pd.Series(1, index=idx), 'current instruction (levels leg OR VIX outrunning VVIX)': INS, 'VIX outrunning VVIX alone': VS}
for name in ('P_off ROC-only (5% drawdown / 21)', 'P_lhll ROC-only (lower highs and lower lows / 21)'):
    P = PR[name]; short = name.split(' (')[0]
    for qo, qi in ((0.95, 0.8), (0.90, 0.8), (0.95, 0.6)):
        S = states(P >= tq(P, qo), P < tq(P, qi)); BK[f'{short}: OUT >= q{qo}, IN < q{qi}'] = S
        BK[f'{short}: OUT >= q{qo}, IN < q{qi}  OR VIX outrunning VVIX'] = (S.astype(bool) & VS.astype(bool)).astype(int)
def stats(S, a, b):
    m = (idx >= pd.Timestamp(a)) & (idx <= pd.Timestamp(b)); w = S.shift(1).fillna(1); x = (w * r)[m]; c = x.cumsum(); dd = c - c.cummax()
    o = pd.Series(((S == 0) & m).to_numpy(bool), index=idx); yrs = m.sum() / 252
    return {'ann %': round(float(x.mean() * 252 * 100), 1), 'worst DD %': round(float((np.exp(dd.min()) - 1) * 100), 1), 'exits/yr': round(float((o & ~o.shift(1, fill_value=False)).sum()) / yrs, 1), 'exposure': round(float(w[m].mean()), 2)}
rows = []
for a, b in (('2010-01-01', '2026-12-31'), ('2016-01-01', '2026-12-31'), ('2023-01-01', '2026-12-31'), ('2006-01-01', '2026-12-31')):
    for k, S in BK.items():
        if a < '2010' and ('VIX outrunning' in k or 'current' in k): continue
        rows.append({'window': f'{a[:4]}-{b[:4]}', 'book': k, **stats(S, a, b)})
B = pd.DataFrame(rows)
for w in B.window.unique(): print(f'\n=== SPY books {w} (2020 included) ==='); print(B[B.window == w].drop(columns='window').to_string(index=False))
P = PR['P_lhll ROC-only (lower highs and lower lows / 21)']; pct = float((P.iloc[-757:-1] <= P.iloc[-1]).mean())
print(f'\ntoday: P(lower highs and lower lows, ROC-only) {P.iloc[-1]:.3f}, trailing percentile {pct:.2f}; ROC-only P_off {PR["P_off ROC-only (5% drawdown / 21)"].iloc[-1]:.3f}')
pd.to_pickle({'pred': PR, 'auc': T, 'books': B, 'states': {k: v for k, v in BK.items()}}, 'ROCONLY.pkl'); print(f'done {time.time() - t0:.0f}s')
