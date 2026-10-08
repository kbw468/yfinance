#!/usr/bin/env python3
"""v2 Stage 2a: multifractal and long-memory features. Run from risk_regime/work.

Rolling MF-DFA (Kantelhardt et al. 2002) on daily log returns: generalized Hurst exponents h(q) for
q in {-4,-2,2,4}, the multifractal width h(-4)-h(4), and the DFA scaling exponent of absolute
returns (volatility persistence). Rolling variance ratios (Lo-MacKinlay) at 5 and 21 days.
Windows of 504 sessions, recomputed every session. Outputs MF.pkl with raw values and trailing ranks.
"""
import pandas as pd, numpy as np, time, warnings; warnings.filterwarnings('ignore')
t0 = time.time()
O = pd.read_pickle('OHLCV.pkl'); idx = O['idx']; T = O['T']; I = O['I']
SERIES = {s: np.log(T[s]['Close']).diff() for s in ('SPY','QQQ','IWM','HYG','TLT','GLD','XLE')}
SERIES['VIX'] = np.log(I['VIX']['Close']).diff()
Q = np.array([-4.0, -2.0, 2.0, 4.0]); SCALES = np.array([8, 12, 16, 24, 32, 48, 64, 96, 126])
W = 504

def _design(s, deg=1):
    x = np.arange(s, dtype=float); X = np.vander(x, deg + 1); return X @ np.linalg.inv(X.T @ X) @ X.T   # hat matrix
HAT = {s: _design(s) for s in SCALES}
def mfdfa(x):
    """x: 1-D array of returns (no NaN). Returns dict of h(q) and the DFA exponent of |x|."""
    out = {}
    for tag, series in (('r', x), ('abs', np.abs(x))):
        Y = np.cumsum(series - series.mean()); N = len(Y); Fq = np.empty((len(SCALES), len(Q))); F2 = np.empty(len(SCALES))
        for i, s in enumerate(SCALES):
            Ns = N // s
            segs = np.concatenate([Y[:Ns * s].reshape(Ns, s), Y[N - Ns * s:].reshape(Ns, s)[::-1]])
            resid = segs - segs @ HAT[s].T
            f2 = (resid ** 2).mean(axis=1) + 1e-12
            F2[i] = f2.mean()
            for j, q in enumerate(Q): Fq[i, j] = (np.mean(f2 ** (q / 2))) ** (1 / q)
        ls = np.log(SCALES)
        if tag == 'r':
            for j, q in enumerate(Q): out[f'h{int(q)}'] = np.polyfit(ls, np.log(Fq[:, j]), 1)[0]
        else:
            out['alpha_abs'] = np.polyfit(ls, 0.5 * np.log(F2), 1)[0]
    out['width'] = out['h-4'] - out['h4']
    return out
def variance_ratio(x, q):
    """Lo-MacKinlay VR(q) on a window of returns."""
    n = len(x); mu = x.mean(); v1 = ((x - mu) ** 2).sum() / (n - 1)
    xq = np.convolve(x, np.ones(q), 'valid'); vq = ((xq - q * mu) ** 2).sum() / ((n - q + 1) * (1 - q / n) * q)
    return vq / v1
MF = {}
for name, r in SERIES.items():
    vals = r.values; cols = {k: np.full(len(vals), np.nan) for k in ('h-4','h-2','h2','h4','width','alpha_abs','vr5','vr21')}
    for i in range(W, len(vals) + 1):
        x = vals[i - W:i]
        if np.isnan(x).sum() > W * 0.05: continue
        x = x[~np.isnan(x)]
        if x.std() == 0: continue
        res = mfdfa(x)
        for k, v in res.items(): cols[k][i - 1] = v
        cols['vr5'][i - 1] = variance_ratio(x[-252:], 5); cols['vr21'][i - 1] = variance_ratio(x[-252:], 21)
    for k, v in cols.items():
        s = pd.Series(v, index=idx); MF[f'MF:{name}:{k}'] = s
        MF[f'MF:{name}:{k}_rank'] = s.rolling(504, min_periods=300).rank(pct=True)
        if k in ('h2','width','alpha_abs'):
            MF[f'MF:{name}:{k}_chg21_rank'] = s.diff(21).rolling(504, min_periods=300).rank(pct=True)
            MF[f'MF:{name}:{k}_chg63_rank'] = s.diff(63).rolling(504, min_periods=300).rank(pct=True)
    print(f'{name:4s} done {time.time()-t0:5.0f}s  last: h2 {cols["h2"][-1]:.3f} width {cols["width"][-1]:.3f} alpha_abs {cols["alpha_abs"][-1]:.3f} vr5 {cols["vr5"][-1]:.2f} vr21 {cols["vr21"][-1]:.2f}')
MF = pd.DataFrame(MF).astype('float32'); MF.to_pickle('MF.pkl')
print('MF', MF.shape, f'{time.time()-t0:.0f}s')
print(MF.filter(regex='SPY:(h2|width|alpha_abs)$').describe().round(3).to_string())
