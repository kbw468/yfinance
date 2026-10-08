#!/usr/bin/env python3
"""Rate of change in credit (HYG, HYG against Treasuries) and vol-of-vol (VVIX), distribution-free throughout.
Run from risk_regime/work. Every series is a trailing 504-session percentile rank of a log rate of change; no means,
variances or correlations. For each: today's rank; out-of-sample AUC of the rank itself against a 5% SPY decline within
21 sessions, by era and 2023-26 (AUC is a rank statistic); fires at the 90th (or 10th) percentile after 10 quiet
sessions since 2006 and since 2023, and the share followed by the 5% decline; first fire in the 2024, 2025 and 2026
drawdowns. Co-movement of HYG and IEF is counted as the share of the last 21 sessions they moved the same way."""
import numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore'); pd.set_option('display.width', 320); pd.set_option('display.max_rows', 200)
from sklearn.metrics import roc_auc_score
F = pd.read_pickle('F.pkl'); idx = F.index; O = pd.read_pickle('OHLCV.pkl'); T = O['T']; L = pd.read_pickle('LAB.pkl'); Y = idx.year.to_numpy()
I = O.get('I', {}) if isinstance(O, dict) else {}
px = lambda k: T[k]['Close'].reindex(idx)
def vvix():
    for key in ('I', 'IDX', 'idx'):
        if isinstance(O, dict) and key in O and '^VVIX' in O[key]: return O[key]['^VVIX']['Close'].reindex(idx)
    raise KeyError
try: VV = vvix()
except Exception: VV = None
rk = lambda s: s.rolling(504, min_periods=252).rank(pct=True)
roc = lambda s, n: np.log(s / s.shift(n))
hyg, ief, spy = px('HYG'), px('IEF'), px('SPY'); cs = np.log(hyg / ief)
S = {
 'HYG 1d ROC': (rk(roc(hyg, 1)), -1), 'HYG 5d ROC': (rk(roc(hyg, 5)), -1), 'HYG 21d ROC': (rk(roc(hyg, 21)), -1),
 'HYG minus Treasuries 5d ROC': (rk(cs - cs.shift(5)), -1), 'HYG minus Treasuries 21d ROC': (rk(cs - cs.shift(21)), -1), 'HYG minus Treasuries 63d ROC': (rk(cs - cs.shift(63)), -1),
 'HYG minus SPY 5d ROC': (rk(roc(hyg, 5) - roc(spy, 5)), 1), 'HYG minus SPY 21d ROC': (rk(roc(hyg, 21) - roc(spy, 21)), 1),
 'HYG range vol 5d ROC': (rk(roc(F['HYG:pk5'].astype(float), 5)), 1), 'HYG range vol 21d ROC': (rk(roc(F['HYG:pk21'].astype(float), 5)), 1),
 'HYG volume 5d ROC': (F['HYG:vroc5_rank'].astype(float), 1), 'HYG volume 21d ROC': (F['HYG:vroc21_rank'].astype(float), 1),
 'VVIX 1d ROC': (F['VVIX:roc1_rank'].astype(float), 1), 'VVIX 3d ROC': (F['VVIX:roc3_rank'].astype(float), 1), 'VVIX 5d ROC': (F['VVIX:roc5_rank'].astype(float), 1),
 'VVIX 10d ROC': (F['VVIX:roc10_rank'].astype(float), 1), 'VVIX 21d ROC': (F['VVIX:roc21_rank'].astype(float), 1), 'VVIX acceleration': (F['VVIX:accel5_rank'].astype(float), 1),
 'VVIX/VIX 5d ROC': (F['VVIX/VIX:roc5_rank'].astype(float), -1), 'VVIX/VIX 21d ROC': (F['VVIX/VIX:roc21_rank'].astype(float), -1),
 'reference: VIX 21d ROC': (F['VIX:roc21_rank'].astype(float), 1), 'reference: VIX 5d ROC': (F['VIX:roc5_rank'].astype(float), 1),
}
EXM = (idx >= pd.Timestamp('2020-02-01')) & (idx <= pd.Timestamp('2020-07-31')); ok = L['ok21'].to_numpy(bool) & ~EXM; y = L['riskoff21'].to_numpy(float)
fmin = spy[::-1].rolling(21, min_periods=21).min()[::-1].shift(-1); hit = (fmin / spy - 1 <= -0.05)
EV = [('2024-07-16', '2024-08-05'), ('2025-02-19', '2025-04-08'), ('2026-01-27', '2026-03-30')]
def fires(s, side):
    on = ((s >= 0.9) if side > 0 else (s <= 0.1)).fillna(False).to_numpy(); out = np.zeros(len(on), bool); q = 10
    for i in range(len(on)):
        if on[i] and q >= 10: out[i] = True
        q = 0 if on[i] else q + 1
    return pd.Series(out, index=idx), pd.Series(on, index=idx)
rows = []
for name, (s, side) in S.items():
    v = (side * s).to_numpy(float); row = {'signal': name, 'today rank': round(float(s.iloc[-1]), 2), 'risk side': 'high' if side > 0 else 'low'}
    for e, (a, b) in (('AUC 06-15', (2006, 2015)), ('AUC 16-26', (2016, 2026)), ('AUC 23-26', (2023, 2026))):
        m = ok & ~np.isnan(v) & (Y >= a) & (Y <= b); row[e] = round(float(roc_auc_score(y[m], v[m])), 2) if m.sum() > 100 and len(np.unique(y[m])) == 2 else None
    fr, on = fires(s, side)
    for a0, lab in (('2006-01-01', '06+'), ('2023-01-01', '23+')):
        m = fr & (idx >= pd.Timestamp(a0)) & hit.notna(); row[f'fires {lab}'] = int(m.sum()); row[f'-5% after, {lab}'] = f'{hit[m].mean() * 100:.0f}%' if m.sum() else ''
    for a, b in EV:
        a = pd.Timestamp(a); b = pd.Timestamp(b); w0 = idx[max(0, idx.get_loc(a) - 10)]; h = on.loc[w0:b]; h = h[h]
        if len(h): t = h.index[0]; row[str(a.year)] = f'{idx.get_loc(t) - idx.get_loc(a):+d}s {spy.loc[t] / spy.loc[a:b].loc[:t].max() * 100 - 100 if t >= a else 0:.1f}%'
        else: row[str(a.year)] = 'never'
    rows.append(row)
R = pd.DataFrame(rows)
m6 = (idx >= pd.Timestamp('2006-01-01')) & hit.notna(); m23 = (idx >= pd.Timestamp('2023-01-01')) & hit.notna()
print(f'base rate, 5% SPY decline within 21 sessions on any day: {hit[m6].mean() * 100:.0f}% since 2006, {hit[m23].mean() * 100:.0f}% since 2023')
print(R.to_string(index=False))
dh = np.sign(np.log(hyg).diff()); di = np.sign(np.log(ief).diff()); same = (dh == di).astype(float).rolling(21).sum()
print(f'\nHYG and IEF moved the same direction on {int(same.iloc[-1])} of the last 21 sessions; two-year median {int(same.iloc[-504:].median())}; that count is at the {float((same.iloc[-505:-1] <= same.iloc[-1]).mean()) * 100:.0f}th percentile of two years')
for n in (5, 21, 63): print(f'{n}d: HYG {float(roc(hyg, n).iloc[-1] * 100):+.2f}%, IEF {float(roc(ief, n).iloc[-1] * 100):+.2f}%, HYG minus IEF {float((cs - cs.shift(n)).iloc[-1] * 100):+.2f}% (rank {float(rk(cs - cs.shift(n)).iloc[-1]):.2f})')
pd.to_pickle(R, 'ROC_HYG_VVIX.pkl')
