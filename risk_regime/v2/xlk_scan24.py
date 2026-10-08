#!/usr/bin/env python3
"""The XLK scan on post-2024 tape only. Run from risk_regime/work after xlk_scan.py.
Same candidates (every rank feature at both tails plus the constructed signals), same requirement (lit on at least one
session from 19 May to 4 June 2026). Each signal is scored only on its fires from 1 Jan 2024 to 18 May 2026, so the
June event it is meant to explain is not in its own record: after each fire, did XLK fall 5% within 21 sessions, and
10% within 63? One-sided binomial p-value against XLK's base rate over the same window, and Benjamini-Hochberg
false-discovery adjustment across every signal tested, since a short window and a long list produce winners by
chance. Writes XLK_SCAN24.pkl."""
import numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore'); pd.set_option('display.width', 330); pd.set_option('display.max_rows', 300)
from scipy.stats import binom
F = pd.read_pickle('F.pkl'); MF = pd.read_pickle('MF.pkl'); X = pd.concat([F, MF], axis=1); O = pd.read_pickle('OHLCV.pkl'); idx = X.index
T = O['T']; xlk = T['XLK']['Close'].reindex(idx); rk = lambda s: s.rolling(504, min_periods=252).rank(pct=True); lr = lambda s, n: np.log(s / s.shift(n))
C = {}
xsd = T['XSD']['Close'].reindex(idx); qqq = T['QQQ']['Close'].reindex(idx)
try:
    qe = pd.read_csv('data_ew/QQQE.csv', index_col=0, parse_dates=True)['Close'].reindex(idx); C['QQQE vs QQQ 21d ROC'] = rk(lr(qe, 21) - lr(qqq, 21)); C['QQQE vs QQQ 10d ROC'] = rk(lr(qe, 10) - lr(qqq, 10))
except Exception: pass
for n in (5, 10, 21): C[f'semis vs XLK {n}d ROC'] = rk(lr(xsd, n) - lr(xlk, n))
non = ['XLE', 'XLB', 'XLU', 'XLP', 'XLI', 'XLF', 'XLV', 'XLY', 'XRT', 'XBI']; C['non-tech realized vol breadth'] = rk(pd.concat([X[f'{t}:cc10_rank'].astype(float) for t in non], axis=1).median(axis=1))
S = {c: X[c].astype(float) for c in X.columns if 'rank' in c.split(':')[-1]}; S.update(C)
f21 = xlk[::-1].rolling(21, min_periods=21).min()[::-1].shift(-1); dn5 = (f21 / xlk - 1 <= -0.05).astype(float).where(f21.notna())
f63 = xlk[::-1].rolling(63, min_periods=63).min()[::-1].shift(-1); dn10 = (f63 / xlk - 1 <= -0.10).astype(float).where(f63.notna())
P = (idx >= pd.Timestamp('2024-01-01')) & (idx <= pd.Timestamp('2026-05-18')); base5 = float(dn5[P].mean()); base10 = float(dn10[P & dn10.notna().to_numpy()].mean())
win = (idx >= pd.Timestamp('2026-05-19')) & (idx <= pd.Timestamp('2026-06-04'))
rows = []
for name, s in S.items():
    for tail, on in (('top 10%', s >= 0.9), ('bottom 10%', s <= 0.1)):
        on = on.fillna(False)
        if not on[win].any(): continue
        prev = on.shift(1, fill_value=False).astype(float).rolling(10, min_periods=1).max() > 0
        fr = (on & ~prev).to_numpy() & P & dn5.notna().to_numpy(); n = int(fr.sum())
        if n < 6: continue
        k = int(dn5[fr].sum()); m10 = fr & dn10.notna().to_numpy()
        first = idx[win & on.to_numpy()][0]
        rows.append({'signal': name, 'tail': tail, 'fires 2024-May26': n, 'XLK -5%/21 hits': k, 'hit rate': round(k / n, 2), 'lift': round(k / n / base5, 2), 'p (binomial)': float(binom.sf(k - 1, n, base5)),
                     'XLK -10%/63': round(float(dn10[m10].mean()), 2) if m10.sum() else None, 'first lit before the break': f'{first.date()} (XLK {xlk[first] / xlk.loc[:first].cummax().iloc[-1] * 100 - 100:.1f}% off high)'})
R = pd.DataFrame(rows).sort_values('p (binomial)')
m = len(R); R['rank'] = np.arange(1, m + 1); R['BH q'] = (R['p (binomial)'] * m / R['rank']).iloc[::-1].cummin().iloc[::-1].clip(upper=1).round(3); R['p (binomial)'] = R['p (binomial)'].round(4)
print(f'scoring window 2024-01-01 to 2026-05-18 (the June event excluded); XLK base rate there: 5% decline within 21 sessions {base5:.2f}, 10% within 63 {base10:.2f}')
print(f'{m} signals were lit between 19 May and 4 Jun 2026 and fired at least 6 times in the window')
print(f'expected by chance at p < 0.05: about {0.05 * m:.0f}; observed: {int((R["p (binomial)"] < 0.05).sum())}; smallest false-discovery q: {R["BH q"].min():.2f}')
print(R.drop(columns='rank').head(25).to_string(index=False))
pd.to_pickle(R, 'XLK_SCAN24.pkl')
