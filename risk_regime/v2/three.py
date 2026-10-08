#!/usr/bin/env python3
"""What the vol and ROC signals did in the three recent drawdowns (Aug 2024, Feb-Apr 2025, Jan-Mar 2026), and what the
same signals did the rest of the time. Run from risk_regime/work. A signal "fires" when its trailing rank reaches 0.90
(0.10 for the ones where low is the risk side) after at least 10 sessions without firing. For each event: the first
fire from 10 sessions before the peak to the trough, sessions after the peak, and SPY's decline at that close. For each
signal: every fire since 2006 and since 2023, and the share followed by a 5% SPY decline within 21 sessions (base rate
about 17%) and by a 10% decline within 63."""
import numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore'); pd.set_option('display.width', 300); pd.set_option('display.max_rows', 200)
F = pd.read_pickle('F.pkl'); idx = F.index; spy = pd.read_pickle('OHLCV.pkl')['T']['SPY']['Close'].reindex(idx); PRED = pd.read_pickle('PRED2.pkl')
def tpct(s, w=756):
    v = s.to_numpy(dtype=float); o = np.full(len(v), np.nan)
    for i in range(252, len(v)):
        x = v[max(0, i - w):i]; x = x[~np.isnan(x)]
        if len(x) >= 252 and not np.isnan(v[i]): o[i] = (x <= v[i]).mean()
    return pd.Series(o, index=idx)
SIG = {  # name: (series, side) side +1 = high is the risk side, -1 = low is
 'VIX level': ('VIX:lvl_rank504', 1), 'VIX 5d ROC': ('VIX:roc5_rank', 1), 'VIX 21d ROC': ('VIX:roc21_rank', 1), 'VIX accel': ('VIX:accel5_rank', 1),
 'VIX9D/VIX': ('VIX9D/VIX:lvl_rank', 1), 'VIX1D 5d ROC': ('VIX1D:roc5_rank', 1), 'VIX/VIX3M level': ('VIX/VIX3M:lvl_rank', 1), 'VIX/VIX3M 5d ROC': ('VIX/VIX3M:roc5_rank', 1),
 'VVIX level': ('VVIX:lvl_rank504', 1), 'VVIX 5d ROC': ('VVIX:roc5_rank', 1), 'VVIX/VIX 5d ROC (low)': ('VVIX/VIX:roc5_rank', -1), 'VXN 5d ROC': ('VXN:roc5_rank', 1),
 'MOVE 5d ROC': ('MOVE:roc5_rank', 1), 'MOVE level': ('MOVE:lvl_rank504', 1), 'OVX 5d ROC': ('OVX:roc5_rank', 1), 'SKEW 5d ROC': ('SKEW:roc5_rank', 1),
 'SPY realized 5d': ('SPY:cc5_rank', 1), 'SPY realized 21d': ('SPY:cc21_rank', 1), 'SPY realized 5d/21d': ('SPY:cc5_21_rank', 1), 'SPY realized 21d ROC': ('SPY:cc21_roc5_rank', 1),
 'SPY Parkinson/close-close': ('SPY:pk21_cc21_rank', 1), 'VIX premium 5d change': ('SPY:VIX_prem_chg5_rank', 1), 'SPY vol-of-volume': ('SPY:volvol21_rank', 1),
 'avg correlation 5d change': ('XS:avgcorr21_chg5_rank', 1), 'dispersion 5d': ('XS:disp5_rank', 1), 'realized-vol breadth': ('XS:rv_breadth_rank', 1), 'HYG Parkinson 5d': ('HYG:pk5_rank', 1),
 'SPY 5d ROC (low)': ('SPY:roc5_rank', -1), 'SPY 21d ROC (low)': ('SPY:roc21_rank', -1),
}
S = {}
for k, (c, side) in SIG.items():
    if c in F: S[k] = (F[c].astype(float), side)
S['model: vol-expansion prob'] = (tpct(PRED['P_vol']), 1); S['model: drawdown prob'] = (tpct(PRED['P_off']), 1); S['model: rally prob'] = (tpct(PRED['P_on']), 1)
def fires(s, side, th=0.90):
    on = (s >= th) if side > 0 else (s <= 1 - th); on = on.fillna(False).to_numpy(); out = np.zeros(len(on), dtype=bool); quiet = 10
    for i in range(len(on)):
        if on[i] and quiet >= 10: out[i] = True
        quiet = 0 if on[i] else quiet + 1
    return pd.Series(out, index=idx)
EV = [('2024-07-16', '2024-08-05', -8.4), ('2025-02-19', '2025-04-08', -18.8), ('2026-01-27', '2026-03-30', -8.9)]
fmin21 = spy[::-1].rolling(21, min_periods=21).min()[::-1].shift(-1); fmin63 = spy[::-1].rolling(63, min_periods=63).min()[::-1].shift(-1)
hit5 = (fmin21 / spy - 1 <= -0.05); hit10 = (fmin63 / spy - 1 <= -0.10)
rows = []
for k, (s, side) in S.items():
    fr = fires(s, side); on = ((s >= 0.9) if side > 0 else (s <= 0.1)).fillna(False); row = {'signal': k}; early = 0
    for a, b, d in EV:
        a = pd.Timestamp(a); b = pd.Timestamp(b); w0 = idx[max(0, idx.get_loc(a) - 10)]; seg = on.loc[w0:b]; h = seg[seg]
        if len(h):
            t = h.index[0]; n = idx.get_loc(t) - idx.get_loc(a); dd = spy.loc[t] / spy.loc[:t].loc[a:].max() * 100 - 100 if t >= a else 0.0
            row[f'{a.year}'] = f'{n:+d}s, SPY {dd:.1f}%'; early += int(dd > -5)
        else: row[f'{a.year}'] = 'never'
    row['fired before SPY -5% in'] = f'{early} of 3'
    for a0, lab in (('2006-01-01', 'since 2006'), ('2023-01-01', 'since 2023')):
        m = fr & (idx >= pd.Timestamp(a0)) & hit5.notna(); n = int(m.sum())
        row[f'fires {lab}'] = n; row[f'then -5% in 21d, {lab}'] = f'{hit5[m].mean() * 100:.0f}%' if n else ''
    m = fr & (idx >= pd.Timestamp('2006-01-01')) & fmin63.notna(); row['then -10% in 63d, since 2006'] = f'{hit10[m].mean() * 100:.0f}%' if m.sum() else ''
    rows.append(row)
T = pd.DataFrame(rows); T['_k'] = T['fired before SPY -5% in'].str[0].astype(int); T = T.sort_values(['_k', 'signal'], ascending=[False, True]).drop(columns='_k')
m = (idx >= pd.Timestamp('2006-01-01')) & hit5.notna(); m23 = (idx >= pd.Timestamp('2023-01-01')) & hit5.notna()
print(f'base rates: 5% decline within 21 sessions, any day: {hit5[m].mean() * 100:.0f}% since 2006, {hit5[m23].mean() * 100:.0f}% since 2023; 10% within 63: {hit10[(idx >= pd.Timestamp("2006-01-01")) & fmin63.notna()].mean() * 100:.0f}% since 2006')
print('events: 2024-07-16 to 08-05 (-8.4%), 2025-02-19 to 04-08 (-18.8%), 2026-01-27 to 03-30 (-8.9%); cell = first reading at the risk extreme from 10 sessions before the peak, sessions after the peak and SPY off its high')
print(T.to_string(index=False)); pd.to_pickle(T, 'THREE.pkl')
