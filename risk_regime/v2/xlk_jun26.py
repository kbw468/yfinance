#!/usr/bin/env python3
"""XLK's top of 2 June 2026 (197.74 close, after +14.4% in ten sessions from 19 May) and the drawdown to 29 July
(166.38, -15.9%; -10.9% by 10 June, intraday -12.8% on 9 June). Run from risk_regime/work.
For every tool in the box that reads XLK, Nasdaq vol or the system's own readings: the first session from 1 May to the
trough where it reached its risk extreme (top or bottom 10% of its trailing two years, or the system's own lines), and
XLK's decline from the high at that close. Then each signal's record on XLK itself: every fire since 2006 and since 2023
(after 10 quiet sessions), and the share followed by a 5% XLK decline within 21 sessions and a 10% decline within 63."""
import numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore'); pd.set_option('display.width', 330); pd.set_option('display.max_rows', 200)
F = pd.read_pickle('F.pkl'); O = pd.read_pickle('OHLCV.pkl'); idx = F.index; PRED = pd.read_pickle('PRED2.pkl'); VS = pd.read_pickle('VVSIG.pkl'); DEC = pd.read_pickle('DECISION.pkl')
xlk = O['T']['XLK']['Close'].reindex(idx); vxn = O['I']['VXN']['Close'].reindex(idx); vix = O['I']['VIX']['Close'].reindex(idx)
rk = lambda s: s.rolling(504, min_periods=252).rank(pct=True)
def tpct(s, w=756):
    v = s.to_numpy(float); o = np.full(len(v), np.nan)
    for i in range(252, len(v)):
        x = v[max(0, i - w):i]; x = x[~np.isnan(x)]
        if len(x) >= 252 and not np.isnan(v[i]): o[i] = (x <= v[i]).mean()
    return pd.Series(o, index=idx)
f = lambda c: F[c].astype(float)
up5 = f('XLK:roc5_rank'); vxn5 = f('VXN:roc5_rank')
SIG = {  # name: (rank series, 'hi' or 'lo', threshold)
 'XLK 5d ROC, top (blow-off)': (f('XLK:roc5_rank'), 'hi', 0.9), 'XLK 10d ROC, top (blow-off)': (f('XLK:roc10_rank'), 'hi', 0.9), 'XLK 21d ROC, top': (f('XLK:roc21_rank'), 'hi', 0.9),
 'XLK vs SPY 10d, top': (f('XLK:rel10_rank'), 'hi', 0.9), 'XLK vs SPY 21d, top': (f('XLK:rel21_rank'), 'hi', 0.9),
 'XLK 1d ROC, bottom (breakdown day)': (f('XLK:roc1_rank'), 'lo', 0.1), 'XLK 5d ROC, bottom': (f('XLK:roc5_rank'), 'lo', 0.1), 'XLK 21d ROC, bottom': (f('XLK:roc21_rank'), 'lo', 0.1),
 'XLK vs SPY 5d, bottom': (f('XLK:rel5_rank'), 'lo', 0.1),
 'XLK realized vol 5d/21d, top': (f('XLK:cc5_21_rank'), 'hi', 0.9), 'XLK realized vol 21d ROC, top': (f('XLK:cc21_roc5_rank'), 'hi', 0.9), 'XLK range vol 5d, top': (f('XLK:pk5_rank'), 'hi', 0.9),
 'XLK range vol 21d, top': (f('XLK:pk21_rank'), 'hi', 0.9), 'XLK gap size, top': (f('XLK:gap5abs_rank'), 'hi', 0.9),
 'XLK volume 5d ROC, top': (f('XLK:vroc5_rank'), 'hi', 0.9), 'XLK volume 21d ROC, top': (f('XLK:vroc21_rank'), 'hi', 0.9), 'XLK vol-of-volume, top': (f('XLK:volvol21_rank'), 'hi', 0.9),
 'XLK illiquidity, top': (f('XLK:amihud21_rank'), 'hi', 0.9), 'XLK range compression, top': (f('XLK:compress5_63_rank'), 'hi', 0.9),
 'VXN 5d ROC, top': (f('VXN:roc5_rank'), 'hi', 0.9), 'VXN 21d ROC, top': (f('VXN:roc21_rank'), 'hi', 0.9), 'VXN/VIX 5d ROC, top': (f('VXN/VIX:roc5_rank'), 'hi', 0.9), 'VXN/VIX level, top': (f('VXN/VIX:lvl_rank'), 'hi', 0.9),
 'spot up, vol up: XLK 5d ROC top 10% and VXN 5d ROC above median': (((up5 >= 0.9) & (vxn5 >= 0.5)).astype(float), 'hi', 0.5),
 'dispersion 5d, top': (f('XS:disp5_rank'), 'hi', 0.9), 'avg correlation 5d change, top': (f('XS:avgcorr21_chg5_rank'), 'hi', 0.9),
 'system: drawdown probability at its 95th': (tpct(PRED['P_off']), 'hi', 0.95), 'system: vol-expansion probability at its 90th': (tpct(PRED['P_vol']), 'hi', 0.9),
 'system: reference rule OUT': ((DEC['states']['C: OUT P_off>q0.95, IN P_off<q0.8'] == 0).astype(float), 'hi', 0.5), 'system: VIX outrunning VVIX live': ((VS['state'] == 0).astype(float), 'hi', 0.5),
}
pk, tr = pd.Timestamp('2026-06-02'), pd.Timestamp('2026-07-29'); w0 = pd.Timestamp('2026-05-01')
hi_run = xlk.cummax()
f21 = xlk[::-1].rolling(21, min_periods=21).min()[::-1].shift(-1); dn5 = (f21 / xlk - 1 <= -0.05)
f63 = xlk[::-1].rolling(63, min_periods=63).min()[::-1].shift(-1); dn10 = (f63 / xlk - 1 <= -0.10)
def fires(on):
    on = on.fillna(False).to_numpy(); out = np.zeros(len(on), bool); q = 10
    for i in range(len(on)):
        if on[i] and q >= 10: out[i] = True
        q = 0 if on[i] else q + 1
    return pd.Series(out, index=idx)
rows = []
for name, (s, side, th) in SIG.items():
    on = (s >= th) if side == 'hi' else (s <= th); on = on.fillna(False); seg = on.loc[w0:tr]; hits = seg[seg].index
    first = hits[0] if len(hits) else None; atpk = s.loc[pk]
    row = {'signal': name, 'at the 2 Jun top': round(float(atpk), 2) if not np.isnan(atpk) else None,
           'first fire 1 May to trough': f"{first.date()} ({(idx.get_loc(first) - idx.get_loc(pk)):+d}s, XLK {xlk[first] / hi_run[first] * 100 - 100:.1f}% off high)" if first is not None else 'never',
           'sessions on, 1 May to trough': int(seg.sum())}
    fr = fires(on)
    for a0, lab in (('2006-01-01', '06+'), ('2023-01-01', '23+')):
        m = fr & (idx >= pd.Timestamp(a0)) & dn5.notna(); row[f'fires {lab}'] = int(m.sum()); row[f'XLK -5%/21 {lab}'] = f'{dn5[m].mean() * 100:.0f}%' if m.sum() else ''
    m = fr & (idx >= pd.Timestamp('2006-01-01')) & dn10.notna(); row['XLK -10%/63 06+'] = f'{dn10[m].mean() * 100:.0f}%' if m.sum() else ''
    rows.append(row)
T = pd.DataFrame(rows)
b6 = (idx >= pd.Timestamp('2006-01-01')) & dn5.notna(); b23 = (idx >= pd.Timestamp('2023-01-01')) & dn5.notna()
print(f'XLK base rates, any day: 5% decline within 21 sessions {dn5[b6].mean() * 100:.0f}% since 2006, {dn5[b23].mean() * 100:.0f}% since 2023; 10% within 63 {dn10[(idx >= pd.Timestamp("2006-01-01")) & dn10.notna()].mean() * 100:.0f}% since 2006')
print('XLK: 172.83 on 19 May, 186.41 on 28 May, top 197.74 on 2 Jun (+14.4% in 10 sessions); 179.88 on 5 Jun (-6.7% day, 2x volume); 176.21 on 10 Jun (-10.9%); lower highs 191.34 / 191.93 / 190.30 (15 / 22 / 30 Jun); 166.38 on 29 Jul (-15.9%)')
print(T.to_string(index=False))
print('\nreadings around the top (rank of each in its trailing two years):')
cols = {'XLK 10d ROC': 'XLK:roc10_rank', 'XLK vs SPY 10d': 'XLK:rel10_rank', 'XLK vol 5d/21d': 'XLK:cc5_21_rank', 'XLK range vol 5d': 'XLK:pk5_rank', 'XLK volume 5d ROC': 'XLK:vroc5_rank', 'VXN 5d ROC': 'VXN:roc5_rank', 'VXN/VIX 5d ROC': 'VXN/VIX:roc5_rank', 'dispersion 5d': 'XS:disp5_rank'}
tab = pd.DataFrame({k: F[c].astype(float) for k, c in cols.items()}).loc['2026-05-19':'2026-06-12'].round(2); tab['P_off pct'] = tpct(PRED['P_off']).loc['2026-05-19':'2026-06-12'].round(2); tab['P_vol pct'] = tpct(PRED['P_vol']).loc['2026-05-19':'2026-06-12'].round(2); tab['XLK'] = xlk.loc['2026-05-19':'2026-06-12'].round(2)
print(tab.to_string())
pd.to_pickle(T, 'XLK_JUN26.pkl')
