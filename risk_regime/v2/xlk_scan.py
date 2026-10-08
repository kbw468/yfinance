#!/usr/bin/env python3
"""Exhaustive scan for anything that was lit before XLK's 5 June 2026 break and carries a record on XLK.
Run from risk_regime/work. Every rank feature in the store (F and MF, about 4,100) at both tails (top and bottom 10%),
plus constructed signals (semis vs tech, equal-weight Nasdaq vs QQQ, upside vol, price/volume divergence, rates into
the top, combinations). A signal qualifies if it was in its tail on at least one session from 19 May to 4 June 2026
(XLK at most 2.5% off its high). For each: fires since 2006 (after 10 quiet sessions), XLK 5% decline within 21
sessions and 10% within 63 after each fire, in 2006-15, 2016-26 and 2023-26, against XLK's base rates. Ranked by the
weaker half's lift on the 5% target, at least 25 fires. ~8,300 tests: some will look good by chance, which is why
both halves must hold. Writes XLK_SCAN.pkl."""
import numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore'); pd.set_option('display.width', 330); pd.set_option('display.max_rows', 300)
F = pd.read_pickle('F.pkl'); MF = pd.read_pickle('MF.pkl'); X = pd.concat([F, MF], axis=1); O = pd.read_pickle('OHLCV.pkl'); idx = X.index; Y = idx.year.to_numpy()
T = O['T']; xlk = T['XLK']['Close'].reindex(idx); rk = lambda s: s.rolling(504, min_periods=252).rank(pct=True); lr = lambda s, n: np.log(s / s.shift(n))
# constructed signals (ranks)
C = {}
xsd = T['XSD']['Close'].reindex(idx); qqq = T['QQQ']['Close'].reindex(idx); spy = T['SPY']['Close'].reindex(idx)
try:
    qe = pd.read_csv('data_ew/QQQE.csv', index_col=0, parse_dates=True)['Close'].reindex(idx); C['QQQE vs QQQ 21d ROC'] = rk(lr(qe, 21) - lr(qqq, 21)); C['QQQE vs QQQ 10d ROC'] = rk(lr(qe, 10) - lr(qqq, 10))
except Exception as e: print('no QQQE', e)
for n in (5, 10, 21): C[f'semis vs XLK {n}d ROC'] = rk(lr(xsd, n) - lr(xlk, n))
C['XLK upside vol: 5d ROC rank x realized 5d rank'] = (X['XLK:roc5_rank'].astype(float) * X['XLK:cc5_rank'].astype(float))
C['XLK at a 63d high with VXN/VIX top 10%'] = ((X['XLK:dd63'].astype(float) >= -0.005) & (X['VXN/VIX:lvl_rank'].astype(float) >= 0.9)).astype(float)
C['XLK 10d ROC top 5% and VXN 5d ROC above median'] = ((X['XLK:roc10_rank'].astype(float) >= 0.95) & (X['VXN:roc5_rank'].astype(float) >= 0.5)).astype(float)
C['XLK 21d ROC top 10% and 10y yield 21d ROC top 10%'] = ((X['XLK:roc21_rank'].astype(float) >= 0.9) & (X['TNX:roc21_rank'].astype(float) >= 0.9)).astype(float)
C['XLK 21d ROC top 10% and signed volume 21d below 0'] = ((X['XLK:roc21_rank'].astype(float) >= 0.9) & (X['XLK:signedvol21'].astype(float) < 0)).astype(float)
C['XLK 10d ROC top 5% and dispersion top 10%'] = ((X['XLK:roc10_rank'].astype(float) >= 0.95) & (X['XS:disp5_rank'].astype(float) >= 0.9)).astype(float)
C['XLK 10d ROC top 5% and breadth up21 low'] = ((X['XLK:roc10_rank'].astype(float) >= 0.95) & (X['XS:breadth_up21'].astype(float) <= X['XS:breadth_up21'].astype(float).rolling(504, min_periods=252).quantile(0.3))).astype(float)
cols = [c for c in X.columns if c.endswith('_rank') or 'rank' in c.split(':')[-1]] 
S = {c: X[c].astype(float) for c in cols}; S.update(C)
f21 = xlk[::-1].rolling(21, min_periods=21).min()[::-1].shift(-1); dn5 = (f21 / xlk - 1 <= -0.05).astype(float).where(f21.notna())
f63 = xlk[::-1].rolling(63, min_periods=63).min()[::-1].shift(-1); dn10 = (f63 / xlk - 1 <= -0.10).astype(float).where(f63.notna())
H = {'06-15': (Y >= 2006) & (Y <= 2015), '16-26': (Y >= 2016) & (Y <= 2026), '23-26': (Y >= 2023) & (Y <= 2026)}
base5 = {k: float(dn5[m].mean()) for k, m in H.items()}; base10 = float(dn10[(Y >= 2006)].mean())
win = (idx >= pd.Timestamp('2026-05-19')) & (idx <= pd.Timestamp('2026-06-04'))
rows = []
for name, s in S.items():
    is_bin = name in C and s.dropna().isin([0, 1]).all()
    tails = [('on', s >= 0.5)] if is_bin else [('top 10%', s >= 0.9), ('bottom 10%', s <= 0.1)]
    for tail, on in tails:
        on = on.fillna(False)
        if not on[win].any(): continue
        prev = on.shift(1, fill_value=False).astype(float).rolling(10, min_periods=1).max().fillna(0) > 0
        fr = (on & ~prev).to_numpy() & (Y >= 2006)
        n = int(fr.sum())
        if n < 25: continue
        row = {'signal': name, 'tail': tail, 'fires 06-26': n}
        lifts = []
        for k, m in H.items():
            mm = fr & m & dn5.notna().to_numpy(); hr = float(dn5[mm].mean()) if mm.sum() else np.nan
            row[f'n {k}'] = int(mm.sum()); row[f'-5%/21 {k}'] = round(hr, 2); 
            if k != '23-26': lifts.append(hr / base5[k] if mm.sum() >= 8 else np.nan)
        mm = fr & dn10.notna().to_numpy(); row['-10%/63 06-26'] = round(float(dn10[mm].mean()), 2)
        row['min half lift'] = round(float(np.nanmin(lifts)), 2) if not np.all(np.isnan(lifts)) else np.nan
        first = idx[win & on.to_numpy()][0]; row['first lit 19 May-4 Jun'] = f'{first.date()} (XLK {xlk[first] / xlk.loc[:first].cummax().iloc[-1] * 100 - 100:.1f}% off high)'
        rows.append(row)
R = pd.DataFrame(rows).sort_values('min half lift', ascending=False)
print(f'XLK base rate, 5% decline within 21 sessions: 2006-15 {base5["06-15"]:.2f}, 2016-26 {base5["16-26"]:.2f}, 2023-26 {base5["23-26"]:.2f}; 10% within 63, 2006-26 {base10:.2f}')
print(f'{len(S)} signals x tails scanned; {len(R)} were lit between 19 May and 4 Jun with at least 25 fires since 2006')
print('\n=== top 30 by the weaker half\'s lift ==='); print(R.head(30).to_string(index=False))
print('\n=== constructed signals ==='); print(R[R.signal.isin(C.keys())].to_string(index=False))
print('\nlift distribution of everything lit (min half lift):', R['min half lift'].describe().round(2).to_dict())
pd.to_pickle(R, 'XLK_SCAN.pkl')
