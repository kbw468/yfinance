#!/usr/bin/env python3
"""Credit (HYG) and vol-of-vol (VVIX) readings, on their own and in the configuration of the 7 October 2026 close:
HYG range vol high while SPY range vol, VIX and VVIX sit near their floors. Run from risk_regime/work.
(1) standalone AUC of each HYG and VVIX feature against a 5% SPY decline within 21 sessions and a 10% decline within 63,
by era and 2023-26 (no fitting: the feature itself is the score); (2) conditional outcomes for each configuration:
sessions, separate episodes, then the 5%/21 and 10%/63 hit rates, the 5% rally rate, vol-expansion rate and mean next
month, against the base rate of the same window. Feb-Jul 2020 excluded."""
import numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore'); pd.set_option('display.width', 300); pd.set_option('display.max_rows', 200)
from sklearn.metrics import roc_auc_score
F = pd.read_pickle('F.pkl'); MF = pd.read_pickle('MF.pkl'); X = pd.concat([F, MF], axis=1); L = pd.read_pickle('LAB.pkl'); idx = X.index
EX = (idx >= pd.Timestamp('2020-02-01')) & (idx <= pd.Timestamp('2020-07-31')); ok21 = L['ok21'].to_numpy(bool) & ~EX; ok63 = L['ok63'].to_numpy(bool) & ~EX
ERAS = {'2006-12': (2006, 2012), '2013-19': (2013, 2019), '2020-26': (2020, 2026), '2023-26': (2023, 2026), 'all 2006-26': (2006, 2026)}
Y = idx.year.to_numpy()
print('=== (1) standalone AUC, the feature itself as the score (sign chosen so above 0.5 means high value = more risk) ===')
FE = [('HYG Parkinson vol 21d', 'HYG:pk21_rank', 1), ('HYG Parkinson vol 5d', 'HYG:pk5_rank', 1), ('HYG realized vol 21d', 'HYG:cc21_rank', 1), ('HYG signed volume 21d, inverted', 'HYG:signedvol21', -1),
      ('HYG 21d ROC, inverted', 'HYG:roc21_rank', -1), ('HYG 63d relative to SPY', 'HYG:rel63_rank', 1), ('HYG drawdown 63d, inverted', 'HYG:dd63', -1), ('HYG persistence h2, inverted', 'MF:HYG:h2_rank', -1),
      ('VIX over HYG realized, inverted', 'HYG:VIX/cc21_rank', -1), ('VVIX level', 'VVIX:lvl_rank504', 1), ('VVIX 5d ROC', 'VVIX:roc5_rank', 1), ('VVIX 21d ROC', 'VVIX:roc21_rank', 1), ('VVIX/VIX level, inverted', 'VVIX/VIX:lvl_rank', -1),
      ('reference: VIX level', 'VIX:lvl_rank504', 1), ('reference: SPY Parkinson 21d', 'SPY:pk21_rank', 1)]
rows = []
for name, c, sg in FE:
    if c not in X: continue
    v = sg * X[c].to_numpy(float); row = {'feature': name}
    for lab, okm, tag in (('riskoff21', ok21, '5%/21'), ('riskoff63', ok63, '10%/63')):
        y = L[lab].to_numpy(float)
        for e, (a, b) in ERAS.items():
            if tag == '10%/63' and e not in ('all 2006-26', '2023-26'): continue
            m = okm & ~np.isnan(v) & ~np.isnan(y) & (Y >= a) & (Y <= b)
            row[f'{tag} {e}'] = round(float(roc_auc_score(y[m], v[m])), 2) if m.sum() > 100 and len(np.unique(y[m])) == 2 else None
    rows.append(row)
print(pd.DataFrame(rows).to_string(index=False))
r = lambda c: X[c].astype(float)
COND = {
 'HYG range vol >= 80th': r('HYG:pk21_rank') >= 0.8,
 'HYG range vol >= 80th and HYG signed volume < 0': (r('HYG:pk21_rank') >= 0.8) & (r('HYG:signedvol21') < 0),
 'credit/equity divergence: HYG range vol >= 80th, SPY range vol <= 30th': (r('HYG:pk21_rank') >= 0.8) & (r('SPY:pk21_rank') <= 0.3),
 'divergence and HYG signed volume < 0': (r('HYG:pk21_rank') >= 0.8) & (r('SPY:pk21_rank') <= 0.3) & (r('HYG:signedvol21') < 0),
 'VVIX level <= 10th': r('VVIX:lvl_rank504') <= 0.1,
 'VVIX <= 10th and VIX <= 20th (both floors)': (r('VVIX:lvl_rank504') <= 0.1) & (r('VIX:lvl_rank504') <= 0.2),
 'divergence and VVIX <= 10th (the 7 Oct configuration)': (r('HYG:pk21_rank') >= 0.8) & (r('SPY:pk21_rank') <= 0.3) & (r('VVIX:lvl_rank504') <= 0.1),
 'HYG persistence h2 <= 10th': r('MF:HYG:h2_rank') <= 0.1,
 'VVIX 5d ROC >= 90th while VIX <= 30th': (r('VVIX:roc5_rank') >= 0.9) & (r('VIX:lvl_rank504') <= 0.3),
}
def episodes(mask):
    d = idx[mask]; n = 0; last = None
    for t in d:
        if last is None or idx.get_loc(t) - idx.get_loc(last) > 10: n += 1
        last = t
    return n
print('\n=== (2) what followed each configuration (5%/21 = SPY fell 5% within 21 sessions; 10%/63; rally = rose 5% within 21; vol x1.5 = realized vol expanded 1.5x within 21) ===')
rows = []
for name, c in COND.items():
    c = c.fillna(False).to_numpy(bool)
    for e, (a, b) in (('all 2006-26', (2006, 2026)), ('2006-15', (2006, 2015)), ('2016-26', (2016, 2026)), ('2023-26', (2023, 2026))):
        w = (Y >= a) & (Y <= b); m = c & w & ok21; m63 = c & w & ok63; base = w & ok21; base63 = w & ok63
        rows.append({'configuration': name, 'window': e, 'sessions': int(m.sum()), 'episodes': episodes(m), '5%/21': round(L['riskoff21'][m].mean(), 2) if m.sum() else None, 'base 5%/21': round(L['riskoff21'][base].mean(), 2),
                     '10%/63': round(L['riskoff63'][m63].mean(), 2) if m63.sum() else None, 'base 10%/63': round(L['riskoff63'][base63].mean(), 2), 'rally 5%/21': round(L['riskon21'][m].mean(), 2) if m.sum() else None,
                     'vol x1.5': round(L['volexp21'][m].mean(), 2) if m.sum() else None, 'mean next month %': round(L['fwd21'][m].mean() * 100, 2) if m.sum() else None, 'base next month %': round(L['fwd21'][base].mean() * 100, 2)})
T = pd.DataFrame(rows); print(T.to_string(index=False))
print('\n7 Oct 2026 close:', {k: round(float(X[c].iloc[-1]), 2) for k, c in (('HYG range vol 21d', 'HYG:pk21_rank'), ('HYG range vol 5d', 'HYG:pk5_rank'), ('HYG signed volume 21d', 'HYG:signedvol21'), ('HYG persistence h2', 'MF:HYG:h2_rank'), ('SPY range vol 21d', 'SPY:pk21_rank'), ('VIX level', 'VIX:lvl_rank504'), ('VVIX level', 'VVIX:lvl_rank504'), ('VVIX 5d ROC', 'VVIX:roc5_rank'))})
print('configurations true today:', [k for k, c in COND.items() if bool(c.fillna(False).iloc[-1])])
pd.to_pickle(T, 'HYGVVIX.pkl')
