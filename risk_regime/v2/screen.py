#!/usr/bin/env python3
"""v2 Stage 3b: univariate screen of every feature, by era. Run from risk_regime/work.

For each feature: Spearman rank correlation with fwdDD21 (drawdown depth, the risk-off target) and
fwdUP21 (the risk-on target) in three eras (to 2012, 2013-19, 2020-26 with Feb-Jul 2020 masked),
plus top- and bottom-decile hit rates for riskoff21 against the era base rate. Overlapping 21-day
labels mean the effective sample is about n/21, so the screen asks for the same sign in all three
eras and a minimum |IC| in each, not for a p-value. Outputs SCREEN.pkl and a printed table.
"""
import pandas as pd, numpy as np, warnings, time; warnings.filterwarnings('ignore')
t0 = time.time(); pd.set_option('display.width', 320)
F = pd.read_pickle('F.pkl'); MF = pd.read_pickle('MF.pkl'); L = pd.read_pickle('LAB.pkl'); idx = F.index
X = pd.concat([F, MF], axis=1)
EX = (idx >= pd.Timestamp('2020-02-01')) & (idx <= pd.Timestamp('2020-07-31')); ok = L['ok21'] & ~EX
ERAS = {'A': ('1990-01-01', '2012-12-31'), 'B': ('2013-01-01', '2019-12-31'), 'C': ('2020-01-01', '2026-12-31')}
res = {}
for e, (a, b) in ERAS.items():
    m = ok & (idx >= pd.Timestamp(a)) & (idx <= pd.Timestamp(b))
    Xe = X[m]; y_dd = L.loc[m, 'fwdDD21']; y_up = L.loc[m, 'fwdUP21']; y_off = L.loc[m, 'riskoff21']; base = y_off.mean()
    Xr = Xe.rank(pct=True)
    ic_dd = Xr.corrwith(y_dd.rank(pct=True)); ic_up = Xr.corrwith(y_up.rank(pct=True))
    top = (Xr >= 0.9); bot = (Xr <= 0.1)
    p_top = (top.mul(y_off, axis=0).sum() / top.sum()); p_bot = (bot.mul(y_off, axis=0).sum() / bot.sum())
    n = Xe.notna().sum()
    res[e] = pd.DataFrame({'n': n, 'ic_dd': ic_dd, 'ic_up': ic_up, 'p_top': p_top, 'p_bot': p_bot, 'base': base})
    print(f'era {e}: {m.sum()} sessions, base P(5% DD/21d) {base:.3f}, {time.time()-t0:.0f}s')
T = pd.concat(res, axis=1)
# consistency: same sign of ic_dd in all eras with enough data; score = min |ic| (sign-adjusted) across eras with n>=400
def consist(col):
    vals = pd.concat([T[(e, col)] for e in ERAS], axis=1); ns = pd.concat([T[(e, 'n')] for e in ERAS], axis=1); vals.columns = list(ERAS); ns.columns = list(ERAS)
    vals = vals.where(ns >= 400); sgn = np.sign(vals); same = (sgn.nunique(axis=1) == 1) & (vals.notna().sum(axis=1) >= 2)
    score = (vals.abs().min(axis=1)).where(same, 0) * np.sign(vals.mean(axis=1))
    return score, vals.notna().sum(axis=1)
T[('cons', 'ic_dd')], T[('cons', 'eras')] = consist('ic_dd'); T[('cons', 'ic_up')], _ = consist('ic_up')
T[('cons', 'lift_top')] = pd.concat([(T[(e, 'p_top')] / T[(e, 'base')]).rename(e) for e in ERAS], axis=1).min(axis=1)
T[('cons', 'lift_bot')] = pd.concat([(T[(e, 'p_bot')] / T[(e, 'base')]).rename(e) for e in ERAS], axis=1).min(axis=1)
T[('cons', 'lift_top_max')] = pd.concat([(T[(e, 'p_top')] / T[(e, 'base')]).rename(e) for e in ERAS], axis=1).max(axis=1)
T.to_pickle('SCREEN.pkl')
fam = lambda c: c.split(':')[0] if c.startswith(('XS', 'MF', 'RSP', 'QQQE')) else ('idx' if c.split(':')[0] in ('VIX','MOVE','VXN','VVIX','TNX','TYX','GVZ','OVX','VIX1D','VIX9D','VIX3M','VIX6M','SKEW','FVX','IRX') or '/' in c.split(':')[0] else 'etf')
T[('cons', 'family')] = [fam(c) for c in T.index]
def show(col, k=40, ascending=False):
    s = T[('cons', col)].sort_values(ascending=ascending).head(k)
    out = pd.DataFrame({'cons': s.round(3), 'icA': T.loc[s.index, ('A', 'ic_dd' if 'dd' in col else 'ic_up')].round(3), 'icB': T.loc[s.index, ('B', 'ic_dd' if 'dd' in col else 'ic_up')].round(3), 'icC': T.loc[s.index, ('C', 'ic_dd' if 'dd' in col else 'ic_up')].round(3),
                        'ptopA': T.loc[s.index, ('A', 'p_top')].round(2), 'ptopB': T.loc[s.index, ('B', 'p_top')].round(2), 'ptopC': T.loc[s.index, ('C', 'p_top')].round(2), 'pbotA': T.loc[s.index, ('A', 'p_bot')].round(2), 'pbotB': T.loc[s.index, ('B', 'p_bot')].round(2), 'pbotC': T.loc[s.index, ('C', 'p_bot')].round(2)})
    print(out.to_string())
print('\n=== features whose HIGH values precede deeper 21d drawdowns in every era (most negative consistent IC with fwdDD21) ===')
show('ic_dd', 45, ascending=True)
print('\n=== features whose HIGH values precede shallower drawdowns / safer tape in every era (most positive consistent IC with fwdDD21) ===')
show('ic_dd', 30, ascending=False)
print('\n=== features whose HIGH values precede bigger 21d rallies in every era (consistent IC with fwdUP21) ===')
show('ic_up', 30, ascending=False)
c = T['cons']; n_cons = int((c['ic_dd'].abs() >= 0.05).sum()); print(f'\nfeatures with |IC| >= 0.05 in every era, same sign: {n_cons} of {len(T)}; >= 0.08: {int((c["ic_dd"].abs() >= 0.08).sum())}; >= 0.10: {int((c["ic_dd"].abs() >= 0.10).sum())}')
print('by family (|IC|>=0.05):', c[c['ic_dd'].abs() >= 0.05].groupby('family').size().to_dict())
print(f'{time.time()-t0:.0f}s')
