#!/usr/bin/env python3
"""v2 Stage 6a: the ticker layer on its own terms. Run from risk_regime/work.

For each of the 38 ETFs: (1) which of its OWN features (realized vol estimators and ratios, volume
rank and ROC, volatility of volume, range compression, liquidity, relative ROC, multifractal where
available) carry an era-consistent rank correlation with its own forward 21-session return and with
its own 5%-drawdown-within-21 label; (2) how the ETF behaves around objective vol-episode starts
(median relative-to-SPY ROC rank at offsets -10..+10) and in the sessions after; (3) the ETF's
forward 10-session return relative to SPY conditional on cross-sectional state (dispersion,
correlation, realized-vol breadth quintiles). Outputs TICK2.pkl and a printed report.
"""
import pandas as pd, numpy as np, warnings, time; warnings.filterwarnings('ignore')
t0 = time.time(); pd.set_option('display.width', 320); pd.set_option('display.max_rows', 500)
F = pd.read_pickle('F.pkl'); O = pd.read_pickle('OHLCV.pkl'); L = pd.read_pickle('LAB.pkl'); idx = F.index; T = O['T']
TICK = ['QQQ','TLT','IEF','HYG','IWM','GLD','SPY','XLC','XLY','XLP','XLE','XLF','XLV','XLI','XLB','XLRE','RWR','XAR','KBE','XBI','KCE','XHE','XHS','XHB','KIE','XME','XES','XOP','XPH','KRE','XRT','XSD','XSW','XTL','XTN','XLK','XLU','BNO']
EX = (idx >= pd.Timestamp('2020-02-01')) & (idx <= pd.Timestamp('2020-07-31'))
ERAS = {'A': ('1990-01-01', '2012-12-31'), 'B': ('2013-01-01', '2019-12-31'), 'C': ('2020-01-01', '2026-12-31')}
spy = T['SPY']['Close']
OWN = ['cc5_21_rank','cc10_21_rank','cc21_63_rank','cc21_126_rank','pk5_21_rank','gk10_21_rank','pk21_cc21_rank','cc21_roc5_rank','cc21_roc21_rank','cc10_roc5_rank','volofvol21_rank','volofvol21_roc5_rank','vol_rank63','vol_rank252','vroc5_rank','vroc21_rank','volvol21_rank','volvol21_roc5_rank','volvol21_roc21_rank','amihud21_rank','amihud_roc5_rank','signedvol5','signedvol21','range_rank','compress5_63_rank','nr5','clv5','gap5abs_rank','roc5_rank','roc21_rank','roc63_rank','rel5_rank','rel21_rank','rel63_rank','relrv21_roc5_rank','cc21_rank','pk21_rank','dd63','dd252']
rows = []; summary = {}
for s in TICK:
    c = T[s]['Close']; fwd21 = np.log(c.shift(-21) / c); dd21 = np.log(c[::-1].rolling(21).min()[::-1].shift(-1) / c); off = (dd21 <= np.log(0.95)).astype(float)
    rel10 = np.log(c.shift(-10) / c) - np.log(spy.shift(-10) / spy)
    ok = L['ok21'] & ~EX & fwd21.notna()
    feats = [f'{s}:{k}' for k in OWN if f'{s}:{k}' in F.columns]
    res = {}
    for e, (a, b) in ERAS.items():
        m = ok & (idx >= pd.Timestamp(a)) & (idx <= pd.Timestamp(b))
        if m.sum() < 400: continue
        Xr = F.loc[m, feats].rank(pct=True)
        res[e] = pd.DataFrame({'ic_ret': Xr.corrwith(fwd21[m].rank(pct=True)), 'ic_dd': Xr.corrwith(dd21[m].rank(pct=True)), 'n': F.loc[m, feats].notna().sum()})
    if len(res) < 2: continue
    R = pd.concat(res, axis=1)
    for col in ('ic_ret', 'ic_dd'):
        vals = pd.concat([R[(e, col)].rename(e) for e in res], axis=1); ns = pd.concat([R[(e, 'n')].rename(e) for e in res], axis=1); vals = vals.where(ns >= 300)
        same = (np.sign(vals).nunique(axis=1) == 1) & (vals.notna().sum(axis=1) >= 2)
        R[('cons', col)] = vals.abs().min(axis=1).where(same, 0) * np.sign(vals.mean(axis=1))
    R[('cons', 'eras')] = len(res); summary[s] = R
    top = R[('cons', 'ic_dd')].sort_values()
    rows.append({'ticker': s, 'eras': len(res), 'n_consistent_|ic|>=0.05': int((R[('cons', 'ic_dd')].abs() >= 0.05).sum()), 'deepest_dd_feature': top.index[0].split(':')[1], 'ic': round(float(top.iloc[0]), 3), 'safest_feature': top.index[-1].split(':')[1], 'ic_safe': round(float(top.iloc[-1]), 3),
                 'best_ret_feature': R[('cons', 'ic_ret')].abs().idxmax().split(':')[1], 'ic_ret': round(float(R[('cons', 'ic_ret')].loc[R[('cons', 'ic_ret')].abs().idxmax()]), 3)})
S = pd.DataFrame(rows).set_index('ticker'); print('=== per-ETF own-feature screen: era-consistent rank IC with the ETF\'s own 21d drawdown depth (negative = high value precedes deeper drawdown) and 21d return ===')
print(S.to_string())
# which own-features are consistent across MANY tickers?
agg = {}
for s, R in summary.items():
    for f, v in R[('cons', 'ic_dd')].items():
        agg.setdefault(f.split(':')[1], []).append(v)
A = pd.DataFrame({k: {'tickers': len(v), 'share_negative(|ic|>=.05)': np.mean(np.array(v) <= -0.05), 'share_positive(|ic|>=.05)': np.mean(np.array(v) >= 0.05), 'median_ic': np.median(v)} for k, v in agg.items()}).T.sort_values('median_ic')
print('\n=== the same own-feature across tickers: how often it is consistently risky (negative) or safe (positive) for the ETF\'s own drawdowns ==='); print(A.round(3).to_string())
# ---- behaviour around vol-episode starts: relative ROC5 rank at offsets, and forward 10d relative return after the start
EP = pd.read_pickle('EPISODES.pkl'); starts = [d for d in EP['start'] if d >= pd.Timestamp('2000-01-01') and not ((d >= pd.Timestamp('2020-02-01')) & (d <= pd.Timestamp('2020-07-31')))]
OFFS = [-5, -3, -1, 0, 1, 3, 5, 10]; lead = {}
for s in TICK:
    if s == 'SPY' or f'{s}:rel5_rank' not in F.columns: continue
    r5 = F[f'{s}:rel5_rank'].astype(float); c = T[s]['Close']; rel10 = np.log(c.shift(-10) / c) - np.log(spy.shift(-10) / spy); rel21 = np.log(c.shift(-21) / c) - np.log(spy.shift(-21) / spy)
    pos = [idx.get_loc(d) for d in starts if not np.isnan(r5.iloc[idx.get_loc(d)])]
    row = {f'rel5rank@{o:+d}': np.nanmedian([r5.iloc[p + o] for p in pos if 0 <= p + o < len(r5)]) for o in OFFS}
    row['fwd10rel%@0'] = np.nanmedian([rel10.iloc[p] for p in pos]) * 100; row['fwd21rel%@0'] = np.nanmedian([rel21.iloc[p] for p in pos]) * 100; row['n'] = len(pos); lead[s] = row
LEAD = pd.DataFrame(lead).T.sort_values('fwd10rel%@0'); print(f'\n=== around {len(starts)} vol-episode starts: median relative-ROC5 rank at offsets, and median forward 10/21d return relative to SPY from the start day ==='); print(LEAD.round(2).to_string())
# ---- conditional on cross-sectional state
COND = {}
for cond in ('XS:disp5_rank', 'XS:avgcorr21_rank', 'XS:rv_breadth_rank', 'XS:vol_breadth_rank', 'XS:compress_median_rank'):
    q = pd.cut(F[cond].astype(float), [-0.01, 0.2, 0.8, 1.01], labels=['low', 'mid', 'high'])
    tab = {}
    for s in TICK:
        if s == 'SPY': continue
        c = T[s]['Close']; rel10 = (np.log(c.shift(-10) / c) - np.log(spy.shift(-10) / spy)) * 100; m = L['ok21'] & ~EX & rel10.notna()
        tab[s] = rel10[m].groupby(q[m]).median()
    COND[cond] = pd.DataFrame(tab).T
    print(f'\n=== median forward 10d return relative to SPY by {cond} tercile (low <0.2, high >0.8) ==='); print(COND[cond].round(2).sort_values('high').to_string())
pd.to_pickle({'summary': summary, 'table': S, 'agg': A, 'lead': LEAD, 'cond': COND}, 'TICK2.pkl'); print(f'{time.time()-t0:.0f}s')
