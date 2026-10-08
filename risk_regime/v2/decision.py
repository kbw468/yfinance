#!/usr/bin/env python3
"""v2 Stage 5: the decision layer, fully out of sample. Run from risk_regime/work after model2.py.

Inputs: PRED2.pkl, the walk-forward probabilities P_vol (vol expansion ahead), P_on (5% rally ahead),
P_off (5% drawdown ahead). (1) What the forward tape looks like by decile of each, by era. (2) The
joint P_vol x P_on cells. (3) A grid of two-state stay-in rules whose thresholds are trailing quantiles
of the probabilities (past out-of-sample values only), each backtested fully invested in SPY when IN
and in cash when OUT, lagged one session, 2006 to date, Feb-Jul 2020 excluded. The whole grid is
reported. The headline rule was fixed before this grid was run (see snapshot.py). Writes DECISION.pkl.
"""
import pandas as pd, numpy as np, time, warnings; warnings.filterwarnings('ignore')
from sklearn.metrics import roc_auc_score
t0 = time.time(); pd.set_option('display.width', 320); pd.set_option('display.max_rows', 400)
F = pd.read_pickle('F.pkl'); L = pd.read_pickle('LAB.pkl'); O = pd.read_pickle('OHLCV.pkl'); PRED = pd.read_pickle('PRED2.pkl'); idx = F.index
spy = O['T']['SPY']['Close']; r = np.log(spy).diff()
EX = (idx >= pd.Timestamp('2020-02-01')) & (idx <= pd.Timestamp('2020-07-31')); ok = L['ok21'] & ~EX
ERAS = {'2005-12': ('2005-01-01', '2012-12-31'), '2013-19': ('2013-01-01', '2019-12-31'), '2020-26': ('2020-01-01', '2026-12-31'), 'all': ('2005-01-01', '2026-12-31')}
LABOF = {'P_vol': 'volexp21', 'P_on': 'riskon21', 'P_off': 'riskoff21', 'P_off63': 'riskoff63'}
print('=== forward tape by out-of-sample probability decile, 2005-2026, Feb-Jul 2020 excluded ===')
DEC = {}
for p in ('P_vol', 'P_on', 'P_off'):
    m = ok & PRED[p].notna(); dec = pd.qcut(PRED[p][m].rank(method='first'), 10, labels=False)
    g = pd.DataFrame({'pred': PRED[p][m], 'fwd21%': L.loc[m, 'fwd21'] * 100, 'fwdDD21%': L.loc[m, 'fwdDD21'] * 100, 'fwdUP21%': L.loc[m, 'fwdUP21'] * 100, 'P(5%DD)': L.loc[m, 'riskoff21'], 'P(5%UP)': L.loc[m, 'riskon21'], 'P(volexp)': L.loc[m, 'volexp21'], 'RV21 ahead/now': np.exp(np.log(F['SPY:cc21'].astype(float).shift(-21) / F['SPY:cc21'].astype(float))[m])}).groupby(dec).mean()
    DEC[p] = g; print(f'\n{p}:'); print(g.round(3).to_string())
    for e, (a, b) in ERAS.items():
        mm = m & (idx >= pd.Timestamp(a)) & (idx <= pd.Timestamp(b)); d2 = pd.qcut(PRED[p][mm].rank(method='first'), 5, labels=False)
        print(f"   {e}: AUC {roc_auc_score(L.loc[mm, LABOF[p]], PRED[p][mm]):.3f}; mean fwd21% by quintile {[round(float(v), 2) for v in (L.loc[mm, 'fwd21'] * 100).groupby(d2).mean()]}; P(5%DD) by quintile {[round(float(v), 2) for v in L.loc[mm, 'riskoff21'].groupby(d2).mean()]}")
m = ok & PRED['P_vol'].notna() & PRED['P_on'].notna()
qv = PRED['P_vol'][m].rank(pct=True); qo = PRED['P_on'][m].rank(pct=True)
cell = pd.DataFrame({'vol': pd.cut(qv, [0, .5, .8, .9, 1.0], labels=['<50', '50-80', '80-90', '>90']), 'on': pd.cut(qo, [0, .5, .8, 1.0], labels=['<50', '50-80', '>80']), 'fwd21': L.loc[m, 'fwd21'] * 100, 'off': L.loc[m, 'riskoff21'], 'dd': L.loc[m, 'fwdDD21'] * 100, 'up': L.loc[m, 'fwdUP21'] * 100})
JOINT = pd.concat({'fwd21': cell.pivot_table(index='vol', columns='on', values='fwd21', aggfunc='mean').round(2), 'P_off': cell.pivot_table(index='vol', columns='on', values='off', aggfunc='mean').round(2), 'DD21': cell.pivot_table(index='vol', columns='on', values='dd', aggfunc='mean').round(1), 'UP21': cell.pivot_table(index='vol', columns='on', values='up', aggfunc='mean').round(1), 'n': cell.pivot_table(index='vol', columns='on', values='off', aggfunc='count')}, axis=1)
print('\n=== joint cells: P_vol percentile (rows) x P_on percentile (cols): mean fwd21 % | P(5% DD/21d) | mean fwdDD21 % | mean fwdUP21 % | n ==='); print(JOINT.to_string())
# ---------- two-state rules on trailing quantiles
def tq(s, q, w=756): return s.rolling(w, min_periods=252).quantile(q).shift(1)
def states(out_cond, in_cond):
    o = out_cond.fillna(False).values; i = in_cond.fillna(False).values; st = np.ones(len(o), dtype=int); s = 1
    for k in range(len(o)):
        if s == 1 and o[k]: s = 0
        elif s == 0 and i[k]: s = 1
        st[k] = s
    return pd.Series(st, index=idx)
m = ~EX & (idx >= pd.Timestamp('2006-01-01')); bh = r[m]
def stats(x):
    c = x.cumsum(); dd = c - c.cummax(); return dict(ann=round(x.mean() * 252 * 100, 2), vol=round(x.std() * np.sqrt(252) * 100, 2), sharpe=round(x.mean() / x.std() * np.sqrt(252), 2), maxDD=round(dd.min() * 100, 1), ulcer=round(np.sqrt((dd ** 2).mean()) * 100, 2))
EP = [('2008-05-19', '2009-03-09'), ('2010-04-23', '2010-07-02'), ('2011-04-29', '2011-10-03'), ('2015-07-20', '2015-08-25'), ('2015-11-03', '2016-02-11'), ('2018-01-26', '2018-02-08'), ('2018-09-20', '2018-12-24'), ('2022-01-03', '2022-10-12'), ('2023-07-31', '2023-10-27'), ('2024-07-16', '2024-08-05'), ('2025-02-19', '2025-04-08'), ('2026-01-27', '2026-03-30')]
def book(ST, name):
    w = ST.shift(1); x = (w * r)[m]; o = (ST == 0) & m; spells = 0; last = None
    for d in idx[o]:
        if last is None or (idx.get_loc(d) - idx.get_loc(last)) > 1: spells += 1
        last = d
    row = {'rule': name, **stats(x), 'exposure': round(float(w[m].mean()), 2), 'spells': spells}
    for e, (a, b) in ERAS.items():
        mm = m & (idx >= pd.Timestamp(a)) & (idx <= pd.Timestamp(b)); row[f'ann {e}'] = round(float((w * r)[mm].mean() * 252 * 100), 1); row[f'maxDD {e}'] = stats((w * r)[mm])['maxDD']
    caps = []
    for a, b in EP:
        a = pd.Timestamp(a); b = pd.Timestamp(b); seg = r.loc[a:b].iloc[1:]; sp = seg.sum(); st_ = (w.loc[seg.index] * seg).sum(); caps.append(round(float(st_ / sp), 2) if sp else np.nan)
    row['loss captured (12 episodes)'] = caps; row['today'] = 'IN' if ST.iloc[-1] == 1 else 'OUT'; return row
rows = [{'rule': 'SPY buy and hold', **stats(bh), 'exposure': 1.0, 'spells': 0, **{f'ann {e}': round(float(r[m & (idx >= pd.Timestamp(a)) & (idx <= pd.Timestamp(b))].mean() * 252 * 100), 1) for e, (a, b) in ERAS.items()}, **{f'maxDD {e}': stats(r[m & (idx >= pd.Timestamp(a)) & (idx <= pd.Timestamp(b))])['maxDD'] for e, (a, b) in ERAS.items()}, 'loss captured (12 episodes)': [1.0] * 12, 'today': 'IN'}]
PV, PO, PF = PRED['P_vol'], PRED['P_on'], PRED['P_off']; STATES = {}
for qout in (0.85, 0.90, 0.95):
    for qin in (0.5, 0.7):
        ST = states(PV >= tq(PV, qout), PV < tq(PV, qin)); name = f'A: OUT P_vol>q{qout:.2f}, IN P_vol<q{qin:.1f}'; rows.append(book(ST, name)); STATES[name] = ST
        ST = states((PV >= tq(PV, qout)) & (PO < tq(PO, 0.5)), (PV < tq(PV, qin)) | (PO >= tq(PO, 0.8))); name = f'B: OUT P_vol>q{qout:.2f} & P_on<q.5, IN P_vol<q{qin:.1f} or P_on>q.8'; rows.append(book(ST, name)); STATES[name] = ST
P63 = PRED['P_off63'] if 'P_off63' in PRED else None
for qout in (0.90, 0.95, 0.98):
    for qin in (0.6, 0.8):
        ST = states(PF >= tq(PF, qout), PF < tq(PF, qin)); name = f'C: OUT P_off>q{qout:.2f}, IN P_off<q{qin:.1f}'; rows.append(book(ST, name)); STATES[name] = ST
    ST = states((PF >= tq(PF, qout)) & (PO < tq(PO, 0.5)), (PF < tq(PF, 0.6)) | (PO >= tq(PO, 0.8))); name = f'D: OUT P_off>q{qout:.2f} & P_on<q.5, IN P_off<q.6 or P_on>q.8'; rows.append(book(ST, name)); STATES[name] = ST
    if P63 is not None:
        ST = states(P63 >= tq(P63, qout), P63 < tq(P63, 0.6)); name = f'E: OUT P_off63>q{qout:.2f}, IN P_off63<q.6'; rows.append(book(ST, name)); STATES[name] = ST
        ST = states((PF >= tq(PF, qout)) & (P63 >= tq(P63, 0.8)), (PF < tq(PF, 0.6)) & (P63 < tq(P63, 0.6))); name = f'F: OUT P_off>q{qout:.2f} & P_off63>q.8, IN both below q.6'; rows.append(book(ST, name)); STATES[name] = ST
TAB = pd.DataFrame(rows); pd.to_pickle({'table': TAB, 'states': pd.DataFrame(STATES), 'deciles': DEC, 'joint': JOINT}, 'DECISION.pkl')
print('\n=== two-state stay-in rules on trailing-quantile thresholds, fully out of sample, 2006 to date, Feb-Jul 2020 excluded ===')
print(TAB.drop(columns=['loss captured (12 episodes)']).to_string(index=False))
print('\nloss captured per episode (share of SPY peak-to-trough loss the book took; 2008-05, 2010-04, 2011-04, 2015-07, 2015-11, 2018-01, 2018-09, 2022-01, 2023-07, 2024-07, 2025-02, 2026-01):')
for _, rw in TAB.iterrows(): print(f"  {rw['rule'][:62]:62s} {rw['loss captured (12 episodes)']}")
# ---- the C family in detail: yearly returns vs SPY and the OUT spells
for name in [n for n in STATES if n in ('C: OUT P_off>q0.95, IN P_off<q0.8', 'C: OUT P_off>q0.90, IN P_off<q0.8', 'F: OUT P_off>q0.95 & P_off63>q.8, IN both below q.6')]:
    ST = STATES[name]; w = ST.shift(1); x = (w * r)[m]
    yr = pd.DataFrame({'book %': (x.groupby(x.index.year).sum() * 100).round(1), 'SPY %': (r[m].groupby(r[m].index.year).sum() * 100).round(1)}); yr['diff'] = (yr['book %'] - yr['SPY %']).round(1)
    print(f'\n=== {name}: yearly ==='); print(yr.T.to_string())
    o = (ST == 0) & m; sp = []; start = None; prev = None
    for d in idx[o]:
        if start is None or (idx.get_loc(d) - idx.get_loc(prev)) > 1:
            if start is not None: sp.append((start, prev))
            start = d
        prev = d
    if start is not None: sp.append((start, prev))
    print('OUT spells (start, end, sessions, SPY % while out):'); print('  ' + '; '.join(f"{a.date()} to {b.date()} ({idx.get_loc(b)-idx.get_loc(a)+1}d, {r.loc[a:b].sum()*100:+.1f}%)" for a, b in sp))
    pd.to_pickle({'yearly': yr, 'spells': sp}, 'DECISION_' + ('F95' if name.startswith('F') else ('C95' if '0.95' in name else 'C90')) + '.pkl')
print(f'\n{time.time()-t0:.0f}s')
