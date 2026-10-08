#!/usr/bin/env python3
"""v2 Stage 5: the decision layer, fully out of sample. Run from risk_regime/work.

Inputs are walk-forward probabilities refit here with the small monotone configuration (30 features
chosen inside each training window, depth-2 trees, monotone in the training-window sign):
  P_vol  = P(SPY 21d realized vol expands 1.5x within 21 sessions)   -- the vol-episode onset side
  P_on   = P(SPY rallies 5% within 21 sessions)                        -- the risk-on onset side
  P_off  = P(SPY draws down 5% within 21 sessions)                     -- reported, weak out of sample
Then: (1) what the forward tape looks like by decile of each probability, out of sample, by era;
(2) a grid of two-state stay-in rules whose thresholds are trailing quantiles of the probabilities
(past out-of-sample values only), each backtested fully invested when IN and in cash when OUT,
lagged one session; the whole grid is reported, not a chosen winner. Outputs PRED2.pkl, DECISION.pkl.
"""
import pandas as pd, numpy as np, time, warnings; warnings.filterwarnings('ignore')
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
t0 = time.time(); pd.set_option('display.width', 320); pd.set_option('display.max_rows', 400)
F = pd.read_pickle('F.pkl'); MF = pd.read_pickle('MF.pkl'); L = pd.read_pickle('LAB.pkl'); O = pd.read_pickle('OHLCV.pkl'); idx = F.index
X = pd.concat([F, MF], axis=1); spy = O['T']['SPY']['Close']; r = np.log(spy).diff()
EX = (idx >= pd.Timestamp('2020-02-01')) & (idx <= pd.Timestamp('2020-07-31'))
ERAS = {'2005-12': ('2005-01-01', '2012-12-31'), '2013-19': ('2013-01-01', '2019-12-31'), '2020-26': ('2020-01-01', '2026-12-31'), 'all': ('2005-01-01', '2026-12-31')}
PAR = dict(learning_rate=0.05, max_iter=200, max_bins=64, random_state=0, max_depth=2, min_samples_leaf=300, l2_regularization=2.0)
_CACHE = {}
def select(Xtr, ytr, k=30):
    key = (ytr.name, str(ytr.index[-1].date()), len(ytr))
    if key not in _CACHE:
        n = len(ytr); h = n // 2; yr = ytr.rank(pct=True)
        ic1 = Xtr.iloc[:h].rank(pct=True).corrwith(yr.iloc[:h]); ic2 = Xtr.iloc[h:].rank(pct=True).corrwith(yr.iloc[h:])
        ok = (np.sign(ic1) == np.sign(ic2)) & ic1.notna() & ic2.notna() & (Xtr.notna().mean() > 0.5)
        score = pd.concat([ic1.abs(), ic2.abs()], axis=1).min(axis=1).where(ok, 0); _CACHE[key] = (score.sort_values(ascending=False), np.sign(ic1 + ic2))
    score, sgn = _CACHE[key]; feats = list(score.head(k).index); return feats, sgn[feats].astype(int).values
def walk(target, okcol):
    y = L[target]; ok = L[okcol] & ~EX & y.notna(); pred = pd.Series(np.nan, index=idx); feats_last = None
    for year in range(2005, idx[-1].year + 1):
        ts = pd.Timestamp(f'{year}-01-01'); te = pd.Timestamp(f'{year}-12-31')
        tr = ok & (idx < ts - pd.Timedelta(days=100)) & (idx >= pd.Timestamp('1993-01-01')); test = (idx >= ts) & (idx <= te)
        if tr.sum() < 1000 or test.sum() == 0: continue
        feats, signs = select(X[tr], y[tr]); clf = HistGradientBoostingClassifier(monotonic_cst=signs, **PAR).fit(X.loc[tr, feats], y[tr])
        pred[test] = clf.predict_proba(X.loc[test, feats])[:, 1]; feats_last = (feats, signs, clf)
    return pred, ok, feats_last
PRED = {}; OK = {}; LASTFIT = {}
for tname, (lab, okc) in {'P_vol': ('volexp21', 'ok21'), 'P_on': ('riskon21', 'ok21'), 'P_off': ('riskoff21', 'ok21')}.items():
    PRED[tname], OK[tname], LASTFIT[tname] = walk(lab, okc); print(f'{tname} walk-forward done {time.time()-t0:.0f}s')
PRED = pd.DataFrame(PRED); PRED.to_pickle('PRED2.pkl')
pd.to_pickle({k: (v[0], v[1]) for k, v in LASTFIT.items()}, 'LASTFIT2.pkl')
ok = L['ok21'] & ~EX
# ---------- what the tape does after each probability decile, out of sample
print('\n=== forward tape by out-of-sample probability decile (all 2005-2026 ex-2020 window) ===')
for p in ('P_vol', 'P_on', 'P_off'):
    m = ok & PRED[p].notna(); dec = pd.qcut(PRED[p][m].rank(method='first'), 10, labels=False)
    g = pd.DataFrame({'pred': PRED[p][m], 'fwd21%': L.loc[m, 'fwd21'] * 100, 'fwdDD21%': L.loc[m, 'fwdDD21'] * 100, 'fwdUP21%': L.loc[m, 'fwdUP21'] * 100, 'P(5%DD)': L.loc[m, 'riskoff21'], 'P(5%UP)': L.loc[m, 'riskon21'], 'P(volexp)': L.loc[m, 'volexp21'], 'RV21 ahead / now': np.exp(np.log(F['SPY:cc21'].astype(float).shift(-21) / F['SPY:cc21'].astype(float))[m])}).groupby(dec).mean()
    print(f'\n{p}:'); print(g.round(3).to_string())
    print('   AUC by era:', {e: round(float(roc_auc_score(L.loc[mm, {'P_vol': 'volexp21', 'P_on': 'riskon21', 'P_off': 'riskoff21'}[p]], PRED[p][mm])), 3) for e, (a, b) in ERAS.items() for mm in [m & (idx >= pd.Timestamp(a)) & (idx <= pd.Timestamp(b))] if mm.sum() > 100})
# joint: vol expansion likely AND no rebound setup
m = ok & PRED['P_vol'].notna() & PRED['P_on'].notna()
qv = PRED['P_vol'][m].rank(pct=True); qo = PRED['P_on'][m].rank(pct=True)
cell = pd.DataFrame({'vol': pd.cut(qv, [0, .5, .8, .9, 1.0], labels=['<50', '50-80', '80-90', '>90']), 'on': pd.cut(qo, [0, .5, .8, 1.0], labels=['<50', '50-80', '>80']), 'fwd21': L.loc[m, 'fwd21'] * 100, 'off': L.loc[m, 'riskoff21'], 'dd': L.loc[m, 'fwdDD21'] * 100})
print('\n=== joint cells: P_vol percentile (rows) x P_on percentile (cols): mean fwd21 % | P(5% DD/21d) | mean fwdDD21 % | n ===')
print(pd.concat({'fwd21': cell.pivot_table(index='vol', columns='on', values='fwd21', aggfunc='mean').round(2), 'P_off': cell.pivot_table(index='vol', columns='on', values='off', aggfunc='mean').round(2), 'DD21': cell.pivot_table(index='vol', columns='on', values='dd', aggfunc='mean').round(1), 'n': cell.pivot_table(index='vol', columns='on', values='off', aggfunc='count')}, axis=1).to_string())
# ---------- two-state rules on trailing quantiles (past OOS values only), stay-in backtest
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
    row['loss captured (12 episodes)'] = caps; row['today'] = 'IN' if ST.iloc[-1] == 1 else 'OUT'
    return row
rows = [{'rule': 'SPY buy and hold', **stats(bh), 'exposure': 1.0, 'spells': 0, **{f'ann {e}': round(float(r[m & (idx >= pd.Timestamp(a)) & (idx <= pd.Timestamp(b))].mean() * 252 * 100), 1) for e, (a, b) in ERAS.items()}, **{f'maxDD {e}': stats(r[m & (idx >= pd.Timestamp(a)) & (idx <= pd.Timestamp(b))])['maxDD'] for e, (a, b) in ERAS.items()}, 'loss captured (12 episodes)': [1.0] * 12, 'today': 'IN'}]
PV, PO = PRED['P_vol'], PRED['P_on']; STATES = {}
for qout in (0.85, 0.90, 0.95):
    for qin in (0.5, 0.7):
        ST = states(PV >= tq(PV, qout), PV < tq(PV, qin)); name = f'A: OUT P_vol>q{qout:.2f}, IN P_vol<q{qin:.1f}'; rows.append(book(ST, name)); STATES[name] = ST
        ST = states((PV >= tq(PV, qout)) & (PO < tq(PO, 0.5)), (PV < tq(PV, qin)) | (PO >= tq(PO, 0.8))); name = f'B: OUT P_vol>q{qout:.2f} & P_on<q.5, IN P_vol<q{qin:.1f} or P_on>q.8'; rows.append(book(ST, name)); STATES[name] = ST
for qout in (0.90, 0.95):
    ST = states(PRED['P_off'] >= tq(PRED['P_off'], qout), PRED['P_off'] < tq(PRED['P_off'], 0.6)); name = f'C: OUT P_off>q{qout:.2f}, IN P_off<q.6 (the weak target, for reference)'; rows.append(book(ST, name)); STATES[name] = ST
TAB = pd.DataFrame(rows); pd.to_pickle({'table': TAB, 'states': pd.DataFrame(STATES)}, 'DECISION.pkl')
print('\n=== two-state stay-in rules on trailing-quantile thresholds, fully out of sample, 2006 to date, Feb-Jul 2020 excluded ===')
print(TAB.drop(columns=['loss captured (12 episodes)']).to_string(index=False))
print('\nloss captured per episode (share of SPY peak-to-trough loss the book took; 2008-05, 2010-04, 2011-04, 2015-07, 2015-11, 2018-01, 2018-09, 2022-01, 2023-07, 2024-07, 2025-02, 2026-01):')
for _, rw in TAB.iterrows(): print(f"  {rw['rule'][:62]:62s} {rw['loss captured (12 episodes)']}")
print(f'\n{time.time()-t0:.0f}s')
