#!/usr/bin/env python3
"""Production, rate-of-change only. Run from risk_regime/work after vvsig.py (daily chain).

Inputs: rates of change of price (1 to 63 days, alone or against SPY), of volume, and of volatility (realized-vol ROC,
ROC of every implied-vol index and of their ratios, the VIX premium's change, dispersion ROC). No levels, no level
ranks, no correlation, no multifractal (the same filter as roconly.py).
Models (same pipeline as model2: in-window same-sign screen, K15 depth-2 monotone trees, yearly refits, 100-day purge):
  P_lhll  lower highs and lower lows: the highest close of the next 21 sessions below the highest of the last 21, and
          the lowest of the next 21 below the lowest of the last 21;
  P_on    a 5% rally within 21 sessions;   P_vol  realized vol expanding 1.5x within 21 sessions.
Instruction: REDUCE when P_lhll reaches its trailing-756 95th percentile (until it falls below the 80th), or when VIX is
outrunning VVIX (vvsig.py); HOLD otherwise. The thresholds are the reference rule's, not tuned. The old levels-based
drawdown leg is kept in the record (DECISION.pkl) but no longer drives the word. Writes LHLL.pkl.
"""
import sys, numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore'); sys.path.insert(0, '../v2'); import wf
cols = wf.COLS; KEEP_TOK = ('roc', ':rel', 'accel', 'vroc', 'prem_chg'); DROP_TOK = ('MF:', 'avgcorr', 'amihud21_rank', 'lvl_rank', 'breadth')
keep = np.array([any(t in c for t in KEEP_TOK) and not any(t in c for t in DROP_TOK) and 'KRE:' not in c for c in cols])
wf.NANSHARE = np.where(keep, wf.NANSHARE, 1.0)
idx = wf.idx; spy = wf.O['T']['SPY']['Close'].reindex(idx); r = np.log(spy).diff(); Y = wf.YEARS; EX = wf.EX
past_hi = spy.rolling(21).max(); past_lo = spy.rolling(21).min()
fut_hi = spy[::-1].rolling(21, min_periods=21).max()[::-1].shift(-1); fut_lo = spy[::-1].rolling(21, min_periods=21).min()[::-1].shift(-1)
lhll = ((fut_hi < past_hi) & (fut_lo < past_lo)).astype(float).where(fut_hi.notna() & past_hi.notna())
ok_l = lhll.notna().to_numpy() & ~EX & wf.L['ok21'].to_numpy(bool)
T = {'P_lhll': (lhll.to_numpy(float), ok_l, 'lower highs and lower lows'), 'P_on': (*wf.label('P_on'), '5% rally'), 'P_vol': (*wf.label('P_vol'), 'vol x1.5')}
PRED = {}; LAST = {}; DEC = {}
f21 = spy[::-1].rolling(21, min_periods=21).min()[::-1].shift(-1); dd21 = (f21 / spy - 1 <= -0.05).astype(float); fwd = np.log(spy.shift(-21) / spy) * 100
for p, (y, ok, lab) in T.items():
    pred, _, last = wf.fit_predict_walk(y, ok, k=15, keep_last=True); s = pd.Series(pred, index=idx); PRED[p] = s; LAST[p] = (last[0], [int(v) for v in last[1]])
    m = ok & ~np.isnan(pred) & (Y >= 2005); dq = pd.qcut(s[m].rank(method='first'), 10, labels=False)
    g = pd.DataFrame({'p_lo': s[m].groupby(dq).min(), 'p_hi': s[m].groupby(dq).max(), 'hit': pd.Series(y, index=idx)[m].groupby(dq).mean(), 'dd5': dd21[m].groupby(dq).mean(), 'fwd21': fwd[m].groupby(dq).mean()})
    DEC[p] = g.round(4)
def tq(s, q, w=756): return s.rolling(w, min_periods=252).quantile(q).shift(1)
def states(o_, i_):
    o = o_.fillna(False).values; i = i_.fillna(False).values; st = np.ones(len(o), dtype=int); s = 1
    for k in range(len(o)):
        if s == 1 and o[k]: s = 0
        elif s == 0 and i[k]: s = 1
        st[k] = s
    return pd.Series(st, index=idx)
P = PRED['P_lhll']; LEG = states(P >= tq(P, 0.95), P < tq(P, 0.8))
VS = pd.read_pickle('VVSIG.pkl')['state']; INS = (LEG.astype(bool) & VS.astype(bool)).astype(int)
def stats(S, a, b):
    m = (idx >= pd.Timestamp(a)) & (idx <= pd.Timestamp(b)); w = S.shift(1).fillna(1); x = (w * r)[m]; c = x.cumsum(); dd = c - c.cummax()
    o = pd.Series(((S == 0) & m).to_numpy(bool), index=idx); yrs = m.sum() / 252
    return {'ann %': round(float(x.mean() * 252 * 100), 1), 'worst DD %': round(float((np.exp(dd.min()) - 1) * 100), 1), 'exits/yr': round(float((o & ~o.shift(1, fill_value=False)).sum()) / yrs, 1), 'exposure': round(float(w[m].mean()), 2)}
OLD = pd.read_pickle('VVSIG.pkl')['instruction']
books = []
for a, b in (('2010-01-01', '2026-12-31'), ('2016-01-01', '2026-12-31'), ('2023-01-01', '2026-12-31')):
    for nm, S in (('SPY buy and hold', pd.Series(1, index=idx)), ('previous instruction (levels leg OR VIX outrunning VVIX)', OLD), ('lower-highs/lower-lows leg alone', LEG), ('instruction: lower-highs/lower-lows OR VIX outrunning VVIX', INS)):
        books.append({'window': f'{a[:4]}-{b[:4]}', 'book': nm, **stats(S, a, b)})
BOOKS = pd.DataFrame(books)
sp = []; o = (LEG == 0); cur = None
for d in idx[o.to_numpy()]:
    if cur is None or idx.get_loc(d) - idx.get_loc(cur[1]) > 1:
        if cur: sp.append(cur)
        cur = [d, d]
    else: cur[1] = d
if cur: sp.append(cur)
SPELLS = [{'from': str(a.date()), 'to': str(b.date()), 'spy_while_out': round(float(np.log(spy.loc[b] / spy.loc[a]) * 100), 1)} for a, b in sp]
pct = lambda p: round(float((PRED[p].iloc[-757:-1] <= PRED[p].iloc[-1]).mean()), 3)
def dec_of(p):
    his = DEC[p]['p_hi'].to_numpy(); return int(min(9, np.searchsorted(his, PRED[p].iloc[-1])))
q95 = float(tq(P, 0.95).iloc[-1]); q80 = float(tq(P, 0.8).iloc[-1])
TODAY = {'P_lhll': round(float(P.iloc[-1]), 4), 'P_lhll_pct': pct('P_lhll'), 'q95': round(q95, 4), 'q80': round(q80, 4), 'leg_live': bool(LEG.iloc[-1] == 0),
         'leg_since': (SPELLS[-1]['from'] if LEG.iloc[-1] == 0 and SPELLS else None), 'instruction': 'REDUCE' if INS.iloc[-1] == 0 else 'HOLD',
         'P_on': round(float(PRED['P_on'].iloc[-1]), 4), 'P_on_pct': pct('P_on'), 'P_vol': round(float(PRED['P_vol'].iloc[-1]), 4), 'P_vol_pct': pct('P_vol'),
         'dec': {p: dec_of(p) for p in PRED}, 'state_since': None}
chg = INS != INS.shift(1); TODAY['state_since'] = str(idx[chg.to_numpy()][-1].date())
print(BOOKS.to_string(index=False)); print('\nlast-window features:'); [print(f'  {p}: ' + ', '.join(f'{f} ({"+" if s > 0 else "-"})' for f, s in zip(*LAST[p]))) for p in LAST]
print('\ntoday:', TODAY); print('last spells of the lower-highs/lower-lows leg:', SPELLS[-6:])
pd.to_pickle({'pred': PRED, 'last': LAST, 'deciles': DEC, 'leg': LEG, 'instruction': INS, 'books': BOOKS, 'spells': SPELLS, 'today': TODAY}, 'LHLL.pkl')
