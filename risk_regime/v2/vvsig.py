#!/usr/bin/env python3
"""v2 Stage 13: the VIX-outrunning-VVIX signal. Run from risk_regime/work after decision.py.

Signal: VIX's 21-session log rate of change is in the top 10% of its trailing 504 sessions AND the 21-session log rate
of change of VIX/VVIX is also in its top 10% (VIX climbing faster than the vol of vol: the level of risk is being
repriced, not a convex spike in tail insurance). Live from the first such session until VIX's 21-session ROC falls
back below its trailing median. Since VVIX starts in 2008 the signal exists from 2009.

Instruction (action page): REDUCE when the reference rule is OUT or this signal is live; HOLD otherwise.
This signal was found in this project's research on 2008-2026 data and adopted after seeing its record; the split
and the books below are in-sample for that choice. Writes VVSIG.pkl.
"""
import numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore'); pd.set_option('display.width', 300)
O = pd.read_pickle('OHLCV.pkl'); F = pd.read_pickle('F.pkl'); idx = F.index
vix = O['I']['VIX']['Close'].reindex(idx); vvix = O['I']['VVIX']['Close'].reindex(idx); spy = O['T']['SPY']['Close'].reindex(idx); r = np.log(spy).diff()
rk = lambda s: s.rolling(504, min_periods=252).rank(pct=True); ratio = np.log(vix / vvix)
V21 = rk(np.log(vix / vix.shift(21))); R21 = rk(ratio - ratio.shift(21)); TRIG = ((V21 >= 0.9) & (R21 >= 0.9)).fillna(False)
def run(out, inn):
    o = out.to_numpy(bool); i = inn.fillna(False).to_numpy(bool); st = np.ones(len(o), int); s = 1
    for k in range(len(o)):
        if s == 1 and o[k]: s = 0
        elif s == 0 and i[k]: s = 1
        st[k] = s
    return pd.Series(st, index=idx)
SIG = run(TRIG, V21 < 0.5)                                             # 1 = quiet, 0 = live
REF = pd.read_pickle('DECISION.pkl')['states']['C: OUT P_off>q0.95, IN P_off<q0.8']
COMB = (REF.astype(bool) & SIG.astype(bool)).astype(int)                # the action page's instruction: 0 = REDUCE
EXM = (idx >= pd.Timestamp('2020-02-01')) & (idx <= pd.Timestamp('2020-07-31'))
f21 = spy[::-1].rolling(21, min_periods=21).min()[::-1].shift(-1); x21 = spy[::-1].rolling(21, min_periods=21).max()[::-1].shift(-1)
dn5 = (f21 / spy - 1 <= -0.05); up5 = (x21 / spy - 1 >= 0.05)
def fires(on):
    on = on.fillna(False).to_numpy(); out = np.zeros(len(on), bool); q = 10
    for i in range(len(on)):
        if on[i] and q >= 10: out[i] = True
        q = 0 if on[i] else q + 1
    return pd.Series(out, index=idx)
FV = fires(V21 >= 0.9); split = []
for a0 in ('2006-01-01', '2023-01-01'):
    base = (idx >= pd.Timestamp(a0)) & ~EXM & dn5.notna() & up5.notna()
    split.append({'from': a0[:4], 'case': 'any day (base rate)', 'n': None, '5% drop within 21': round(float(dn5[base].mean()), 2), '5% rally within 21': round(float(up5[base].mean()), 2)})
    for lab, c in (('VIX 21d ROC fires, VIX outrunning VVIX', R21 >= 0.9), ('VIX 21d ROC fires, VVIX keeping up', R21 < 0.9)):
        m = FV & c.fillna(False) & base; split.append({'from': a0[:4], 'case': lab, 'n': int(m.sum()), '5% drop within 21': round(float(dn5[m].mean()), 2), '5% rally within 21': round(float(up5[m].mean()), 2)})
SPLIT = pd.DataFrame(split)
def stats(S, a, b):
    m = (idx >= pd.Timestamp(a)) & (idx <= pd.Timestamp(b)); w = S.shift(1).fillna(1); x = (w * r)[m]; c = x.cumsum(); dd = c - c.cummax()
    o = pd.Series(((S == 0) & m).to_numpy(bool), index=idx)
    return {'ann %': round(float(x.mean() * 252 * 100), 1), 'worst DD %': round(float((np.exp(dd.min()) - 1) * 100), 1), 'times OUT': int((o & ~o.shift(1, fill_value=False)).sum()), 'exposure': round(float(w[m].mean()), 2)}
books = []
for a, b in (('2010-01-01', '2026-12-31'), ('2010-01-01', '2017-12-31'), ('2018-01-01', '2026-12-31'), ('2023-01-01', '2026-12-31')):
    for name, S in (('SPY buy and hold', pd.Series(1, index=idx)), ('reference rule alone', REF), ('VIX outrunning VVIX alone', SIG), ('instruction: reference OR VIX outrunning VVIX', COMB)):
        books.append({'window': f'{a[:4]}-{b[:4]}', 'book': name, **stats(S, a, b)})
BOOKS = pd.DataFrame(books)
sp = []; o = (SIG == 0); cur = None
for d in idx[o.to_numpy()]:
    if cur is None or idx.get_loc(d) - idx.get_loc(cur[1]) > 1:
        if cur: sp.append(cur)
        cur = [d, d]
    else: cur[1] = d
if cur: sp.append(cur)
SPELLS = [{'from': str(a.date()), 'to': str(b.date()), 'sessions': int(idx.get_loc(b) - idx.get_loc(a) + 1), 'spy_while_live': round(float(np.log(spy.loc[b] / spy.loc[a]) * 100), 1)} for a, b in sp]
live = bool(SIG.iloc[-1] == 0); since = SPELLS[-1]['from'] if live and SPELLS else None
TODAY = {'live': live, 'since': since, 'vix_roc21_rank': round(float(V21.iloc[-1]), 3), 'ratio_roc21_rank': round(float(R21.iloc[-1]), 3), 'vix': round(float(vix.iloc[-1]), 2), 'vvix': round(float(vvix.iloc[-1]), 2),
         'vix_roc21_pct': round(float(np.log(vix.iloc[-1] / vix.iloc[-22]) * 100), 1), 'ratio_roc21_pct': round(float((ratio.iloc[-1] - ratio.iloc[-22]) * 100), 1), 'last_spell': SPELLS[-1] if SPELLS else None,
         'reference_out': bool(REF.iloc[-1] == 0), 'instruction': 'REDUCE' if COMB.iloc[-1] == 0 else 'HOLD'}
print('=== split of VIX 21d ROC fires by the VIX/VVIX 21d ROC ==='); print(SPLIT.to_string(index=False))
print('\n=== books, 2020 included ==='); print(BOOKS.to_string(index=False))
print(f'\n{len(SPELLS)} live spells since 2009; last five:', SPELLS[-5:])
print('\ntoday:', TODAY)
pd.to_pickle({'split': SPLIT, 'books': BOOKS, 'spells': SPELLS, 'today': TODAY, 'state': SIG, 'instruction': COMB, 'V21': V21, 'R21': R21}, 'VVSIG.pkl')
