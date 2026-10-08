#!/usr/bin/env python3
"""A tech-break leg: does an exit on a hard break in XLK help an XLK book and the SPY book? Run from risk_regime/work.
Trigger (the two post-break readings with a record in xlk_jun26.py): XLK's 5-day rate of change in the bottom 10% of
its trailing two years AND XLK's 5-day range vol in the top 10%, on the same session. Four re-entries, fixed before
scoring: XLK 5-day ROC back above its median; 21 sessions; XLK back above its 21-day closing high at the trigger;
the earlier of the first two. Scored on XLK and on SPY (alone, and OR'd with the current instruction), 2006 to date,
2020 included, lagged one session. The trigger was picked after looking at June 2026, so its record there is not
evidence; the other years are. Writes TECHBREAK.pkl."""
import numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore'); pd.set_option('display.width', 320); pd.set_option('display.max_rows', 200)
F = pd.read_pickle('F.pkl'); O = pd.read_pickle('OHLCV.pkl'); idx = F.index; T = O['T']
xlk = T['XLK']['Close'].reindex(idx); spy = T['SPY']['Close'].reindex(idx); rx = np.log(xlk).diff(); rs = np.log(spy).diff()
r5 = F['XLK:roc5_rank'].astype(float); pk5 = F['XLK:pk5_rank'].astype(float)
TRIG = ((r5 <= 0.1) & (pk5 >= 0.9)).fillna(False).to_numpy()
hi21 = xlk.rolling(21, min_periods=21).max().to_numpy()
def machine(kind):
    st = np.ones(len(idx), int); s = 1; n = 0; lvl = np.nan
    r5v = r5.to_numpy(); xv = xlk.to_numpy()
    for k in range(len(idx)):
        if s == 1 and TRIG[k]: s = 0; n = 0; lvl = hi21[k]
        elif s == 0:
            n += 1
            if kind == 'roc5 above median' and r5v[k] >= 0.5: s = 1
            elif kind == '21 sessions' and n >= 21: s = 1
            elif kind == 'back above 21d high' and xv[k] >= lvl: s = 1
            elif kind == 'earlier of roc5 median or 21 sessions' and (r5v[k] >= 0.5 or n >= 21): s = 1
        st[k] = s
    return pd.Series(st, index=idx)
KINDS = ['roc5 above median', '21 sessions', 'back above 21d high', 'earlier of roc5 median or 21 sessions']
TB = {k: machine(k) for k in KINDS}
INS = pd.read_pickle('VVSIG.pkl')['instruction']
def stats(S, r, a, b):
    m = (idx >= pd.Timestamp(a)) & (idx <= pd.Timestamp(b)); w = S.shift(1).fillna(1); x = (w * r)[m]; c = x.cumsum(); dd = c - c.cummax()
    o = pd.Series(((S == 0) & m).to_numpy(bool), index=idx)
    return {'ann %': round(float(x.mean() * 252 * 100), 1), 'worst DD %': round(float((np.exp(dd.min()) - 1) * 100), 1), 'times OUT': int((o & ~o.shift(1, fill_value=False)).sum()), 'exposure': round(float(w[m].mean()), 2)}
WIN = [('2006-01-01', '2026-12-31'), ('2006-01-01', '2015-12-31'), ('2016-01-01', '2026-12-31'), ('2023-01-01', '2026-12-31'), ('2010-01-01', '2026-12-31')]
rows = []
for a, b in WIN:
    lab = f'{a[:4]}-{b[:4]}'
    rows.append({'book': 'XLK buy and hold', 'window': lab, **stats(pd.Series(1, index=idx), rx, a, b)})
    for k in KINDS: rows.append({'book': f'XLK, tech-break exit, re-entry {k}', 'window': lab, **stats(TB[k], rx, a, b)})
    rows.append({'book': 'SPY buy and hold', 'window': lab, **stats(pd.Series(1, index=idx), rs, a, b)})
    if a >= '2010': rows.append({'book': 'SPY, current instruction (reference OR VIX outrunning VVIX)', 'window': lab, **stats(INS, rs, a, b)})
    for k in KINDS:
        rows.append({'book': f'SPY, tech-break exit alone, re-entry {k}', 'window': lab, **stats(TB[k], rs, a, b)})
        if a >= '2010': rows.append({'book': f'SPY, current instruction OR tech-break, re-entry {k}', 'window': lab, **stats((INS.astype(bool) & TB[k].astype(bool)).astype(int), rs, a, b)})
R = pd.DataFrame(rows)
for lab in ['2006-2026', '2006-2015', '2016-2026', '2023-2026', '2010-2026']:
    print(f'\n=== {lab} (2020 included) ==='); print(R[R.window == lab].drop(columns='window').to_string(index=False))
on = pd.Series(TRIG, index=idx); fr = on & ~on.shift(1, fill_value=False)
print('\ntrigger sessions since 2023:', [d.date().isoformat() for d in idx[fr.to_numpy() & (idx >= pd.Timestamp('2023-01-01'))]])
k = 'roc5 above median'; S = TB[k]; seg = S.loc['2026-05-25':'2026-08-10']; ch = seg[seg != seg.shift(1)]
print(f'June-July 2026 with re-entry "{k}":', [(d.date().isoformat(), int(v), round(float(xlk[d]), 2)) for d, v in ch.items()])
for k in KINDS:
    S = TB[k]; seg = S.loc['2026-06-01':'2026-08-10']; ch = seg[seg != seg.shift(1)]; print(f'  {k}: ' + ', '.join(f"{d.date()} {'IN' if v else 'OUT'} @ {xlk[d]:.2f}" for d, v in ch.items()))
pd.to_pickle({'table': R, 'states': TB}, 'TECHBREAK.pkl')
