#!/usr/bin/env python3
"""The fast leg: rates of change over 1 to 5 sessions only, every series together. Run from risk_regime/work.
Inputs: 1-, 2-, 3- and 5-day rates of change (and their trailing ranks) of every ETF and against SPY, of volume (5-day),
of every vol index and their ratios, short realized-vol rates of change and acceleration. Nothing longer than 5 days.
Targets (horizon 10 sessions): FAST LH/LL = the highest close of the next 10 sessions below the highest of the last 10
and the lowest of the next 10 below the lowest of the last 10; FAST DD = SPY falls 3% from today's close within 10.
Same pipeline (in-window same-sign screen, K15 depth-2 monotone trees, yearly refits from 2005, purge). Decision: REDUCE
at the trailing-756 95th (or 90th) percentile, HOLD below the 80th. Scored alone and OR'd with the current instruction.
Writes FAST.pkl."""
import sys, re, time, numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore'); sys.path.insert(0, '../v2'); import wf
from sklearn.metrics import roc_auc_score
t0 = time.time(); cols = wf.COLS
short = re.compile(r':(roc|rel)(1|2|3|5)(_rank)?$|:vroc5(_rank)?$|:accel5_rank$|:cc(5|10)_roc5_rank$|:cc21_roc5(_rank)?$|prem_chg5_rank$|:disp(1|5)_roc5_rank$|:volvol21_roc5_rank$|:amihud_roc5_rank$')
keep = np.array([bool(short.search(c)) and not c.startswith('KRE:') and not c.startswith('MF:') for c in cols])
print(f'fast candidates: {keep.sum()}; sample: {list(cols[keep][::max(1, keep.sum() // 20)])[:20]}')
wf.NANSHARE = np.where(keep, wf.NANSHARE, 1.0)
idx = wf.idx; spy = wf.O['T']['SPY']['Close'].reindex(idx); r = np.log(spy).diff(); Y = wf.YEARS; EX = wf.EX; okb = wf.L['ok21'].to_numpy(bool) & ~EX
ph = spy.rolling(10).max(); pl = spy.rolling(10).min(); fh = spy[::-1].rolling(10, min_periods=10).max()[::-1].shift(-1); fl = spy[::-1].rolling(10, min_periods=10).min()[::-1].shift(-1)
lh10 = ((fh < ph) & (fl < pl)).astype(float).where(fh.notna() & ph.notna())
dd3 = (fl / spy - 1 <= -0.03).astype(float).where(fl.notna())
T = {'FAST LH/LL (10 sessions)': lh10, 'FAST DD (3% within 10)': dd3}
PRED = {}; LAST = {}
for name, y in T.items():
    yv = y.to_numpy(float); ok = okb & ~np.isnan(yv)
    pred, _, last = wf.fit_predict_walk(yv, ok, k=15, keep_last=True, purge=30); PRED[name] = pd.Series(pred, index=idx); LAST[name] = (last[0], [int(s) for s in last[1]])
    a = wf.auc_by_era(pred, yv, ok); m = ok & ~np.isnan(pred) & (Y >= 2023)
    print(f'{name}: base {np.nanmean(yv[ok & (Y >= 2005)]):.2f} | AUC {a} 2023-26 {roc_auc_score(yv[m], pred[m]):.3f} | quintiles {wf.quintiles(pred, yv, ok)} ({time.time()-t0:.0f}s)')
    print('   last fit:', ', '.join(f'{f} ({"+" if s > 0 else "-"})' for f, s in zip(last[0], last[1])))
def tq(s, q, w=756): return s.rolling(w, min_periods=252).quantile(q).shift(1)
def states(o_, i_):
    o = o_.fillna(False).values; i = i_.fillna(False).values; st = np.ones(len(o), dtype=int); s = 1
    for k in range(len(o)):
        if s == 1 and o[k]: s = 0
        elif s == 0 and i[k]: s = 1
        st[k] = s
    return pd.Series(st, index=idx)
INS = pd.read_pickle('LHLL.pkl')['instruction']
BK = {'SPY buy and hold': pd.Series(1, index=idx), 'current instruction': INS}
for name, P in PRED.items():
    sh = name.split(' (')[0]
    for qo in (0.95, 0.90):
        S = states(P >= tq(P, qo), P < tq(P, 0.8)); BK[f'{sh} at q{qo}'] = S; BK[f'current OR {sh} at q{qo}'] = (S.astype(bool) & INS.astype(bool)).astype(int)
def stats(S, a):
    m = idx >= pd.Timestamp(a); w = S.shift(1).fillna(1); x = (w * r)[m]; c = x.cumsum(); dd = c - c.cummax(); o = pd.Series(((S == 0) & m).to_numpy(bool), index=idx)
    return {'ann %': round(float(x.mean() * 252 * 100), 1), 'worst DD %': round(float((np.exp(dd.min()) - 1) * 100), 1), 'exits/yr': round(float((o & ~o.shift(1, fill_value=False)).sum()) / (m.sum() / 252), 1), 'exposure': round(float(w[m].mean()), 2)}
rows = [{'from': a[:4], 'book': k, **stats(S, a)} for a in ('2010-01-01', '2016-01-01', '2023-01-01') for k, S in BK.items()]
B = pd.DataFrame(rows); pd.set_option('display.width', 250)
for a in B['from'].unique(): print(f'\n=== SPY books from {a}, 2020 included ==='); print(B[B['from'] == a].drop(columns='from').to_string(index=False))
pd.to_pickle({'pred': PRED, 'last': LAST, 'books': B}, 'FAST.pkl'); print(f'done {time.time()-t0:.0f}s')
