#!/usr/bin/env python3
"""v2 Stage 8: the 15% layer. Run from risk_regime/work after model2.py and decision.py.

The loss that matters is a 15% decline, not 5%. This script
 (1) lists every SPY peak-to-trough decline of 15% or more since 1993 (zigzag: a 15% fall from the running
     high opens an event, a 10% rebound off the low or a new high closes it), and the 10% ones for reference;
 (2) defines forward labels: a 15% decline from today's close within 63 / 126 sessions (and 10% within 63);
     sessions whose forward window touches Feb-Jul 2020 are masked, as everywhere else in v2;
 (3) scores the existing out-of-sample probabilities (P_off, P_off63, P_vol, P_on) against those labels;
 (4) walk-forward fits dedicated models for the new labels with the same protocol as model2 (K15, depth-2
     monotone trees, yearly refits, purge longer than the label window), with capacity variants and raw
     single-feature baselines; descriptive screen of the raw features for the 15% label;
 (5) for every 15% event since 2005: the trailing percentile of each probability at the peak, the first
     session it crossed its trailing 95th percentile and how far SPY had already fallen by then;
 (6) re-scores the decision grid (states from DECISION.pkl) plus rules on the new probabilities on what
     matters here: the share of each 15% decline the book took, the OUT spells that were not followed by
     any 10% decline, and the return given up outside the 15% events; the whole table is printed;
 (7) a 20-draw permutation test of the dedicated 15%/63 pipeline (same construction as nulltest.py).
Writes PRED15.pkl, LASTFIT15.pkl, BIG.pkl.
"""
import sys, time, numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore'); sys.path.insert(0, '../v2'); import wf
from sklearn.metrics import roc_auc_score
pd.set_option('display.width', 340); pd.set_option('display.max_rows', 500); pd.set_option('display.max_columns', 60)
t0 = time.time(); idx = wf.idx; YEARS = wf.YEARS; O = wf.O
spy = O['T']['SPY']['Close'].reindex(idx); r = np.log(spy).diff()
PRED = pd.read_pickle('PRED2.pkl'); DEC = pd.read_pickle('DECISION.pkl')
EX0, EX1 = pd.Timestamp('2020-02-01'), pd.Timestamp('2020-07-31')
R_NULL = int(sys.argv[1]) if len(sys.argv) > 1 else 20
OUT = {}
# ---------- 1. events
def zigzag(px, fall, rebound=0.10):
    ev = []; hi = px.iloc[0]; hi_t = px.index[0]; lo = None; lo_t = None; down = False
    for t, p in px.items():
        if not down:
            if p > hi: hi, hi_t = p, t
            elif p / hi - 1 <= -fall: down = True; lo, lo_t = p, t
        else:
            if p < lo: lo, lo_t = p, t
            elif p / lo - 1 >= rebound or p > hi: ev.append((hi_t, lo_t, lo / hi - 1)); down = False; hi, hi_t = p, t
    if down: ev.append((hi_t, lo_t, lo / hi - 1))
    E = pd.DataFrame(ev, columns=['peak', 'trough', 'depth'])
    E['depth%'] = (E['depth'] * 100).round(1); E['sessions'] = [idx.get_loc(b) - idx.get_loc(a) for a, b in zip(E['peak'], E['trough'])]
    E['covid'] = [(a <= EX1) and (b >= EX0) for a, b in zip(E['peak'], E['trough'])]
    return E
EV15 = zigzag(spy.dropna(), 0.15); EV10 = zigzag(spy.dropna(), 0.10)
print('=== SPY peak-to-trough declines of 15% or more since 1993 (zigzag, closes; a 10% rebound or a new high closes the event) ===')
print(EV15[['peak', 'trough', 'depth%', 'sessions', 'covid']].to_string(index=False))
print(f'\n10% or more: {len(EV10)} events; since 2005 excluding covid: {int(((EV10.peak >= "2005-01-01") & ~EV10.covid).sum())}')
print(EV10[(EV10.peak >= '2005-01-01')][['peak', 'trough', 'depth%', 'sessions', 'covid']].to_string(index=False))
OUT['EV15'] = EV15; OUT['EV10'] = EV10
# ---------- 2. labels
def fwd_label(N, thr):
    fmin = spy[::-1].rolling(N, min_periods=N).min()[::-1].shift(-1)          # lowest close over the next N sessions
    y = (fmin / spy - 1 <= -thr).astype(float).to_numpy()
    touches = pd.Series(False, index=idx); pos0 = idx.get_loc(idx[idx >= EX0][0]); touches.iloc[max(0, pos0 - N):] = idx[max(0, pos0 - N):] <= EX1
    ok = fmin.notna().to_numpy() & spy.notna().to_numpy() & ~touches.to_numpy()
    return y, ok
LABS = {'dd15_63': fwd_label(63, 0.15), 'dd15_126': fwd_label(126, 0.15), 'dd10_63': fwd_label(63, 0.10)}
PURGE = {'dd15_63': 110, 'dd15_126': 200, 'dd10_63': 110}
print('\n=== labels: base rate 2005+ (forward windows touching Feb-Jul 2020 masked) ===')
for ln, (y, ok) in LABS.items():
    m = ok & (YEARS >= 2005); print(f'{ln}: base rate {y[m].mean():.3f}, positive sessions {int(y[m].sum())}, years with positives {sorted(set(YEARS[m][y[m] == 1]))}; pre-2005 positives {int(y[ok & (YEARS < 2005)].sum())}')
# ---------- 3. existing probabilities vs the new labels
print('\n=== existing out-of-sample probabilities scored against the 15% labels (AUC by era; quintile hit rates 2005-26) ===')
rows = []
for ln, (y, ok) in LABS.items():
    for p in ('P_off', 'P_off63', 'P_vol', 'P_on'):
        pr = PRED[p].to_numpy(); a = wf.auc_by_era(pr, y, ok); q = wf.quintiles(pr, y, ok); rows.append({'label': ln, 'prob': p, **{f'auc {k}': v for k, v in a.items()}, 'quintiles': q})
EXIST = pd.DataFrame(rows); print(EXIST.to_string(index=False)); OUT['existing_vs_labels'] = EXIST
# ---------- 4. dedicated models
print('\n=== dedicated walk-forward models (K15 depth-2 monotone, yearly refits from 2005, purge > label window) ===')
PRED15 = {}; LAST15 = {}; rows = []
def report(name, pred, y, ok, extra=''):
    lo, md, hi = wf.year_bootstrap_auc(pred, y, ok); a = wf.auc_by_era(pred, y, ok); q = wf.quintiles(pred, y, ok)
    print(f'{name}: AUC {a}  CI95 {lo:.3f}-{hi:.3f}  quintiles {q}{extra}  ({time.time()-t0:.0f}s)'); return {'model': name, **{f'auc {k}': v for k, v in a.items()}, 'ci95': f'{lo:.3f}-{hi:.3f}', 'quintiles': q}
for ln in ('dd15_63', 'dd15_126', 'dd10_63'):
    y, ok = LABS[ln]; pred, sel, last = wf.fit_predict_walk(y, ok, k=15, purge=PURGE[ln], keep_last=True)
    PRED15[ln] = pd.Series(pred, index=idx); LAST15[ln] = (last[0], last[1])
    rows.append({'label': ln, **report(f'{ln} K15 d2 mono', pred, y, ok)})
    top = sorted(sel.items(), key=lambda kv: -kv[1])[:12]; print('   most selected:', ', '.join(f'{wf.COLS[f]} ({n})' for f, n in top))
    print('   last-window:', ', '.join(f'{f} ({"+" if s > 0 else "-"})' for f, s in zip(last[0], last[1])))
y, ok = LABS['dd15_63']
for name, kw in (('dd15_63 K8 d1 mono', dict(k=8, depth=1, leaf=300)), ('dd15_63 K30 d2 mono', dict(k=30)), ('dd15_63 K5 d1 mono', dict(k=5, depth=1)), ('dd15_63 K150 d3 free (overfit reference)', dict(k=150, depth=3, mono=False, leaf=100))):
    pred, _, _ = wf.fit_predict_walk(y, ok, purge=110, **kw); rows.append({'label': 'dd15_63', **report(name, pred, y, ok)})
print('\n--- raw single features against dd15_63, no fitting (AUC of the feature itself, 2005-26 by era) ---')
RAW = ['VIX:lvl_rank504', 'VIX3M:lvl_rank252', 'SPY:cc21_rank', 'SPY:cc63_rank', 'HYG:pk21_rank', 'XLV:rel63', 'XS:avgcorr21_rank', 'XS:disp21_rank', 'SPY:roc63_rank', 'SPY:roc126_rank', 'MF:SPY:h2_rank', 'MF:SPY:width_rank', 'MF:VIX:vr21_rank', 'MF:HYG:h2_rank', 'SPY:VIX/cc21_rank', 'MOVE:lvl_rank504', 'TNX:roc63_rank', 'RSP/SPY:roc63_rank', 'HYG:roc63_rank', 'SPY:volvol21_rank']
for f in RAW:
    if f not in wf.X.columns: print('   (missing)', f); continue
    v = wf.X[f].to_numpy(dtype='float64'); okf = ok & ~np.isnan(v)
    for sgn in (1, -1):
        a = wf.auc_by_era(sgn * v, y, okf)
        if a.get('all', 0) >= 0.5: rows.append({'label': 'dd15_63', 'model': f'raw {"-" if sgn < 0 else ""}{f}', **{f'auc {k}': vv for k, vv in a.items()}, 'ci95': '', 'quintiles': wf.quintiles(sgn * v, y, okf)}); print(f'   {"-" if sgn < 0 else " "}{f:28s} {a}')
MODELS15 = pd.DataFrame(rows); OUT['models15'] = MODELS15
print('\n--- descriptive screen (NOT out of sample): raw features ranked by the smaller of their rank correlations with dd15_63 in 2005-15 and 2016-26, same sign required ---')
rows_ = np.where(ok & (YEARS >= 2005))[0]; top, signs, score = wf.screen(rows_, y, 25)
SCR = pd.DataFrame({'feature': wf.COLS[top], 'sign': signs, 'min |ic| across halves': score.round(3)}); print(SCR.to_string(index=False)); OUT['screen15'] = SCR
pd.to_pickle(PRED15, 'PRED15.pkl'); pd.to_pickle(LAST15, 'LASTFIT15.pkl')
# ---------- 5. event table: when did each probability cross its trailing 95th percentile
print('\n=== every 15% decline since 2005 (covid excluded): trailing-756 percentile of each probability at the peak, and the first session at or after the peak where it crossed its trailing 95th percentile, with the SPY decline already in place at that session ===')
def trailing_pct(s, w=756):
    v = s.to_numpy(dtype='float64'); out = np.full(len(v), np.nan)
    for i in range(252, len(v)):
        if np.isnan(v[i]): continue
        win = v[max(0, i - w):i]; win = win[~np.isnan(win)]
        if len(win) >= 252: out[i] = (win <= v[i]).mean()
    return pd.Series(out, index=s.index)
PROBS = {'P_off (5%/21)': PRED['P_off'], 'P_off63 (10%/63)': PRED['P_off63'], 'P15/63 (new)': PRED15['dd15_63'], 'P15/126 (new)': PRED15['dd15_126'], 'P_vol': PRED['P_vol']}
PCT = {k: trailing_pct(v) for k, v in PROBS.items()}; OUT['pct'] = PCT
ev_rows = []
for _, e in EV15[(EV15.peak >= '2005-06-01') & ~EV15.covid].iterrows():
    a, b = e.peak, e.trough; ia, ib = idx.get_loc(a), idx.get_loc(b); row = {'peak': a.date(), 'trough': b.date(), 'depth%': e['depth%'], 'sessions': e.sessions}
    for k, pc in PCT.items():
        seg = pc.iloc[ia:ib + 1]; hit = seg[seg >= 0.95]
        row[f'{k} @peak'] = round(float(pc.iloc[ia]), 2) if not np.isnan(pc.iloc[ia]) else None
        if len(hit): d = hit.index[0]; row[f'{k} first>95th'] = f'{d.date()} (+{idx.get_loc(d) - ia}s, SPY {spy.loc[d] / spy.loc[a] * 100 - 100:.1f}%)'
        else: row[f'{k} first>95th'] = 'never'
        pre = pc.iloc[max(0, ia - 21):ia]; row[f'{k} max pct 21s before peak'] = round(float(pre.max()), 2) if len(pre) and not pre.isna().all() else None
    ev_rows.append(row)
EVT = pd.DataFrame(ev_rows); OUT['event_table'] = EVT
for k in PCT: print(f'\n{k}:'); print(EVT[['peak', 'trough', 'depth%', 'sessions', f'{k} @peak', f'{k} max pct 21s before peak', f'{k} first>95th']].to_string(index=False))
# ---------- 6. decision rules scored on the 15% events
print('\n=== decision rules scored on what matters: share of each 15% decline taken (1.0 = took it all, 0 = sidestepped it), spells not followed by any 10% decline, return given up outside the 15% events; 2006-2026, Feb-Jul 2020 excluded ===')
EXM = ((idx >= EX0) & (idx <= EX1)); m = ~EXM & (idx >= pd.Timestamp('2006-01-01'))
def tq(s, q, w=756): return s.rolling(w, min_periods=252).quantile(q).shift(1)
def states(out_cond, in_cond):
    o = out_cond.fillna(False).values; i = in_cond.fillna(False).values; st = np.ones(len(o), dtype=int); s = 1
    for k in range(len(o)):
        if s == 1 and o[k]: s = 0
        elif s == 0 and i[k]: s = 1
        st[k] = s
    return pd.Series(st, index=idx)
ST = dict(DEC['states'])
P63, P126, PF, PF63 = PRED15['dd15_63'], PRED15['dd15_126'], PRED['P_off'], PRED['P_off63']
for qo, qi in ((0.95, 0.8), (0.90, 0.8), (0.90, 0.7), (0.95, 0.6), (0.98, 0.8), (0.85, 0.7)):
    ST[f'G: OUT P15/63>q{qo}, IN P15/63<q{qi}'] = states(P63 > tq(P63, qo), P63 < tq(P63, qi))
    ST[f'H: OUT P15/126>q{qo}, IN P15/126<q{qi}'] = states(P126 > tq(P126, qo), P126 < tq(P126, qi))
for qo, qi in ((0.95, 0.8), (0.90, 0.8)):
    ST[f'I: OUT P_off>q{qo} or P15/63>q{qo}, IN both<q{qi}'] = states((PF > tq(PF, qo)) | (P63 > tq(P63, qo)), (PF < tq(PF, qi)) & (P63 < tq(P63, qi)))
    ST[f'J: OUT P_off>q{qo} and P15/63>q0.8, IN P_off<q{qi}'] = states((PF > tq(PF, qo)) & (P63 > tq(P63, 0.8)), PF < tq(PF, qi))
    ST[f'K: OUT P15/63>q{qo} and P15/126>q0.8, IN both<q{qi}'] = states((P63 > tq(P63, qo)) & (P126 > tq(P126, 0.8)), (P63 < tq(P63, qi)) & (P126 < tq(P126, qi)))
EVS = EV15[(EV15.peak >= '2006-01-01') & ~EV15.covid]; EV10S = EV10[(EV10.peak >= '2005-06-01')]
in15 = pd.Series(False, index=idx)
for _, e in EVS.iterrows(): in15.loc[e.peak:e.trough] = True
def stats(x):
    c = x.cumsum(); dd = c - c.cummax(); return dict(ann=round(x.mean() * 252 * 100, 2), vol=round(x.std() * np.sqrt(252) * 100, 2), sharpe=round(x.mean() / x.std() * np.sqrt(252), 2), maxDD=round(dd.min() * 100, 1))
def score(S, name):
    w = S.shift(1); x = (w * r)[m]; row = {'rule': name, **stats(x), 'exposure': round(float(w[m].mean()), 2)}
    caps = []
    for _, e in EVS.iterrows():
        seg = r.loc[e.peak:e.trough].iloc[1:]; sp = seg.sum(); caps.append(round(float((w.loc[seg.index] * seg).sum() / sp), 2) if sp else np.nan)
    row['taken per 15% event'] = caps; row['mean taken'] = round(float(np.nanmean(caps)), 2); row['worst taken'] = round(float(np.nanmax(caps)), 2)
    o = (S == 0) & m; spells = []; cur = None
    for d in idx[o]:
        if cur is None or idx.get_loc(d) - idx.get_loc(cur[1]) > 1:
            if cur: spells.append(cur)
            cur = [d, d]
        else: cur[1] = d
    if cur: spells.append(cur)
    row['spells'] = len(spells)
    just = 0
    for a, b in spells:
        if any((e.peak <= b + pd.Timedelta(days=45)) and (e.trough >= a) for _, e in EV10S.iterrows()): just += 1
    row['spells w/o any 10% decline'] = len(spells) - just
    out_outside = (w == 0) & m & ~in15; row['return given up outside 15% events, total %'] = round(float(r[out_outside].sum() * 100), 1)
    row['return avoided inside 15% events, total %'] = round(float(-r[(w == 0) & m & in15].sum() * 100), 1)
    row['today'] = 'IN' if S.iloc[-1] == 1 else 'OUT'; return row
rows = [{'rule': 'SPY buy and hold', **stats(r[m]), 'exposure': 1.0, 'taken per 15% event': [1.0] * len(EVS), 'mean taken': 1.0, 'worst taken': 1.0, 'spells': 0, 'spells w/o any 10% decline': 0, 'return given up outside 15% events, total %': 0.0, 'return avoided inside 15% events, total %': 0.0, 'today': 'IN'}]
for name, S in ST.items(): rows.append(score(S, name))
TAB = pd.DataFrame(rows); OUT['rules15'] = TAB; OUT['events_scored'] = EVS
print('15% events scored (peak -> trough):', [f'{a.date()}->{b.date()} {d}%' for a, b, d in zip(EVS.peak, EVS.trough, EVS['depth%'])])
cols = ['rule', 'ann', 'sharpe', 'maxDD', 'exposure', 'spells', 'spells w/o any 10% decline', 'mean taken', 'worst taken', 'return avoided inside 15% events, total %', 'return given up outside 15% events, total %', 'today']
print(TAB.sort_values('mean taken')[cols].to_string(index=False))
print('\nshare taken per event, by rule (same event order as above):')
for _, rw in TAB.sort_values('mean taken').iterrows(): print(f"  {rw['rule'][:60]:60s} {rw['taken per 15% event']}")
pd.to_pickle({k: v for k, v in OUT.items() if k != 'pct'}, 'BIG.pkl'); pd.to_pickle({k: v for k, v in ST.items() if k[0] in 'GHIJK'}, 'STATES15.pkl')
print(f'\nsections 1-6 done {time.time()-t0:.0f}s', flush=True)
# ---------- 7. permutation null for the dedicated 15%/63 pipeline
if R_NULL > 0:
    y, ok = LABS['dd15_63']; real = wf.auc_by_era(PRED15['dd15_63'].to_numpy(), y, ok); rng = np.random.default_rng(1); nulls = []
    for i in range(R_NULL):
        off = int(rng.integers(252, len(y) - 252)); ys = np.roll(y, off); oks = np.roll(ok, off) & ~wf.EX
        pred, _, _ = wf.fit_predict_walk(ys, oks, k=15, purge=110); mm = oks & ~np.isnan(pred) & (YEARS >= 2005)
        nulls.append(float(roc_auc_score(ys[mm], pred[mm]))); print(f'dd15_63 null {i+1}/{R_NULL}: auc {nulls[-1]:.3f}  ({time.time()-t0:.0f}s)', flush=True)
    nulls = np.array(nulls); z = (real['all'] - nulls.mean()) / max(nulls.std(), 1e-6)
    print(f'=== dd15_63 K15 d2 mono: real AUC {real}  null mean {nulls.mean():.3f} sd {nulls.std():.3f} p95 {np.percentile(nulls, 95):.3f} max {nulls.max():.3f}  z {z:.1f}')
    OUT['null15'] = {'real': real, 'null_mean': float(nulls.mean()), 'null_sd': float(nulls.std()), 'null_p95': float(np.percentile(nulls, 95)), 'null_max': float(nulls.max()), 'n': R_NULL, 'z': float(z), 'nulls': nulls.tolist()}
    pd.to_pickle({k: v for k, v in OUT.items() if k != 'pct'}, 'BIG.pkl')
print(f'done {time.time()-t0:.0f}s')
