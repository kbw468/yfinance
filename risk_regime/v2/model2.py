#!/usr/bin/env python3
"""v2 Stage 4 (final): walk-forward monotone models for the four targets, plus the variant table and
year-block bootstrap intervals. Run from risk_regime/work. Writes PRED2.pkl, LASTFIT2.pkl, VARIANTS2.pkl.

Production configuration (fixed before the variants were compared, see README): 15 features chosen
inside each training window by same-signed rank correlation in both halves, depth-2 gradient-boosted
trees constrained to be monotone in the training-window sign, 200 rounds, no tuning. Nothing is
Gaussian or linear here: inputs are ranks, the learner is a sum of shallow step functions.
"""
import sys, time, numpy as np, pandas as pd; sys.path.insert(0, '../v2'); import wf
t0 = time.time(); pd.set_option('display.width', 320)
PRED = {}; LAST = {}; SEL = {}; rows = []
for name in ('P_vol', 'P_on', 'P_off', 'P_off63'):
    y, ok = wf.label(name); pred, selcount, last = wf.fit_predict_walk(y, ok, k=15, keep_last=True)
    PRED[name] = pred; LAST[name] = (last[0], last[1]); SEL[name] = pd.Series({wf.COLS[k]: v for k, v in selcount.items()}).sort_values(ascending=False)
    lo, med, hi = wf.year_bootstrap_auc(pred, y, ok)
    rows.append({'target': name, 'model': 'K15 d2 mono (production)', **{f'auc {e}': v for e, v in wf.auc_by_era(pred, y, ok).items()}, 'auc CI95 (year bootstrap)': f'{lo:.3f}-{hi:.3f}', 'quintile hit rates 2005-26': wf.quintiles(pred, y, ok), 'base': round(float(y[ok & (wf.YEARS >= 2005)].mean()), 3)})
    print(f'{name}: {rows[-1]}  {time.time()-t0:.0f}s'); print('  most selected:', ', '.join(f'{k} ({v})' for k, v in SEL[name].head(12).items()))
    print('  last-window features:', ', '.join(f'{f} ({"+" if s > 0 else "-"})' for f, s in zip(last[0], last[1])))
pd.DataFrame(PRED, index=wf.idx).to_pickle('PRED2.pkl'); pd.to_pickle(LAST, 'LASTFIT2.pkl'); pd.to_pickle(SEL, 'SELCOUNT2.pkl')
# ---- variants: the same walk-forward with other capacities, and raw single features, for the record
def single(feat, sign):
    j = int(np.where(wf.COLS == feat)[0][0]); v = wf.XR[:, j] * sign; return v
VAR = []
for name, variants in {'P_off': [('K150 d3 free (the overfit reference)', dict(k=150, depth=3, leaf=150, mono=False, iters=250)), ('K30 d2 mono', dict(k=30)), ('K15 d2 mono (production)', dict(k=15)), ('K8 d1 mono (additive)', dict(k=8, depth=1)), ('K5 d1 mono', dict(k=5, depth=1))],
                       'P_vol': [('K30 d2 mono', dict(k=30)), ('K15 d2 mono (production)', dict(k=15)), ('K8 d1 mono', dict(k=8, depth=1))],
                       'P_on': [('K30 d2 mono', dict(k=30)), ('K15 d2 mono (production)', dict(k=15)), ('K8 d1 mono', dict(k=8, depth=1))]}.items():
    y, ok = wf.label(name)
    for lab, kw in variants:
        pred = PRED[name] if lab.endswith('(production)') else wf.fit_predict_walk(y, ok, **kw)[0]
        VAR.append({'target': name, 'model': lab, **{f'auc {e}': v for e, v in wf.auc_by_era(pred, y, ok).items()}, 'quintiles': wf.quintiles(pred, y, ok)}); print(f'  {name} {lab}: {VAR[-1]}  {time.time()-t0:.0f}s')
y, ok = wf.label('P_off')
for feat, sg in [('VIX:lvl_rank504', 1), ('VIX3M:lvl_rank252', 1), ('SPY:cc21_rank', 1), ('HYG:pk21_rank', 1), ('XLV:rel63', 1), ('XS:avgcorr21_rank', 1), ('HYG:signedvol21', -1), ('VIX:roc21_rank', 1), ('VIX:roc5_rank', 1), ('SPY:VIX/cc21_rank', -1)]:
    pred = single(feat, sg); pred[wf.YEARS < 2005] = np.nan
    VAR.append({'target': 'P_off', 'model': f'raw {"-" if sg < 0 else ""}{feat}', **{f'auc {e}': v for e, v in wf.auc_by_era(pred, y, ok).items()}, 'quintiles': wf.quintiles(pred, y, ok)})
VAR = pd.DataFrame(VAR); VAR.to_pickle('VARIANTS2.pkl'); print('\n=== variants, all walk-forward out of sample ==='); print(VAR.to_string(index=False)); print(f'{time.time()-t0:.0f}s')
