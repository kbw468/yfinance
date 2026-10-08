#!/usr/bin/env python3
"""v2 robustness: a permutation test of the WHOLE walk-forward pipeline. Run from risk_regime/work.

For each target the label series is shifted circularly by a random offset of at least 252 sessions,
which keeps its autocorrelation and base rate but breaks its link to the features, and the complete
pipeline (yearly feature selection inside the window, monotone fit, out-of-sample prediction) is rerun.
The pooled out-of-sample AUC under this null is what selection and fitting produce from nothing. The
real AUC has to sit outside that distribution to mean anything. Writes NULLTEST2.pkl.
"""
import sys, time, numpy as np, pandas as pd; sys.path.insert(0, '../v2'); import wf
from sklearn.metrics import roc_auc_score
t0 = time.time(); R = int(sys.argv[1]) if len(sys.argv) > 1 else 20; rng = np.random.default_rng(1)
PRED = pd.read_pickle('PRED2.pkl'); out = {}
for name in ('P_off', 'P_vol', 'P_on'):
    y, ok = wf.label(name); real = wf.auc_by_era(PRED[name].to_numpy(), y, ok); nulls = []
    for r in range(R):
        off = int(rng.integers(252, len(y) - 252)); ys = np.roll(y, off); oks = np.roll(ok, off) & ~wf.EX
        pred, _, _ = wf.fit_predict_walk(ys, oks, k=15)
        m = oks & ~np.isnan(pred) & (wf.YEARS >= 2005); nulls.append(float(roc_auc_score(ys[m], pred[m])))
        print(f'{name} null {r+1}/{R}: auc {nulls[-1]:.3f}  ({time.time()-t0:.0f}s)', flush=True)
    nulls = np.array(nulls); out[name] = {'real': real, 'null_mean': float(nulls.mean()), 'null_sd': float(nulls.std()), 'null_p95': float(np.percentile(nulls, 95)), 'null_max': float(nulls.max()), 'n': R, 'nulls': nulls.tolist(), 'z': float((real['all'] - nulls.mean()) / max(nulls.std(), 1e-6))}
    print(f'=== {name}: real AUC {real}  null mean {nulls.mean():.3f} sd {nulls.std():.3f} p95 {np.percentile(nulls, 95):.3f} max {nulls.max():.3f}  z {out[name]["z"]:.1f}', flush=True)
pd.to_pickle(out, 'NULLTEST2.pkl'); print(f'done {time.time()-t0:.0f}s')
