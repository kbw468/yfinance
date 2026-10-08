#!/usr/bin/env python3
"""v2 Stage 4b: can the risk-off side be predicted out of sample at all? Run from risk_regime/work.

The first walk-forward fit (model.py) found riskon21 and volexp21 predictable (AUC 0.76 / 0.71) but
riskoff21 and riskoff63 at chance (AUC 0.50 / 0.47) with an inverted top decile: the trees overfit
vol-level features. This script gives the risk-off side every reasonable chance before the design
accepts that result: fewer features, monotonic constraints taken from the training-window sign,
shallower trees, a linear model, raw single-feature baselines, a continuous drawdown-depth target,
and the 63-session / 10% target. Everything is walk-forward with the same purge.
"""
import pandas as pd, numpy as np, time, warnings; warnings.filterwarnings('ignore')
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from scipy.stats import spearmanr
t0 = time.time(); pd.set_option('display.width', 320)
F = pd.read_pickle('F.pkl'); MF = pd.read_pickle('MF.pkl'); L = pd.read_pickle('LAB.pkl'); idx = F.index
X = pd.concat([F, MF], axis=1)
EX = (idx >= pd.Timestamp('2020-02-01')) & (idx <= pd.Timestamp('2020-07-31'))
ERAS = {'2005-12': ('2005-01-01', '2012-12-31'), '2013-19': ('2013-01-01', '2019-12-31'), '2020-26': ('2020-01-01', '2026-12-31'), 'all': ('2005-01-01', '2026-12-31')}
_CACHE = {}
def select(Xtr, ytr, k):
    """same-signed rank IC in both halves of the training window; cached per (target name, window end)"""
    key = (ytr.name, str(ytr.index[-1].date()), len(ytr))
    if key not in _CACHE:
        n = len(ytr); h = n // 2; yr = ytr.rank(pct=True)
        ic1 = Xtr.iloc[:h].rank(pct=True).corrwith(yr.iloc[:h]); ic2 = Xtr.iloc[h:].rank(pct=True).corrwith(yr.iloc[h:])
        ok = (np.sign(ic1) == np.sign(ic2)) & ic1.notna() & ic2.notna() & (Xtr.notna().mean() > 0.5)
        score = pd.concat([ic1.abs(), ic2.abs()], axis=1).min(axis=1).where(ok, 0)
        _CACHE[key] = (score.sort_values(ascending=False), np.sign(ic1 + ic2))
    score, sgn = _CACHE[key]
    feats = list(score.head(k).index); signs = sgn[feats].astype(int).values
    return feats, signs
def walk(target, okcol, fit_predict, first=2005, label=''):
    y = L[target]; ok = L[okcol] & ~EX & y.notna(); pred = pd.Series(np.nan, index=idx)
    for year in range(first, idx[-1].year + 1):
        ts = pd.Timestamp(f'{year}-01-01'); te = pd.Timestamp(f'{year}-12-31')
        tr = ok & (idx < ts - pd.Timedelta(days=100)) & (idx >= pd.Timestamp('1993-01-01')); test = (idx >= ts) & (idx <= te)
        if tr.sum() < 1000 or test.sum() == 0: continue
        pred[test] = fit_predict(X[tr], y[tr], X[test])
    return evaluate(pred, y, ok, label)
def evaluate(pred, y, ok, label, binary=True):
    rows = []
    for e, (a, b) in ERAS.items():
        m = ok & pred.notna() & (idx >= pd.Timestamp(a)) & (idx <= pd.Timestamp(b)); yy, pp = y[m], pred[m]
        if m.sum() < 100: continue
        dec = pd.qcut(pp.rank(method='first'), 5, labels=False); q = yy.groupby(dec).mean()
        if binary: rows.append({'model': label, 'era': e, 'n': int(m.sum()), 'base': round(float(yy.mean()), 3), 'auc': round(float(roc_auc_score(yy, pp)), 3), 'q1': round(float(q.iloc[0]), 3), 'q5': round(float(q.iloc[-1]), 3), 'lift_q5': round(float(q.iloc[-1] / yy.mean()), 2)})
        else: rows.append({'model': label, 'era': e, 'n': int(m.sum()), 'spearman': round(float(spearmanr(pp, yy)[0]), 3), 'q1_meanDD%': round(float(q.iloc[0] * 100), 2), 'q5_meanDD%': round(float(q.iloc[-1] * 100), 2)})
    return pd.DataFrame(rows)
OUT = []
PAR = dict(learning_rate=0.05, max_iter=200, max_bins=64, random_state=0)
def gbm(k, depth, leaf, mono):
    def fp(Xtr, ytr, Xte):
        feats, signs = select(Xtr, ytr, k)
        clf = HistGradientBoostingClassifier(max_depth=depth, min_samples_leaf=leaf, l2_regularization=2.0, monotonic_cst=(signs if mono else None), **PAR).fit(Xtr[feats], ytr)
        return clf.predict_proba(Xte[feats])[:, 1]
    return fp
def logit(k):
    def fp(Xtr, ytr, Xte):
        feats, signs = select(Xtr, ytr, k)
        A = Xtr[feats].rank(pct=True).fillna(0.5); B = Xte[feats].copy()
        for f in feats:   # map test values onto the training distribution's percentile
            srt = np.sort(Xtr[f].dropna().values); B[f] = np.searchsorted(srt, Xte[f].values) / max(len(srt), 1)
        B = B.fillna(0.5); clf = LogisticRegression(C=0.05, max_iter=500).fit(A, ytr); return clf.predict_proba(B)[:, 1]
    return fp
def single(feat, sign):
    def fp(Xtr, ytr, Xte): return sign * Xte[feat].fillna(Xtr[feat].median()).values
    return fp
print('=== riskoff21: 5% SPY drawdown within 21 sessions ===')
for lab, fp in [('gbm K150 d3 (model.py)', gbm(150, 3, 150, False)), ('gbm K30 d2 mono', gbm(30, 2, 300, True)), ('gbm K15 d2 mono', gbm(15, 2, 300, True)), ('gbm K8 d1 mono (additive)', gbm(8, 1, 300, True)), ('logit K20 ranks', logit(20)), ('logit K8 ranks', logit(8)),
                ('raw VIX:lvl_rank504', single('VIX:lvl_rank504', 1)), ('raw VIX3M:lvl_rank252', single('VIX3M:lvl_rank252', 1)), ('raw HYG:pk21_rank', single('HYG:pk21_rank', 1)), ('raw SPY:cc21_rank', single('SPY:cc21_rank', 1)), ('raw XS:avgcorr21_rank', single('XS:avgcorr21_rank', 1)), ('raw -HYG:signedvol21', single('HYG:signedvol21', -1)), ('raw XLV:rel63', single('XLV:rel63', 1)), ('raw VIX:roc21_rank', single('VIX:roc21_rank', 1))]:
    r = walk('riskoff21', 'ok21', fp, label=lab); OUT.append(r); print(r.to_string(index=False)); print(f'   {time.time()-t0:.0f}s')
print('\n=== riskoff63: 10% SPY drawdown within 63 sessions ===')
for lab, fp in [('gbm K30 d2 mono', gbm(30, 2, 300, True)), ('gbm K8 d1 mono', gbm(8, 1, 300, True)), ('logit K8 ranks', logit(8)), ('raw VIX:lvl_rank504', single('VIX:lvl_rank504', 1)), ('raw MOVE:lvl_rank504', single('MOVE:lvl_rank504', 1))]:
    r = walk('riskoff63', 'ok63', fp, label=lab); OUT.append(r); print(r.to_string(index=False)); print(f'   {time.time()-t0:.0f}s')
print('\n=== drawdown depth (fwdDD21, continuous) with a monotone regressor; Spearman OOS and mean drawdown by predicted quintile ===')
def gbr(k):
    def fp(Xtr, ytr, Xte):
        feats, signs = select(Xtr, ytr, k)
        m = HistGradientBoostingRegressor(max_depth=2, min_samples_leaf=300, l2_regularization=2.0, monotonic_cst=signs, **PAR).fit(Xtr[feats], ytr); return m.predict(Xte[feats])
    return fp
y = L['fwdDD21']; ok = L['ok21'] & ~EX & y.notna(); pred = pd.Series(np.nan, index=idx)
for year in range(2005, idx[-1].year + 1):
    ts = pd.Timestamp(f'{year}-01-01'); te = pd.Timestamp(f'{year}-12-31'); tr = ok & (idx < ts - pd.Timedelta(days=100)); test = (idx >= ts) & (idx <= te)
    if tr.sum() < 1000 or test.sum() == 0: continue
    pred[test] = gbr(20)(X[tr], y[tr], X[test])
r = evaluate(pred, y, ok, 'gbr K20 d2 mono on fwdDD21', binary=False); OUT.append(r); print(r.to_string(index=False))
print('\n=== for contrast, the two targets that did predict: monotone small models ===')
for tgt, okc in (('riskon21', 'ok21'), ('volexp21', 'ok21')):
    for lab, fp in [('gbm K30 d2 mono', gbm(30, 2, 300, True)), ('gbm K8 d1 mono', gbm(8, 1, 300, True)), ('logit K8 ranks', logit(8))]:
        r = walk(tgt, okc, fp, label=f'{tgt} {lab}'); OUT.append(r); print(r.to_string(index=False)); print(f'   {time.time()-t0:.0f}s')
pd.concat(OUT).to_pickle('MODEL_OFF_VARIANTS.pkl'); print(f'done {time.time()-t0:.0f}s')
