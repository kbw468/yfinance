#!/usr/bin/env python3
"""v2 Stage 4: walk-forward multi-factor model. Run from risk_regime/work.

Gradient-boosted trees (sklearn HistGradientBoosting, depth 3, heavy regularization) on the rank
feature store, refit once per calendar year on all data before that year with a 63-session purge so
no training label overlaps the test year. Feature selection happens INSIDE each training window:
features must carry the same-signed rank correlation with the target in both halves of the training
window; the top K by the weaker half are used. Targets: 5% SPY drawdown within 21 sessions (riskoff21),
10% within 63 (riskoff63), 5% rally within 21 (riskon21), realized-vol expansion ahead (volexp21).
Outputs PRED.pkl (out-of-sample probabilities), MODEL_REPORT (calibration, AUC by era, importances).
"""
import pandas as pd, numpy as np, json, time, warnings; warnings.filterwarnings('ignore')
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score, brier_score_loss
from sklearn.inspection import permutation_importance
t0 = time.time(); pd.set_option('display.width', 320)
F = pd.read_pickle('F.pkl'); MF = pd.read_pickle('MF.pkl'); L = pd.read_pickle('LAB.pkl'); idx = F.index
X = pd.concat([F, MF], axis=1)
EX = (idx >= pd.Timestamp('2020-02-01')) & (idx <= pd.Timestamp('2020-07-31'))
TARGETS = {'riskoff21': ('riskoff21', 'ok21', 21), 'riskoff63': ('riskoff63', 'ok63', 63), 'riskon21': ('riskon21', 'ok21', 21), 'volexp21': ('volexp21', 'ok21', 21)}
K = 150; FIRST_TEST = 2005; LAST = idx[-1].year
PARAMS = dict(max_depth=3, learning_rate=0.05, max_iter=250, l2_regularization=2.0, min_samples_leaf=150, max_bins=64, random_state=0)
def select(Xtr, ytr, k=K):
    """same-signed rank IC in both halves of the training window; rank by the weaker half"""
    n = len(ytr); h = n // 2; yr = ytr.rank(pct=True)
    ic1 = Xtr.iloc[:h].rank(pct=True).corrwith(yr.iloc[:h]); ic2 = Xtr.iloc[h:].rank(pct=True).corrwith(yr.iloc[h:])
    ok = (np.sign(ic1) == np.sign(ic2)) & ic1.notna() & ic2.notna() & (Xtr.notna().mean() > 0.5)
    score = pd.concat([ic1.abs(), ic2.abs()], axis=1).min(axis=1).where(ok, 0)
    return list(score.sort_values(ascending=False).head(k).index), score
PRED = {}; REPORT = {}; SEL = {}
for tname, (lab, okcol, h) in TARGETS.items():
    y = L[lab]; ok = L[okcol] & ~EX & y.notna()
    pred = pd.Series(np.nan, index=idx); selcount = {}; yearly = []
    for year in range(FIRST_TEST, LAST + 1):
        ts = pd.Timestamp(f'{year}-01-01'); te = pd.Timestamp(f'{year}-12-31')
        tr = ok & (idx < ts - pd.Timedelta(days=100)) & (idx >= pd.Timestamp('1993-01-01'))
        # purge: a training row whose label window could reach into the test year is dropped (100 calendar days ~ 63 sessions + slack)
        test = (idx >= ts) & (idx <= te)
        if tr.sum() < 1000 or test.sum() == 0: continue
        Xtr = X[tr]; ytr = y[tr]
        feats, score = select(Xtr, ytr)
        for f in feats: selcount[f] = selcount.get(f, 0) + 1
        clf = HistGradientBoostingClassifier(**PARAMS).fit(Xtr[feats], ytr)
        p = clf.predict_proba(X.loc[test, feats])[:, 1]; pred[test] = p
        m = test & ok
        if m.sum() > 50 and y[m].nunique() == 2:
            yearly.append((year, int(m.sum()), round(float(y[m].mean()), 3), round(float(roc_auc_score(y[m], pred[m])), 3), round(float(brier_score_loss(y[m], pred[m])), 4), round(float(brier_score_loss(y[m], np.full(m.sum(), ytr.mean()))), 4)))
        print(f'{tname} {year}: train {tr.sum()} rows, test {test.sum()}, top feature {feats[0]} ({score[feats[0]]:.3f}), {time.time()-t0:.0f}s')
    PRED[tname] = pred
    yt = pd.DataFrame(yearly, columns=['year', 'n', 'base', 'auc', 'brier', 'brier_base'])
    # pooled by era
    eras = {'2005-12': ('2005-01-01', '2012-12-31'), '2013-19': ('2013-01-01', '2019-12-31'), '2020-26': ('2020-01-01', '2026-12-31'), 'all': ('2005-01-01', '2026-12-31')}
    era_rows = []
    for e, (a, b) in eras.items():
        m = ok & pred.notna() & (idx >= pd.Timestamp(a)) & (idx <= pd.Timestamp(b))
        yy, pp = y[m], pred[m]
        dec = pd.qcut(pp.rank(method='first'), 10, labels=False)
        calib = pd.DataFrame({'pred': pp.groupby(dec).mean(), 'real': yy.groupby(dec).mean(), 'n': yy.groupby(dec).size()})
        era_rows.append({'era': e, 'n': int(m.sum()), 'base': round(float(yy.mean()), 3), 'auc': round(float(roc_auc_score(yy, pp)), 3), 'brier': round(float(brier_score_loss(yy, pp)), 4), 'brier_base': round(float(brier_score_loss(yy, np.full(m.sum(), yy.mean()))), 4),
                         'top_decile_real': round(float(calib['real'].iloc[-1]), 3), 'bottom_decile_real': round(float(calib['real'].iloc[0]), 3), 'top_decile_pred': round(float(calib['pred'].iloc[-1]), 3), 'bottom_decile_pred': round(float(calib['pred'].iloc[0]), 3)})
        if e == 'all': calib_all = calib
    REPORT[tname] = {'yearly': yt, 'eras': pd.DataFrame(era_rows), 'calib': calib_all, 'selcount': pd.Series(selcount).sort_values(ascending=False)}
    SEL[tname] = selcount
    print(f'\n=== {tname} out-of-sample ===\n' + pd.DataFrame(era_rows).to_string(index=False)); print('calibration deciles (all OOS):'); print(calib_all.round(3).to_string())
    print('most-selected features across windows:'); print(pd.Series(selcount).sort_values(ascending=False).head(25).to_string())
    print(yt.to_string(index=False))
PRED = pd.DataFrame(PRED); PRED.to_pickle('PRED.pkl'); pd.to_pickle(REPORT, 'MODEL_REPORT.pkl')
# ---- permutation importance of the final models on the last three test years, for the dashboard narrative
IMP = {}
for tname, (lab, okcol, h) in TARGETS.items():
    y = L[lab]; ok = L[okcol] & ~EX & y.notna()
    tr = ok & (idx < pd.Timestamp(f'{LAST-2}-01-01') - pd.Timedelta(days=100)); test = ok & (idx >= pd.Timestamp(f'{LAST-2}-01-01'))
    feats, _ = select(X[tr], y[tr]); clf = HistGradientBoostingClassifier(**PARAMS).fit(X.loc[tr, feats], y[tr])
    pi = permutation_importance(clf, X.loc[test, feats], y[test], scoring='roc_auc', n_repeats=5, random_state=0, n_jobs=-1)
    IMP[tname] = pd.Series(pi.importances_mean, index=feats).sort_values(ascending=False)
    print(f'\n{tname}: permutation importance (AUC drop) on {LAST-2}-{LAST} test, final-window model'); print(IMP[tname].head(20).round(4).to_string())
pd.to_pickle(IMP, 'MODEL_IMP.pkl')
print(f'done {time.time()-t0:.0f}s')
