"""Same features, same walk-forward, different learning objectives (all non-Gaussian): which one ranks superior 21-session
trades best out of sample?
  binary_sup21   log loss on the superior label (model.py)
  l1_marpct      median (L1) regression on the trade's same-day MAR percentile (uses the whole ordering, not just the top 20%)
  rank_grade     LambdaRank inside each date on the MAR quintile grade 0..4 (learns the within-date ordering directly)
  binary_sup42   log loss on the 42-session superior label, scored against the 21-session one
Test years 2017-2026; skill = mean per-date AUC for sup_21 and the superior rate of the top decile."""
import time, warnings
import numpy as np
import pandas as pd
import lightgbm as lgb
from .common import date_auc, D, OUT
from .model import prepare, PARAMS, TEST_YEARS, EMBARGO_DAYS
warnings.filterwarnings("ignore")


def fit_predict(T, feats, kind, Y):
    te = (T.date >= f"{Y}-01-01") & (T.date <= f"{Y}-12-31") & T.sup_21.notna()
    tr = (T.date < pd.Timestamp(f"{Y}-01-01") - pd.Timedelta(days=EMBARGO_DAYS)) & T.thin
    p = {k: v for k, v in PARAMS.items() if k != "objective"}
    if kind == "binary_sup21":
        tr &= T.sup_21.notna(); m = lgb.LGBMClassifier(objective="binary", **p).fit(T.loc[tr, feats], T.loc[tr, "sup_21"]); return te, m.predict_proba(T.loc[te, feats])[:, 1]
    if kind == "binary_sup42":
        tr &= T.sup_42.notna(); m = lgb.LGBMClassifier(objective="binary", **p).fit(T.loc[tr, feats], T.loc[tr, "sup_42"]); return te, m.predict_proba(T.loc[te, feats])[:, 1]
    if kind == "l1_marpct":
        tr &= T.marpct_21.notna(); m = lgb.LGBMRegressor(objective="regression_l1", **p).fit(T.loc[tr, feats], T.loc[tr, "marpct_21"]); return te, m.predict(T.loc[te, feats])
    if kind == "rank_grade":
        tr &= T.marpct_21.notna()
        X = T.loc[tr].sort_values("date")
        grade = np.clip((X.marpct_21 * 5).astype(int), 0, 4)
        m = lgb.LGBMRanker(objective="lambdarank", lambdarank_truncation_level=100, **{k: v for k, v in p.items() if k not in ("subsample", "subsample_freq")})
        m.fit(X[feats], grade, group=X.groupby("date", sort=True).size().to_numpy())
        return te, m.predict(T.loc[te, feats])


def main(kinds=("l1_marpct", "rank_grade", "binary_sup42")):
    t0 = time.time()
    T, sf, mf = prepare(); feats = sf          # stock features only: the market block added nothing out of sample (model_walkforward.txt)
    res = {}; tops = {}
    for kind in kinds:
        s = pd.Series(np.nan, index=T.index)
        for Y in TEST_YEARS:
            te, pr = fit_predict(T, feats, kind, Y); s[te] = pr
        a = date_auc(s.to_numpy(float), T.sup_21.to_numpy(float), T.dcode.to_numpy())
        d = T.groupby("dcode").date.first()
        res[kind] = a.groupby(d.reindex(a.index).dt.year.values).mean()
        top = (s.groupby(T.date).rank(pct=True) >= 0.9) & s.notna()
        tops[kind] = {"top decile superior_21": float(T.sup_21[top].mean()), "2024+": float(T.sup_21[top & (T.date >= "2024-01-01")].mean()),
                      "top decile superior_42": float(T.sup_42[top].mean()), "top decile median xs_21": float(T.xs_21[top].median()), "top decile stopped_21": float(T.stopped_21[top].mean())}
        T[f"s_{kind}"] = s
        print(kind, res[kind].round(3).to_dict(), "mean", round(res[kind].mean(), 3), tops[kind], f"{time.time()-t0:.0f}s", flush=True)
    R = pd.DataFrame(res); txt = "Within-date AUC for sup_21 by test year, by learning objective:\n" + R.round(3).to_string() + "\nmean:\n" + R.mean().round(3).to_string() + "\n\n" + pd.DataFrame(tops).round(3).to_string()
    (OUT / "model_variants.txt").write_text(txt); print(txt)
    T[["date", "ticker"] + [f"s_{k}" for k in kinds]].to_parquet(D / "oos_variants.parquet")


if __name__ == "__main__":
    main()
