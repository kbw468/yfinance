"""Multifactor model for sup_21: gradient-boosted trees (threshold splits only, no distributional assumption) on same-day
universe percentiles of every stock feature, plus the raw market-state block (same value for every name on a date, usable
only through interactions: which kind of name wins in which tape).
Walk-forward: test years 2017..2026; training = every 3rd session from 2015 ending 100 calendar days before the test year,
COVID Feb-Jun 2020 out. Fixed hyperparameters. Skill = mean per-date AUC (name selection only), also inside beta terciles and
inside typical-move terciles. Controls: labels shuffled within each date (two test years); and the model without market state.
Writes the out-of-sample scores (CACHE_DIR/ra21/oos.parquet) and the summary tables (results/ra21/)."""
import sys, time, warnings
import numpy as np
import pandas as pd
import lightgbm as lgb
from .common import load, feature_sets, date_auc, D, OUT, MARKET
warnings.filterwarnings("ignore")

PARAMS = dict(objective="binary", learning_rate=0.03, num_leaves=31, min_child_samples=1000, subsample=0.7, subsample_freq=1,
              colsample_bytree=0.5, reg_lambda=1.0, n_estimators=400, verbose=-1, n_jobs=4)
TEST_YEARS = range(2017, 2027)
EMBARGO_DAYS = 100
TARGET = "sup_21"
KEEP = ["sector", "industry", "beta_l1_252", "mad_63", "sup_21", "sup_42", "sup_63", "xs_21", "ret_21", "mar_21", "dd_21", "stopped_21", "effx_21",
        "mae_21", "mfe_21", "xs_42", "mar_42", "stopped_42", "xs_63", "mar_63", "stopped_63", "xsns_21", "ddns_21", "marpct_21"]


def prepare():
    fs = feature_sets()
    T = load(fs["stock"] + fs["market"] + [c for c in KEEP if c not in fs["stock"]])
    g = T.groupby("date")
    R = g[fs["stock"]].rank(pct=True).astype("float32"); R.columns = [f"r_{c}" for c in fs["stock"]]
    T = pd.concat([T, R], axis=1)
    T["dcode"] = T.date.factorize(sort=True)[0]
    T["thin"] = T.dcode % 3 == 0
    return T, [f"r_{c}" for c in fs["stock"]], fs["market"]


def walk_forward(T, feats, y=TARGET, years=TEST_YEARS, shuffle=False, seed=0):
    pred = pd.Series(np.nan, index=T.index, dtype="float64"); models = {}
    lab = T[y].copy()
    if shuffle:
        rng = np.random.default_rng(seed)
        lab = lab.groupby(T.date).transform(lambda s: pd.Series(rng.permutation(s.values), index=s.index))
    for Y in years:
        te = (T.date >= f"{Y}-01-01") & (T.date <= f"{Y}-12-31") & T[y].notna()
        tr = (T.date < pd.Timestamp(f"{Y}-01-01") - pd.Timedelta(days=EMBARGO_DAYS)) & lab.notna() & T.thin
        if tr.sum() < 50000 or te.sum() == 0: continue
        m = lgb.LGBMClassifier(**PARAMS).fit(T.loc[tr, feats], lab[tr])
        pred[te] = m.predict_proba(T.loc[te, feats])[:, 1]
        models[Y] = m
    return pred, models


def auc_by_year(T, s, y=TARGET) -> pd.Series:
    a = date_auc(s.to_numpy(float), T[y].to_numpy(float), T.dcode.to_numpy())
    d = T.groupby("dcode").date.first()
    return a.groupby(d.reindex(a.index).dt.year.values).mean()


def bucket_auc(T, s, col, y=TARGET) -> float:
    q = np.clip((T.groupby("date")[col].rank(pct=True).fillna(0.5).to_numpy() * 3).astype(int), 0, 2)
    a = date_auc(s.to_numpy(float), T[y].to_numpy(float), T.dcode.to_numpy() * 3 + q)
    return float(np.nanmean(a))


def deciles(T, s, mask) -> pd.DataFrame:
    X = T[mask & s.notna()].copy(); X["s"] = s[mask & s.notna()]
    X["dec"] = (X.groupby("date").s.rank(pct=True) * 10).clip(upper=9.999).astype(int) + 1
    f = {"rows": ("sup_21", "size"), "superior_21": ("sup_21", "mean"), "superior_42": ("sup_42", "mean"), "superior_63": ("sup_63", "mean"),
         "mean_xs_21": ("xs_21", "mean"), "med_xs_21": ("xs_21", "median"), "beat_spy_21": ("xs_21", lambda v: (v > 0).mean()),
         "med_mar_21": ("mar_21", "median"), "med_dd_21": ("dd_21", "median"), "stopped_21": ("stopped_21", "mean"),
         "med_xs_63": ("xs_63", "median"), "mean_xs_63": ("xs_63", "mean"), "med_beta": ("beta_l1_252", "median"), "med_move": ("mad_63", "median")}
    return X.groupby("dec").agg(**f)


def main():
    t0 = time.time(); OUT.mkdir(parents=True, exist_ok=True)
    T, sf, mf = prepare(); print(f"rows {len(T):,} stock features {len(sf)} market {len(mf)} {time.time()-t0:.0f}s", flush=True)
    full, models = walk_forward(T, sf + mf); print(f"walk-forward (stock + market) {time.time()-t0:.0f}s", flush=True)
    stock_only, _ = walk_forward(T, sf); print(f"walk-forward (stock only) {time.time()-t0:.0f}s", flush=True)
    ctrl, _ = walk_forward(T, sf + mf, years=[2021, 2024], shuffle=True); print(f"shuffled control {time.time()-t0:.0f}s", flush=True)
    res = pd.DataFrame({"stock + market": auc_by_year(T, full), "stock only": auc_by_year(T, stock_only), "shuffled control": auc_by_year(T, ctrl),
                        "superior_42 (same score)": auc_by_year(T, full, "sup_42"), "superior_63 (same score)": auc_by_year(T, full, "sup_63")})
    oos = T.date >= "2017-01-01"
    extra = {"inside beta terciles": bucket_auc(T[oos], full[oos], "beta_l1_252"), "inside typical-move terciles": bucket_auc(T[oos], full[oos], "mad_63")}
    lines = ["Within-date AUC of the walk-forward score for a superior 21-session trade (0.5 = no skill), by test year:", res.round(3).to_string(),
             f"\nmean 2017+: {res.mean().round(3).to_dict()}", f"inside risk buckets (2017+): {({k: round(v, 3) for k, v in extra.items()})}"]
    for lab, m in (("2017-2023", oos & (T.date < "2024-01-01")), ("2024 onward", T.date >= "2024-01-01")):
        lines += [f"\n{lab}: deciles of the score within each date (10 = highest)", deciles(T, full, m).round(3).to_string()]
    imp = pd.DataFrame({Y: pd.Series(m.booster_.feature_importance("gain"), index=sf + mf) for Y, m in models.items()})
    imp = imp.div(imp.sum()).mean(axis=1).sort_values(ascending=False)
    imp.round(4).to_csv(OUT / "model_importance.csv")
    lines += ["\nfeature importance (share of split gain, mean over folds), top 30:", imp.head(30).round(4).to_string()]
    txt = "\n".join(lines); (OUT / "model_walkforward.txt").write_text(txt); print(txt)
    T[["date", "ticker"]].assign(score=full, score_stock_only=stock_only).to_parquet(D / "oos.parquet")
    print(f"done {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
