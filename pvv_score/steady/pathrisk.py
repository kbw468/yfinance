"""Path-risk model: probability that the next h sessions stay shallow (close-to-close drawdown shallower than CAP[h]) and never
trade 8% below entry. Rank features, walk-forward 2018-2026, 100-day embargo, within-date AUC. Then the question that matters:
holding beta equal, does low predicted path risk cost excess return?"""
import numpy as np, pandas as pd, lightgbm as lgb, warnings
from .model import PARAMS, feature_list, load, EMBARGO_DAYS, _daily, D
from .outcomes import HORIZONS, CAP, STOP
warnings.filterwarnings("ignore")


def main():
    feats = feature_list(); T = load(); T["thin"] = T.date.isin(set(np.sort(T.date.unique())[::3]))
    keep = ["date", "ticker", "sector", "beta_l1_252", "steady"] + [f"{k}_{h}" for h in HORIZONS for k in ("xs", "ptt", "stop", "ret")]
    O = T[keep].copy()
    for h in HORIZONS:
        y = f"safe_{h}"; T[y] = ((T[f"ptt_{h}"] >= CAP[h]) & (T[f"stop_{h}"] > -STOP)).astype(float).where(T[f"xs_{h}"].notna())
        O[y] = T[y]; O[f"ps_{h}"] = np.nan; auc = {}
        for Y in range(2018, 2027):
            te = (T.date >= f"{Y}-01-01") & (T.date <= f"{Y}-12-31")
            tr = (T.date < pd.Timestamp(f"{Y}-01-01") - pd.Timedelta(days=EMBARGO_DAYS)) & T[y].notna() & T.thin
            m = lgb.LGBMClassifier(**PARAMS).fit(T.loc[tr, feats], T.loc[tr, y])
            O.loc[te, f"ps_{h}"] = m.predict_proba(T.loc[te, feats])[:, 1]
            ok = te & T[y].notna()
            if ok.sum() > 1000: auc[Y] = round(_daily(O.loc[ok], y, f"ps_{h}"), 3)
        print(f"[{h}] path-risk within-date AUC {auc} mean {np.mean(list(auc.values())):.3f}", flush=True)
    O.to_parquet(D / "pathrisk_oos.parquet")
    pd.set_option("display.width", 240)
    for h in HORIZONS:
        X = O[(O.date >= "2019-01-01")].dropna(subset=[f"ps_{h}", f"xs_{h}", "beta_l1_252"]).copy()
        X["beta_q"] = X.groupby("date").beta_l1_252.transform(lambda v: pd.qcut(v.rank(method="first"), 5, labels=False) + 1)
        X["risk_t"] = X.groupby(["date", "beta_q"])[f"ps_{h}"].transform(lambda v: pd.qcut(v.rank(method="first"), 3, labels=["high path risk", "mid", "low path risk"]))
        for per, g in [("2019-2023", X[X.date < "2024-01-01"]), ("2024 onward", X[X.date >= "2024-01-01"])]:
            t = g.groupby(["beta_q", "risk_t"]).agg(rows=(f"xs_{h}", "size"), med_beta=("beta_l1_252", "median"), mean_excess=(f"xs_{h}", "mean"), med_excess=(f"xs_{h}", "median"),
                                                    beat_spy=(f"xs_{h}", lambda v: (v > 0).mean()), med_drawdown=(f"ptt_{h}", "median"), stopped=(f"stop_{h}", lambda v: (v <= -STOP).mean()))
            print(f"\n==== {h} sessions, {per}: within each beta quintile (1 = lowest L1 beta), terciles of predicted path risk"); print(t.round(3).to_string())


if __name__ == "__main__":
    main()
