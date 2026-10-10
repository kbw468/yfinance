"""Score every eligible name on the latest session with the path-risk models (final fit on all labelled history, isotonic
calibration on the full walk-forward record) and the descriptive columns: numeric L1 beta and captures, price / volume /
volatility rates of change, distance to the 52-week high, trailing path drawdown, excess vs SPY. Writes results/steady/."""
import time, warnings
import numpy as np, pandas as pd, lightgbm as lgb
from sklearn.isotonic import IsotonicRegression
from ..config import CACHE_DIR, RESULTS_DIR
from .model import PARAMS, feature_list, load
from .outcomes import HORIZONS, CAP, STOP
warnings.filterwarnings("ignore")
D = CACHE_DIR / "steady"; OUT = RESULTS_DIR / "steady"


def main():
    t0 = time.time(); OUT.mkdir(exist_ok=True)
    feats = feature_list(); T = load(); O = pd.read_parquet(D / "pathrisk_oos.parquet")
    d = T.date.max(); N = T[T.date == d].copy(); thin = T.date.isin(set(np.sort(T.date.unique())[::3]))
    for h in HORIZONS:
        y = ((T[f"ptt_{h}"] >= CAP[h]) & (T[f"stop_{h}"] > -STOP)).astype(float).where(T[f"xs_{h}"].notna())
        tr = y.notna() & thin
        m = lgb.LGBMClassifier(**PARAMS).fit(T.loc[tr, feats], y[tr])
        ok = O[f"ps_{h}"].notna() & O[f"safe_{h}"].notna()
        iso = IsotonicRegression(out_of_bounds="clip").fit(O.loc[ok, f"ps_{h}"], O.loc[ok, f"safe_{h}"])
        N[f"P_safe_{h}"] = iso.predict(m.predict_proba(N[feats])[:, 1])
        print(h, "trained through", T.loc[tr, "date"].max().date(), f"{time.time()-t0:.0f}s", flush=True)
    b = N.beta_l1_252.notna()
    N["beta_quintile"] = np.nan
    N.loc[b, "beta_quintile"] = pd.qcut(N.loc[b, "beta_l1_252"].rank(method="first"), 5, labels=[1, 2, 3, 4, 5]).astype(int)
    N["safe_tercile_in_beta"] = N.loc[b].groupby("beta_quintile").P_safe_42.transform(lambda v: pd.qcut(v.rank(method="first"), 3, labels=[3, 2, 1]).astype(int))
    L = pd.read_csv(RESULTS_DIR / "THE_LIST.csv").set_index("ticker"); N["list_tier"] = N.ticker.map(L.tier)
    cols = ["ticker", "sector", "beta_l1_252", "beta_quintile", "up_capture_252", "down_capture_252", "P_safe_21", "P_safe_42", "P_safe_63", "safe_tercile_in_beta",
            "roc_5", "roc_21", "roc_63", "xs_spy_63", "volume_roc_5_63", "volume_roc_21_252", "vol_roc_5_63", "vol_roc_21_252",
            "off_high_252", "days_since_high_252", "mdd_63", "steady", "list_tier"]
    out = N[cols].sort_values(["beta_quintile", "P_safe_42"], ascending=[True, False])
    out.to_csv(OUT / f"pathrisk_{d.date()}.csv", index=False); out.to_csv(OUT / "pathrisk_latest.csv", index=False)
    print(f"scored {len(out)} names for {d.date()} {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
