"""Score every eligible name on the latest session: final models on all labelled history (same hyperparameters), isotonic
calibration on the full out-of-sample record, and the descriptive columns asked for (numeric L1 beta and captures, price /
volume / volatility rates of change, distance to the 52-week high, trailing path drawdown). Writes results/steady/."""
import time, warnings
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.isotonic import IsotonicRegression
from ..config import CACHE_DIR, RESULTS_DIR
from .model import PARAMS, feature_list, load, EMBARGO_DAYS
from .outcomes import HORIZONS
warnings.filterwarnings("ignore")
D = CACHE_DIR / "steady"; OUT = RESULTS_DIR / "steady"


def main():
    t0 = time.time(); OUT.mkdir(exist_ok=True)
    feats = feature_list(); T = load(); O = pd.read_parquet(D / "oos.parquet")
    d = T.date.max(); N = T[T.date == d].copy()
    thin = set(np.sort(T.date.unique())[::3])
    for h in HORIZONS:
        y = f"hit_{h}"; tr = T[y].notna() & T.date.isin(thin)
        m = lgb.LGBMClassifier(**PARAMS).fit(T.loc[tr, feats], T.loc[tr, y])
        ok = O[f"raw_{h}"].notna() & O[y].notna()
        iso = IsotonicRegression(out_of_bounds="clip").fit(O.loc[ok, f"raw_{h}"], O.loc[ok, y])
        N[f"P{h}"] = iso.predict(m.predict_proba(N[feats])[:, 1])
        N[f"pctile{h}"] = N[f"P{h}"].rank(pct=True)
        print(h, "trained through", T.loc[tr, "date"].max().date(), f"{time.time()-t0:.0f}s", flush=True)
    L = pd.read_csv(RESULTS_DIR / "THE_LIST.csv").set_index("ticker")
    N["list_tier"] = N.ticker.map(L.tier)
    cols = ["ticker", "sector", "P21", "P42", "P63", "steady", "beta_l1_252", "up_capture_252", "down_capture_252",
            "roc_5", "roc_21", "roc_63", "xs_spy_63", "volume_roc_5_63", "volume_roc_21_252", "vol_roc_5_63", "vol_roc_21_252",
            "off_high_252", "days_since_high_252", "mdd_63", "eff_63", "up_share_63", "list_tier"]
    out = N[cols].sort_values("P42", ascending=False)
    out.to_csv(OUT / f"steady_{d.date()}.csv", index=False); out.to_csv(OUT / "steady_latest.csv", index=False)
    print(f"scored {len(out)} names for {d.date()} {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
