"""Whole-profile similarity to the nine case breakouts. Distance = mean absolute difference of same-day universe percentiles
across all 75 features (L1, no distributional assumption); similarity to the set = distance to the nearest case.
Out-of-time test: the profiles come from 2026, the test runs on 2015-2025. Each date, the 1% / 5% most similar names vs the rest."""
import numpy as np, pandas as pd
from ..config import CACHE_DIR, RESULTS_DIR
from .outcomes import HORIZONS, STOP, alpha_target
D = CACHE_DIR / "steady"; OUT = RESULTS_DIR / "steady" / "cases"


def main():
    feats = pd.read_csv(D / "features.csv").iloc[:, 0].tolist(); cs = [f"cs_{f}" for f in feats]
    B = pd.read_csv(OUT / "breakout_percentiles.csv", index_col=0)[feats].values.astype("float32")      # 9 x 75
    T = pd.read_parquet(D / "table.parquet", columns=["date", "ticker", "sector", "beta_l1_252"] + cs + [f"{k}_{h}" for h in HORIZONS for k in ("xs", "ptt", "stop", "ret", "hit")])
    T = alpha_target(T)
    X = T[cs].values.astype("float32"); nearest = np.full(len(T), np.inf, dtype="float32"); which = np.zeros(len(T), dtype=int)
    for k in range(len(B)):
        d = np.nanmean(np.abs(X - B[k]), axis=1)
        upd = d < nearest; nearest[upd] = d[upd]; which[upd] = k
    T["distance"] = nearest; T["nearest_case"] = np.array(pd.read_csv(OUT / "breakout_percentiles.csv", index_col=0).index)[which]
    T["sim_pct"] = T.groupby("date").distance.rank(pct=True)          # low = most similar
    rows = []
    H = T[T.date < "2026-01-01"]
    for lab, m in (("1% most similar", H.sim_pct <= 0.01), ("5% most similar", H.sim_pct <= 0.05), ("middle 90%", H.sim_pct.between(0.05, 0.95)), ("all names", H.sim_pct.notna())):
        for h in HORIZONS:
            x = H[m].dropna(subset=[f"xs_{h}"])
            rows.append({"group": lab, "h": h, "rows": len(x), "mean_excess": x[f"xs_{h}"].mean(), "median_excess": x[f"xs_{h}"].median(), "beat_spy": (x[f"xs_{h}"] > 0).mean(),
                         "median_drawdown": x[f"ptt_{h}"].median(), "stopped_8pct": (x[f"stop_{h}"] <= -STOP).mean(), "alpha_per_pain_hit": x[f"ah_{h}"].mean()})
    R = pd.DataFrame(rows); R.to_csv(OUT / "lookalike_test.csv", index=False)
    pd.set_option("display.width", 240)
    for h in HORIZONS: print(f"\n=== {h} sessions, 2015-2025"); print(R[R.h == h].drop(columns="h").set_index("group").round(3).to_string())
    top = H[H.sim_pct <= 0.01].dropna(subset=["xs_42"])
    print("\n1% most similar, 42 sessions, by year:"); print(top.groupby(top.date.dt.year).agg(rows=("xs_42", "size"), mean_excess=("xs_42", "mean"), median_excess=("xs_42", "median"), beat=("xs_42", lambda v: (v > 0).mean())).round(3).to_string())
    print("\n1% most similar, 42 sessions, by nearest case:"); print(top.groupby("nearest_case").agg(rows=("xs_42", "size"), mean_excess=("xs_42", "mean"), median_excess=("xs_42", "median"), beat=("xs_42", lambda v: (v > 0).mean()), stopped=("stop_42", lambda v: (v <= -STOP).mean())).round(3).to_string())
    T[T.date == T.date.max()][["date", "ticker", "sector", "beta_l1_252", "distance", "sim_pct", "nearest_case"]].sort_values("distance").to_csv(OUT / "lookalikes_today.csv", index=False)


if __name__ == "__main__":
    main()
