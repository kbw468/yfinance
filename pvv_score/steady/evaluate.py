"""Out-of-sample evaluation, non-Gaussian measures only: AUC (rank-based), calibration, decile hit rates, medians of excess
return and path drawdown, stop-out frequency, L1 beta; model vs transparent baseline vs steady flag; MSI 2024 path."""
import numpy as np
import pandas as pd
from ..config import CACHE_DIR
from .outcomes import HORIZONS, CAP, STOP
D = CACHE_DIR / "steady"


def main():
    O = pd.read_parquet(D / "oos.parquet"); log = pd.read_csv(D / "walkforward_log.csv")
    pd.set_option("display.width", 240)
    print("walk-forward AUC by year (model / transparent baseline / steady flag):")
    print(log.pivot_table(index="year", columns="h", values=["auc_model", "auc_baseline", "auc_steady_flag"]).round(3).to_string())
    for h in HORIZONS:
        X = O.dropna(subset=[f"raw_{h}", f"hit_{h}"]).copy()
        X["per"] = np.where(X.date < "2024-01-01", "2018-2023", "2024 onward")
        for s in ("raw", "baseline"):
            col = f"raw_{h}" if s == "raw" else "baseline"
            X[f"dec_{s}"] = X.groupby("date")[col].transform(lambda v: np.ceil(v.rank(method="first", pct=True) * 10).clip(1, 10))
        print(f"\n======== {h} sessions: hit = beat SPY, close-to-close drawdown shallower than {CAP[h]:.0%}, intraday low never {STOP:.0%} below entry")
        c = X.dropna(subset=[f"p_{h}"])
        print("calibration (isotonic from earlier years only):")
        print(c.groupby(pd.cut(c[f"p_{h}"], [0, .05, .1, .15, .2, .25, .3, .4, 1])).agg(rows=(f"hit_{h}", "size"), printed=(f"p_{h}", "mean"), realized=(f"hit_{h}", "mean")).round(3).to_string())
        agg = dict(rows=(f"hit_{h}", "size"), hit=(f"hit_{h}", "mean"), med_excess=(f"xs_{h}", "median"), med_drawdown=(f"ptt_{h}", "median"),
                   stopped=(f"stop_{h}", lambda v: (v <= -STOP).mean()), med_beta_l1=("beta_l1_252", "median"), steady=("steady", "mean"))
        for per, g in X.groupby("per"):
            print(f"\n{per}: model deciles within each date (10 = best)"); print(g.groupby("dec_raw").agg(**agg).round(3).to_string())
            top = pd.DataFrame({"model top decile": g[g.dec_raw == 10].agg({f"hit_{h}": "mean", f"xs_{h}": "median", f"ptt_{h}": "median", f"stop_{h}": lambda v: (v <= -STOP).mean()}),
                                "baseline top decile": g[g.dec_baseline == 10].agg({f"hit_{h}": "mean", f"xs_{h}": "median", f"ptt_{h}": "median", f"stop_{h}": lambda v: (v <= -STOP).mean()}),
                                "steady flag": g[g.steady == 1].agg({f"hit_{h}": "mean", f"xs_{h}": "median", f"ptt_{h}": "median", f"stop_{h}": lambda v: (v <= -STOP).mean()}),
                                "all names": g.agg({f"hit_{h}": "mean", f"xs_{h}": "median", f"ptt_{h}": "median", f"stop_{h}": lambda v: (v <= -STOP).mean()})})
            top.index = ["hit", "median excess", "median drawdown", "stopped"]; print(top.round(3).to_string())
        m = X[(X.ticker == "MSI") & (X.date.dt.year == 2024)]
        print(f"\nMSI 2024 by month ({h}d): model decile, calibrated P, realized hit")
        print(m.groupby(m.date.dt.to_period("M")).agg(decile=("dec_raw", "mean"), P=(f"p_{h}", "mean"), hit=(f"hit_{h}", "mean")).round(2).T.to_string())


if __name__ == "__main__":
    main()
