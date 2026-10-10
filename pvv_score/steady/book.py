"""Book test of the path-risk product. Each month-start, inside a beta band (L1 beta, 252 sessions), hold the 20 names with the
lowest predicted path risk (out-of-sample) for h sessions, equal weight, one new sleeve per month; compare with every name in the
same band (same beta, no selection), the 20 HIGHEST path-risk names in the band, and SPY. Non-Gaussian metrics only."""
import numpy as np, pandas as pd
from ..config import CACHE_DIR
from .portfolio import metrics
from .outcomes import HORIZONS, STOP
D = CACHE_DIR / "steady"
BANDS = {"beta 0.8-1.5": (0.8, 1.5), "beta 1.5+": (1.5, 9), "beta under 0.8": (-9, 0.8)}


def sleeves(O, C, Rt, h, lo, hi, pick):
    dates = np.sort(O.dropna(subset=[f"ps_{h}"]).date.unique()); starts = pd.Series(dates).groupby(pd.Series(dates).dt.to_period("M")).min().tolist()
    idx = C.index; out = []; pos = []
    for d in starts:
        s = O[(O.date == d) & O.beta_l1_252.between(lo, hi)].dropna(subset=[f"ps_{h}"])
        s = s.nlargest(20, f"ps_{h}") if pick == "low risk" else s.nsmallest(20, f"ps_{h}") if pick == "high risk" else s
        tk = [t for t in s.ticker if t in C.columns]; days = idx[idx > d][:h]
        if tk and len(days): out.append(Rt.loc[days, tk].mean(axis=1)); pos.append(s)
    return pd.concat(out, axis=1).mean(axis=1), pd.concat(pos)


def main():
    O = pd.read_parquet(D / "pathrisk_oos.parquet"); O = O[O.date >= "2018-01-01"]
    C = pd.read_parquet(CACHE_DIR / "Close.parquet").sort_index(); Rt = C.pct_change(fill_method=None); spy = Rt.SPY
    pd.set_option("display.width", 240)
    for h in HORIZONS:
        for band, (lo, hi) in BANDS.items():
            res = {}; P = {}
            for pick in ("low risk", "all in band", "high risk"):
                res[pick], P[pick] = sleeves(O, C, Rt, h, lo, hi, pick)
            print(f"\n======== {h}-session sleeves, {band}")
            for start, end, lab in [("2018-01-01", "2023-12-31", "2018-2023"), ("2024-01-01", "2026-12-31", "2024 to date")]:
                M = pd.DataFrame({k: metrics(v, spy, start, end) for k, v in res.items()} | {"SPY": metrics(spy, spy, start, end)}).T
                print(f"-- {lab}"); print(M.round(3).to_string())
            for pick in ("low risk", "all in band", "high risk"):
                X = P[pick].dropna(subset=[f"stop_{h}"])
                print(f"   positions, {pick:<12}: stopped at -8% intraday {(X[f'stop_{h}'] <= -STOP).mean():.1%}   median excess {X[f'xs_{h}'].median():+.2%}   "
                      f"median path drawdown {X[f'ptt_{h}'].median():.1%}   median L1 beta {X.beta_l1_252.median():.2f}")


if __name__ == "__main__":
    main()
