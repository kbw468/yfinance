"""V-reversal breakout, defined with information available on the day it triggers:
  laggard   : at some point in the last 63 sessions the name's 63-session return ranked in the bottom 30% of the universe
  shock     : at some point in the last 63 sessions its worst single day (vs its own typical move) ranked in the bottom 30%
  breakout  : today's close is a new 63-session closing high, and there was no new 63-session high in the prior 42 sessions
  thrust    : 21-session return in the top 20% of the universe, efficiency and up-day share (21) in the top 30%
  below_52w : close still more than 2% under the 52-week high
Event = all of the above (variants drop one leg at a time). One event per name per 63 sessions."""
import numpy as np, pandas as pd
from ..config import CACHE_DIR
from .outcomes import HORIZONS, STOP, FLOOR, alpha_target
D = CACHE_DIR / "steady"
COLS = ["date", "ticker", "sector", "beta_l1_252", "off_high_63", "off_high_252", "cs_roc_63", "cs_worst_day_63", "cs_roc_21", "cs_eff_21", "cs_up_share_21",
        "roc_21", "xs_spy_63"] + [f"{k}_{h}" for h in HORIZONS for k in ("xs", "ptt", "stop", "ret", "hit")]


def flags(T: pd.DataFrame) -> pd.DataFrame:
    T = T.sort_values(["ticker", "date"]).reset_index(drop=True)
    g = T.groupby("ticker")
    T["laggard"] = g.cs_roc_63.transform(lambda s: s.rolling(63, min_periods=20).min()) <= 0.30
    T["shock"] = g.cs_worst_day_63.transform(lambda s: s.rolling(63, min_periods=20).min()) <= 0.30
    newhi = T.off_high_63 >= -1e-9
    T["fresh_breakout"] = newhi & ~g.off_high_63.transform(lambda s: (s >= -1e-9).shift(1).rolling(42, min_periods=1).max().fillna(0).astype(bool))
    T["thrust"] = (T.cs_roc_21 >= 0.8) & (T.cs_eff_21 >= 0.7) & (T.cs_up_share_21 >= 0.7)
    T["below_52w"] = T.off_high_252 < -0.02
    return T


def dedupe(E: pd.DataFrame, gap=63) -> pd.DataFrame:
    keep, last = [], {}
    for i, (t, d) in enumerate(zip(E.ticker.values, E.date.values)):
        if t in last and (d - last[t]) < np.timedelta64(int(gap * 1.45), "D"): continue
        last[t] = d; keep.append(i)
    return E.iloc[keep]


def spy_drawdown(dates) -> pd.Series:
    s = pd.read_parquet(CACHE_DIR / "Close.parquet", columns=["SPY"]).SPY
    dd252 = s / s.rolling(252, min_periods=60).max() - 1
    recent_min = dd252.rolling(42, min_periods=1).min()          # deepest SPY drawdown in the last ~2 months
    return recent_min.reindex(dates).values


def main():
    T = pd.read_parquet(D / "table.parquet", columns=COLS)
    T = alpha_target(T)
    T = flags(T)
    T["spy_dd_42"] = spy_drawdown(T.date)
    T["after_selloff"] = T.spy_dd_42 <= -0.10
    variants = {
        "all names (every day)": pd.Series(True, index=T.index),
        "fresh 63d breakout, any": T.fresh_breakout,
        "fresh breakout, NOT from laggard": T.fresh_breakout & ~T.laggard,
        "CORE: laggard + fresh breakout (9/9 cases)": T.fresh_breakout & T.laggard,
        "core + below 52w high (8/9)": T.fresh_breakout & T.laggard & T.below_52w,
        "core + shock (7/9)": T.fresh_breakout & T.laggard & T.shock,
        "core + thrust (5/9)": T.fresh_breakout & T.laggard & T.thrust,
        "FULL: all five legs (3/9)": T.fresh_breakout & T.laggard & T.shock & T.thrust & T.below_52w,
    }
    rows = []
    for ctx, cm in (("all markets", pd.Series(True, index=T.index)), ("within 2 months of a 10%+ SPY drawdown", T.after_selloff), ("other times", ~T.after_selloff)):
        for v, m in variants.items():
            E = T[m & cm & (T.date < "2026-01-01")]
            if not v.startswith("all names"): E = dedupe(E.sort_values(["ticker", "date"]))
            for h in HORIZONS:
                X = E.dropna(subset=[f"xs_{h}"])
                if len(X) < 30: continue
                rows.append({"context": ctx, "variant": v, "h": h, "events": len(X), "names": X.ticker.nunique(),
                             "mean_excess": X[f"xs_{h}"].mean(), "median_excess": X[f"xs_{h}"].median(), "beat_spy": (X[f"xs_{h}"] > 0).mean(),
                             "p75_excess": X[f"xs_{h}"].quantile(.75), "p25_excess": X[f"xs_{h}"].quantile(.25),
                             "median_drawdown": X[f"ptt_{h}"].median(), "stopped_8pct": (X[f"stop_{h}"] <= -STOP).mean(), "alpha_per_pain_hit": X[f"ah_{h}"].mean()})
    R = pd.DataFrame(rows); R.to_csv(D / "vbreak_test.csv", index=False)
    pd.set_option("display.width", 260, "display.max_rows", 300)
    for ctx in R.context.unique():
        for h in HORIZONS:
            print(f"\n=== {ctx}, {h} sessions, events 2015-2025 (one per name per 63 sessions)")
            print(R[(R.context == ctx) & (R.h == h)].drop(columns=["context", "h"]).set_index("variant").round(3).to_string())
    core = T[T.fresh_breakout & T.laggard & (T.date < "2026-01-01")]; core = dedupe(core.sort_values(["ticker", "date"])).dropna(subset=["xs_42"])
    print("\nCORE by year, 42 sessions:"); print(core.groupby(core.date.dt.year).agg(events=("xs_42", "size"), mean_excess=("xs_42", "mean"), median_excess=("xs_42", "median"), beat=("xs_42", lambda v: (v > 0).mean()), stopped=("stop_42", lambda v: (v <= -STOP).mean())).round(3).to_string())
    T.to_parquet(D / "vbreak_flags.parquet")


if __name__ == "__main__":
    main()
