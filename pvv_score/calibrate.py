"""Calibrate the actionability score into probabilities using the out-of-sample walk-forward predictions.

For each score vigintile (20 bins) over the OOS history (recent window primary, full window reported), the realised
frequency that the name's forward 42d Sharpe landed in the top half / top quartile of the universe that day, and the
mean forward 42d Sharpe excess over the universe mean. Today's names inherit the probability of their bin.
"""
import numpy as np
import pandas as pd
from .config import CACHE_DIR, RESULTS_DIR, RECENT_START
from .run_eval import load_research

BINS = 20


def main():
    df = load_research()[["date", "ticker", "xs_spy_sharpe_42", "xs_spy_sharpe_63"]]
    st = pd.read_parquet(CACHE_DIR / "composite_state_oos_preds.parquet")[["date", "ticker", "comp_bucketed"]].rename(columns={"comp_bucketed": "state"})
    lv = pd.read_parquet(CACHE_DIR / "composite_level_oos_preds.parquet")[["date", "ticker", "comp_bucketed"]].rename(columns={"comp_bucketed": "level"})
    d = df.merge(st, on=["date", "ticker"]).merge(lv, on=["date", "ticker"]).dropna()
    for c in ["state", "level"]:
        d[c + "_p"] = d.groupby("date")[c].rank(pct=True)
    d["avg_p"] = (d.state_p + d.level_p) / 2
    for h in (42, 63):
        y = d[f"xs_spy_sharpe_{h}"]
        d[f"top_half_{h}"] = (y > y.groupby(d.date).transform("median")).astype(float)
        d[f"top_q_{h}"] = (y > y.groupby(d.date).transform(lambda s: s.quantile(0.75))).astype(float)
        d[f"xs_mean_{h}"] = y - y.groupby(d.date).transform("mean")
    d["bin"] = np.ceil(d.avg_p * BINS).clip(1, BINS).astype(int)
    tabs = {}
    for win, start in [("recent", RECENT_START), ("full", None)]:
        s = d if start is None else d[d.date >= start]
        g = s.groupby("bin").agg(n=("avg_p", "size"), p_top_half_42d=("top_half_42", "mean"), p_top_quartile_42d=("top_q_42", "mean"),
                                 xs_sharpe_42_vs_universe=("xs_mean_42", "mean"), p_top_half_63d=("top_half_63", "mean"),
                                 p_top_quartile_63d=("top_q_63", "mean"), xs_sharpe_63_vs_universe=("xs_mean_63", "mean"))
        tabs[win] = g
        g.to_csv(RESULTS_DIR / f"calibration_{win}.csv")
    cal = tabs["recent"]
    u = pd.read_csv(RESULTS_DIR / "universe_scores.csv")
    b = np.ceil(u.avg_score * BINS).clip(1, BINS)
    for c in ["p_top_half_42d", "p_top_quartile_42d", "xs_sharpe_42_vs_universe", "p_top_half_63d", "p_top_quartile_63d"]:
        u[c] = b.map(cal[c]).round(3)
    u.to_csv(RESULTS_DIR / "universe_scores.csv", index=False)
    pd.set_option("display.width", 200)
    print("recent-window calibration by score vigintile (20 = best):")
    print(cal.round(3).to_string())
    print("\nfull-window:")
    print(tabs["full"][["n", "p_top_half_42d", "p_top_quartile_42d", "xs_sharpe_42_vs_universe"]].round(3).to_string())


if __name__ == "__main__":
    main()
