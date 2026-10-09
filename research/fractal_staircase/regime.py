"""When do staircases cluster? Time-series view of the weekly base rate.

For each week: the realised share of names that went on to staircase (y63, y21),
against market-state variables known that week. Spearman rank correlation across
weeks, tercile tables with quarter-block bootstrap ranges, and today's reading.
"""
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from common import RES_DIR, block_bootstrap_rate, load_panel

STATE = ["mkt_brd_up63", "mkt_brd_hi", "mkt_brd_eff", "mkt_med_roc21", "mkt_med_roc63", "mkt_med_vc21_126",
         "mkt_med_vroc21", "mkt_med_rv10_126", "mkt_med_mar21", "mkt_spy_roc21", "mkt_spy_roc63",
         "mkt_spy_ddh252", "mkt_rsp_spy_roc63", "mkt_iwm_spy_roc63", "mkt_vix", "mkt_vix_roc21",
         "mkt_vix_pct252", "mkt_vix_term", "mkt_vvix_vix", "stair_prev63", "stair_prev21"]


def main():
    df = load_panel()
    df = df[df["date"] >= "2006-01-01"]
    df["stair_prev63"] = df["stair_b63"]
    df["stair_prev21"] = df["stair_b21"]
    agg = {c: "median" for c in STATE if c.startswith("mkt_")}
    agg.update({"stair_prev63": "mean", "stair_prev21": "mean", "y63": "mean", "y21": "mean", "y_hold": "mean"})
    wk = df.groupby("date").agg(agg)
    wk["d_brd_up63"] = wk["mkt_brd_up63"] - wk["mkt_brd_up63"].shift(4)
    wk["d_stair_prev63"] = wk["stair_prev63"] - wk["stair_prev63"].shift(4)
    states = STATE + ["d_brd_up63", "d_stair_prev63"]
    lab = wk.dropna(subset=["y63"])
    rows = []
    for c in states:
        d = lab[[c, "y63", "y21"]].dropna()
        rho63 = spearmanr(d[c], d["y63"]).statistic
        rho21 = spearmanr(d[c], d["y21"]).statistic
        t = pd.qcut(d[c], 3, labels=["low", "mid", "high"])
        rates = d.groupby(t, observed=True)["y63"].mean()
        lo_ci = block_bootstrap_rate(d.index[t == "low"], d.loc[t == "low", "y63"].to_numpy())
        hi_ci = block_bootstrap_rate(d.index[t == "high"], d.loc[t == "high", "y63"].to_numpy())
        today = wk[c].iloc[-1]
        pct_today = (d[c] <= today).mean()
        rows.append(dict(state=c, rho_y63=rho63, rho_y21=rho21, y63_low=rates.get("low"), y63_mid=rates.get("mid"),
                         y63_high=rates.get("high"), low_ci=f"{lo_ci[0]:.3f}-{lo_ci[2]:.3f}",
                         high_ci=f"{hi_ci[0]:.3f}-{hi_ci[2]:.3f}", today=today, today_pctile=pct_today))
    out = pd.DataFrame(rows).sort_values("rho_y63", key=np.abs, ascending=False)
    out.to_csv(f"{RES_DIR}/regime.csv", index=False)
    wk.to_csv(f"{RES_DIR}/regime_weekly.csv")
    pd.set_option("display.width", 250)
    print("weekly base y63: median", round(lab["y63"].median(), 4), "range", round(lab["y63"].min(), 3), "-", round(lab["y63"].max(), 3))
    print(out.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
