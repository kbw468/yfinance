"""Point out anything that looks like MSI 2024.

Two forms:
  pre-run   MSI's base before the run (Jun-Oct 2023): the JNJ/MSI base profile
            (63-day range tightest 10%, volatility bottom 10%, 6-month drawdown bottom 15%,
            within 10% of the 52-week high, 6-month return upper half), not pinned.
  in-run    MSI's run itself: trailing 126 sessions up 20%+ with a worst pullback <= 6%,
            volatility bottom 20%, within 5% of the 52-week high.
Shape match: mean absolute gap between min-max scaled log paths and MSI's own paths
(Apr 18 -> Oct 16 2024, 126 sessions; Oct 27 2023 -> Nov 8 2024, 260 sessions). 0 = identical.
The backtest odds of the in-run state come from the weekly panel (regime-neutral lift).
"""
import os

import numpy as np
import pandas as pd

from common import DATA_DIR, RES_DIR, add_targets, adj_lift

T126 = ("2024-04-18", "2024-10-16")
T260 = ("2023-10-27", "2024-11-08")


def scaled(x):
    lo, hi = np.nanmin(x), np.nanmax(x)
    return (x - lo) / (hi - lo) if hi > lo else np.full(len(x), np.nan)


def template(px, a, b):
    m = px[(px["ticker"] == "MSI") & (px["date"] >= a) & (px["date"] <= b)].sort_values("date")
    return scaled(np.log(m["close"].to_numpy(float)))


def shape_gap(x, tmpl):
    n = len(tmpl)
    if len(x) < n:
        return np.nan
    w = scaled(x[-n:])
    return float(np.nanmean(np.abs(w - tmpl)))


def odds(panel):
    """Forward outcomes after the in-run state, from the weekly history."""
    from prototype import forward126
    d = panel.merge(forward126(), on=["date", "ticker"], how="left")
    d = d[d["date"] >= "2006-01-01"]
    d["proto"] = ((d["fr126"] >= np.log(1.25)) & (d["fdd126"] <= 0.06)).astype(float).where(d["fr126"].notna())
    for t in ("proto", "y63", "y21"):
        d["_wb_" + t] = d.groupby("date")[t].transform("mean")
    state = ((d["roc126"] >= np.log(1.20)) & (d["mdd126"] <= 0.06) & (d["q_mar63"] <= 0.2) & (d["ddh252"] > np.log(0.95))).to_numpy()
    first = (d["date"] <= "2016-12-31").to_numpy()
    rows = []
    for t, label in (("proto", "MSI-grade next 126 sessions"), ("y63", "63-session staircase"), ("y21", "21-session staircase")):
        lab = d[t].notna().to_numpy()
        m = state & lab
        rows.append(dict(outcome=label, weeks=int(m.sum()), hit=d.loc[m, t].mean(), lift=adj_lift(d.loc[m, t], d.loc[m, "_wb_" + t]),
                         lift_2006_16=adj_lift(d.loc[m & first, t], d.loc[m & first, "_wb_" + t]),
                         lift_2017_26=adj_lift(d.loc[m & ~first, t], d.loc[m & ~first, "_wb_" + t]),
                         base=d.loc[lab, t].mean()))
    m = state & d["fr63"].notna().to_numpy()
    rows.append(dict(outcome="next 63 sessions: up / median return / median worst pullback", weeks=int(m.sum()),
                     hit=(d.loc[m, "fr63"] > 0).mean(), lift=np.exp(d.loc[m, "fr63"].median()) - 1,
                     lift_2006_16=d.loc[m, "fdd63"].median(), lift_2017_26=np.nan, base=np.nan))
    return pd.DataFrame(rows)


def main():
    px = pd.read_parquet(f"{DATA_DIR}/ohlcv.parquet", columns=["date", "ticker", "close"])
    px["date"] = pd.to_datetime(px["date"]).astype("datetime64[ns]")
    t126, t260 = template(px, *T126), template(px, *T260)
    last_date = px["date"].max()
    rows = []
    for tkr, g in px.groupby("ticker"):
        g = g.sort_values("date")
        if g["date"].iloc[-1] != last_date or len(g) < 300:
            continue
        x = np.log(g["close"].to_numpy(float))
        w = x[-127:]
        rows.append(dict(ticker=tkr, gain126=np.exp(w[-1] - w[0]) - 1, worst_pullback126=1 - np.exp(-(np.maximum.accumulate(w) - w).max()),
                         eff126=(w[-1] - w[0]) / np.abs(np.diff(w)).sum(), gap126=shape_gap(x, t126), gap260=shape_gap(x, t260)))
    sh = pd.DataFrame(rows)
    cols = ["date", "ticker", "sector", "industry", "mcap", "q_mar63", "q_rng63", "q_mdd126", "q_roc126", "ddh252", "roc126", "mdd126",
            "fr21", "fdd21", "fr42", "fdd42", "fr63", "fdd63"]
    panel = pd.read_parquet(f"{DATA_DIR}/panel.parquet", columns=cols)
    panel = add_targets(panel)
    last = panel[panel["date"] == panel["date"].max()].merge(sh, on="ticker", how="left")
    last["pinned"] = (last["q_mar63"] <= 0.01) & (last["q_rng63"] <= 0.01)
    last["pre_run"] = ((last["q_rng63"] <= 0.10) & (last["q_mar63"] <= 0.10) & (last["q_mdd126"] <= 0.15) & (last["ddh252"] > -0.10)
                       & (last["q_roc126"] >= 0.5) & ~last["pinned"])
    last["in_run"] = (last["gain126"] >= 0.20) & (last["worst_pullback126"] <= 0.06) & (last["q_mar63"] <= 0.2) & (last["ddh252"] > np.log(0.95))
    last.to_csv(f"{RES_DIR}/msi_like_{last['date'].iloc[0]:%Y-%m-%d}.csv", index=False)
    od = odds(panel)
    od.to_csv(f"{RES_DIR}/msi_like_odds.csv", index=False)
    pd.set_option("display.width", 250)
    show = ["ticker", "sector", "industry", "mcap", "gain126", "worst_pullback126", "eff126", "gap126", "gap260", "q_mar63", "q_rng63", "q_mdd126", "q_roc126", "ddh252"]
    print("scan date", last["date"].iloc[0].date(), "| MSI's own 126-session gap to itself = 0; universe median gap126 =", round(last["gap126"].median(), 3))
    print("\nIN-RUN (running like MSI 2024):")
    print(last[last["in_run"]].sort_values("gap126")[show].round(3).to_string(index=False))
    print("\nPRE-RUN (MSI's base before the run):")
    print(last[last["pre_run"]].sort_values("mcap", ascending=False)[show].round(3).to_string(index=False))
    print("\nclosest 126-session shapes to MSI 2024 (any name, worst pullback <= 8%):")
    print(last[last["worst_pullback126"] <= 0.08].sort_values("gap126")[show].head(15).round(3).to_string(index=False))
    print("\nodds after the in-run state:")
    print(od.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
