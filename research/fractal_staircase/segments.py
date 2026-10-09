"""Stage 5: where does the behaviour live, and does the signal hold everywhere?

Using out-of-sample model scores:
* base rate, regime-neutral base lift, and top-decile hit rate by sector, industry,
  size (point-in-time dollar-volume quintile and current market-cap bucket),
  market regime, calendar month, year
* per-sector single-feature leaders (regime-neutral): which rate of change matters
  most inside each sector
"""
import sys

import numpy as np
import pandas as pd

from common import DATA_DIR, RES_DIR, add_week_base, adj_lift, load_panel


def seg_table(o, key, target, min_n=2000):
    g = o.groupby(key, observed=True)
    top = o[o["pct"] >= 0.9]
    gt = top.groupby(key, observed=True)
    t = pd.DataFrame({
        "n": g.size(),
        "base_hit": g[target].mean(),
        "base_lift_neutral": g.apply(lambda d: adj_lift(d[target], d["_wb"]), include_groups=False),
        "top_n": gt.size(),
        "top_hit": gt[target].mean(),
        "top_lift_neutral": gt.apply(lambda d: adj_lift(d[target], d["_wb"]), include_groups=False),
        "top_med_fr63": gt["fr63"].median(),
        "top_med_fdd63": gt["fdd63"].median(),
    })
    return t[t["n"] >= min_n].sort_values("top_lift_neutral", ascending=False)


def main(tag="rank", target="y63"):
    o = pd.read_parquet(f"{DATA_DIR}/oos_{tag}.parquet")
    keep = ["date", "ticker", "mkt_brd_up63", "mkt_vix_pct252", "mkt_spy_roc63", "q_ldv63", "mkt_med_vc21_126",
            "mkt_rsp_spy_roc63"]
    pnl = pd.read_parquet(f"{DATA_DIR}/panel.parquet", columns=keep)
    o = o.merge(pnl, on=["date", "ticker"], how="left")
    o = add_week_base(o, target)
    o["pct"] = o.groupby("date")["score"].rank(pct=True)
    o["year"] = o["date"].dt.year
    o["month"] = o["date"].dt.month
    o["size_pit"] = pd.cut(o["q_ldv63"], [0, .2, .4, .6, .8, 1.0], labels=["Q1 small", "Q2", "Q3", "Q4", "Q5 large"])
    o["breadth"] = pd.qcut(o["mkt_brd_up63"], 3, labels=["weak", "mid", "strong"])
    o["vix_state"] = pd.qcut(o["mkt_vix_pct252"], 3, labels=["low", "mid", "high"])
    o["spy_trend"] = np.where(o["mkt_spy_roc63"] > 0, "SPY up 63d", "SPY down 63d")
    o["mkt_volcomp"] = pd.qcut(o["mkt_med_vc21_126"], 3, labels=["compressing", "flat", "expanding"])
    o["equal_vs_cap"] = pd.qcut(o["mkt_rsp_spy_roc63"], 3, labels=["narrow", "mid", "broad"])
    pd.set_option("display.width", 250)
    for key in ["sector", "size_pit", "mcap_bucket", "breadth", "vix_state", "spy_trend", "mkt_volcomp",
                "equal_vs_cap", "month", "year"]:
        t = seg_table(o, key, target)
        t.to_csv(f"{RES_DIR}/seg_{key}_{tag}_{target}.csv")
        print("\n==", key)
        print(t.round(3).to_string())
    t = seg_table(o, "industry", target, min_n=3000)
    t.to_csv(f"{RES_DIR}/seg_industry_{tag}_{target}.csv")
    print("\n== industry (top 25 / bottom 10 by top-decile neutral lift)")
    print(t.head(25).round(3).to_string())
    print(t.tail(10).round(3).to_string())


def sector_leaders(target="y63"):
    import pyarrow.parquet as pq
    names = pq.read_schema(f"{DATA_DIR}/panel.parquet").names
    qcols = [c for c in names if c.startswith("q_") and c[2:] not in ("stair_b21", "stair_b63", "stair_b126", "stair_scales", "roc_align")]
    df = load_panel(columns=["date", "ticker", "sector", "fr21", "fdd21", "fr42", "fdd42", "fr63", "fdd63"] + qcols)
    df = df[(df["date"] >= "2006-01-01") & df[target].notna()]
    df = add_week_base(df, target)
    rows = []
    for sec, d in df.groupby("sector"):
        for c in qcols:
            v = d[c]
            for side, m in (("top quintile", v >= 0.8), ("bottom quintile", v <= 0.2)):
                n = int(m.sum())
                if n < 1500:
                    continue
                half = d.loc[m, "date"] <= "2016-06-30"
                l1 = adj_lift(d.loc[m][half][target], d.loc[m][half]["_wb"])
                l2 = adj_lift(d.loc[m][~half][target], d.loc[m][~half]["_wb"])
                rows.append(dict(sector=sec, feature=c, side=side, n=n, hit=d.loc[m, target].mean(),
                                 lift=adj_lift(d.loc[m, target], d.loc[m, "_wb"]), lift_2006_16=l1, lift_2016_26=l2))
    lead = pd.DataFrame(rows)
    lead["lift_min_half"] = lead[["lift_2006_16", "lift_2016_26"]].min(axis=1)
    lead = lead.sort_values(["sector", "lift_min_half"], ascending=[True, False])
    lead.to_csv(f"{RES_DIR}/sector_leaders_{target}.csv", index=False)
    pd.set_option("display.width", 250)
    print(lead.groupby("sector").head(6).round(3).to_string(index=False))


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "leaders":
        sector_leaders(sys.argv[2] if len(sys.argv) > 2 else "y63")
    else:
        main(sys.argv[1] if len(sys.argv) > 1 else "rank", sys.argv[2] if len(sys.argv) > 2 else "y63")
