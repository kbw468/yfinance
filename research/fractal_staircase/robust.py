"""Robustness of the quiet-base setup, and which confirmers add to it.

1. Neighbourhood grid: every combination of nearby thresholds. A real effect
   shows up across the neighbourhood, not at one tuned point.
2. Inside the base setup: add each rate-of-change confirmer one at a time
   (price, volatility, volume, their accelerations) and read the lift change.
3. By sector, by size, by year.
Lifts are regime-neutral (hits / same-week universe expectation).
"""
import itertools

import numpy as np
import pandas as pd

from common import ERAS, RES_DIR, add_week_base, adj_lift, load_panel

GRID = dict(vol=(0.1, 0.2, 0.3, 0.4), dd=(0.2, 0.3, 0.4, 0.5), lag=(0.3, 0.4, 0.5, 0.6, 1.0), hi=(-0.05, -0.105, -0.15, -0.25, -9))

CONFIRMERS = {
    "price ROC 21d > 0": lambda d: d["roc21"] > 0,
    "price ROC 21d < 0": lambda d: d["roc21"] < 0,
    "price ROC accelerating (acc21 > 0)": lambda d: d["acc21"] > 0,
    "price ROC decelerating (acc21 < 0)": lambda d: d["acc21"] < 0,
    "vol compressing (vroc21 < 0)": lambda d: d["vroc21"] < 0,
    "vol expanding (vroc21 > 0)": lambda d: d["vroc21"] > 0,
    "compression accelerating (vacc21 < 0)": lambda d: d["vacc21"] < 0,
    "range compressing 21v252 (rc21_252 < 0)": lambda d: d["rc21_252"] < 0,
    "rel volume up (rv10_126 > 0)": lambda d: d["rv10_126"] > 0,
    "rel volume down (rv10_126 < 0)": lambda d: d["rv10_126"] < 0,
    "volume ROC up (vlroc21 > 0)": lambda d: d["vlroc21"] > 0,
    "volume ROC down (vlroc21 < 0)": lambda d: d["vlroc21"] < 0,
    "up-volume > down-volume 21d (udv21 > 0)": lambda d: d["udv21"] > 0,
    "price up & vol down & volume up": lambda d: (d["roc21"] > 0) & (d["vroc21"] < 0) & (d["vlroc21"] > 0),
    "drift-to-noise top half (q_dn63 >= 0.5)": lambda d: d["q_dn63"] >= 0.5,
    "efficiency 63 top half (q_er63 >= 0.5)": lambda d: d["q_er63"] >= 0.5,
    "monotone 63 (kt63 > 0.3)": lambda d: d["kt63"] > 0.3,
    "holding above prior base (bo_hold63 > 0)": lambda d: d["bo_hold63"] > 0,
    "staircase memory (mem_y63_long > 0.1)": lambda d: d["mem_y63_long"] > 0.1,
    "industry momentum rising (ind_d_roc63 > 0)": lambda d: d["ind_d_roc63"] > 0,
    "high co-movement (q_co126 >= 0.6)": lambda d: d["q_co126"] >= 0.6,
    "defensive on down days (q_rel_dn126 >= 0.6)": lambda d: d["q_rel_dn126"] >= 0.6,
}


def base_mask(d, vol=0.2, dd=0.3, lag=0.4, hi=-0.105):
    return ((d["q_mar63"] <= vol) & (d["q_mdd126"] <= dd) & (d["q_roc126"] <= lag) & (d["ddh252"] > hi)).to_numpy()


def stats(d, m):
    s = d[m]
    out = dict(n=int(m.sum()), per_week=m.sum() / d["date"].nunique(), hit=s["y63"].mean(),
               lift=adj_lift(s["y63"], s["_wb"]), lift21=adj_lift(s["y21"], s["_wb21"]),
               med_fr63=s["fr63"].median(), med_fdd63=s["fdd63"].median(), p_up63=(s["fr63"] > 0).mean())
    for a, b in ERAS:
        e = s[(s["date"] >= a) & (s["date"] <= b)]
        out[f"lift_{a[:4]}"] = adj_lift(e["y63"], e["_wb"]) if len(e) >= 150 else np.nan
    return out


def main():
    cols = ["date", "ticker", "sector", "mcap_bucket", "fr21", "fdd21", "fr42", "fdd42", "fr63", "fdd63", "q_mar63",
            "q_mdd126", "q_roc126", "ddh252", "roc21", "acc21", "vroc21", "vacc21", "rc21_252", "rv10_126", "vlroc21",
            "udv21", "q_dn63", "q_er63", "kt63", "bo_hold63", "mem_y63_long", "ind_d_roc63", "q_co126", "q_rel_dn126",
            "q_ldv63"]
    d = load_panel(columns=cols)
    d = d[(d["date"] >= "2006-01-01") & d["y63"].notna()].reset_index(drop=True)
    d = add_week_base(d, "y63")
    d["_wb21"] = d.groupby("date")["y21"].transform("mean")
    pd.set_option("display.width", 250)

    rows = []
    for vol, dd, lag, hi in itertools.product(*GRID.values()):
        m = base_mask(d, vol, dd, lag, hi)
        if m.sum() < 1000:
            continue
        rows.append(dict(vol=vol, dd=dd, lag=lag, hi=hi, **stats(d, m)))
    g = pd.DataFrame(rows)
    g.to_csv(f"{RES_DIR}/quietbase_grid.csv", index=False)
    era = [c for c in g.columns if c.startswith("lift_20")]
    g["era_min"] = g[era].min(axis=1)
    print("grid cells:", len(g), " lift>1.15:", (g["lift"] > 1.15).mean().round(3),
          " lift>1.25:", (g["lift"] > 1.25).mean().round(3), " era_min>1.1:", (g["era_min"] > 1.1).mean().round(3))
    print(g.sort_values("lift", ascending=False).head(12).round(3).to_string(index=False))

    m0 = base_mask(d)
    ref = stats(d, m0)
    crow = [dict(confirmer="(quiet base alone)", **ref)]
    for name, fn in CONFIRMERS.items():
        m = m0 & fn(d).fillna(False).to_numpy()
        if m.sum() < 300:
            continue
        crow.append(dict(confirmer=name, **stats(d, m)))
    c = pd.DataFrame(crow)
    c["lift_change"] = c["lift"] / ref["lift"]
    c.to_csv(f"{RES_DIR}/quietbase_confirmers.csv", index=False)
    print("\n== confirmers added to the quiet base")
    print(c.round(3).to_string(index=False))

    out = []
    for key in ("sector", "mcap_bucket"):
        for val, idx in d.groupby(key, observed=True).indices.items():
            m = np.zeros(len(d), bool)
            m[idx] = True
            mm = m & m0
            if mm.sum() < 300:
                continue
            out.append(dict(split=key, value=val, base_lift_segment=adj_lift(d.loc[m, "y63"], d.loc[m, "_wb"]), **stats(d, mm)))
    d["size_pit"] = pd.cut(d["q_ldv63"], [0, .2, .4, .6, .8, 1.0], labels=["Q1 small", "Q2", "Q3", "Q4", "Q5 large"])
    for val, idx in d.groupby("size_pit", observed=True).indices.items():
        m = np.zeros(len(d), bool)
        m[idx] = True
        mm = m & m0
        if mm.sum() >= 300:
            out.append(dict(split="size_pit (dollar volume)", value=val, base_lift_segment=adj_lift(d.loc[m, "y63"], d.loc[m, "_wb"]), **stats(d, mm)))
    for yr, idx in d.groupby(d["date"].dt.year).indices.items():
        m = np.zeros(len(d), bool)
        m[idx] = True
        mm = m & m0
        if mm.sum() >= 50:
            out.append(dict(split="year", value=yr, base_lift_segment=1.0, **stats(d, mm)))
    s = pd.DataFrame(out)
    s.to_csv(f"{RES_DIR}/quietbase_segments.csv", index=False)
    print("\n== quiet base by segment")
    print(s[["split", "value", "n", "per_week", "hit", "lift", "base_lift_segment", "med_fr63", "med_fdd63", "p_up63"]].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
