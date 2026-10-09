"""Direct test of the stated thesis, condition by condition, against the setup the
data itself prefers. Each ladder adds one condition at a time.

For every rung: names per week, hit rates (y21 / y63 / y_hold), regime-neutral lift
(hits / same-week universe expectation), median forward 63-session return and max
drawdown, and the lift in each era.
"""
import numpy as np
import pandas as pd

from common import DATA_DIR, ERAS, RES_DIR, add_week_base, adj_lift, load_panel

THESIS = [
    ("price rising (roc63 > 0)", lambda d: d["roc63"] > 0),
    ("+ volatility compressing (vroc21 < 0)", lambda d: d["vroc21"] < 0),
    ("+ relative volume rising (rv10_126 > 0)", lambda d: d["rv10_126"] > 0),
    ("+ drawdown below universe median (q_mdd63 < 0.5)", lambda d: d["q_mdd63"] < 0.5),
    ("+ price ROC accelerating (acc21 > 0)", lambda d: d["acc21"] > 0),
    ("+ compression accelerating (vacc21 < 0)", lambda d: d["vacc21"] < 0),
    ("+ volume ROC rising (vlroc21 > 0)", lambda d: d["vlroc21"] > 0),
]
THESIS_STRONG = [
    ("price ROC top quintile (q_roc63 >= 0.8)", lambda d: d["q_roc63"] >= 0.8),
    ("+ vol ROC bottom quintile (q_vroc21 <= 0.2)", lambda d: d["q_vroc21"] <= 0.2),
    ("+ rel volume top 40% (q_rv10_126 >= 0.6)", lambda d: d["q_rv10_126"] >= 0.6),
    ("+ drawdown bottom 40% (q_mdd63 <= 0.4)", lambda d: d["q_mdd63"] <= 0.4),
]
CHART_NOW = [  # the reference chart as it looked in August: staircase already running
    ("in a 63-bar staircase (+10%, maxDD <= gain/3)", lambda d: d["stair_b63"] == 1),
    ("+ volatility compressing (vroc21 < 0)", lambda d: d["vroc21"] < 0),
    ("+ relative volume rising (rv10_126 > 0)", lambda d: d["rv10_126"] > 0),
    ("+ holding above the prior base (bo_hold63 > 0)", lambda d: d["bo_hold63"] > 0),
]
QUIET_BASE = [  # what the data prefers
    ("low volatility (q_mar63 <= 0.2)", lambda d: d["q_mar63"] <= 0.2),
    ("+ shallow 6-month drawdown (q_mdd126 <= 0.3)", lambda d: d["q_mdd126"] <= 0.3),
    ("+ lagging 6-month return (q_roc126 <= 0.4)", lambda d: d["q_roc126"] <= 0.4),
    ("+ within 10% of 52-week high (ddh252 > -0.105)", lambda d: d["ddh252"] > -0.105),
]


def rung_stats(d, m, label):
    s = d[m]
    weeks = d["date"].nunique()
    row = dict(rung=label, per_week=len(s) / weeks, n=len(s),
               hit_y21=s["y21"].mean(), hit_y63=s["y63"].mean(), hit_hold=s["y_hold"].mean(),
               lift_y63=adj_lift(s["y63"], s["_wb"]), lift_y21=adj_lift(s["y21"], s["_wb21"]),
               med_fr63=s["fr63"].median(), med_fdd63=s["fdd63"].median(), p_up63=(s["fr63"] > 0).mean())
    for a, b in ERAS:
        e = s[(s["date"] >= a) & (s["date"] <= b)]
        row[f"lift_{a[:4]}"] = adj_lift(e["y63"], e["_wb"]) if len(e) >= 200 else np.nan
    return row


def ladder(d, steps, name):
    m = np.ones(len(d), bool)
    rows = []
    for label, fn in steps:
        m &= fn(d).fillna(False).to_numpy()
        r = rung_stats(d, m, label)
        r["ladder"] = name
        rows.append(r)
    return rows


def main():
    cols = ["date", "ticker", "fr21", "fdd21", "fr42", "fdd42", "fr63", "fdd63", "roc63", "vroc21", "rv10_126",
            "q_mdd63", "acc21", "vacc21", "vlroc21", "q_roc63", "q_vroc21", "q_rv10_126", "stair_b63", "bo_hold63",
            "q_mar63", "q_mdd126", "q_roc126", "ddh252"]
    d = load_panel(columns=cols)
    d = d[(d["date"] >= "2006-01-01") & d["y63"].notna()].reset_index(drop=True)
    d = add_week_base(d, "y63")
    d["_wb21"] = d.groupby("date")["y21"].transform("mean")
    rows = [rung_stats(d, np.ones(len(d), bool), "universe (all names, all weeks)") | {"ladder": "base"}]
    rows += ladder(d, THESIS, "thesis")
    rows += ladder(d, THESIS_STRONG, "thesis_strong")
    rows += ladder(d, CHART_NOW, "chart_in_august")
    rows += ladder(d, QUIET_BASE, "quiet_base")
    out = pd.DataFrame(rows)
    out.to_csv(f"{RES_DIR}/hypothesis_ladders.csv", index=False)
    pd.set_option("display.width", 260)
    pd.set_option("display.max_colwidth", 55)
    show = ["ladder", "rung", "per_week", "hit_y21", "hit_y63", "hit_hold", "lift_y63", "lift_y21", "med_fr63",
            "med_fdd63", "p_up63", "lift_2006", "lift_2013", "lift_2020"]
    print(out[show].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
