"""Conditional single-feature study (regime-neutral).

Within a starting state (e.g. the stock is ALREADY in a staircase, like the reference
chart in August), which rates of change separate the names that keep going from the
ones that stop? Each continuous feature is split into quintiles of its universe rank;
discrete features (flags, counts) by raw value.

Lift is regime-neutral: hits / sum of each row's same-week universe base rate, so a
bucket that merely concentrates in bull weeks gets no credit.
"""
import sys

import numpy as np
import pandas as pd

from common import DATA_DIR, ERAS, RES_DIR, add_week_base, adj_lift, load_panel

STATES = {
    "all": lambda d: d["fr63"].notna(),
    "in_stair63": lambda d: d["stair_b63"] == 1,
    "in_stair21": lambda d: d["stair_b21"] == 1,
    "post_breakout63": lambda d: (d["bo_hold63"] > 0) & (d["bo_gain63"] > 0),
    "low_vol_third": lambda d: d["q_mar63"] <= 1 / 3,
    "quiet_uptrend": lambda d: (d["q_mar63"] <= 0.5) & (d["roc63"] > 0) & (d["er63"] > 0.1),
}
DISCRETE = ("stair_b21", "stair_b63", "stair_b126", "stair_scales", "roc_align")


def buckets(sub, c):
    raw = c[2:] if c.startswith("q_") else None
    if raw in DISCRETE:
        return sub[raw].round(0)
    return np.floor(np.minimum(sub[c] * 5, 4.999))


def main(target="y63"):
    import pyarrow.parquet as pq
    names = pq.read_schema(f"{DATA_DIR}/panel.parquet").names
    need = ["date", "ticker", "fr21", "fdd21", "fr42", "fdd42", "fr63", "fdd63", "bo_hold63", "bo_gain63",
            "roc63", "er63", *DISCRETE] + [c for c in names if c.startswith(("q_", "sq_", "iq_"))]
    df = load_panel(columns=list(dict.fromkeys(need)))
    df = df[(df["date"] >= "2006-01-01") & df[target].notna()].reset_index(drop=True)
    df = add_week_base(df, target)
    qcols = [c for c in df.columns if c.startswith(("q_", "sq_", "iq_"))]
    rows = []
    for sname, fn in STATES.items():
        sub = df[fn(df).fillna(False).to_numpy()]
        s_adj = adj_lift(sub[target], sub["_wb"])
        print(f"{sname}: n={len(sub)} hit={sub[target].mean():.4f} regime-neutral lift={s_adj:.3f}", flush=True)
        for c in qcols:
            if sub[c].notna().sum() < 2000:
                continue
            b = buckets(sub, c)
            ok = b.notna()
            g = pd.DataFrame({"b": b[ok], "y": sub.loc[ok, target], "wb": sub.loc[ok, "_wb"], "date": sub.loc[ok, "date"]})
            agg = g.groupby("b").agg(n=("y", "size"), hits=("y", "sum"), exp=("wb", "sum"))
            agg = agg[agg["n"] >= 400]
            if len(agg) < 2:
                continue
            agg["lift"] = agg["hits"] / agg["exp"]
            best = agg["lift"].idxmax()
            el = []
            for a, bnd in ERAS:
                e = g[(g["date"] >= a) & (g["date"] <= bnd) & (g["b"] == best)]
                el.append(adj_lift(e["y"], e["wb"]) if len(e) >= 150 else np.nan)
            rows.append(dict(state=sname, feature=c, best_bucket=best, n=int(agg.loc[best, "n"]),
                             hit=agg.loc[best, "hits"] / agg.loc[best, "n"], lift=agg.loc[best, "lift"],
                             lift_vs_state=agg.loc[best, "lift"] / s_adj, era_min=np.nanmin(el) if any(~np.isnan(el)) else np.nan,
                             lift_low=agg["lift"].iloc[0], lift_high=agg["lift"].iloc[-1]))
    out = pd.DataFrame(rows)
    out.to_csv(f"{RES_DIR}/conditional_{target}.csv", index=False)
    pd.set_option("display.width", 250)
    for sname in STATES:
        t = out[(out["state"] == sname) & (out["era_min"] > 1.0)].sort_values("lift", ascending=False)
        print(f"\n== {sname}: top era-stable buckets (lift is regime-neutral, vs same-week universe)")
        print(t.head(18).round(3).to_string(index=False))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "y63")
