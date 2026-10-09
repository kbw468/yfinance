"""Stage 1: every feature on its own (regime-neutral).

Each week's cross-section is bucketed into deciles of the feature's per-week rank
(flags and counts by raw value). Per decile: hit rate and regime-neutral lift =
hits / sum of the same rows' weekly universe base rate. Also reported: rank
monotonicity across deciles (Spearman), and the best decile's neutral lift in each
of three eras.
"""
import sys

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from common import DATA_DIR, ERAS, RES_DIR, add_week_base, family, load_panel

DISCRETE = ("stair_b21", "stair_b63", "stair_b126", "stair_scales", "roc_align")


def tie_heavy(d, col):
    """Counts / fractions where one value dominates: per-week ranks of ties shift
    with the week's mix, so bucket these on pooled raw values instead."""
    v = d[col].dropna()
    return col in DISCRETE or (len(v) and v.value_counts(normalize=True).iloc[0] > 0.15)


def table(d, col, target, raw_bins=None):
    if isinstance(raw_bins, str):          # flags / small counts: bucket on the raw value
        b = d[col].round(0)
    elif raw_bins is not None:
        b = pd.cut(d[col], raw_bins, labels=False, include_lowest=True)
    else:
        b = np.floor(np.minimum(d["q_" + col] * 10, 9.999))
    g = pd.DataFrame({"b": b, "y": d[target], "wb": d["_wb"]}).dropna().groupby("b")
    agg = g.agg(n=("y", "size"), hits=("y", "sum"), exp=("wb", "sum"))
    agg["hit"] = agg["hits"] / agg["n"]
    agg["lift"] = agg["hits"] / agg["exp"]
    return agg[agg["n"] >= 300]


def main(target="y63"):
    import pyarrow.parquet as pq
    names = pq.read_schema(f"{DATA_DIR}/panel.parquet").names
    feats = [c[2:] for c in names if c.startswith("q_") and c[2:] in names] + list(DISCRETE)
    need = ["date", "ticker", "fr21", "fdd21", "fr42", "fdd42", "fr63", "fdd63"] + \
        [c for c in names if c.startswith("q_")] + list(DISCRETE)
    df = load_panel(columns=list(dict.fromkeys(need)))
    df = df[(df["date"] >= "2006-01-01") & df[target].notna()].reset_index(drop=True)
    df = add_week_base(df, target)
    print(target, "rows", len(df), "base", round(df[target].mean(), 4), flush=True)
    rows = []
    raw_needed = [c for c in dict.fromkeys(feats) if c not in df.columns and c in names]
    if raw_needed:
        extra = pd.read_parquet(f"{DATA_DIR}/panel.parquet", columns=["date", "ticker"] + raw_needed)
        df = df.merge(extra, on=["date", "ticker"], how="left")
    era_masks = [((df["date"] >= a) & (df["date"] <= b)).to_numpy() for a, b in ERAS]
    for col in dict.fromkeys(feats):
        bins = None
        if col in DISCRETE:
            bins = "discrete"
        elif tie_heavy(df, col):
            edges = np.unique(np.nanquantile(df[col], np.linspace(0, 1, 11)))
            if len(edges) < 3:
                continue
            bins = edges
        t = table(df, col, target, bins)
        if len(t) < 2:
            continue
        best = t["lift"].idxmax()
        worst = t["lift"].idxmin()
        row = dict(feature=col, family=family(col), rho=spearmanr(np.asarray(t.index, float), t["lift"]).statistic,
                   best_bucket=best, best_lift=t.loc[best, "lift"], best_hit=t.loc[best, "hit"],
                   worst_bucket=worst, worst_lift=t.loc[worst, "lift"],
                   lift_low=t["lift"].iloc[0], lift_high=t["lift"].iloc[-1])
        for i, v in t["lift"].items():
            row[f"L{int(i)}"] = v
        for (a, _), m in zip(ERAS, era_masks):
            te = table(df[m], col, target, bins)
            row[f"best_{a[:4]}"] = te["lift"].get(best, np.nan)
        rows.append(row)
    out = pd.DataFrame(rows)
    out["era_min"] = out[[c for c in out.columns if c.startswith("best_20")]].min(axis=1)
    out = out.sort_values("best_lift", ascending=False)
    out.to_csv(f"{RES_DIR}/univariate_{target}.csv", index=False)
    pd.set_option("display.width", 250)
    show = ["feature", "family", "rho", "best_bucket", "best_lift", "best_hit", "worst_bucket", "worst_lift", "era_min"]
    print(out[show].head(40).round(3).to_string(index=False))
    print("\n-- by family: best era-stable feature")
    st = out[out["era_min"] > 1.0].sort_values("best_lift", ascending=False)
    print(st.groupby("family").head(4)[show].round(3).to_string(index=False))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "y63")
