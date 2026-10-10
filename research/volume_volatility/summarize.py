"""Render backtest CSVs into a markdown summary.

Usage: python summarize.py <results_dir>
"""
import sys
from pathlib import Path

import pandas as pd

H = [7, 14, 21, 42]


def cell(mean, t, pct=False):
    m = f"{mean * 100:+.2f}%" if pct else f"{mean:+.3f}"
    return f"{m} ({t:+.1f})"


def quintile_grid(df, factor, day, metric="RAX"):
    s = df[(df.factor == factor) & (df.day == day) & (df.metric == metric)]
    order = ["Q1", "Q2", "Q3", "Q4", "Q5", "BASE", "Q5-Q1"]
    lines = ["| bucket | " + " | ".join(f"{h}d" for h in H) + " |", "|---|" + "---|" * len(H)]
    for b in order:
        row = []
        for h in H:
            r = s[(s.bucket == b) & (s.h == h)].iloc[0]
            row.append(cell(r["mean"], r["t"], pct=(metric == "RET")))
        lines.append(f"| {b} | " + " | ".join(row) + " |")
    return "\n".join(lines)


def spread_table(df, metric):
    s = df[(df.bucket == "Q5-Q1") & (df.metric == metric) & (df.day != "ALL")]
    lines = ["| factor | day | " + " | ".join(f"{h}d" for h in H) + " |", "|---|---|" + "---|" * len(H)]
    for (f, d), g in s.groupby(["factor", "day"], sort=False):
        row = [cell(g[g.h == h]["mean"].iloc[0], g[g.h == h]["t"].iloc[0], pct=(metric == "RET")) for h in H]
        lines.append(f"| {f} | {d} | " + " | ".join(row) + " |")
    return "\n".join(lines)


def level_table(df, metric="RA"):
    s = df[(df.metric == metric) & (df.day != "ALL") & df.bucket.isin(["Q1", "Q5", "BASE"])]
    lines = ["| factor | day | bucket | " + " | ".join(f"{h}d" for h in H) + " |", "|---|---|---|" + "---|" * len(H)]
    for (f, d, b), g in s.groupby(["factor", "day", "bucket"], sort=False):
        row = [cell(g[g.h == h]["mean"].iloc[0], g[g.h == h]["t"].iloc[0]) for h in H]
        lines.append(f"| {f} | {d} | {b} | " + " | ".join(row) + " |")
    return "\n".join(lines)


def breadth_table(pt):
    pt = pt[(pt.n_q5 >= 30) & (pt.n_q1 >= 30)]
    g = pt.groupby(["factor", "day", "h"]).agg(
        tickers=("ticker", "nunique"),
        pct_pos=("q5_minus_q1", lambda x: (x > 0).mean()),
        median_spread=("q5_minus_q1", "median"),
        median_ic=("ic", "median"),
    ).reset_index()
    lines = ["| factor | day | h | tickers | % tickers Q5>Q1 | median Q5-Q1 RAX | median IC |",
             "|---|---|---|---|---|---|---|"]
    for _, r in g.iterrows():
        lines.append(f"| {r.factor} | {r.day} | {r.h}d | {r.tickers} | {r.pct_pos:.0%} | "
                     f"{r.median_spread:+.3f} | {r.median_ic:+.3f} |")
    return "\n".join(lines)


def yearly_table(y):
    lines = ["| factor | day | " + " | ".join(f"{h}d yrs>0" for h in H) + " |", "|---|---|" + "---|" * len(H)]
    for (f, d), g in y.groupby(["factor", "day"], sort=False):
        row = []
        for h in H:
            v = g[g.h == h].q5_minus_q1_rax.dropna()
            row.append(f"{(v > 0).sum()}/{len(v)}")
        lines.append(f"| {f} | {d} | " + " | ".join(row) + " |")
    return "\n".join(lines)


def combo_grid(c, day, h):
    s = c[(c.day == day) & (c.h == h)]
    lines = [f"| VV \\ VV_ROC | ROC low | ROC mid | ROC high |", "|---|---|---|---|"]
    for a, name in zip((1, 2, 3), ("VV low", "VV mid", "VV high")):
        row = [cell(*s[(s.vv_tercile == a) & (s.roc_tercile == b)][["mean_rax", "t"]].iloc[0]) for b in (1, 2, 3)]
        lines.append(f"| {name} | " + " | ".join(row) + " |")
    return "\n".join(lines)


def main(res_dir):
    d = Path(res_dir)
    ts = pd.read_csv(d / "buckets_time_series.csv")
    xs = pd.read_csv(d / "buckets_cross_section.csv")
    pt = pd.read_csv(d / "per_ticker.csv")
    y = pd.read_csv(d / "yearly_q5_minus_q1.csv")
    c = pd.read_csv(d / "combo_vv_x_roc.csv")

    out = ["# Volume volatility backtest — results", "",
           "Cells: mean (Newey-West t). RAX = forward return / (trailing 20d sigma x sqrt(h)), minus that day's universe average (0 = in line with universe).", ""]
    for f in ["VV", "VV_ROC"]:
        for day in ["UP", "DOWN"]:
            out += [f"## {f} quintile (own-history) | {day} days | RAX", "", quintile_grid(ts, f, day), ""]
    out += ["## Q5-Q1 spread, own-history rank, RAX", "", spread_table(ts, "RAX"), ""]
    out += ["## Absolute RA levels (not market-adjusted; includes the 10y equity drift)", "", level_table(ts), ""]
    out += ["## Q5-Q1 spread, own-history rank, raw forward return", "", spread_table(ts, "RET"), ""]
    out += ["## Q5-Q1 spread, cross-sectional rank, RAX (robustness)", "", spread_table(xs, "RAX"), ""]
    out += ["## Per-ticker breadth (own-history quintiles, RAX)", "", breadth_table(pt), ""]
    out += ["## Year-by-year consistency of Q5-Q1 (RAX)", "", yearly_table(y), ""]
    for day in ["UP", "DOWN"]:
        for h in H:
            out += [f"## VV x VV_ROC terciles | {day} days | {h}d RAX", "", combo_grid(c, day, h), ""]
    (d / "RESULTS.md").write_text("\n".join(out))
    print("\n".join(out))


if __name__ == "__main__":
    main(*sys.argv[1:])
