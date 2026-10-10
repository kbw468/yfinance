"""Side-by-side VV results across backtest runs (different windows / definitions / rank lookbacks).

Usage: python compare.py <out.md> <label>=<run_dir> [<label>=<run_dir> ...]
Each run_dir is a backtest.py output directory.
"""
import sys
from pathlib import Path

import pandas as pd

H = [7, 14, 21, 42]
CELLS = [(d, h) for d in ("DOWN", "UP") for h in H]


def spread_line(label, df, rank="time_series", factor="VV", metric="RAX"):
    s = df[(df["rank"] == rank) & (df.factor == factor) & (df.metric == metric) & (df.bucket == "Q5-Q1")]
    cells = []
    for d, h in CELLS:
        r = s[(s.day == d) & (s.h == h)].iloc[0]
        cells.append(f"{r['mean']:+.3f} ({r['t']:+.1f})")
    return f"| {label} | " + " | ".join(cells) + " |"


def header(first):
    return [f"| {first} | " + " | ".join(f"{d} {h}d" for d, h in CELLS) + " |", "|---|" + "---|" * len(CELLS)]


def quintiles(label, df, day):
    s = df[(df["rank"] == "time_series") & (df.factor == "VV") & (df.metric == "RAX") & (df.day == day)]
    lines = [f"**{label} | {day} days**", "", "| bucket | " + " | ".join(f"{h}d" for h in H) + " |",
             "|---|" + "---|" * len(H)]
    for b in ["Q1", "Q2", "Q3", "Q4", "Q5", "BASE", "Q5-Q1"]:
        row = [f"{r['mean']:+.3f} ({r['t']:+.1f})" for _, r in s[s.bucket == b].sort_values("h").iterrows()]
        lines.append(f"| {b} | " + " | ".join(row) + " |")
    return "\n".join(lines)


def yearly(runs):
    lines = ["| year | " + " | ".join(f"{lab} DOWN | {lab} UP" for lab in runs) + " |",
             "|---|" + "---|---|" * len(runs)]
    tabs = {}
    for lab, d in runs.items():
        y = pd.read_csv(Path(d) / "yearly_q5_minus_q1.csv")
        y = y[y.factor == "VV"].groupby(["year", "day"]).q5_minus_q1_rax.mean().unstack("day").dropna(how="all")
        tabs[lab] = y
    years = sorted(set().union(*[t.index for t in tabs.values()]))
    for yr in years:
        cells = []
        for lab in runs:
            t = tabs[lab]
            cells += [f"{t.loc[yr, 'DOWN']:+.3f}" if yr in t.index else "", f"{t.loc[yr, 'UP']:+.3f}" if yr in t.index else ""]
        lines.append(f"| {yr} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def breadth(label, d):
    pt = pd.read_csv(Path(d) / "per_ticker.csv")
    pt = pt[(pt.factor == "VV") & (pt.n_q5 >= 30) & (pt.n_q1 >= 30)]
    cells = []
    for dd, h in CELLS:
        g = pt[(pt.day == dd) & (pt.h == h)]
        cells.append(f"{(g.q5_minus_q1 < 0).mean():.0%} / {g.ic.median():+.3f}")
    return f"| {label} | " + " | ".join(cells) + " |"


def main(out_md, *pairs):
    runs = dict(p.split("=", 1) for p in pairs)
    data = {lab: pd.read_csv(Path(d) / "buckets_time_series.csv") for lab, d in runs.items()}
    xs = {lab: pd.read_csv(Path(d) / "buckets_cross_section.csv") for lab, d in runs.items()}
    out = ["# VV across settings — Q5-Q1 RAX (Newey-West t)", ""]
    out += ["## Own-history rank", ""] + header("run") + [spread_line(l, df) for l, df in data.items()] + [""]
    out += ["## Cross-sectional rank", ""] + header("run") + [spread_line(l, df, rank="cross_section")
                                                            for l, df in xs.items()] + [""]
    out += ["## Per-ticker: % of tickers with Q5 < Q1 / median IC", ""] + header("run") + \
           [breadth(l, d) for l, d in runs.items()] + [""]
    out += ["## Q5-Q1 RAX by calendar year (mean of the 4 horizons)", "", yearly(runs), ""]
    for lab, df in data.items():
        out += [quintiles(lab, df, "DOWN"), "", quintiles(lab, df, "UP"), ""]
    (Path(out_md)).write_text("\n".join(out))
    print("\n".join(out))


if __name__ == "__main__":
    main(*sys.argv[1:])
