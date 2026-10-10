"""Combination test: 20-session VV level x VV_ROC (several window / lag settings), by day type.

For each ROC setting it scores:
  - the 3x3 grid of VV tercile x ROC tercile (mean RAX per cell vs the universe)
  - the 5x5 corners (VV Q1/Q5 x ROC Q1/Q5)
  - a long-short spread, long minus short, with sides set by each factor's sign:
      VV level: low is long.  Fast ROC (window <= 15): high is long.  Slower ROC: low is long.
    LS_T3 uses the tercile corners, LS_Q5 the quintile corners.
  - the single-factor long-shorts on the same names and dates: VV alone (T1-T3, Q1-Q5), ROC alone.
All stats use Newey-West t (lag = horizon), FULL and per period, same universe and dates as sweep.py.

Usage: python combo.py <bars.parquet> <out_dir> [--eval-start D] [--splits D1,D2] [--min-history N]
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

import backtest as bt
import sweep as sw

VV_WIN = 20
ROC_CONFIGS = [(5, 5), (10, 5), (15, 3), (15, 5), (20, 10), (50, 20), (120, 20)]


def bucket(pct, n):
    return np.where(np.isnan(pct), np.nan, np.clip(np.ceil(pct * n), 1, n))


def rows_for(G, label, mask, h, extra):
    d, cnt = bt.daily_bucket_mean(G["rax"][h], mask)
    st = sw.stats(d, h)
    names = float(np.nanmean(cnt[cnt > 0])) if (cnt > 0).any() else np.nan
    return [dict(extra, kind=label, period=p, mean=st[p][0], t=st[p][1], avg_names=names) for p in G["period"]]


def ls_rows(G, label, long_mask, short_mask, h, extra):
    dl, _ = bt.daily_bucket_mean(G["rax"][h], long_mask)
    ds, _ = bt.daily_bucket_mean(G["rax"][h], short_mask)
    st = sw.stats(dl - ds, h)
    return [dict(extra, kind=label, period=p, mean=st[p][0], t=st[p][1], avg_names=np.nan) for p in G["period"]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("bars")
    ap.add_argument("out_dir")
    ap.add_argument("--eval-start", default="2009-01-01")
    ap.add_argument("--splits", default="2016-10-01,2022-07-01")
    ap.add_argument("--min-history", type=int, default=756)
    a = ap.parse_args()
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    sw.build_base(pd.read_parquet(a.bars), a.eval_start, [x for x in a.splits.split(",") if x], a.min_history)
    G = sw.G
    vv = G["dlv"].rolling(VV_WIN, min_periods=VV_WIN - 2).std()
    pv = sw.ts_rank(vv)
    v3, v5 = bucket(pv, 3), bucket(pv, 5)
    rows = []
    for (w, lag) in ROC_CONFIGS:
        base = vv if w == VV_WIN else G["dlv"].rolling(w, min_periods=max(w - 2, 3)).std()
        pr = sw.ts_rank(base / base.shift(lag) - 1)
        r3, r5 = bucket(pr, 3), bucket(pr, 5)
        fast = w <= 15
        for dt, day in G["day"].items():
            for h in bt.HORIZONS:
                ex = dict(roc=f"{w}/{lag}", day=dt, h=h)
                for i in (1, 2, 3):
                    for j in (1, 2, 3):
                        rows += rows_for(G, "T3", day & (v3 == i) & (r3 == j), h, dict(ex, vv=i, rocb=j))
                for i in (1, 5):
                    for j in (1, 5):
                        rows += rows_for(G, "Q5", day & (v5 == i) & (r5 == j), h, dict(ex, vv=i, rocb=j))
                rl3, rs3 = (3, 1) if fast else (1, 3)
                rl5, rs5 = (5, 1) if fast else (1, 5)
                rows += ls_rows(G, "LS_T3", day & (v3 == 1) & (r3 == rl3), day & (v3 == 3) & (r3 == rs3), h, ex)
                rows += ls_rows(G, "LS_Q5", day & (v5 == 1) & (r5 == rl5), day & (v5 == 5) & (r5 == rs5), h, ex)
                rows += ls_rows(G, "ROC_ONLY_Q5", day & (r5 == rl5), day & (r5 == rs5), h, ex)
                if (w, lag) == ROC_CONFIGS[0]:  # VV alone, once per day x horizon
                    exv = dict(roc="none", day=dt, h=h)
                    rows += ls_rows(G, "VV_ONLY_T3", day & (v3 == 1), day & (v3 == 3), h, exv)
                    rows += ls_rows(G, "VV_ONLY_Q5", day & (v5 == 1), day & (v5 == 5), h, exv)
        print(f"ROC {w}/{lag} done", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(out / "combo.csv", index=False)

    P = G["labels"] + ["FULL"]
    lab = {p: p for p in P}
    md = ["# VV20 x VV_ROC combinations — market-neutral RAX (Newey-West t)", "",
          "Long-short = long minus short, positive = the combination works. VV low is long. Fast ROC "
          "(window <= 15) high is long; slower ROC low is long.", ""]

    def ls_table(kind_combo, kind_single):
        lines = ["| ROC | day | " + " | ".join(f"{h}d" for h in bt.HORIZONS) + " | avg t by period (" +
                 ", ".join(G["labels"]) + ") |", "|---|---|" + "---|" * (len(bt.HORIZONS) + 1)]
        f = df[df.period == "FULL"]
        for dt in ("DOWN", "UP"):
            s = f[(f.kind == kind_single) & (f.day == dt)].set_index("h")
            per = df[(df.kind == kind_single) & (df.day == dt)].groupby("period").t.mean()
            lines.append(f"| VV alone | {dt} | " + " | ".join(f"{s.loc[h, 'mean']:+.3f} ({s.loc[h, 't']:+.1f})"
                                                                for h in bt.HORIZONS) +
                         " | " + ", ".join(f"{per[p]:+.1f}" for p in G["labels"]) + " |")
            for cfg in [f"{w}/{l}" for w, l in ROC_CONFIGS]:
                s = f[(f.kind == kind_combo) & (f.day == dt) & (f.roc == cfg)].set_index("h")
                per = df[(df.kind == kind_combo) & (df.day == dt) & (df.roc == cfg)].groupby("period").t.mean()
                lines.append(f"| VV + ROC {cfg} | {dt} | " +
                             " | ".join(f"{s.loc[h, 'mean']:+.3f} ({s.loc[h, 't']:+.1f})" for h in bt.HORIZONS) +
                             " | " + ", ".join(f"{per[p]:+.1f}" for p in G["labels"]) + " |")
        return "\n".join(lines)

    md += ["## Long-short, tercile corners", "", ls_table("LS_T3", "VV_ONLY_T3"), ""]
    md += ["## Long-short, quintile corners", "", ls_table("LS_Q5", "VV_ONLY_Q5"), ""]
    f = df[(df.period == "FULL") & df.kind.isin(["T3", "Q5"])].copy()
    f["abs_t"] = f.t.abs()
    top = f.sort_values("abs_t", ascending=False).head(15)
    md += ["## Strongest single cells (FULL), vs universe", "",
           "| ROC | day | h | grid | VV bucket | ROC bucket | mean | t | avg names |", "|---|---|---|---|---|---|---|---|---|"]
    for _, r in top.iterrows():
        md.append(f"| {r.roc} | {r.day} | {r.h}d | {r.kind} | {int(r.vv)} | {int(r.rocb)} | {r['mean']:+.3f} | "
                  f"{r.t:+.1f} | {r.avg_names:.0f} |")
    (out / "COMBO.md").write_text("\n".join(md) + "\n")
    print(f"wrote {out / 'COMBO.md'}")


if __name__ == "__main__":
    main()
