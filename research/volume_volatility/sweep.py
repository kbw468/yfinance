"""Lookback sweep: VV window and VV_ROC lag.

Holds everything else fixed (own-history 252d quintiles, 20d sigma for risk
adjustment, same filters) and reports Q5-Q1 RAX with Newey-West t for each
VV window and each (window, ROC lag) pair, split by UP / DOWN days.

Usage: python sweep.py <bars.parquet> <out_dir>
"""
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

import backtest as bt

WINDOWS = [5, 10, 20, 40, 60]
LAGS = [3, 5, 10, 20]
G = {}


def build_base(bars):
    close = bars.pivot(index="date", columns="ticker", values="close").sort_index()
    close = close.where(close > 0)
    vol = bars.pivot(index="date", columns="ticker", values="volume").reindex(close.index)
    vol = vol.where(vol > 0)
    ret = close.pct_change(fill_method=None)
    sigma = np.log(close).diff().rolling(20, min_periods=18).std()
    dollar_adv = (close * vol).rolling(20, min_periods=18).median()
    tradable = (close >= bt.MIN_PRICE) & (dollar_adv >= bt.MIN_DOLLAR_ADV) & (sigma > 0) & vol.notna()
    rax = {}
    for h in bt.HORIZONS:
        f = (close.shift(-h) / close - 1).where(tradable)
        ra = bt.winsorize((f / (sigma * np.sqrt(h))).values)
        rax[h] = ra - np.nanmean(ra, axis=1, keepdims=True)
    G.update(dlv=np.log(vol).diff(), rax=rax,
             day={"UP": (ret > 0).values & tradable.values, "DOWN": (ret < 0).values & tradable.values})


def spread_rows(raw, factor, window, lag):
    pct = raw.replace([np.inf, -np.inf], np.nan).rolling(bt.RANK_WIN, min_periods=bt.RANK_MIN).rank(pct=True)
    q = bt.quintile(pct.values)
    rows = []
    for dt, day in G["day"].items():
        for h in bt.HORIZONS:
            d5, _ = bt.daily_bucket_mean(G["rax"][h], day & (q == 5))
            d1, _ = bt.daily_bucket_mean(G["rax"][h], day & (q == 1))
            t, n = bt.newey_west_t(d5 - d1, h)
            rows.append(dict(factor=factor, window=window, lag=lag, day=dt, h=h, q1=np.nanmean(d1),
                             q5=np.nanmean(d5), spread=np.nanmean(d5 - d1), t=t, n_dates=n))
    return rows


def run_window(w):
    vv = G["dlv"].rolling(w, min_periods=max(w - 2, 3)).std()
    rows = spread_rows(vv, "VV", w, np.nan)
    for lag in LAGS:
        rows += spread_rows(vv / vv.shift(lag) - 1, "VV_ROC", w, lag)
    print(f"window {w} done", flush=True)
    return rows


def grid(df, factor):
    s = df[df.factor == factor]
    keys = ["window"] if factor == "VV" else ["window", "lag"]
    cols = [(d, h) for d in ("DOWN", "UP") for h in bt.HORIZONS]
    head = " | ".join(keys) + " | " + " | ".join(f"{d} {h}d" for d, h in cols)
    lines = [f"| {head} |", "|" + "---|" * (len(keys) + len(cols))]
    for k, g in s.groupby(keys):
        k = k if isinstance(k, tuple) else (k,)
        cells = []
        for d, h in cols:
            r = g[(g.day == d) & (g.h == h)].iloc[0]
            cells.append(f"{r.spread:+.3f} ({r.t:+.1f})")
        lines.append("| " + " | ".join(str(int(x)) for x in k) + " | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def main(bars_path, out_dir):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    build_base(pd.read_parquet(bars_path))
    with Pool(min(4, len(WINDOWS))) as pool:
        rows = [r for chunk in pool.map(run_window, WINDOWS) for r in chunk]
    df = pd.DataFrame(rows)
    df.to_csv(out / "sweep.csv", index=False)
    md = ["# Lookback sweep — Q5-Q1 RAX (Newey-West t)", "",
          "## VV window (sessions)", "", grid(df, "VV"), "",
          "## VV_ROC: VV window x ROC lag (sessions)", "", grid(df, "VV_ROC"), ""]
    (out / "SWEEP.md").write_text("\n".join(md))
    print("\n".join(md))


if __name__ == "__main__":
    main(*sys.argv[1:])
