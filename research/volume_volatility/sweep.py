"""Lookback sweep: VV window and VV_ROC lag, with an out-of-sample check.

Everything else is held fixed: own-history 252d quintiles, 20d sigma for risk
adjustment, same liquidity filter. Every config is scored on the same dates
(EVAL_START onward, after the slowest config has a full rank history), split
into in-sample (before SPLIT) and out-of-sample (SPLIT onward).

Per config, day type (UP / DOWN) and horizon it reports Q5-Q1 RAX with a
Newey-West t for FULL, IS and OOS. For VV_ROC it also reports the spread
holding VV level fixed: ROC Q5-Q1 inside each same-window VV tercile, averaged
across terciles ("ctrl"). That is the part of ROC that is not just VV level.

Usage: python sweep.py <bars.parquet> <out_dir>
"""
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

import backtest as bt

WINDOWS = [5, 10, 15, 20, 30, 40, 50, 60, 80, 100, 120]
LAGS = [3, 5, 10, 15, 20, 30, 40, 60]
EVAL_START = "2018-07-01"
SPLIT = "2022-07-01"
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
    d = close.index
    live = tradable.values & np.asarray(d >= EVAL_START)[:, None]
    G.update(dlv=np.log(vol).diff(), rax=rax, dates=d, tradable=tradable.values,
             day={"UP": (ret > 0).values & live, "DOWN": (ret < 0).values & live},
             period={"FULL": np.asarray(d >= EVAL_START),
                     "IS": np.asarray((d >= EVAL_START) & (d < SPLIT)),
                     "OOS": np.asarray(d >= SPLIT)})


def ts_rank(raw):
    raw = raw.replace([np.inf, -np.inf], np.nan)
    return raw.rolling(bt.RANK_WIN, min_periods=bt.RANK_MIN).rank(pct=True).values


def coverage_start(pct):
    """First date where at least half the tradable names have a rank."""
    cov = (~np.isnan(pct) & G["tradable"]).sum(1) / np.maximum(G["tradable"].sum(1), 1)
    return G["dates"][np.argmax(cov >= 0.5)]


def stats(series, h):
    out = {}
    for p, m in G["period"].items():
        t, _ = bt.newey_west_t(series[m], h)
        out[p] = (np.nanmean(series[m]), t)
    return out


def spread_rows(pct, factor, window, lag, ctrl_terciles=None):
    q = bt.quintile(pct)
    rows = []
    for dt, day in G["day"].items():
        for h in bt.HORIZONS:
            rax = G["rax"][h]
            d5, _ = bt.daily_bucket_mean(rax, day & (q == 5))
            d1, _ = bt.daily_bucket_mean(rax, day & (q == 1))
            raw = stats(d5 - d1, h)
            ctrl = None
            if ctrl_terciles is not None:
                parts = []
                for a in (1, 2, 3):
                    c5, _ = bt.daily_bucket_mean(rax, day & (ctrl_terciles == a) & (q == 5))
                    c1, _ = bt.daily_bucket_mean(rax, day & (ctrl_terciles == a) & (q == 1))
                    parts.append(c5 - c1)
                ctrl = stats(np.nanmean(np.vstack(parts), axis=0), h)
            for p in G["period"]:
                rows.append(dict(factor=factor, window=window, lag=lag, day=dt, h=h, period=p,
                                 spread=raw[p][0], t=raw[p][1],
                                 ctrl_spread=ctrl[p][0] if ctrl else np.nan,
                                 ctrl_t=ctrl[p][1] if ctrl else np.nan))
    return rows


def run_window(w):
    vv = G["dlv"].rolling(w, min_periods=max(w - 2, 3)).std()
    pv = ts_rank(vv)
    starts = [coverage_start(pv)]
    rows = spread_rows(pv, "VV", w, 0)  # lag 0 = not applicable
    terc = np.ceil(pv * 3)
    for lag in LAGS:
        pr = ts_rank(vv / vv.shift(lag) - 1)
        starts.append(coverage_start(pr))
        rows += spread_rows(pr, "VV_ROC", w, lag, ctrl_terciles=terc)
    print(f"window {w} done, latest coverage start {max(starts).date()}", flush=True)
    return rows, max(starts)


def score(df, col):
    """Mean over the 8 day x horizon cells, per config and period."""
    return df.groupby(["factor", "window", "lag", "period"])[col].mean().unstack("period")


def fmt_grid(df, factor, period, val, tcol):
    s = df[(df.factor == factor) & (df.period == period)]
    keys = ["window"] if factor == "VV" else ["window", "lag"]
    cols = [(d, h) for d in ("DOWN", "UP") for h in bt.HORIZONS]
    lines = ["| " + " | ".join(keys) + " | " + " | ".join(f"{d} {h}d" for d, h in cols) + " |",
             "|" + "---|" * (len(keys) + len(cols))]
    for k, g in s.groupby(keys):
        k = k if isinstance(k, tuple) else (k,)
        cells = []
        for d, h in cols:
            r = g[(g.day == d) & (g.h == h)].iloc[0]
            cells.append(f"{r[val]:+.3f} ({r[tcol]:+.1f})")
        lines.append("| " + " | ".join(str(int(x)) for x in k) + " | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def fmt_score(sc_spread, sc_t, factor):
    a = sc_spread.loc[factor]
    b = sc_t.loc[factor]
    keys = "window" if factor == "VV" else "window | lag"
    lines = [f"| {keys} | IS spread | IS avg t | OOS spread | OOS avg t | FULL spread | FULL avg t |",
             "|" + "---|" * (len(keys.split("|")) + 6)]
    for idx in a.index:
        k = idx[0] if factor == "VV" else idx
        kk = str(int(k)) if factor == "VV" else f"{int(k[0])} | {int(k[1])}"
        lines.append(f"| {kk} | {a.loc[idx, 'IS']:+.4f} | {b.loc[idx, 'IS']:+.2f} | {a.loc[idx, 'OOS']:+.4f} | "
                     f"{b.loc[idx, 'OOS']:+.2f} | {a.loc[idx, 'FULL']:+.4f} | {b.loc[idx, 'FULL']:+.2f} |")
    return "\n".join(lines)


def main(bars_path, out_dir):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    build_base(pd.read_parquet(bars_path))
    with Pool(4) as pool:
        res = pool.map(run_window, sorted(WINDOWS, reverse=True))
    latest = max(r[1] for r in res)
    assert latest <= pd.Timestamp(EVAL_START), f"config warmup ends {latest}, after EVAL_START"
    df = pd.DataFrame([row for r in res for row in r[0]])
    df.to_csv(out / "sweep.csv", index=False)

    vv = df[df.factor == "VV"]
    roc = df[df.factor == "VV_ROC"]
    sc_vv = (score(vv, "spread"), score(vv, "t"))
    sc_roc_raw = (score(roc, "spread"), score(roc, "t"))
    sc_roc_ctrl = (score(roc, "ctrl_spread"), score(roc, "ctrl_t"))

    # Does the IS ranking of configs carry into OOS?
    def is_oos_corr(sc):
        return sc[0]["IS"].corr(sc[0]["OOS"], method="spearman")

    md = [f"# Lookback sweep — Q5-Q1 RAX (Newey-West t)", "",
          f"Common sample from {EVAL_START} (slowest config warm by {latest.date()}). "
          f"IS = {EVAL_START} to {SPLIT}, OOS = {SPLIT} onward. "
          "Score = mean over the 8 day x horizon cells; negative = high factor underperforms.", "",
          "## VV window — score by period", "", fmt_score(*sc_vv, "VV"), "",
          f"IS vs OOS rank correlation of window scores: {is_oos_corr(sc_vv):+.2f}", "",
          "## VV window — full-sample cells", "", fmt_grid(df, "VV", "FULL", "spread", "t"), "",
          "## VV window — OOS cells", "", fmt_grid(df, "VV", "OOS", "spread", "t"), "",
          "## VV_ROC raw — score by period", "", fmt_score(*sc_roc_raw, "VV_ROC"), "",
          f"IS vs OOS rank correlation (raw): {is_oos_corr(sc_roc_raw):+.2f}", "",
          "## VV_ROC holding VV level fixed — score by period", "", fmt_score(*sc_roc_ctrl, "VV_ROC"), "",
          f"IS vs OOS rank correlation (ctrl): {is_oos_corr(sc_roc_ctrl):+.2f}", "",
          "## VV_ROC holding VV level fixed — full-sample cells", "",
          fmt_grid(df, "VV_ROC", "FULL", "ctrl_spread", "ctrl_t"), ""]
    (out / "SWEEP.md").write_text("\n".join(md))
    print(f"wrote {out / 'SWEEP.md'}")


if __name__ == "__main__":
    main(*sys.argv[1:])
