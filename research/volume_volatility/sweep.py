"""Lookback sweep: VV window and VV_ROC lag, scored period by period.

Everything else is held fixed: own-history 252d quintiles, 20d sigma for risk
adjustment, same liquidity filter, same volume cleaning as backtest.py. Every
config is scored on the same dates (--eval-start onward, after the slowest
config has a full rank history), split into periods at --splits.

Per config, day type (UP / DOWN) and horizon it reports Q5-Q1 RAX with a
Newey-West t for FULL and each period. For VV_ROC it also reports the spread
holding VV level fixed: ROC Q5-Q1 inside each same-window VV tercile, averaged
across terciles ("ctrl"). That is the part of ROC that is not just VV level.

Only names with --min-history sessions of data are scored, so every config is
tested on the same names as well as the same dates.

Usage: python sweep.py <bars.parquet> <out_dir> [--eval-start D] [--splits D1,D2]
                       [--min-history N] [--raw-volume]
"""
import argparse
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

import backtest as bt

WINDOWS = [5, 10, 15, 20, 30, 40, 50, 60, 80, 100, 120, 150, 180, 210, 252]
LAGS = [3, 5, 10, 15, 20, 30, 40, 60, 90, 126, 189, 252]
G = {}


def build_base(bars, eval_start, splits, min_history):
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
    # Same universe for every config: names with enough history for the slowest one.
    seasoned = (close.notna().cumsum() >= min_history).values
    live = tradable.values & seasoned & np.asarray(d >= eval_start)[:, None]
    edges = [pd.Timestamp(eval_start)] + [pd.Timestamp(x) for x in splits] + [d[-1] + pd.Timedelta(days=1)]
    period = {"FULL": np.asarray(d >= eval_start)}
    labels = []
    for a, b in zip(edges[:-1], edges[1:]):
        lab = f"{a:%Y-%m}..{(b - pd.Timedelta(days=1)):%Y-%m}"
        period[lab] = np.asarray((d >= a) & (d < b))
        labels.append(lab)
    G.update(dlv=bt.log_volume_change(vol), rax=rax, dates=d, tradable=tradable.values, labels=labels,
             day={"UP": (ret > 0).values & live, "DOWN": (ret < 0).values & live}, period=period)


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
    a, b = sc_spread.loc[factor], sc_t.loc[factor]
    periods = G["labels"] + ["FULL"]
    keys = "window" if factor == "VV" else "window | lag"
    lines = [f"| {keys} | " + " | ".join(f"{p} spread (avg t)" for p in periods) + " |",
             "|" + "---|" * (len(keys.split("|")) + len(periods))]
    for idx in a.index:
        kk = str(int(idx[0])) if factor == "VV" else f"{int(idx[0])} | {int(idx[1])}"
        lines.append(f"| {kk} | " + " | ".join(f"{a.loc[idx, p]:+.4f} ({b.loc[idx, p]:+.2f})" for p in periods) + " |")
    return "\n".join(lines)


def period_corr(sc):
    """Rank correlation of config scores between each pair of periods."""
    labs = G["labels"]
    out = []
    for i in range(len(labs)):
        for j in range(i + 1, len(labs)):
            out.append(f"{labs[i]} vs {labs[j]}: {sc[labs[i]].corr(sc[labs[j]], method='spearman'):+.2f}")
    return "; ".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("bars")
    ap.add_argument("out_dir")
    ap.add_argument("--eval-start", default="2009-01-01")
    ap.add_argument("--splits", default="2016-10-01,2022-07-01")
    ap.add_argument("--raw-volume", action="store_true")
    ap.add_argument("--min-history", type=int, default=756,
                    help="sessions of history a name needs before any config scores it (0 = off)")
    a = ap.parse_args()
    bt.CLEAN = not a.raw_volume
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    build_base(pd.read_parquet(a.bars), a.eval_start, [x for x in a.splits.split(",") if x], a.min_history)
    with Pool(4) as pool:
        res = pool.map(run_window, sorted(WINDOWS, reverse=True))
    latest = max(r[1] for r in res)
    assert latest <= pd.Timestamp(a.eval_start), f"config warmup ends {latest}, after eval start"
    df = pd.DataFrame([row for r in res for row in r[0]])
    df.to_csv(out / "sweep.csv", index=False)

    vv, roc = df[df.factor == "VV"], df[df.factor == "VV_ROC"]
    sc_vv = (score(vv, "spread"), score(vv, "t"))
    sc_raw = (score(roc, "spread"), score(roc, "t"))
    sc_ctrl = (score(roc, "ctrl_spread"), score(roc, "ctrl_t"))
    md = ["# Lookback sweep — Q5-Q1 RAX (Newey-West t)", "",
          f"Common sample from {a.eval_start} (slowest config warm by {latest.date()}). "
          f"Volume cleaning: {'on' if bt.CLEAN else 'off'}. Min history per name: {a.min_history} sessions. "
          "Score = mean over the 8 day x horizon cells; negative = high factor underperforms.", "",
          "## VV window — score by period", "", fmt_score(*sc_vv, "VV"), "",
          f"Rank correlation of window scores across periods: {period_corr(sc_vv[0])}", ""]
    for p in G["labels"] + ["FULL"]:
        md += [f"## VV window — cells, {p}", "", fmt_grid(df, "VV", p, "spread", "t"), ""]
    md += ["## VV_ROC raw — score by period", "", fmt_score(*sc_raw, "VV_ROC"), "",
           f"Rank correlation across periods (raw): {period_corr(sc_raw[0])}", "",
           "## VV_ROC holding VV level fixed — score by period", "", fmt_score(*sc_ctrl, "VV_ROC"), "",
           f"Rank correlation across periods (ctrl): {period_corr(sc_ctrl[0])}", "",
           "## VV_ROC holding VV level fixed — full-sample cells", "",
           fmt_grid(df, "VV_ROC", "FULL", "ctrl_spread", "ctrl_t"), ""]
    (out / "SWEEP.md").write_text("\n".join(md))
    print(f"wrote {out / 'SWEEP.md'}")


if __name__ == "__main__":
    main()
