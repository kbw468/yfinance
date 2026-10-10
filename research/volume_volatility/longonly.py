"""Long-only view: which buckets beat the average stock, from --eval-start onward.

Metric: forward return of the bucket minus the average forward return of all tradable names that day
(raw %, winsorized at 0.5% / 99.5%). Newey-West t (lag = horizon). Per-year means for consistency.

Candidates (each on DOWN days, UP days and ANY day):
  calm volume      VV level in the bottom fifth of its own past year, windows 5-252
  jump             fast ROC (5/5, 10/5, 15/3, 15/5) in the top fifth
  falling (slow)   slow ROC (50/20, 120/20) in the bottom fifth
  calm + jump      VV bottom fifth AND fast ROC top fifth; also bottom third AND top third

Usage: python longonly.py <bars.parquet> <out_dir> [--eval-start D] [--min-history N]
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

import backtest as bt

WINDOWS = [5, 10, 15, 20, 30, 40, 50, 60, 80, 100, 120, 150, 180, 210, 252]
FAST = [(5, 5), (10, 5), (15, 3), (15, 5)]
SLOW = [(50, 20), (120, 20)]
COMBO_LEVELS = [20, 50, 60, 120]


def rank(x):
    return x.replace([np.inf, -np.inf], np.nan).rolling(252, min_periods=126).rank(pct=True).values


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("bars")
    ap.add_argument("out_dir")
    ap.add_argument("--eval-start", default="2023-01-01")
    ap.add_argument("--min-history", type=int, default=400)
    a = ap.parse_args()
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    bars = pd.read_parquet(a.bars)
    close = bars.pivot(index="date", columns="ticker", values="close").sort_index()
    close = close.where(close > 0)
    vol = bars.pivot(index="date", columns="ticker", values="volume").reindex(close.index)
    vol = vol.where(vol > 0)
    ret = close.pct_change(fill_method=None)
    sigma = np.log(close).diff().rolling(20, min_periods=18).std()
    adv = (close * vol).rolling(20, min_periods=18).median()
    tradable = ((close >= bt.MIN_PRICE) & (adv >= bt.MIN_DOLLAR_ADV) & (sigma > 0) & vol.notna()).values
    dates = close.index
    seasoned = (close.notna().cumsum() >= a.min_history).values
    live = tradable & seasoned & np.asarray(dates >= a.eval_start)[:, None]
    up, dn = (ret > 0).values, (ret < 0).values
    days = {"DOWN": live & dn, "UP": live & up, "ANY": live}
    years = dates.year.values
    yr_list = sorted(set(years[dates >= a.eval_start]))

    excess, rawf = {}, {}
    for h in bt.HORIZONS:
        f = np.where(tradable, (close.shift(-h) / close - 1).values, np.nan)
        f = bt.winsorize(f)
        excess[h] = f - np.nanmean(f, axis=1, keepdims=True)
        rawf[h] = f
    dlv = bt.log_volume_change(vol)

    sig = {}
    vv_cache = {}
    for w in sorted(set(WINDOWS) | {w for w, _ in FAST + SLOW}):
        vv_cache[w] = dlv.rolling(w, min_periods=max(w - 2, 3)).std()
    for w in WINDOWS:
        p = rank(vv_cache[w])
        sig[("calm volume", f"{w}", "fifth")] = p <= 0.2
        if w in COMBO_LEVELS:
            sig[("_lvl3", f"{w}", "")] = p <= 1 / 3
    for w, l in FAST:
        p = rank(vv_cache[w] / vv_cache[w].shift(l) - 1)
        sig[("jump", f"{w}/{l}", "fifth")] = p > 0.8
        sig[("_jump3", f"{w}/{l}", "")] = p > 2 / 3
    for w, l in SLOW:
        p = rank(vv_cache[w] / vv_cache[w].shift(l) - 1)
        sig[("falling (slow)", f"{w}/{l}", "fifth")] = p <= 0.2
    for lw in COMBO_LEVELS:
        for w, l in FAST:
            sig[("calm + jump", f"{lw} + {w}/{l}", "fifth")] = sig[("calm volume", f"{lw}", "fifth")] & \
                sig[("jump", f"{w}/{l}", "fifth")]
            sig[("calm + jump", f"{lw} + {w}/{l}", "third")] = sig[("_lvl3", f"{lw}", "")] & sig[("_jump3", f"{w}/{l}", "")]
    print(f"{len(sig)} signals built", flush=True)

    rows = []
    for (fam, setting, cut), m in sig.items():
        if fam.startswith("_"):
            continue
        for dname, dmask in days.items():
            mask = m & dmask
            for h in bt.HORIZONS:
                d, cnt = bt.daily_bucket_mean(excess[h], mask)
                rf, _ = bt.daily_bucket_mean(rawf[h], mask)
                uni, _ = bt.daily_bucket_mean(rawf[h], dmask)
                t, n = bt.newey_west_t(d, h)
                beat = np.nanmean(np.where(np.isnan(d), np.nan, (d > 0).astype(float)))
                row = dict(family=fam, setting=setting, cut=cut, day=dname, h=h, excess=np.nanmean(d), t=t,
                           bucket_return=np.nanmean(rf), day_avg_return=np.nanmean(uni), beat_share=beat,
                           names=np.nanmean(cnt[cnt > 0]) if (cnt > 0).any() else np.nan, dates=n)
                for y in yr_list:
                    row[f"y{y}"] = np.nanmean(d[years == y])
                rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(out / "longonly.csv", index=False)

    ycols = [f"y{y}" for y in yr_list]
    g = df.groupby(["family", "setting", "cut", "day"]).agg(
        avg_excess=("excess", "mean"), avg_t=("t", "mean"), names=("names", "mean"),
        **{c: (c, "mean") for c in ycols}).reset_index()
    g["years_positive"] = (g[ycols] > 0).sum(axis=1)
    g = g.sort_values("avg_t", ascending=False)
    g.to_csv(out / "longonly_ranked.csv", index=False)
    print(g.head(25).round(4).to_string(index=False))


if __name__ == "__main__":
    main()
