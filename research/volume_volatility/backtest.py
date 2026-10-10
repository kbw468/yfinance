"""Volume-volatility factor backtest.

Factors (measured at the close of day t, per ticker):
  VV     = N-session stdev of daily log volume change, ln(V_t / V_{t-1})   (--vv-win, default 20)
  VV_ROC = VV_t / VV_{t-L} - 1   (L-session rate of change of VV)           (--roc-lag, default 10)

Each factor is ranked against the ticker's own trailing 252-session history (--rank-win)
(time-series percentile, no look-ahead) and bucketed into quintiles
(Q1 = lowest, Q5 = highest). A cross-sectional (rank vs. the universe that day)
version is run as a robustness check.

Day type: UP if close_t > close_{t-1}, DOWN if close_t < close_{t-1}.

Forward risk-adjusted return over h sessions (entry at close t):
  RA_h  = fwd_ret_h / (sigma20_t * sqrt(h))      (winsorized at 0.5% / 99.5%)
  RAX_h = RA_h - universe mean RA_h that date     (market-neutral risk-adjusted)
sigma20_t is trailing 20-session stdev of daily log returns, known at t. It and
the liquidity filter stay on a fixed 20-session window whatever --vv-win is.

Statistics: per date, average the metric across tickers in a bucket, then
take the time-series mean of that daily series. t-stats are Newey-West with
lag = h to handle overlapping forward windows and cross-sectional correlation.

Usage: python backtest.py <bars.parquet> <out_dir> [--def dlogv|cv] [--vv-win N]
                          [--roc-lag L] [--rank-win R] [--eval-start YYYY-MM-DD]
  --def cv swaps the VV definition to the coefficient of variation of volume levels.
  --eval-start drops signal dates before it, to compare settings on one sample.
"""
import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

HORIZONS = [7, 14, 21, 42]
VV_WIN = 20
ROC_LAG = 10
RANK_WIN = 252
RANK_MIN = 126
SIGMA_WIN = 20
EVAL_START = None
MIN_PRICE = 1.0
MIN_DOLLAR_ADV = 500_000
MIN_NAMES = 5
warnings.filterwarnings("ignore", message="Mean of empty slice")
VV_DEF = "dlogv"


def newey_west_t(x, lag):
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    n = len(x)
    if n < 30:
        return np.nan, n
    mu = x.mean()
    e = x - mu
    var = e @ e / n
    for k in range(1, lag + 1):
        w = 1 - k / (lag + 1)
        var += 2 * w * (e[k:] @ e[:-k]) / n
    return mu / np.sqrt(var / n), n


def daily_bucket_mean(metric, mask):
    """Cross-sectional mean of metric over tickers where mask is True, per date."""
    m = mask & ~np.isnan(metric)
    cnt = m.sum(axis=1)
    tot = np.where(m, metric, 0.0).sum(axis=1)
    out = np.full(len(cnt), np.nan)
    ok = cnt >= MIN_NAMES
    out[ok] = tot[ok] / cnt[ok]
    return out, cnt


def winsorize(a, lo=0.005, hi=0.995):
    v = a[~np.isnan(a)]
    ql, qh = np.quantile(v, [lo, hi])
    return np.clip(a, ql, qh)


def build_panels(bars):
    close = bars.pivot(index="date", columns="ticker", values="close").sort_index()
    close = close.where(close > 0)
    vol = bars.pivot(index="date", columns="ticker", values="volume").reindex(close.index)
    vol = vol.where(vol > 0)

    ret = close.pct_change(fill_method=None)
    logret = np.log(close).diff()
    sigma = logret.rolling(SIGMA_WIN, min_periods=SIGMA_WIN - 2).std()

    if VV_DEF == "cv":  # dispersion of volume levels: stdev / mean over the window
        roll = vol.rolling(VV_WIN, min_periods=VV_WIN - 2)
        vv = roll.std() / roll.mean()
    else:  # default: realized vol of volume, stdev of daily log volume change
        vv = np.log(vol).diff().rolling(VV_WIN, min_periods=VV_WIN - 2).std()
    vv_roc = (vv / vv.shift(ROC_LAG) - 1).replace([np.inf, -np.inf], np.nan)

    dollar_adv = (close * vol).rolling(SIGMA_WIN, min_periods=SIGMA_WIN - 2).median()
    tradable = (close >= MIN_PRICE) & (dollar_adv >= MIN_DOLLAR_ADV) & (sigma > 0) & vol.notna()
    in_sample = np.asarray(close.index >= pd.Timestamp(EVAL_START) if EVAL_START else np.ones(len(close), bool))

    ts_rank = {
        "VV": vv.rolling(RANK_WIN, min_periods=RANK_MIN).rank(pct=True),
        "VV_ROC": vv_roc.rolling(RANK_WIN, min_periods=RANK_MIN).rank(pct=True),
    }
    xs_rank = {
        "VV": vv.where(tradable).rank(axis=1, pct=True),
        "VV_ROC": vv_roc.where(tradable).rank(axis=1, pct=True),
    }

    fwd, ra, rax = {}, {}, {}
    for h in HORIZONS:
        f = close.shift(-h) / close - 1
        f = f.where(tradable)
        fwd[h] = f.values
        ra[h] = winsorize((f / (sigma * np.sqrt(h))).values)
        rax[h] = ra[h] - np.nanmean(ra[h], axis=1, keepdims=True)

    live = tradable.values & in_sample[:, None]
    day = {
        "ALL": live,
        "UP": (ret > 0).values & live,
        "DOWN": (ret < 0).values & live,
    }
    return dict(close=close, dates=close.index, tickers=close.columns, day=day,
                ts_rank=ts_rank, xs_rank=xs_rank, fwd=fwd, ra=ra, rax=rax)


def quintile(pct):
    q = np.ceil(pct * 5)
    return np.where(np.isnan(pct), np.nan, np.clip(q, 1, 5))


def bucket_table(P, ranks, label):
    rows = []
    for fac, pct in ranks.items():
        q = quintile(pct.values)
        for dt in ["ALL", "UP", "DOWN"]:
            base = P["day"][dt] & ~np.isnan(q)
            for h in HORIZONS:
                up = np.where(np.isnan(P["fwd"][h]), np.nan, (P["fwd"][h] > 0).astype(float))
                daily = {}
                for qq in range(1, 6):
                    mask = base & (q == qq)
                    hit_d, _ = daily_bucket_mean(up, mask)
                    for mname, metric in [("RA", P["ra"][h]), ("RAX", P["rax"][h]), ("RET", P["fwd"][h])]:
                        d, cnt = daily_bucket_mean(metric, mask)
                        daily[(qq, mname)] = d
                        t, n = newey_west_t(d, h)
                        rows.append(dict(rank=label, factor=fac, day=dt, h=h, bucket=f"Q{qq}", metric=mname,
                                         mean=np.nanmean(d), t=t, n_dates=n, avg_names=np.nanmean(cnt[cnt > 0]),
                                         hit=np.nanmean(hit_d)))
                for mname, metric in [("RA", P["ra"][h]), ("RAX", P["rax"][h]), ("RET", P["fwd"][h])]:
                    d, cnt = daily_bucket_mean(metric, base)
                    t, n = newey_west_t(d, h)
                    rows.append(dict(rank=label, factor=fac, day=dt, h=h, bucket="BASE", metric=mname,
                                     mean=np.nanmean(d), t=t, n_dates=n, avg_names=np.nanmean(cnt[cnt > 0]),
                                     hit=np.nanmean(daily_bucket_mean(up, base)[0])))
                    s = daily[(5, mname)] - daily[(1, mname)]
                    t, n = newey_west_t(s, h)
                    rows.append(dict(rank=label, factor=fac, day=dt, h=h, bucket="Q5-Q1", metric=mname,
                                     mean=np.nanmean(s), t=t, n_dates=n, avg_names=np.nan, hit=np.nan))
    return pd.DataFrame(rows)


def combo_table(P):
    """VV tercile x VV_ROC tercile grid, by day type, RAX metric."""
    t_vv = np.ceil(P["ts_rank"]["VV"].values * 3)
    t_roc = np.ceil(P["ts_rank"]["VV_ROC"].values * 3)
    rows = []
    for dt in ["UP", "DOWN"]:
        for h in HORIZONS:
            for a in (1, 2, 3):
                for b in (1, 2, 3):
                    mask = P["day"][dt] & (t_vv == a) & (t_roc == b)
                    d, cnt = daily_bucket_mean(P["rax"][h], mask)
                    t, n = newey_west_t(d, h)
                    rows.append(dict(day=dt, h=h, vv_tercile=a, roc_tercile=b, mean_rax=np.nanmean(d), t=t,
                                     avg_names=np.nanmean(cnt[cnt > 0])))
    return pd.DataFrame(rows)


def yearly_spread(P):
    rows = []
    years = P["dates"].year
    for fac, pct in P["ts_rank"].items():
        q = quintile(pct.values)
        for dt in ["UP", "DOWN"]:
            for h in HORIZONS:
                d5, _ = daily_bucket_mean(P["rax"][h], P["day"][dt] & (q == 5))
                d1, _ = daily_bucket_mean(P["rax"][h], P["day"][dt] & (q == 1))
                s = pd.Series(d5 - d1, index=years)
                for y, v in s.groupby(level=0).mean().items():
                    rows.append(dict(factor=fac, day=dt, h=h, year=y, q5_minus_q1_rax=v))
    return pd.DataFrame(rows)


def per_ticker(P):
    rows = []
    tickers = P["tickers"]
    for fac, pct in P["ts_rank"].items():
        q = quintile(pct.values)
        pv = pct.values
        for dt in ["UP", "DOWN"]:
            day = P["day"][dt]
            for h in HORIZONS:
                ra = P["rax"][h]
                valid = day & ~np.isnan(ra) & ~np.isnan(pv)
                hi = valid & (q == 5)
                lo = valid & (q == 1)
                with np.errstate(invalid="ignore", divide="ignore"):
                    m_hi = np.where(hi, ra, 0).sum(0) / hi.sum(0)
                    m_lo = np.where(lo, ra, 0).sum(0) / lo.sum(0)
                    # Pearson IC between factor percentile and forward RAX, per ticker
                    x = np.where(valid, pv, np.nan)
                    y = np.where(valid, ra, np.nan)
                    xm = x - np.nanmean(x, 0)
                    ym = y - np.nanmean(y, 0)
                    ic = np.nansum(xm * ym, 0) / np.sqrt(np.nansum(xm ** 2, 0) * np.nansum(ym ** 2, 0))
                for i, tk in enumerate(tickers):
                    rows.append(dict(ticker=tk, factor=fac, day=dt, h=h, n_obs=int(valid[:, i].sum()),
                                     n_q5=int(hi[:, i].sum()), n_q1=int(lo[:, i].sum()),
                                     rax_q5=m_hi[i], rax_q1=m_lo[i], q5_minus_q1=m_hi[i] - m_lo[i], ic=ic[i]))
    return pd.DataFrame(rows)


def main(bars_path, out_dir):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    bars = pd.read_parquet(bars_path)
    P = build_panels(bars)
    print(f"panel: {len(P['dates'])} dates x {len(P['tickers'])} tickers "
          f"({P['dates'][0].date()} -> {P['dates'][-1].date()})", flush=True)

    ts = bucket_table(P, P["ts_rank"], "time_series")
    ts.to_csv(out / "buckets_time_series.csv", index=False)
    print("time-series buckets done", flush=True)
    xs = bucket_table(P, P["xs_rank"], "cross_section")
    xs.to_csv(out / "buckets_cross_section.csv", index=False)
    print("cross-section buckets done", flush=True)
    combo_table(P).to_csv(out / "combo_vv_x_roc.csv", index=False)
    yearly_spread(P).to_csv(out / "yearly_q5_minus_q1.csv", index=False)
    pt = per_ticker(P)
    pt.to_csv(out / "per_ticker.csv", index=False)
    print("done", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("bars")
    ap.add_argument("out_dir")
    ap.add_argument("--def", dest="vv_def", default="dlogv", choices=["dlogv", "cv"])
    ap.add_argument("--vv-win", type=int, default=VV_WIN)
    ap.add_argument("--roc-lag", type=int, default=ROC_LAG)
    ap.add_argument("--rank-win", type=int, default=RANK_WIN)
    ap.add_argument("--eval-start", default=None)
    a = ap.parse_args()
    VV_DEF, VV_WIN, ROC_LAG, EVAL_START = a.vv_def, a.vv_win, a.roc_lag, a.eval_start
    RANK_WIN, RANK_MIN = a.rank_win, a.rank_win // 2
    main(a.bars, a.out_dir)
