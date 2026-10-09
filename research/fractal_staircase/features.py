"""Per-ticker feature + forward-outcome engine.

Every statistic here is distribution-free: medians, ranks, ranges, sums of
absolute moves, drawdowns. No standard deviation, variance, z-score, OLS or
moving average appears anywhere.

Notation (all on adjusted daily bars):
    x   = log(close)
    r   = x_t - x_{t-1}                     daily log return
    a   = |r|                               absolute move
    g   = log(high / low)                   daily log range
"""
import os
from multiprocessing import Pool

import numpy as np
import pandas as pd
import yfinance as yf

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.environ.get("FS_DATA", os.path.join(HERE, "data"))
SAMPLE_EVERY = 5          # one observation per week per ticker
MIN_HISTORY = 300         # bars required before a ticker is eligible
HORIZONS = (21, 42, 63)


# ---------------------------------------------------------------- helpers
def roll_med(s, w, minp=None):
    return s.rolling(w, min_periods=minp or w).median()


def roll_sum(s, w, minp=None):
    return s.rolling(w, min_periods=minp or w).sum()


def kendall_trend(x, w):
    """Rolling Kendall tau of x against time over a w-bar window.

    tau_t = mean over all pairs i<j in the window of sign(x_j - x_i).
    Computed in O(N*w) as a sum over lags d of rolling sums of sign(x_t - x_{t-d}).
    """
    n = len(x)
    out = np.zeros(n)
    xs = pd.Series(x)
    for d in range(1, w):
        sd = np.sign(xs - xs.shift(d))
        out += sd.rolling(w - d, min_periods=w - d).sum().to_numpy()
    out /= w * (w - 1) / 2
    out[: w - 1] = np.nan
    return out


def scaling_exponent(x, w, ks=(1, 2, 4, 8, 16)):
    """Median-based scaling exponent (robust Hurst analogue).

    For each k, m_k = median |x_t - x_{t-k}| over the last w bars.
    Exponent = Theil-Sen slope (median of pairwise slopes) of log m_k on log k.
    ~0.5 random-walk scaling, -> 1 persistent trend, < 0.5 mean reverting.
    """
    xs = pd.Series(x)
    logm = []
    for k in ks:
        logm.append(np.log(roll_med((xs - xs.shift(k)).abs(), w).to_numpy() + 1e-12))
    logk = np.log(np.array(ks, dtype=float))
    slopes = []
    for i in range(len(ks)):
        for j in range(i + 1, len(ks)):
            slopes.append((logm[j] - logm[i]) / (logk[j] - logk[i]))
    return np.median(np.vstack(slopes), axis=0)


def rolling_maxdd(x, w):
    """Max peak-to-trough log drawdown of x within each trailing w-bar window."""
    n = len(x)
    out = np.full(n, np.nan)
    if n < w:
        return out
    from numpy.lib.stride_tricks import sliding_window_view
    W = sliding_window_view(x, w)
    peak = np.maximum.accumulate(W, axis=1)
    out[w - 1:] = (peak - W).max(axis=1)
    return out


def forward_path_stats(x, r, h):
    """Forward outcome over (t, t+h]: return, max drawdown, efficiency, median |r|."""
    n = len(x)
    fr = np.full(n, np.nan)
    fdd = np.full(n, np.nan)
    fae = np.full(n, np.nan)
    fer = np.full(n, np.nan)
    fmar = np.full(n, np.nan)
    if n <= h:
        return fr, fdd, fae, fer, fmar
    from numpy.lib.stride_tricks import sliding_window_view
    W = sliding_window_view(x, h + 1)              # rows: x_t .. x_{t+h}
    m = W.shape[0]
    fr[:m] = W[:, -1] - W[:, 0]
    peak = np.maximum.accumulate(W, axis=1)
    fdd[:m] = (peak - W).max(axis=1)               # max drawdown along the path
    fae[:m] = W[:, 0] - W.min(axis=1)              # worst excursion below entry
    A = sliding_window_view(np.abs(r[1:]), h)      # |r_{t+1}| .. |r_{t+h}|
    path = A.sum(axis=1)[:m]
    fer[:m] = fr[:m] / np.where(path > 0, path, np.nan)
    fmar[:m] = np.median(A, axis=1)[:m]
    return fr, fdd, fae, fer, fmar


# ---------------------------------------------------------------- per ticker
def ticker_features(df, sample_dates, mkt=None):
    df = df.sort_values("date").reset_index(drop=True)
    df = df[(df["close"] > 0) & (df["high"] > 0) & (df["low"] > 0)].reset_index(drop=True)
    n = len(df)
    if n < MIN_HISTORY:
        return None
    c = df["close"].to_numpy(float)
    o = df["open"].to_numpy(float)
    h = np.maximum(df["high"].to_numpy(float), c)
    lo = np.minimum(df["low"].to_numpy(float), c)
    v = df["volume"].to_numpy(float)
    x = np.log(c)
    xs = pd.Series(x)
    r = xs.diff()
    a = r.abs()
    g = pd.Series(np.log(h / lo))
    V = pd.Series(v).where(lambda s: s > 0)

    f = {}
    f["bars"] = np.arange(1, n + 1)
    f["logp"] = x

    # ---- price rate of change, multi-scale, plus acceleration
    for k in (5, 10, 21, 42, 63, 126, 252):
        f[f"roc{k}"] = (xs - xs.shift(k)).to_numpy()
    for k in (10, 21, 63):
        f[f"acc{k}"] = f[f"roc{k}"] - pd.Series(f[f"roc{k}"]).shift(k).to_numpy()
    f["roc21_vs_63"] = f["roc21"] - f["roc63"] / 3.0          # pace now vs pace of quarter
    f["roc63_vs_252"] = f["roc63"] - f["roc252"] / 4.0

    # ---- volatility: median absolute move and median range (no variance)
    for k in (10, 21, 63, 126, 252):
        f[f"mar{k}"] = roll_med(a, k).to_numpy()
    for k in (10, 21, 63, 252):
        f[f"mrg{k}"] = roll_med(g, k).to_numpy()
    f["vc10_63"] = np.log(f["mar10"] / f["mar63"])
    f["vc21_126"] = np.log(f["mar21"] / f["mar126"])
    f["vc63_252"] = np.log(f["mar63"] / f["mar252"])
    f["rc10_63"] = np.log(f["mrg10"] / f["mrg63"])
    f["rc21_252"] = np.log(f["mrg21"] / f["mrg252"])
    f["rc63_252"] = np.log(f["mrg63"] / f["mrg252"])
    # rate of change of volatility (compression speed) and its acceleration
    f["vroc21"] = np.log(f["mar21"] / pd.Series(f["mar21"]).shift(21).to_numpy())
    f["vroc63"] = np.log(f["mar63"] / pd.Series(f["mar63"]).shift(63).to_numpy())
    f["rroc21"] = np.log(f["mrg21"] / pd.Series(f["mrg21"]).shift(21).to_numpy())
    f["rroc63"] = np.log(f["mrg63"] / pd.Series(f["mrg63"]).shift(63).to_numpy())
    f["vacc21"] = f["vroc21"] - pd.Series(f["vroc21"]).shift(21).to_numpy()
    f["racc21"] = f["rroc21"] - pd.Series(f["rroc21"]).shift(21).to_numpy()
    # where today's volatility sits inside its own 1y history (rank, not z)
    f["mar21_pct252"] = pd.Series(f["mar21"]).rolling(252, min_periods=200).rank(pct=True).to_numpy()
    f["mrg21_pct252"] = pd.Series(f["mrg21"]).rolling(252, min_periods=200).rank(pct=True).to_numpy()
    # up-move vs down-move median size
    up = r.where(r > 0)
    dn = (-r).where(r < 0)
    f["asym63"] = np.log(roll_med(up, 63, 15).to_numpy() / roll_med(dn, 63, 15).to_numpy())
    f["asym21"] = np.log(roll_med(up, 21, 5).to_numpy() / roll_med(dn, 21, 5).to_numpy())

    # ---- price ROC scaled by typical move (drift-to-noise, linear scaling)
    for k in (21, 63, 126):
        f[f"dn{k}"] = f[f"roc{k}"] / (k * f[f"mar{min(k, 126)}"])

    # ---- volume: relative volume on medians, volume ROC, up/down volume
    mv = {k: roll_med(V, k, int(k * 0.8)).to_numpy() for k in (5, 10, 21, 63, 126, 252)}
    f["rv5_63"] = np.log(mv[5] / mv[63])
    f["rv10_126"] = np.log(mv[10] / mv[126])
    f["rv21_252"] = np.log(mv[21] / mv[252])
    f["rv63_252"] = np.log(mv[63] / mv[252])
    f["vlroc21"] = np.log(mv[21] / pd.Series(mv[21]).shift(21).to_numpy())
    f["vlroc63"] = np.log(mv[63] / pd.Series(mv[63]).shift(63).to_numpy())
    f["vlacc21"] = f["vlroc21"] - pd.Series(f["vlroc21"]).shift(21).to_numpy()
    vu = V.where(r > 0, 0.0)
    vd = V.where(r < 0, 0.0)
    for k in (21, 63):
        f[f"udv{k}"] = np.log((roll_sum(vu, k).to_numpy() + 1) / (roll_sum(vd, k).to_numpy() + 1))
    f["udv21_roc"] = f["udv21"] - pd.Series(f["udv21"]).shift(21).to_numpy()
    f["ldv63"] = np.log(roll_med(pd.Series(c * v), 63, 50).to_numpy() + 1)
    # volume turbulence: median absolute day-over-day log volume change
    lv = np.log(V)
    f["vnoise21"] = roll_med((lv - lv.shift(1)).abs(), 21).to_numpy()
    f["vnoise_roc21"] = np.log(f["vnoise21"] / pd.Series(f["vnoise21"]).shift(21).to_numpy())

    # ---- drawdown structure
    for k in (63, 252):
        f[f"ddh{k}"] = x - xs.rolling(k, min_periods=int(k * 0.8)).max().to_numpy()
    for k in (21, 63, 126):
        f[f"mdd{k}"] = rolling_maxdd(x, k)
    f["mdd63_rel"] = np.log((f["mdd63"] + 1e-4) / (roll_med(pd.Series(f["mdd63"]), 504, 250).to_numpy() + 1e-4))
    f["mdd63_mar"] = f["mdd63"] / f["mar63"]               # drawdown in typical-move units
    f["gp63"] = f["roc63"] / (f["mdd63"] + 0.01)           # gain-to-pain, trailing
    f["gp21"] = f["roc21"] / (f["mdd21"] + 0.01)
    f["gp126"] = f["roc126"] / (f["mdd126"] + 0.01)
    f["mdd_roc"] = f["mdd63"] - pd.Series(f["mdd63"]).shift(63).to_numpy()

    # ---- path geometry / fractal
    for k in (5, 10, 21, 63, 126):
        f[f"er{k}"] = f[f"roc{k}"] / roll_sum(a, k).to_numpy()
    f["er_min"] = np.minimum(np.minimum(f["er10"], f["er21"]), f["er63"])
    f["er_prod"] = np.sign(f["er21"]) * np.abs(f["er10"] * f["er21"] * f["er63"]) ** (1 / 3)
    f["er_roc21"] = f["er21"] - pd.Series(f["er21"]).shift(21).to_numpy()
    f["er_roc63"] = f["er63"] - pd.Series(f["er63"]).shift(21).to_numpy()
    f["er63_126"] = f["er63"] - f["er126"]
    f["kt21"] = kendall_trend(x, 21)
    f["kt63"] = kendall_trend(x, 63)
    f["kt126"] = kendall_trend(x, 126)
    f["kt_roc"] = f["kt63"] - pd.Series(f["kt63"]).shift(21).to_numpy()
    f["hexp126"] = scaling_exponent(x, 126)
    f["hexp252"] = scaling_exponent(x, 252)
    f["hexp_roc"] = f["hexp126"] - pd.Series(f["hexp126"]).shift(63).to_numpy()
    # staircase: share of 5-bar blocks whose low/high steps above the prior block
    low5 = pd.Series(lo).rolling(5).min()
    high5 = pd.Series(h).rolling(5).max()
    hl = (low5 > low5.shift(5)).astype(float).where(low5.shift(5).notna())
    hh = (high5 > high5.shift(5)).astype(float).where(high5.shift(5).notna())
    hl_blocks = sum(hl.shift(5 * i) for i in range(12)) / 12.0
    hh_blocks = sum(hh.shift(5 * i) for i in range(12)) / 12.0
    f["hl_steps63"] = hl_blocks.to_numpy()
    f["hh_steps63"] = hh_blocks.to_numpy()
    f["hl_steps21"] = (sum(hl.shift(5 * i) for i in range(4)) / 4.0).to_numpy()
    cmax63 = xs.rolling(63, min_periods=50).max()
    f["nh_freq21"] = (xs >= cmax63 - 1e-12).astype(float).rolling(21).mean().to_numpy()
    f["nh_freq63"] = (xs >= cmax63 - 1e-12).astype(float).rolling(63).mean().to_numpy()
    f["upfrac21"] = (r > 0).astype(float).rolling(21).mean().to_numpy()
    f["upfrac63"] = (r > 0).astype(float).rolling(63).mean().to_numpy()
    rngd = np.where(h > lo, h - lo, np.nan)
    clv = pd.Series(np.where(np.isnan(rngd), 0.5, (c - lo) / rngd))
    f["clv21"] = roll_med(clv, 21).to_numpy()
    f["clv63"] = roll_med(clv, 63).to_numpy()
    # overnight vs session contribution
    ov = pd.Series(np.log(o / np.roll(c, 1)))
    ov.iloc[0] = np.nan
    ses = pd.Series(np.log(c / o))
    f["ovn63"] = roll_sum(ov, 63).to_numpy() / (63 * f["mar63"])
    f["ses63"] = roll_sum(ses, 63).to_numpy() / (63 * f["mar63"])
    f["gapfreq63"] = (ov.abs() > 2 * pd.Series(f["mar63"])).astype(float).rolling(63).mean().to_numpy()
    # base: tightness of the 63-bar window that ended 63 bars ago, and now
    rng63 = (xs.rolling(63).max() - xs.rolling(63).min()).to_numpy()
    f["rng63"] = rng63
    f["rng63_lag63"] = pd.Series(rng63).shift(63).to_numpy()
    f["rng63_ratio"] = np.log((rng63 + 1e-4) / (f["rng63_lag63"] + 1e-4))
    f["base_tight"] = f["rng63_lag63"] / (63 * pd.Series(f["mar63"]).shift(63).to_numpy())

    # ---- base -> breakout -> hold, at three scales (same geometry, stretched)
    for w in (21, 63, 126):
        base_hi = xs.rolling(w).max().shift(w)
        base_lo = xs.rolling(w).min().shift(w)
        base_path = roll_sum(a, w).shift(w)
        f[f"bo_hold{w}"] = (xs.rolling(w).min() - base_hi).to_numpy()   # >0: never fell back into the base
        f[f"bo_gain{w}"] = (xs - base_hi).to_numpy()                     # distance above the base top
        f[f"bo_tight{w}"] = ((base_hi - base_lo) / base_path).to_numpy() # base range / base path length

    # ---- return-distribution shape (quantiles, no moments)
    f["tail63"] = np.log(a.rolling(63, min_periods=50).quantile(0.9).to_numpy() / f["mar63"])
    f["worst63"] = (-r).rolling(63, min_periods=50).max().to_numpy() / f["mar63"]
    f["best63"] = r.rolling(63, min_periods=50).max().to_numpy() / f["mar63"]
    f["jump_asym63"] = np.log(np.maximum(f["best63"], 1e-6) / np.maximum(f["worst63"], 1e-6))
    # ---- time spent under water
    uw = (xs.rolling(63, min_periods=50).max() - xs)
    f["med_uw63"] = roll_med(uw, 63, 50).to_numpy()
    f["med_uw63_mar"] = f["med_uw63"] / f["mar63"]
    f["hi_frac63"] = (uw <= 0.02).astype(float).rolling(63).mean().to_numpy()
    # ---- volatility of volatility (median absolute weekly change of log vol)
    lm = np.log(pd.Series(f["mar21"]))
    f["vov63"] = roll_med((lm - lm.shift(5)).abs(), 63, 50).to_numpy()

    # ---- co-movement with the market (sign agreement, conditional medians)
    if mkt is not None:
        rm = mkt.reindex(df["date"]).to_numpy()
        rms = pd.Series(rm)
        agree = pd.Series(np.where((r.to_numpy() != 0) & (rm != 0) & ~np.isnan(rm),
                                   (np.sign(r.to_numpy()) == np.sign(rm)).astype(float), np.nan))
        f["co126"] = agree.rolling(126, min_periods=90).mean().to_numpy()
        ex = r - rms
        f["rel_dn126"] = roll_med(ex.where(rms < 0), 126, 40).to_numpy()   # vs market on down days
        f["rel_up126"] = roll_med(ex.where(rms > 0), 126, 40).to_numpy()   # vs market on up days
        f["rel_dn126_mar"] = f["rel_dn126"] / f["mar126"]
        f["rs63"] = f["roc63"] - (rms.rolling(63).sum()).to_numpy()
        f["rs_roc21"] = f["rs63"] - pd.Series(f["rs63"]).shift(21).to_numpy()

    # ---- forward outcomes
    rr = r.to_numpy()
    for hz in HORIZONS:
        fr, fdd, fae, fer, fmar = forward_path_stats(x, rr, hz)
        f[f"fr{hz}"] = fr
        f[f"fdd{hz}"] = fdd
        f[f"fae{hz}"] = fae
        f[f"fer{hz}"] = fer
        f[f"fmar{hz}"] = fmar

    out = pd.DataFrame(f)
    out.insert(0, "date", df["date"].to_numpy())
    out = out[out["date"].isin(sample_dates) & (out["bars"] >= MIN_HISTORY)]
    return out


def _work(args):
    tkr, df, sample_dates, mkt = args
    try:
        out = ticker_features(df, sample_dates, mkt)
    except Exception as e:  # keep the run going; report at the end
        print("FAIL", tkr, e, flush=True)
        return None
    if out is None or out.empty:
        return None
    out.insert(1, "ticker", tkr)
    fcols = [c for c in out.columns if c not in ("date", "ticker")]
    out[fcols] = out[fcols].astype("float32")
    return out


def main():
    px = pd.read_parquet(os.path.join(DATA_DIR, "ohlcv.parquet"))
    cal = np.sort(px["date"].unique())
    # weekly grid anchored on the last bar so the latest session is always included
    sample_dates = set(cal[::-1][::SAMPLE_EVERY])
    spy = yf.download("SPY", start="2003-01-01", auto_adjust=True, progress=False)["Close"].squeeze()
    spy.index = pd.to_datetime(spy.index).tz_localize(None)
    mkt = np.log(spy).diff()
    groups = [(t, g, sample_dates, mkt) for t, g in px.groupby("ticker", sort=False)]
    with Pool(4) as pool:
        parts = [p for p in pool.imap_unordered(_work, groups, chunksize=8) if p is not None]
    panel = pd.concat(parts, ignore_index=True).sort_values(["date", "ticker"])
    panel = panel.replace([np.inf, -np.inf], np.nan)
    panel.to_parquet(os.path.join(DATA_DIR, "panel_raw.parquet"), index=False)
    print("panel", panel.shape, panel["ticker"].nunique(), "tickers",
          panel["date"].min(), "->", panel["date"].max())


if __name__ == "__main__":
    main()
