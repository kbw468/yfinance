"""Forward-looking targets. Everything here uses returns strictly AFTER date t (t+1 .. t+h)."""
import numpy as np
import pandas as pd

from .config import HORIZONS, MARKET, SECTOR_ETF_FALLBACK

ANN = np.sqrt(252.0)


def _fwd_window_stats(r: pd.DataFrame, h: int) -> dict:
    """Mean, std, downside dev over the window t+1..t+h, aligned to t.
    rolling(h) ending at t+h covers r[t+1..t+h]; shift(-h) moves that value back to row t."""
    mean = r.rolling(h, min_periods=h).mean().shift(-h)
    std = r.rolling(h, min_periods=h).std(ddof=1).shift(-h)
    neg = r.clip(upper=0.0)
    ddev = np.sqrt((neg ** 2).rolling(h, min_periods=h).mean()).shift(-h)
    return {"mean": mean, "std": std, "ddev": ddev}


def sharpe_sortino(r: pd.DataFrame, h: int):
    s = _fwd_window_stats(r, h)
    sharpe = (s["mean"] / s["std"]) * ANN
    sortino = (s["mean"] / s["ddev"].replace(0, np.nan)) * ANN
    return sharpe, sortino


def sector_bench_returns(r: pd.DataFrame, universe: pd.DataFrame) -> pd.DataFrame:
    """Per-ticker benchmark return series = its sector ETF, falling back to the parent ETF before inception."""
    out = {}
    for t in universe.index:
        if t not in r.columns:
            continue
        etf = universe.loc[t, "SectorETF"]
        b = r[etf]
        fb = SECTOR_ETF_FALLBACK.get(etf)
        if fb is not None:
            b = b.where(b.notna(), r[fb])
        out[t] = b
    return pd.DataFrame(out, index=r.index)


def build_targets(close: pd.DataFrame, universe: pd.DataFrame, horizons=HORIZONS) -> dict:
    """Returns dict name -> wide DataFrame (date x stock). Names:
         xs_spy_sharpe_{h}:  fwd Sharpe(stock) - fwd Sharpe(SPY)
         xs_sec_sharpe_{h}:  fwd Sharpe(stock) - fwd Sharpe(sector ETF)
         xs_spy_sortino_{h}, xs_sec_sortino_{h}: same with Sortino
         act_spy_sharpe_{h}: Sharpe of the active return (stock - SPY)   (information-ratio style)
         fwd_ret_{h}:        plain forward return (diagnostic only)
    """
    r = close.pct_change()
    stocks = [t for t in universe.index if t in r.columns]
    rs = r[stocks]
    r_spy = r[MARKET]
    r_sec = sector_bench_returns(r, universe)[stocks]
    out = {}
    for h in horizons:
        sh_s, so_s = sharpe_sortino(rs, h)
        sh_m, so_m = sharpe_sortino(r_spy.to_frame(MARKET), h)
        sh_sec, so_sec = sharpe_sortino(r_sec, h)
        out[f"xs_spy_sharpe_{h}"] = sh_s.sub(sh_m[MARKET], axis=0)
        out[f"xs_spy_sortino_{h}"] = so_s.sub(so_m[MARKET], axis=0)
        out[f"xs_sec_sharpe_{h}"] = sh_s - sh_sec
        out[f"xs_sec_sortino_{h}"] = so_s - so_sec
        act, _ = sharpe_sortino(rs.sub(r_spy, axis=0), h)
        out[f"act_spy_sharpe_{h}"] = act
        out[f"fwd_ret_{h}"] = close[stocks].shift(-h) / close[stocks] - 1
    return out


def forward_path_metrics(close: pd.DataFrame, h: int) -> dict:
    """Shape of the forward path t+1..t+h: max drawdown inside the window, straightness (R^2 of log price on time),
    share of up days. A steadily rising, drawdown-free path scores mdd ~ 0, r2 ~ 1, up_frac high."""
    logc = np.log(close)
    run_max = close.copy()
    mdd = pd.DataFrame(0.0, index=close.index, columns=close.columns)
    k = np.arange(1, h + 1, dtype=float)
    kbar = k.mean(); skk = ((k - kbar) ** 2).sum()
    sy = pd.DataFrame(0.0, index=close.index, columns=close.columns); syy = sy.copy(); sky = sy.copy(); valid = pd.DataFrame(True, index=close.index, columns=close.columns)
    for i in range(1, h + 1):
        px = close.shift(-i)
        run_max = np.maximum(run_max, px)
        mdd = np.minimum(mdd, px / run_max - 1)
        y = logc.shift(-i)
        sy += y; syy += y ** 2; sky += (i - kbar) * y
        valid &= px.notna()
    n = float(h)
    syy_c = syy - sy ** 2 / n
    r2 = (sky ** 2 / skk) / syy_c.where(syy_c > 1e-12)
    r = close.pct_change()
    up_frac = (r > 0).astype(float).where(r.notna()).rolling(h, min_periods=h).mean().shift(-h)
    slope_sign = np.sign(sky)
    return {"mdd": mdd.where(valid), "r2": (r2 * slope_sign).where(valid), "up_frac": up_frac}


def smooth_components(close: pd.DataFrame, horizons=HORIZONS) -> dict:
    """Per-ticker forward path components only (fwd_sharpe/mdd/r2/up per horizon); percentiles are taken later."""
    r = close.pct_change()
    out = {}
    for h in horizons:
        sh, _ = sharpe_sortino(r, h)
        pm = forward_path_metrics(close, h)
        out[f"fwd_sharpe_{h}"] = sh; out[f"fwd_mdd_{h}"] = pm["mdd"]; out[f"fwd_r2_{h}"] = pm["r2"]; out[f"fwd_up_{h}"] = pm["up_frac"]
    return out


def smooth_path_target(close: pd.DataFrame, horizons=HORIZONS, beta: pd.DataFrame | None = None) -> dict:
    """Forward 'smooth climb' score. Four path components over t+1..t+h: Sharpe, -max drawdown, signed R^2 of the
    climb, up-day share. Each is converted to a per-date percentile WITHIN THE NAME'S BETA TERCILE (so a low-vol
    utility is only credited for a smoother path than other low-beta names), then averaged.
    smooth_{h} per horizon; y_smooth = mean across horizons. Raw components are returned too."""
    r = close.pct_change()
    if beta is not None:
        rk = beta.rank(axis=1, pct=True)
        bucket = pd.DataFrame(np.select([rk <= 1 / 3, rk <= 2 / 3], [0, 1], 2), index=rk.index, columns=rk.columns).where(rk.notna())
    else:
        bucket = None

    def pct_within(x: pd.DataFrame) -> pd.DataFrame:
        if bucket is None:
            return x.rank(axis=1, pct=True)
        out = pd.DataFrame(np.nan, index=x.index, columns=x.columns)
        for b in (0, 1, 2):
            m = bucket == b
            out = out.where(~m, x.where(m).rank(axis=1, pct=True))
        return out

    out = {}
    for h in horizons:
        sh, _ = sharpe_sortino(r, h)
        pm = forward_path_metrics(close, h)
        out[f"fwd_sharpe_{h}"] = sh
        out[f"fwd_mdd_{h}"] = pm["mdd"]
        out[f"fwd_r2_{h}"] = pm["r2"]
        out[f"fwd_up_{h}"] = pm["up_frac"]
        parts = [pct_within(sh), pct_within(-pm["mdd"]), pct_within(pm["r2"]), pct_within(pm["up_frac"])]
        out[f"smooth_{h}"] = sum(parts) / 4
    out["y_smooth"] = sum(out[f"smooth_{h}"] for h in horizons) / len(horizons)
    return out


def blended_rank_target(targets: dict, key="xs_spy_sharpe", horizons=HORIZONS) -> pd.DataFrame:
    """Cross-sectional percentile (per date) of excess Sharpe, averaged across horizons. Used as the ML label."""
    ranks = [targets[f"{key}_{h}"].rank(axis=1, pct=True) for h in horizons]
    return sum(ranks) / len(ranks)
