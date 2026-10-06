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


def blended_rank_target(targets: dict, key="xs_spy_sharpe", horizons=HORIZONS) -> pd.DataFrame:
    """Cross-sectional percentile (per date) of excess Sharpe, averaged across horizons. Used as the ML label."""
    ranks = [targets[f"{key}_{h}"].rank(axis=1, pct=True) for h in horizons]
    return sum(ranks) / len(ranks)
