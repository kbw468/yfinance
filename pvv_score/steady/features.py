"""Non-Gaussian feature library, computed from the cleaned OHLCV panels (dates x tickers), point-in-time.

No standard deviations, variances, z-scores, Sharpe ratios, OLS regressions, correlations, skew or kurtosis anywhere.
Building blocks: returns over windows, medians, empirical percentiles (own history and cross-section), running maxima and
drawdown paths, counts and shares of days, sums of gains vs sums of losses, and up/down capture against SPY.
Beta is reported as the L1 capture beta: sum(r_stock * sign(r_spy)) / sum(|r_spy|) over the window."""
import numpy as np
import pandas as pd


def _days_since_max(C: pd.DataFrame, w: int) -> pd.DataFrame:
    """Sessions since the close last printed its own trailing-w maximum (= sessions since the current w-session high)."""
    at = C.ge(C.rolling(w, min_periods=w).max())
    pos = pd.DataFrame(np.where(at.values, np.arange(len(C))[:, None], np.nan), index=C.index, columns=C.columns).ffill()
    return pd.DataFrame(np.arange(len(C))[:, None] - pos.values, index=C.index, columns=C.columns).where(C.notna())


def _path_drawdowns(C: pd.DataFrame, w: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Inside each trailing w-session window: deepest close-to-running-high drawdown, and the mean drawdown (time under water)."""
    a = C.values; n, k = a.shape
    mdd = np.full((n, k), np.nan); mean_dd = np.full((n, k), np.nan)
    for i in range(w - 1, n):
        seg = a[i - w + 1:i + 1]
        dd = seg / np.fmax.accumulate(seg, axis=0) - 1
        full = ~np.isnan(seg).any(axis=0)
        mdd[i, full] = dd[:, full].min(axis=0); mean_dd[i, full] = dd[:, full].mean(axis=0)
    return pd.DataFrame(mdd, C.index, C.columns), pd.DataFrame(mean_dd, C.index, C.columns)


def compute(panel: dict, sector_bench: pd.DataFrame, stocks: list) -> dict:
    C, H, L, O, V = (panel[k][stocks] for k in ("Close", "High", "Low", "Open", "Volume"))
    spy = panel["Close"]["SPY"]
    r = C.pct_change(fill_method=None); rs = spy.pct_change(fill_method=None)
    absr = r.abs(); up = (r > 0).astype(float).where(r.notna()); dn = (r < 0).astype(float).where(r.notna())
    F = {}
    # price rate of change, excess vs SPY and vs sector
    for w in (5, 10, 21, 63, 126, 252):
        F[f"roc_{w}"] = C / C.shift(w) - 1
    for w in (21, 63, 126, 252):
        F[f"xs_spy_{w}"] = F[f"roc_{w}"].sub(spy / spy.shift(w) - 1, axis=0)
    sb = sector_bench[stocks]
    for w in (63, 126):
        F[f"xs_sec_{w}"] = F[f"roc_{w}"] - (sb / sb.shift(w) - 1)
    # position vs highs / lows
    for w in (21, 63, 126, 252):
        F[f"off_high_{w}"] = C / C.rolling(w, min_periods=w).max() - 1
    F["off_low_252"] = C / C.rolling(252, min_periods=252).min() - 1
    F["days_since_high_252"] = _days_since_max(C, 252); F["days_since_high_63"] = _days_since_max(C, 63)
    # drawdown paths
    for w in (21, 63, 126, 252):
        F[f"mdd_{w}"], md = _path_drawdowns(C, w)
        if w in (63, 126): F[f"mean_dd_{w}"] = md
    # path quality: efficiency, shares of up days / up pairs / new highs, gain-to-pain, higher lows
    lr = np.log(C).diff()
    for w in (21, 63, 126):
        F[f"eff_{w}"] = (np.log(C) - np.log(C.shift(w))) / lr.abs().rolling(w, min_periods=w).sum()
        F[f"up_share_{w}"] = up.rolling(w, min_periods=w).mean()
    for lag in (10, 21):
        F[f"up_pair_{lag}_63"] = (C > C.shift(lag)).astype(float).where(C.shift(lag).notna()).rolling(63, min_periods=63).mean()
    F["new_high63_share_63"] = C.ge(C.rolling(63, min_periods=63).max()).astype(float).where(C.notna()).rolling(63, min_periods=63).mean()
    F["new_high252_count_63"] = C.ge(C.rolling(252, min_periods=252).max()).astype(float).where(C.notna()).rolling(63, min_periods=63).sum()
    for w in (63, 126):
        F[f"gain_pain_{w}"] = r.clip(lower=0).rolling(w, min_periods=w).sum() / (-r.clip(upper=0)).rolling(w, min_periods=w).sum()
    lmin10 = L.rolling(10, min_periods=10).min()
    F["higher_lows_6"] = sum((lmin10.shift(10 * k) > lmin10.shift(10 * (k + 1))).astype(float) for k in range(6)).where(lmin10.shift(60).notna())
    F["close_loc_21"] = ((C - L) / (H - L).replace(0, np.nan)).rolling(21, min_periods=15).median()
    F["gap_share_21"] = (O / C.shift(1) - 1).abs().rolling(21, min_periods=21).sum() / absr.rolling(21, min_periods=21).sum()
    # volatility without variance: medians of absolute moves and ranges, their rates of change, own-history percentile
    mad = {w: absr.rolling(w, min_periods=w).median() for w in (5, 21, 63, 252)}
    for w in (21, 63, 252): F[f"mad_{w}"] = mad[w]
    F["vol_roc_21_252"] = mad[21] / mad[252]; F["vol_roc_5_63"] = mad[5] / mad[63]; F["vol_roc_21_vs_21ago"] = mad[21] / mad[21].shift(21)
    F["vol_pctile_own_252"] = mad[21].rolling(252, min_periods=252).rank(pct=True)
    rng = (H - L) / C
    mr = {w: rng.rolling(w, min_periods=w).median() for w in (10, 21, 252)}
    F["range_comp_10_252"] = mr[10] / mr[252]; F["range_roc_21_252"] = mr[21] / mr[252]
    F["down_up_move_63"] = absr.where(r < 0).rolling(63, min_periods=20).median() / r.where(r > 0).rolling(63, min_periods=20).median()
    F["worst_day_63"] = r.rolling(63, min_periods=63).min() / mad[252]
    # volume, medians only
    mv = {w: V.rolling(w, min_periods=w).median() for w in (5, 10, 21, 63, 252)}
    F["volume_roc_5_63"] = mv[5] / mv[63]; F["volume_roc_10_63"] = mv[10] / mv[63]; F["volume_roc_21_252"] = mv[21] / mv[252]
    F["volume_pctile_own_252"] = mv[5].rolling(252, min_periods=252).rank(pct=True)
    for w in (21, 63):
        F[f"up_volume_share_{w}"] = (V * up).rolling(w, min_periods=w).sum() / V.rolling(w, min_periods=w).sum()
        F[f"pullback_volume_{w}"] = V.where(r < 0).rolling(w, min_periods=max(5, w // 4)).median() / V.where(r > 0).rolling(w, min_periods=max(5, w // 4)).median()
    vup = V > V.shift(1)
    F["accum_minus_dist_50"] = ((r > 0) & vup).astype(float).rolling(50, min_periods=50).sum() - ((r < 0) & vup).astype(float).rolling(50, min_periods=50).sum()
    F["log_dollar_vol_63"] = np.log((C * V).rolling(63, min_periods=40).median())
    # market relation, no regression: capture and L1 beta
    sp_up = (rs > 0).astype(float); sp_dn = (rs < 0).astype(float)
    for w in (63, 252):
        F[f"up_capture_{w}"] = r.mul(sp_up, axis=0).rolling(w, min_periods=w).sum().div((rs * sp_up).rolling(w, min_periods=w).sum(), axis=0)
        F[f"down_capture_{w}"] = r.mul(sp_dn, axis=0).rolling(w, min_periods=w).sum().div((rs * sp_dn).rolling(w, min_periods=w).sum(), axis=0)
        F[f"beta_l1_{w}"] = r.mul(np.sign(rs), axis=0).rolling(w, min_periods=w).sum().div(rs.abs().rolling(w, min_periods=w).sum(), axis=0)
    F["capture_spread_252"] = F["up_capture_252"] - F["down_capture_252"]
    F["defend_share_63"] = (r > 0).astype(float).mul(sp_dn, axis=0).rolling(63, min_periods=63).sum().div(sp_dn.rolling(63, min_periods=63).sum(), axis=0)
    rsl = C.div(spy, axis=0)
    F["rs_off_high_252"] = rsl / rsl.rolling(252, min_periods=252).max() - 1
    # own-history percentiles of the core state variables
    for k in ("roc_63", "xs_spy_63", "off_high_252", "volume_roc_21_252", "up_capture_63"):
        F[f"{k}_pctile_own"] = F[k].rolling(252, min_periods=252).rank(pct=True)
    return {k: v.astype("float32") for k, v in F.items()}
