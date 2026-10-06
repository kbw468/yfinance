"""Point-in-time factor library. Every factor at row t uses only data with index <= t.

Inputs are wide (date x ticker) adjusted O/H/L/C and split-adjusted Volume. All rolling windows are
in trading days. Where a factor is naturally a ratio to the ticker's own history (percentile-style),
we use ratio-to-rolling-median or a rolling z-score, which is equivalent in rank space per ticker and
far cheaper than rolling percentile ranks on a 500-column panel.

Registry: FACTORS = {name: (group, hypothesis_sign, description)}. Sign is the PRIOR only; the
backtest decides, and a factor whose empirical sign disagrees with the prior is still kept if robust.
"""
import numpy as np
import pandas as pd

from .config import MARKET

EPS = 1e-12
ANN = np.sqrt(252.0)

FACTORS: dict[str, tuple[str, int, str]] = {}


def _reg(name, group, sign, desc):
    FACTORS[name] = (group, sign, desc)


# ----------------------------------------------------------------------------- helpers
def _days_since(flag: pd.DataFrame, cap: int) -> pd.DataFrame:
    """Trading days since `flag` was last True (per column). cap where never seen."""
    idx = np.arange(len(flag))[:, None] * np.ones((1, flag.shape[1]))
    marks = pd.DataFrame(np.where(flag.values, idx, np.nan), index=flag.index, columns=flag.columns).ffill()
    ds = pd.DataFrame(idx, index=flag.index, columns=flag.columns) - marks
    return ds.fillna(cap).clip(upper=cap)


def _streak(up: pd.DataFrame) -> pd.DataFrame:
    """Length of the current run of consecutive True values (0 if last value False)."""
    u = up.fillna(False).astype(int)
    grp = (u != u.shift()).cumsum()
    out = pd.DataFrame(index=u.index, columns=u.columns, dtype=float)
    for c in u.columns:
        out[c] = u[c].groupby(grp[c]).cumsum() * u[c]
    return out


def _zs(x: pd.DataFrame, w: int) -> pd.DataFrame:
    return (x - x.rolling(w).mean()) / (x.rolling(w).std() + EPS)


def _beta_corr(rs: pd.DataFrame, rm: pd.Series, w: int):
    cov = rs.rolling(w).cov(rm)
    var_m = rm.rolling(w).var()
    beta = cov.div(var_m, axis=0)
    corr = rs.rolling(w).corr(rm)
    return beta, corr


# ----------------------------------------------------------------------------- main
def compute_features(panel: dict, universe: pd.DataFrame, sector_bench: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """panel: cleaned dict of wide frames. sector_bench: wide frame of each stock's sector-ETF CLOSE (aligned).
    Returns dict factor -> wide float32 DataFrame restricted to universe stocks."""
    stocks = [t for t in universe.index if t in panel["Close"].columns]
    O, H, L, C, V = (panel[f][stocks] for f in ["Open", "High", "Low", "Close", "Volume"])
    Cm = panel["Close"][MARKET]
    F: dict[str, pd.DataFrame] = {}

    r = C.pct_change()
    lr = np.log(C).diff()
    rm = Cm.pct_change()
    lrm = np.log(Cm).diff()
    dv = C * V
    rng = (H - L) / C                         # daily range as % of close
    gap = O / C.shift(1) - 1                  # overnight gap
    intra = C / O - 1                         # open-to-close

    # ====================================================== PRICE STRUCTURE
    hi252 = H.rolling(252).max()
    F["dist_52w_high"] = C / hi252 - 1
    _reg("dist_52w_high", "price", +1, "close vs 252d high (0 = at high)")

    F["days_since_52w_high"] = _days_since(H >= hi252, 252)
    _reg("days_since_52w_high", "price", -1, "sessions since last 52w high print")

    F["base_depth_126"] = C / H.rolling(126).max() - 1
    _reg("base_depth_126", "price", +1, "depth below 126d high")
    F["base_depth_63"] = C / H.rolling(63).max() - 1
    _reg("base_depth_63", "price", +1, "depth below 63d high")

    F["base_tightness_63"] = (H.rolling(63).max() - L.rolling(63).min()) / C
    _reg("base_tightness_63", "price", -1, "63d high-low range / close (tight base = small)")

    F["range_comp_10_252"] = rng.rolling(10).mean() / (rng.rolling(252).median() + EPS)
    _reg("range_comp_10_252", "price", -1, "10d avg daily range vs own 252d median (compression < 1)")
    F["range_comp_z"] = _zs(rng.rolling(10).mean(), 252)
    _reg("range_comp_z", "price", -1, "z-score of 10d avg range vs 252d")

    bbw = 4 * C.rolling(20).std() / C.rolling(20).mean()
    F["bbw_pct_252"] = bbw / (bbw.rolling(252).median() + EPS)
    _reg("bbw_pct_252", "price", -1, "Bollinger width vs own 252d median")
    F["bbw_squeeze_126"] = bbw / (bbw.rolling(126).min() + EPS)
    _reg("bbw_squeeze_126", "price", -1, "Bollinger width / 126d min width (1 = tightest in 6m)")

    nr7 = (rng <= rng.rolling(7).min() + EPS).astype(float)
    F["nr7_count_20"] = nr7.rolling(20).sum()
    _reg("nr7_count_20", "price", +1, "number of NR7 days in last 20")

    low10 = L.rolling(10).min()
    hl = sum((low10.shift(10 * k) > low10.shift(10 * (k + 1))).astype(float) for k in range(6)) / 6
    F["higher_low_cadence"] = hl.where(low10.shift(60).notna())
    _reg("higher_low_cadence", "price", +1, "share of 6 successive 10d-lows that were higher than the prior")

    F["gap_share_21"] = gap.abs().rolling(21).sum() / (r.abs().rolling(21).sum() + EPS)
    _reg("gap_share_21", "price", 0, "share of 21d absolute movement that occurred overnight")
    F["net_gap_21"] = gap.rolling(21).sum()
    _reg("net_gap_21", "price", +1, "sum of overnight gaps 21d")
    F["up_gap_freq_21"] = (gap > 0.005).astype(float).rolling(21).sum() / 21
    _reg("up_gap_freq_21", "price", +1, "frequency of >0.5% up gaps, 21d")

    clv = ((C - L) / (H - L + EPS)).where(H > L)
    F["clv_21"] = clv.rolling(21).mean()
    _reg("clv_21", "price", +1, "mean close location in daily range, 21d")
    F["clv_5"] = clv.rolling(5).mean()
    _reg("clv_5", "price", +1, "mean close location in daily range, 5d")

    F["mom_12_1"] = C.shift(21) / C.shift(252) - 1
    _reg("mom_12_1", "price", +1, "12-1 month momentum")
    F["mom_6_1"] = C.shift(21) / C.shift(126) - 1
    _reg("mom_6_1", "price", +1, "6-1 month momentum")
    F["ret_63"] = C / C.shift(63) - 1
    _reg("ret_63", "price", +1, "63d return")
    F["ret_21"] = C / C.shift(21) - 1
    _reg("ret_21", "price", +1, "21d return")
    F["ret_5"] = C / C.shift(5) - 1
    _reg("ret_5", "price", -1, "5d return (short-term reversal prior)")

    mx20, mn20 = H.rolling(20).max(), L.rolling(20).min()
    F["pos_in_20d_range"] = (C - mn20) / (mx20 - mn20 + EPS)
    _reg("pos_in_20d_range", "price", +1, "close position within 20d high-low range")

    F["up_day_frac_21"] = (r > 0).astype(float).where(r.notna()).rolling(21).mean()
    _reg("up_day_frac_21", "price", +1, "fraction of up days, 21d")
    F["up_day_frac_63"] = (r > 0).astype(float).where(r.notna()).rolling(63).mean()
    _reg("up_day_frac_63", "price", +1, "fraction of up days, 63d")

    F["pullback_21"] = C / H.rolling(21).max() - 1
    _reg("pullback_21", "run", +1, "pullback from 21d high")
    F["pullback_duration_21"] = _days_since(H >= H.rolling(21).max(), 21)
    _reg("pullback_duration_21", "run", -1, "sessions since 21d high")

    roll_max = C.rolling(63, min_periods=1).max()
    F["mdd_63"] = (C / roll_max - 1).rolling(63).min()
    _reg("mdd_63", "run", +1, "worst 63d drawdown (less negative = smoother)")

    lo252 = L.rolling(252).min()
    F["dist_52w_low"] = C / lo252 - 1
    _reg("dist_52w_low", "run", 0, "gain off 252d low")
    F["ceiling_2x_low"] = (F["dist_52w_low"] >= 1.0).astype(float).where(lo252.notna())
    _reg("ceiling_2x_low", "run", -1, "already >= 100% off 52w low (user ceiling)")
    cross2x = (F["dist_52w_low"] >= 1.0) & (F["dist_52w_low"].shift(1) < 1.0)
    F["days_since_2x_low"] = _days_since(cross2x, 252).where(lo252.notna())
    _reg("days_since_2x_low", "run", 0, "sessions since crossing 2x the 52w low (252 if not)")

    # ====================================================== RUN CHARACTER
    F["streak_now"] = _streak(r > 0) - _streak(r < 0)
    _reg("streak_now", "run", 0, "signed length of current consecutive close streak")
    F["max_up_streak_21"] = _streak(r > 0).rolling(21).max()
    _reg("max_up_streak_21", "run", +1, "longest up-streak within 21d")

    dd252 = C / C.rolling(252, min_periods=126).max() - 1
    F["days_since_20pct_dd"] = _days_since(dd252 < -0.20, 504)
    _reg("days_since_20pct_dd", "run", +1, "sessions since drawdown from 252d high exceeded 20%")
    F["dd_from_252_high"] = dd252
    _reg("dd_from_252_high", "run", +1, "current drawdown from 252d closing high")

    # ====================================================== VOLUME
    v50 = V.rolling(50).mean()
    v250 = V.rolling(250).mean()
    F["vol_dry_10_50"] = V.rolling(10).mean() / (v50 + EPS)
    _reg("vol_dry_10_50", "volume", -1, "10d avg volume / 50d avg (dry-up < 1)")
    F["vol_dry_20_250"] = V.rolling(20).mean() / (v250 + EPS)
    _reg("vol_dry_20_250", "volume", -1, "20d avg volume / 250d avg")
    F["vol_ratio_5_50"] = V.rolling(5).mean() / (v50 + EPS)
    _reg("vol_ratio_5_50", "volume", 0, "5d avg volume / 50d avg")
    F["vol_z_1"] = (V - v50) / (V.rolling(50).std() + EPS)
    _reg("vol_z_1", "volume", 0, "today's volume z-score vs 50d")

    upv = (r > 0) & (V > V.shift(1))
    dnv = (r < 0) & (V > V.shift(1))
    F["accum_dist_50"] = (upv.astype(float) - dnv.astype(float)).where(r.notna()).rolling(50).sum()
    _reg("accum_dist_50", "volume", +1, "accumulation days minus distribution days, 50d")

    vup = (V * (r > 0)).rolling(50).sum()
    vdn = (V * (r < 0)).rolling(50).sum()
    F["updown_vol_ratio_50"] = np.log((vup + 1) / (vdn + 1))
    _reg("updown_vol_ratio_50", "volume", +1, "log up-volume / down-volume, 50d")
    vup21 = (V * (r > 0)).rolling(21).sum()
    vdn21 = (V * (r < 0)).rolling(21).sum()
    F["updown_vol_ratio_21"] = np.log((vup21 + 1) / (vdn21 + 1))
    _reg("updown_vol_ratio_21", "volume", +1, "log up-volume / down-volume, 21d")

    sd20 = r.rolling(20).std().shift(1)
    ign_day = (r > 2 * sd20) & (r > 0)
    vmult = (V / (v50.shift(1) + EPS)).where(ign_day)
    F["ignition_mult_10"] = vmult.rolling(10, min_periods=1).max().fillna(0.0).where(v50.notna())
    _reg("ignition_mult_10", "volume", +1, "max volume multiple on a +2sd up day within last 10 sessions (0 if none)")
    F["days_since_ignition"] = _days_since(ign_day & (V > 1.5 * v50.shift(1)), 63)
    _reg("days_since_ignition", "volume", 0, "sessions since last +2sd up day on 1.5x volume (63 cap)")

    dn_mask = (r < 0)
    up_mask = (r > 0)
    vol_dn = (V.where(dn_mask)).rolling(21, min_periods=3).mean()
    vol_up = (V.where(up_mask)).rolling(21, min_periods=3).mean()
    F["pullback_vol_21"] = np.log((vol_dn + 1) / (vol_up + 1))
    _reg("pullback_vol_21", "volume", -1, "log avg volume on down days / up days, 21d (quiet pullbacks < 0)")

    nvf63 = (np.sign(r) * V).rolling(63).sum() / (V.rolling(63).sum() + EPS)
    F["net_vol_flow_63"] = nvf63
    _reg("net_vol_flow_63", "volume", +1, "signed volume / total volume, 63d (OBV slope normalised)")
    F["obv_price_div_63"] = _zs(nvf63, 252) - _zs(C / C.shift(63) - 1, 252)
    _reg("obv_price_div_63", "volume", +1, "OBV-flow z minus price-return z (positive = volume leading price)")

    F["dvol_trend_21_126"] = dv.rolling(21).mean() / (dv.rolling(126).mean() + EPS)
    _reg("dvol_trend_21_126", "volume", 0, "21d dollar volume / 126d dollar volume")
    F["vol_cv_20"] = V.rolling(20).std() / (V.rolling(20).mean() + EPS)
    _reg("vol_cv_20", "volume", 0, "coefficient of variation of volume, 20d")

    big = V > 1.5 * v50.shift(1)
    F["big_vol_updays_net_21"] = ((big & (r > 0)).astype(float) - (big & (r < 0)).astype(float)).where(r.notna()).rolling(21).sum()
    _reg("big_vol_updays_net_21", "volume", +1, "1.5x-volume up days minus 1.5x-volume down days, 21d")

    F["amihud_21"] = np.log((r.abs() / (dv + 1)).rolling(21).mean() * 1e9 + EPS)
    _reg("amihud_21", "volume", 0, "log Amihud illiquidity (|ret| / $vol), 21d")
    F["log_dvol_63"] = np.log(dv.rolling(63).median() + 1)
    _reg("log_dvol_63", "volume", 0, "log median dollar volume 63d (size/liquidity control)")

    # ====================================================== VOLATILITY
    rv5 = lr.rolling(5).std() * ANN
    rv10 = lr.rolling(10).std() * ANN
    rv20 = lr.rolling(20).std() * ANN
    rv60 = lr.rolling(60).std() * ANN
    rv63 = lr.rolling(63).std() * ANN
    rv126 = lr.rolling(126).std() * ANN
    F["rv20"] = rv20
    _reg("rv20", "vol", 0, "20d realised vol (annualised)")
    F["rv20_pct_252"] = rv20 / (rv20.rolling(252).median() + EPS)
    _reg("rv20_pct_252", "vol", -1, "20d RV vs own 252d median RV")
    F["rv20_vs_min_126"] = rv20 / (rv20.rolling(126).min() + EPS)
    _reg("rv20_vs_min_126", "vol", -1, "20d RV / 126d minimum of 20d RV (1 = most compressed)")
    F["rv_ratio_5_20"] = rv5 / (rv20 + EPS)
    _reg("rv_ratio_5_20", "vol", 0, "5d RV / 20d RV (expansion > 1)")
    F["rv_ratio_20_60"] = rv20 / (rv60 + EPS)
    _reg("rv_ratio_20_60", "vol", -1, "20d RV / 60d RV")
    F["vol_of_vol_63"] = rv20.rolling(63).std() / (rv20.rolling(63).mean() + EPS)
    _reg("vol_of_vol_63", "vol", -1, "std / mean of 20d RV over 63d")
    F["comp_then_exp"] = (1 - F["rv20_pct_252"].clip(upper=2)) * (F["rv_ratio_5_20"] - 1)
    _reg("comp_then_exp", "vol", +1, "compressed 20d RV x nascent 5d expansion (sequence factor)")

    pos = lr.clip(lower=0)
    neg = lr.clip(upper=0)
    upvol63 = np.sqrt((pos ** 2).rolling(63).mean()) * ANN
    dnvol63 = np.sqrt((neg ** 2).rolling(63).mean()) * ANN
    F["dn_up_vol_asym_63"] = np.log((dnvol63 + EPS) / (upvol63 + EPS))
    _reg("dn_up_vol_asym_63", "vol", -1, "log downside semi-vol / upside semi-vol, 63d")
    F["dn_vol_63"] = dnvol63
    _reg("dn_vol_63", "vol", -1, "downside semi-vol 63d")

    park = np.sqrt((np.log(H / L) ** 2).rolling(20).mean() / (4 * np.log(2))) * ANN
    F["parkinson_cc_20"] = park / (rv20 + EPS)
    _reg("parkinson_cc_20", "vol", 0, "Parkinson (range) vol / close-close vol, 20d")
    F["overnight_intraday_vol_20"] = np.log((gap.rolling(20).std() + EPS) / (intra.rolling(20).std() + EPS))
    _reg("overnight_intraday_vol_20", "vol", 0, "log overnight vol / intraday vol, 20d")

    F["skew_63"] = lr.rolling(63).skew()
    _reg("skew_63", "vol", 0, "return skewness 63d")
    F["kurt_63"] = lr.rolling(63).kurt()
    _reg("kurt_63", "vol", 0, "return excess kurtosis 63d")

    F["sharpe_63"] = (r.rolling(63).mean() / (r.rolling(63).std() + EPS)) * ANN
    _reg("sharpe_63", "vol", +1, "trailing 63d Sharpe")
    F["sharpe_126"] = (r.rolling(126).mean() / (r.rolling(126).std() + EPS)) * ANN
    _reg("sharpe_126", "vol", +1, "trailing 126d Sharpe")
    F["sharpe_21"] = (r.rolling(21).mean() / (r.rolling(21).std() + EPS)) * ANN
    _reg("sharpe_21", "vol", 0, "trailing 21d Sharpe")
    F["jump_21"] = r.abs().rolling(21).max() / (r.rolling(21).std() + EPS)
    _reg("jump_21", "vol", 0, "largest |daily move| / daily sd, 21d")

    # ====================================================== RELATIVE
    rsline = C.div(Cm, axis=0)
    for w in (21, 63, 126):
        F[f"rs_spy_{w}"] = rsline / rsline.shift(w) - 1
        _reg(f"rs_spy_{w}", "relative", +1, f"{w}d return relative to SPY")
    S = sector_bench[stocks]
    rsec = C / S
    for w in (21, 63):
        F[f"rs_sec_{w}"] = rsec / rsec.shift(w) - 1
        _reg(f"rs_sec_{w}", "relative", +1, f"{w}d return relative to sector ETF")

    rs_dist = rsline / rsline.rolling(126).max() - 1
    px_dist = C / C.rolling(126).max() - 1
    F["rs_dist_high_126"] = rs_dist
    _reg("rs_dist_high_126", "relative", +1, "RS line vs its 126d high")
    F["rs_lead_126"] = rs_dist - px_dist
    _reg("rs_lead_126", "relative", +1, "RS line nearer its high than price is (RS leading price)")
    F["rs_newhigh_price_not_20"] = ((rsline >= rsline.rolling(126).max()) & (C < 0.97 * C.rolling(126).max())).astype(float).where(rsline.notna()).rolling(20).sum()
    _reg("rs_newhigh_price_not_20", "relative", +1, "days in last 20 where RS made 126d high while price >3% below its high")

    b63, c63 = _beta_corr(r, rm, 63)
    b252, _ = _beta_corr(r, rm, 252)
    F["beta_63"] = b63
    _reg("beta_63", "relative", 0, "63d beta to SPY")
    F["beta_252"] = b252
    _reg("beta_252", "relative", 0, "252d beta to SPY")
    F["beta_shift"] = b63 - b252
    _reg("beta_shift", "relative", 0, "63d beta minus 252d beta")
    F["corr_spy_63"] = c63
    _reg("corr_spy_63", "relative", -1, "63d correlation to SPY")
    var_s = r.rolling(63).var()
    var_m = rm.rolling(63).var()
    F["idio_vol_63"] = np.sqrt((var_s - (b63 ** 2).mul(var_m, axis=0)).clip(lower=0)) * ANN
    _reg("idio_vol_63", "relative", 0, "idiosyncratic vol 63d")

    mkt_up = rm > 0
    up_cap = (r.where(mkt_up, axis=0)).rolling(63, min_periods=10).mean().div(rm.where(mkt_up).rolling(63, min_periods=10).mean(), axis=0)
    dn_cap = (r.where(~mkt_up, axis=0)).rolling(63, min_periods=10).mean().div(rm.where(~mkt_up).rolling(63, min_periods=10).mean(), axis=0)
    F["up_capture_63"] = up_cap
    _reg("up_capture_63", "relative", +1, "up-market capture ratio 63d")
    F["down_capture_63"] = dn_cap
    _reg("down_capture_63", "relative", -1, "down-market capture ratio 63d")
    F["capture_spread_63"] = up_cap - dn_cap
    _reg("capture_spread_63", "relative", +1, "up capture minus down capture 63d")

    # interactions the user explicitly described as setups
    F["dryup_x_tight"] = (-F["vol_dry_20_250"]).rank(axis=1, pct=True) + (-F["range_comp_10_252"]).rank(axis=1, pct=True) + F["dist_52w_high"].rank(axis=1, pct=True)
    _reg("dryup_x_tight", "interaction", +1, "rank sum: volume dry-up + range compression + proximity to 52w high (VCP-style)")
    F["ignite_x_rs"] = F["ignition_mult_10"].rank(axis=1, pct=True) + F["rs_spy_63"].rank(axis=1, pct=True) + F["updown_vol_ratio_50"].rank(axis=1, pct=True)
    _reg("ignite_x_rs", "interaction", +1, "rank sum: recent ignition volume + 63d RS + up/down volume")

    return {k: v.astype("float32") for k, v in F.items()}


def eligibility(panel: dict, stocks: list, min_price=5.0, min_dvol=10e6, min_hist=252) -> pd.DataFrame:
    """Point-in-time investable mask: price >= $5, 63d median dollar volume >= $10M, >= 252 sessions of history."""
    C, V = panel["Close"][stocks], panel["Volume"][stocks]
    dv = (C * V).rolling(63, min_periods=40).median()
    hist = C.notna().cumsum()
    return (C >= min_price) & (dv >= min_dvol) & (hist >= min_hist) & C.notna()
