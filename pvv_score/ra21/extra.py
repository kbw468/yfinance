"""Additional non-Gaussian feature families (point-in-time; everything at the close of t uses data through t only).

  shock_*    the largest volume shock of the last 63 sessions (volume / prior 50-session median): its size, age, the
             day's excess return and gap, and whether price has held it since (an earnings-reaction proxy from tape alone)
  earn_due   how many of the last four quarters (and last year) had a volume shock >= 2.5x in the window that lines up with
             the next 21 sessions (quarterly cadence = report likely due); since_spike = sessions since the last one
  seas_*     same calendar window in prior years (up to 10, at least 3): median excess return over the next 21 sessions,
             share of years positive, median of the trade outcome's percentile, share of years it was superior
  ind_*      leave-one-out median of the industry peers (sector if fewer than 3 peers): 5/21/63-session return, share of peers
             within 5% of their 63-session high; peer_gap = peers' 21-session return minus the name's
  vshock_*   high-volume premium inputs: today's and the last 5 sessions' volume vs the prior 50-session median, own 50-day
             volume percentile, the largest shock on an up day and on a down day in the last 10
  max/min    lottery and tails: largest and 5 largest daily returns in 21 sessions, largest vs the name's typical move,
             worst day, 95th / |5th| percentile of daily returns over 126 sessions
  mom_12_1, id_12_1   twelve-month return skipping the last month, and its information discreteness (frog in the pan)
  effx_*     trailing excess-path efficiency vs SPY / sector: sum of daily excess returns / sum of their absolute values
  mar_trail  trailing excess return / max(|drawdown|, floor), 21 and 63 sessions;  lead_persist = share of the last 63 / 252
             sessions the trailing 21-session MAR ranked in the universe's top 20%;  rank_mom_21 = change in the 63-session
             return's universe percentile over 21 sessions
  on_/in_    overnight and intraday components of return, 21 / 63 sessions; on_pos_63 = share of up gaps
  rpos_*     position in the high-low range, 21 / 63 / 252 sessions;  bounce_21 = close vs the 21-session low
  streak     current run of up (+) or down (-) closes;  sign_persist_63 = share of days repeating the previous day's sign
  log_price  log of the unadjusted close
  mkt_*      market state, the same for every name on a date: SPY vs its 252-session high, SPY 21/63-session return, SPY
             typical move vs its own year, VIX and MOVE vs their own year, breadth (share near 63-session highs, share up
             over 21 sessions), cross-sectional spread (interquartile range) of 21-session returns"""
import numpy as np
import pandas as pd

def _vr(V: pd.DataFrame) -> pd.DataFrame:
    return V / V.rolling(50, min_periods=40).median().shift(1)


def _argmax_lag(A: pd.DataFrame, w: int) -> tuple[np.ndarray, np.ndarray]:
    """Max over the trailing w rows (lags 0..w-1) and the lag where it sits (most recent on ties)."""
    a = A.to_numpy(np.float64)
    best = np.full(a.shape, -np.inf); lag = np.full(a.shape, -1, np.int32)
    for j in range(w):
        s = np.full(a.shape, np.nan); s[j:] = a[:len(a) - j] if j else a
        better = s > best
        best = np.where(better, s, best); lag = np.where(better, j, lag)
    best[np.isinf(best)] = np.nan
    return best, lag


def _at_lag(X: pd.DataFrame, lag: np.ndarray, w: int) -> np.ndarray:
    x = X.to_numpy(np.float64); out = np.full(x.shape, np.nan)
    for j in range(w):
        s = np.full(x.shape, np.nan); s[j:] = x[:len(x) - j] if j else x
        out = np.where(lag == j, s, out)
    return out


def _loo_median(X: pd.DataFrame, groups: dict, fallback: dict) -> pd.DataFrame:
    """For each name, the median of its group peers (excluding itself); fallback group when fewer than 3 peers."""
    out = pd.DataFrame(np.nan, index=X.index, columns=X.columns)
    a = X.to_numpy(np.float64); col = {c: i for i, c in enumerate(X.columns)}
    for t in X.columns:
        peers = [p for p in groups.get(t, []) if p != t and p in col]
        if len(peers) < 3:
            peers = [p for p in fallback.get(t, []) if p != t and p in col]
        if peers:
            out[t] = np.nanmedian(a[:, [col[p] for p in peers]], axis=1)
    return out


def _streak(r: pd.DataFrame) -> pd.DataFrame:
    s = np.sign(r.to_numpy(np.float64)); s = np.nan_to_num(s)
    out = np.zeros_like(s)
    for i in range(1, len(s)):
        out[i] = np.where(s[i] == 0, 0, np.where(np.sign(out[i - 1]) == s[i], out[i - 1] + s[i], s[i]))
    return pd.DataFrame(out, index=r.index, columns=r.columns).where(r.notna())


def _seasonal(X: pd.DataFrame, years=range(1, 11), min_years=3):
    """Values of X (a frame indexed by date) at the trading day on/after the same calendar date 1..10 years earlier."""
    idx = X.index; a = X.to_numpy(np.float64); stack = []
    for y in years:
        target = idx - pd.DateOffset(years=y)
        pos = idx.searchsorted(target)
        ok = (target >= idx[0]) & (pos < len(idx))
        s = np.full(a.shape, np.nan); s[ok] = a[pos[ok]]
        stack.append(s)
    S = np.stack(stack)
    return S, (np.sum(~np.isnan(S), axis=0) >= min_years)


def compute(panel: dict, stocks: list, uni: pd.DataFrame, sector_bench: pd.DataFrame, outcome_wide: dict, vol: pd.DataFrame) -> dict:
    C, H, L, O, V = (panel[k][stocks] for k in ("Close", "High", "Low", "Open", "Volume"))
    spy = panel["Close"]["SPY"]; r = C.pct_change(fill_method=None); rs = spy.pct_change(fill_method=None)
    F = {}
    # volume shocks and the earnings-reaction proxy
    vr = _vr(V)
    best, lag = _argmax_lag(vr, 63)
    F["shock_volx_63"] = pd.DataFrame(best, C.index, C.columns)
    F["shock_age_63"] = pd.DataFrame(np.where(lag >= 0, lag, np.nan), C.index, C.columns)
    xs1 = r.sub(rs, axis=0); gap = O / C.shift(1) - 1
    F["shock_xs_63"] = pd.DataFrame(_at_lag(xs1, lag, 63), C.index, C.columns)
    F["shock_gap_63"] = pd.DataFrame(_at_lag(gap, lag, 63), C.index, C.columns)
    cd = _at_lag(C, lag, 63); ld = _at_lag(L, lag, 63); sd = _at_lag(pd.DataFrame(np.repeat(spy.to_numpy()[:, None], len(stocks), 1), C.index, C.columns), lag, 63)
    F["shock_hold_63"] = C / cd - 1
    F["shock_lowhold_63"] = C / ld - 1
    F["shock_xs_since_63"] = (C / cd - 1) - (spy.to_numpy()[:, None] / sd - 1)
    F["shock_signed_63"] = F["shock_xs_63"] * np.log(F["shock_volx_63"].clip(lower=1))
    spike = (vr >= 2.5).astype(float).where(vr.notna())
    s21 = spike.rolling(21, min_periods=15).max()
    F["earn_due_21"] = sum(s21.shift(63 * q - 21).fillna(0) for q in (1, 2, 3, 4)).where(s21.shift(63 * 4 - 21).notna())
    pos = pd.DataFrame(np.where(spike.to_numpy() == 1, np.arange(len(C))[:, None], np.nan), C.index, C.columns).ffill()
    F["since_spike"] = (pd.DataFrame(np.arange(len(C))[:, None] - pos.to_numpy(), C.index, C.columns)).clip(upper=252).where(C.notna())
    # high-volume premium inputs
    F["vshock_1"] = vr
    F["vshock_5"] = V.rolling(5, min_periods=5).sum() / (5 * V.rolling(50, min_periods=40).median().shift(5))
    F["vrank_50"] = V.rolling(50, min_periods=40).rank(pct=True)
    F["vshock_up_10"] = vr.where(r > 0).rolling(10, min_periods=1).max().fillna(0).where(vr.notna())
    F["vshock_dn_10"] = vr.where(r < 0).rolling(10, min_periods=1).max().fillna(0).where(vr.notna())
    # lottery and tails
    mad252 = r.abs().rolling(252, min_periods=200).median()
    F["max1_21"] = r.rolling(21, min_periods=21).max()
    a = r.to_numpy(np.float64); top5 = np.full(a.shape, np.nan)
    from numpy.lib.stride_tricks import sliding_window_view
    for s in range(0, a.shape[1], 100):
        blk = a[:, s:s + 100]; wv = sliding_window_view(blk, 21, axis=0)              # (n-20, k, 21)
        srt = np.sort(np.where(np.isnan(wv), -np.inf, wv), axis=2)[:, :, -5:]
        val = np.where(np.isinf(srt), np.nan, srt).mean(axis=2)
        val[np.isnan(wv).any(axis=2)] = np.nan
        top5[20:, s:s + 100] = val
    F["max5_21"] = pd.DataFrame(top5, C.index, C.columns)
    F["max_rel_21"] = F["max1_21"] / mad252
    F["min1_21"] = r.rolling(21, min_periods=21).min()
    F["tail_ratio_126"] = r.rolling(126, min_periods=100).quantile(0.95) / (-r.rolling(126, min_periods=100).quantile(0.05))
    # momentum variants
    F["mom_12_1"] = C.shift(21) / C.shift(252) - 1
    upd = (r > 0).astype(float).where(r.notna()); dnd = (r < 0).astype(float).where(r.notna())
    F["id_12_1"] = np.sign(F["mom_12_1"]) * (dnd.shift(21).rolling(231, min_periods=200).mean() - upd.shift(21).rolling(231, min_periods=200).mean())
    xspy = r.sub(rs, axis=0); sbr = sector_bench[stocks].pct_change(fill_method=None); xsec = r - sbr
    for w in (21, 63):
        F[f"effx_spy_{w}"] = xspy.rolling(w, min_periods=w).sum() / xspy.abs().rolling(w, min_periods=w).sum()
    F["effx_spy_252"] = xspy.rolling(252, min_periods=200).sum() / xspy.abs().rolling(252, min_periods=200).sum()
    F["effx_sec_63"] = xsec.rolling(63, min_periods=50).sum() / xsec.abs().rolling(63, min_periods=50).sum()
    F["effx_sec_252"] = xsec.rolling(252, min_periods=200).sum() / xsec.abs().rolling(252, min_periods=200).sum()
    # trailing analogs of the target
    for w, fl in ((21, 0.02), (63, 0.04)):
        xs_w = (C / C.shift(w) - 1).sub(spy / spy.shift(w) - 1, axis=0)
        a = C.to_numpy(np.float64); d = np.full(a.shape, np.nan)       # deepest close-to-running-high drawdown inside the window
        for i in range(w, len(a)):
            seg = a[i - w:i + 1]; d[i] = np.min(seg / np.fmax.accumulate(seg, axis=0) - 1, axis=0)
        dd = pd.DataFrame(d, C.index, C.columns)
        F[f"mar_trail_{w}"] = xs_w / np.maximum(-dd, fl)
    top = F["mar_trail_21"].rank(axis=1, pct=True) >= 0.8
    top = top.astype(float).where(F["mar_trail_21"].notna())
    F["lead_persist_63"] = top.rolling(63, min_periods=50).mean(); F["lead_persist_252"] = top.rolling(252, min_periods=200).mean()
    r63rank = (C / C.shift(63) - 1).rank(axis=1, pct=True)
    F["rank_mom_21"] = r63rank - r63rank.shift(21)
    # overnight vs intraday
    lon = np.log(O / C.shift(1)); lin = np.log(C / O)
    for w in (21, 63):
        F[f"on_{w}"] = lon.rolling(w, min_periods=w).sum(); F[f"in_{w}"] = lin.rolling(w, min_periods=w).sum()
    F["on_pos_63"] = (lon > 0).astype(float).where(lon.notna()).rolling(63, min_periods=50).mean()
    # range position, bounce, streaks, sign persistence
    for w in (21, 63, 252):
        hi = H.rolling(w, min_periods=w).max(); lo = L.rolling(w, min_periods=w).min()
        F[f"rpos_{w}"] = (C - lo) / (hi - lo).replace(0, np.nan)
    F["bounce_21"] = C / L.rolling(21, min_periods=21).min() - 1
    F["streak"] = _streak(r)
    sg = np.sign(r)
    F["sign_persist_63"] = ((sg == sg.shift(1)) & (sg != 0)).astype(float).where(r.notna() & r.shift(1).notna()).rolling(63, min_periods=50).mean()
    F["log_price"] = np.log(panel["RawClose"][stocks])
    # industry leave-one-out
    ind = uni.Industry.reindex(stocks); sec = uni.Sector.reindex(stocks)
    g_ind = {t: list(ind.index[ind == ind[t]]) for t in stocks}; g_sec = {t: list(sec.index[sec == sec[t]]) for t in stocks}
    for w in (5, 21, 63):
        F[f"ind_roc_{w}"] = _loo_median(C / C.shift(w) - 1, g_ind, g_sec)
    near = (C / C.rolling(63, min_periods=63).max() - 1 >= -0.05).astype(float).where(C.notna())
    F["ind_breadth_63h"] = _loo_mean(near, g_ind, g_sec)
    F["peer_gap_21"] = F["ind_roc_21"] - (C / C.shift(21) - 1)
    # seasonality: same calendar window in prior years
    xs21c = (C.shift(-21) / C - 1).sub(spy.shift(-21) / spy - 1, axis=0)
    S, ok = _seasonal(xs21c)
    F["seas_xs_med"] = pd.DataFrame(np.where(ok, np.nanmedian(S, axis=0), np.nan), C.index, C.columns)
    F["seas_pos"] = pd.DataFrame(np.where(ok, np.nansum(S > 0, axis=0) / np.maximum(np.sum(~np.isnan(S), axis=0), 1), np.nan), C.index, C.columns)
    del S
    mar = outcome_wide["mar_21"].reindex(index=C.index, columns=stocks); xs = outcome_wide["xs_21"].reindex(index=C.index, columns=stocks)
    mpct = mar.rank(axis=1, pct=True); supw = ((mpct >= 0.8) & (xs > 0)).astype(float).where(mar.notna())
    S, ok = _seasonal(mpct)
    F["seas_marpct_med"] = pd.DataFrame(np.where(ok, np.nanmedian(S, axis=0), np.nan), C.index, C.columns); del S
    S, ok = _seasonal(supw)
    F["seas_sup_share"] = pd.DataFrame(np.where(ok, np.nanmean(S, axis=0), np.nan), C.index, C.columns); del S
    # market state (same on every name of a date)
    v = vol.reindex(C.index).ffill()
    smad = rs.abs().rolling(21, min_periods=21).median()
    mk = {"mkt_off_high_252": spy / spy.rolling(252, min_periods=252).max() - 1, "mkt_roc_21": spy / spy.shift(21) - 1, "mkt_roc_63": spy / spy.shift(63) - 1,
          "mkt_mad_pctile": smad.rolling(252, min_periods=252).rank(pct=True), "mkt_vix": v["vix"], "mkt_vix_pctile": v["vix"].rolling(252, min_periods=200).rank(pct=True),
          "mkt_move_pctile": v["move"].rolling(252, min_periods=200).rank(pct=True), "mkt_breadth_63h": near.mean(axis=1), "mkt_breadth_up21": ((C / C.shift(21) - 1) > 0).astype(float).where(C.shift(21).notna()).mean(axis=1),
          "mkt_disp_iqr_21": (C / C.shift(21) - 1).quantile(0.75, axis=1) - (C / C.shift(21) - 1).quantile(0.25, axis=1)}
    for k, s in mk.items():
        F[k] = pd.DataFrame(np.repeat(s.to_numpy(np.float64)[:, None], len(stocks), 1), C.index, C.columns)
    return {k: v.astype("float32") for k, v in F.items()}


def _loo_mean(X: pd.DataFrame, groups: dict, fallback: dict) -> pd.DataFrame:
    out = pd.DataFrame(np.nan, index=X.index, columns=X.columns)
    a = X.to_numpy(np.float64); col = {c: i for i, c in enumerate(X.columns)}
    for t in X.columns:
        peers = [p for p in groups.get(t, []) if p != t and p in col]
        if len(peers) < 3:
            peers = [p for p in fallback.get(t, []) if p != t and p in col]
        if peers:
            out[t] = np.nanmean(a[:, [col[p] for p in peers]], axis=1)
    return out
