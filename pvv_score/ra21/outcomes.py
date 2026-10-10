"""Trade outcomes, as the trade is actually held. Signal at the close of t, buy the open of t+1, stop 8% below entry
(a gap through the stop fills at that open), otherwise exit at the close of t+h. Everything measured on the trade's own path:

  ret      trade return                         xs   ret minus SPY over the same holding period (SPY bought at the same open)
  dd       deepest drawdown of the position (closes, entry as the first peak, the stop fill on a stop day)
  mar      xs / max(|dd|, FLOOR[h])             excess return per unit of drawdown suffered (MAR / Calmar form)
  effx     sum of daily excess returns / sum of their absolute values, entry to exit (Omega form: omega = (1+effx)/(1-effx))
  mae/mfe  lowest low / highest high vs entry while held;   stopped, k_exit
  sup      superior: mar in the top 20% of the universe for that entry date and xs > 0
No standard deviations, variances or regressions."""
import numpy as np
import pandas as pd

HORIZONS = (21, 42, 63)
STOP = 0.08
FLOOR = {21: 0.02, 42: 0.03, 63: 0.04}
TOP = 0.20


def trades(panel: dict, stocks: list, h: int, stop: float = STOP) -> dict:
    O, H, L, C = (panel[k][stocks].to_numpy(np.float64) for k in ("Open", "High", "Low", "Close"))
    so, sc = panel["Open"]["SPY"].to_numpy(np.float64), panel["Close"]["SPY"].to_numpy(np.float64)
    n, m = C.shape
    def fwd(A, k):                                   # row i -> value at row i+k
        out = np.full_like(A, np.nan); out[:n - k] = A[k:]; return out
    entry = fwd(O, 1); spy_in = fwd(so[:, None], 1)[:, 0]
    sp = entry * (1 - stop)
    alive = ~np.isnan(entry); valid = alive.copy()
    runmax = entry.copy(); dd = np.zeros((n, m)); mae = np.zeros((n, m)); mfe = np.zeros((n, m))
    exit_px = np.full((n, m), np.nan); k_exit = np.zeros((n, m), np.int16)
    sx = np.zeros((n, m)); sa = np.zeros((n, m)); prev = entry.copy(); sprev = np.repeat(spy_in[:, None], m, 1)
    for k in range(1, h + 1):
        Ok, Hk, Lk, Ck = fwd(O, k), fwd(H, k), fwd(L, k), fwd(C, k)
        sck = fwd(sc[:, None], k)[:, 0][:, None]
        valid &= ~(alive & (np.isnan(Ck) | np.isnan(Lk)))      # a missing bar while held voids the trade
        hit = alive & (Lk <= sp)
        fill = np.where((k > 1) & (Ok <= sp), Ok, sp)
        px = np.where(hit, fill, Ck)
        mae = np.where(alive, np.fmin(mae, np.where(hit, fill, Lk) / entry - 1), mae)
        mfe = np.where(alive, np.fmax(mfe, Hk / entry - 1), mfe)
        dd = np.where(alive, np.fmin(dd, px / runmax - 1), dd)
        dx = (px / prev - 1) - (sck / sprev - 1)
        sx = np.where(alive, sx + dx, sx); sa = np.where(alive, sa + np.abs(dx), sa)
        runmax = np.where(alive & ~hit, np.fmax(runmax, Ck), runmax)
        prev = np.where(alive, px, prev); sprev = np.where(alive, sck, sprev)
        exit_px = np.where(hit, fill, exit_px); k_exit = np.where(hit, k, k_exit)
        alive &= ~hit
    Ch = fwd(C, h)
    exit_px = np.where(alive, Ch, exit_px); k_exit = np.where(alive, h, k_exit)
    valid &= ~np.isnan(exit_px) & ~np.isnan(spy_in)[:, None]
    # an outcome counts only once the whole window has printed: otherwise only the trades already stopped would be known,
    # and the most recent sessions would carry a stopped-only (biased) sample
    valid &= (np.arange(n) + h <= n - 1)[:, None]
    # SPY over the same holding period: bought at the same open, sold at the close of the exit day
    idx = np.clip(np.arange(n)[:, None] + k_exit, 0, n - 1)
    spy_out = sc[idx]
    ret = exit_px / entry - 1
    spy_ret = spy_out / spy_in[:, None] - 1
    xs = ret - spy_ret
    mar = xs / np.maximum(-dd, FLOOR.get(h, 0.02))
    effx = np.divide(sx, sa, out=np.zeros_like(sx), where=sa > 0)
    out = {"ret": ret, "xs": xs, "dd": dd, "mar": mar, "effx": effx, "mae": mae, "mfe": mfe,
           "stopped": ~alive, "k_exit": k_exit.astype(float)}
    idx_ = panel["Close"].index
    return {f"{k}_{h}": pd.DataFrame(np.where(valid, v, np.nan).astype("float32"), index=idx_, columns=stocks) for k, v in out.items()}


def no_stop(panel: dict, stocks: list, h: int) -> dict:
    """Same entry, no stop: excess return to the close of t+h and the deepest close drawdown (persistence checks)."""
    O, C = panel["Open"][stocks], panel["Close"][stocks]
    so, sc = panel["Open"]["SPY"], panel["Close"]["SPY"]
    entry = O.shift(-1); ret = C.shift(-h) / entry - 1
    xs = ret.sub(sc.shift(-h) / so.shift(-1) - 1, axis=0)
    rm = entry.copy(); dd = pd.DataFrame(0.0, index=C.index, columns=C.columns)
    for k in range(1, h + 1):
        p = C.shift(-k); rm = np.fmax(rm, p); dd = np.fmin(dd, p / rm - 1)
    ok = ret.notna()
    return {f"xsns_{h}": xs.where(ok).astype("float32"), f"ddns_{h}": dd.where(ok).astype("float32")}


def compute(panel: dict, stocks: list) -> dict:
    out = {}
    for h in HORIZONS:
        out.update(trades(panel, stocks, h)); out.update(no_stop(panel, stocks, h))
    return out


def label(T: pd.DataFrame) -> pd.DataFrame:
    """sup_h: top 20% of the entry date on mar_h with xs_h > 0. Also the within-date percentile of mar_h and effx_h."""
    for h in HORIZONS:
        ok = T[f"mar_{h}"].notna()
        r = T[f"mar_{h}"].groupby(T.date).rank(pct=True)
        T[f"marpct_{h}"] = r.astype("float32")
        T[f"sup_{h}"] = ((r >= 1 - TOP) & (T[f"xs_{h}"] > 0)).astype("float32").where(ok)
        T[f"effpct_{h}"] = T[f"effx_{h}"].groupby(T.date).rank(pct=True).astype("float32")
    return T
