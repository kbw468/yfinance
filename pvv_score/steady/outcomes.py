"""Forward path outcomes, per name and date (all non-Gaussian, measured on the actual path).

For h in 21/42/63 sessions after the close at t:
  ret_h     close-to-close return;  xs_h = ret_h minus SPY's
  ptt_h     deepest peak-to-trough drawdown on closes inside (t, t+h], with the entry close as the first peak
  stop_h    lowest intraday low inside (t, t+h] relative to the entry close
  hit_h     xs_h > 0 AND ptt_h >= CAP[h] AND stop_h > -STOP   (beat SPY, shallow path, never stopped out)  -- first target, retired:
            the model met it by picking low-beta names that rarely draw down and lag SPY.
  ah_h      alpha target: excess return per unit of path pain, xs_h / max(|ptt_h|, FLOOR[h]), ranked within the date;
            hit = top 20% of the universe that date AND xs_h > 0 AND never stopped out. Base rate is the same in every regime."""
import numpy as np
import pandas as pd

HORIZONS = (21, 42, 63)
FLOOR = {21: 0.02, 42: 0.03, 63: 0.04}     # drawdowns shallower than this count as this (no division blow-ups)
TOP = 0.20
CAP = {21: -0.05, 42: -0.07, 63: -0.08}
STOP = 0.08


def compute(panel: dict, stocks: list) -> dict:
    C = panel["Close"][stocks]; L = panel["Low"][stocks]; spy = panel["Close"]["SPY"]
    out = {}
    for h in HORIZONS:
        end = C.shift(-h)
        ret = end / C - 1
        xs = ret.sub(spy.shift(-h) / spy - 1, axis=0)
        rm = C.copy(); ptt = pd.DataFrame(0.0, index=C.index, columns=C.columns); lo = pd.DataFrame(np.inf, index=C.index, columns=C.columns)
        for k in range(1, h + 1):
            p = C.shift(-k); rm = np.fmax(rm, p); ptt = np.fmin(ptt, p / rm - 1); lo = np.fmin(lo, L.shift(-k))
        stop = lo / C - 1
        valid = end.notna()
        hit = ((xs > 0) & (ptt >= CAP[h]) & (stop > -STOP)).astype(float)
        out[f"ret_{h}"] = ret.where(valid); out[f"xs_{h}"] = xs.where(valid); out[f"ptt_{h}"] = ptt.where(valid)
        out[f"stop_{h}"] = stop.where(valid); out[f"hit_{h}"] = hit.where(valid)
    return {k: v.astype("float32") for k, v in out.items()}


def alpha_target(T: pd.DataFrame) -> pd.DataFrame:
    for h in HORIZONS:
        ratio = T[f"xs_{h}"] / np.maximum(-T[f"ptt_{h}"], FLOOR[h])
        ratio = ratio.where(T[f"stop_{h}"] > -STOP, -np.inf).where(T[f"xs_{h}"].notna())
        T[f"ratio_{h}"] = ratio.astype("float32")
        rk = ratio.groupby(T.date).rank(pct=True)
        T[f"ah_{h}"] = ((rk >= 1 - TOP) & (T[f"xs_{h}"] > 0)).astype("float32").where(T[f"xs_{h}"].notna())
    return T
