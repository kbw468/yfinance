"""Forward path outcomes, per name and date (all non-Gaussian, measured on the actual path).

For h in 21/42/63 sessions after the close at t:
  ret_h     close-to-close return;  xs_h = ret_h minus SPY's
  ptt_h     deepest peak-to-trough drawdown on closes inside (t, t+h], with the entry close as the first peak
  stop_h    lowest intraday low inside (t, t+h] relative to the entry close
  hit_h     xs_h > 0 AND ptt_h >= CAP[h] AND stop_h > -STOP   (beat SPY, shallow path, never stopped out)"""
import numpy as np
import pandas as pd

HORIZONS = (21, 42, 63)
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
