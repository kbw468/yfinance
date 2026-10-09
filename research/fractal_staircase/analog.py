"""Shape-analog features: how closely each stock's recent path matches the
pre-entry geometry of the reference chart (base -> breakout -> first higher step),
measured at three time scales (the same shape stretched over 40, 80, 160 bars).

Distance = mean absolute gap between min-max normalised log paths (L1, no variance).
Writes tmpl_* columns (and their per-date ranks q_tmpl_*) into panel.parquet.
"""
import os

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view

from common import DATA_DIR
from exemplar import CHART

SCALES = (40, 80, 160)
ENTRY = "2026-08-03"


def template(cal):
    pts = pd.Series(CHART)
    pts.index = pd.to_datetime(pts.index)
    pts = pts[pts.index <= ENTRY]
    days = cal[(cal >= pts.index[0]) & (cal <= pts.index[-1])]
    pos = np.searchsorted(days, pts.index)
    path = np.interp(np.arange(len(days)), pos, np.log(pts.to_numpy()))
    return (path - path.min()) / (path.max() - path.min())


def norm_rows(W):
    lo = W.min(axis=1, keepdims=True)
    hi = W.max(axis=1, keepdims=True)
    return (W - lo) / np.where(hi > lo, hi - lo, np.nan)


def main():
    px = pd.read_parquet(os.path.join(DATA_DIR, "ohlcv.parquet"), columns=["date", "ticker", "close"])
    cal = np.sort(px["date"].unique())
    base = template(pd.DatetimeIndex(cal))
    temps = {L: np.interp(np.linspace(0, len(base) - 1, L), np.arange(len(base)), base) for L in SCALES}
    panel = pd.read_parquet(os.path.join(DATA_DIR, "panel.parquet"))
    keys = panel[["date", "ticker"]]
    parts = []
    for tkr, g in px.groupby("ticker", sort=False):
        g = g.sort_values("date")
        x = np.log(g["close"].to_numpy(float))
        d = g["date"].to_numpy()
        res = {"date": d}
        for L, T in temps.items():
            out = np.full(len(x), np.nan)
            if len(x) >= L:
                W = norm_rows(sliding_window_view(x, L))
                out[L - 1:] = np.nanmean(np.abs(W - T), axis=1)
            res[f"tmpl_d{L}"] = out
        r = pd.DataFrame(res)
        r["ticker"] = tkr
        parts.append(r)
    an = pd.concat(parts, ignore_index=True)
    an["tmpl_min"] = an[[f"tmpl_d{L}" for L in SCALES]].min(axis=1)
    an["tmpl_mean"] = an[[f"tmpl_d{L}" for L in SCALES]].mean(axis=1)
    an = keys.merge(an, on=["date", "ticker"], how="left")
    cols = [c for c in an.columns if c.startswith("tmpl_")]
    panel = panel.drop(columns=[c for c in panel.columns if "tmpl_" in c], errors="ignore")
    for c in cols:
        panel[c] = an[c].astype("float32").to_numpy()
        # smaller distance = better match -> rank so that 1.0 = closest match
        panel["q_" + c] = (1 - panel.groupby("date")[c].rank(pct=True)).astype("float32")
    panel.to_parquet(os.path.join(DATA_DIR, "panel.parquet"), index=False)
    print("analog features added", cols)


if __name__ == "__main__":
    main()
