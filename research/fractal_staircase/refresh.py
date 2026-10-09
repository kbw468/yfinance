"""Append the latest (possibly intraday) daily bars to ohlcv.parquet.

Downloads the last few sessions for every ticker, replaces overlapping dates, and
rescales a ticker's stored history when Yahoo's dividend/split adjustment moved
(detected from the overlapping session's close). Keeps a backup of the previous file.
"""
import os
import shutil
import time

import numpy as np
import pandas as pd
import yfinance as yf

from download import BATCH, to_long

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.environ.get("FS_DATA", os.path.join(HERE, "data"))


def main(period="5d"):
    path = os.path.join(DATA_DIR, "ohlcv.parquet")
    old = pd.read_parquet(path)
    old["date"] = pd.to_datetime(old["date"]).astype("datetime64[ns]")
    stamp = old["date"].max().strftime("%Y%m%d")
    backup = os.path.join(DATA_DIR, f"ohlcv_upto_{stamp}.parquet")
    if not os.path.exists(backup):
        shutil.copy(path, backup)
    tickers = sorted(old["ticker"].unique())
    parts = []
    for i in range(0, len(tickers), BATCH):
        chunk = tickers[i:i + BATCH]
        for attempt in range(3):
            try:
                df = yf.download(chunk, period=period, interval="1d", auto_adjust=True, actions=False,
                                 progress=False, threads=True, group_by="column")
                parts.append(to_long(df, chunk))
                break
            except Exception as e:
                print("retry", attempt, e, flush=True)
                time.sleep(3 * 2 ** attempt)
    new = pd.concat(parts, ignore_index=True)
    new["date"] = pd.to_datetime(new["date"]).dt.tz_localize(None).astype("datetime64[ns]")
    first_new = new["date"].min()
    # adjustment drift: compare closes on the overlapping dates
    ov = old[old["date"] >= first_new].merge(new, on=["date", "ticker"], suffixes=("_old", ""))
    ratio = (ov["close"] / ov["close_old"]).groupby(ov["ticker"]).median()
    drift = ratio[(ratio - 1).abs() > 1e-4]
    if len(drift):
        cols = ["open", "high", "low", "close"]
        m = old["ticker"].isin(drift.index)
        old.loc[m, cols] = old.loc[m, cols].mul(old.loc[m, "ticker"].map(drift), axis=0)
        print("rescaled history for", len(drift), "tickers (dividend/split adjustment moved)")
    keep = old[old["date"] < first_new]
    out = pd.concat([keep, new], ignore_index=True).drop_duplicates(["date", "ticker"], keep="last")
    out = out.sort_values(["ticker", "date"])
    out.to_parquet(path, index=False)
    last = out["date"].max()
    print("ohlcv now through", last.date(), "| tickers with a bar on that date:", out.loc[out["date"] == last, "ticker"].nunique())


if __name__ == "__main__":
    main()
