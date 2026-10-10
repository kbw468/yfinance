"""Download daily OHLCV for every ticker in the finviz screen export.

Usage: python download.py <finviz.csv> <out_dir> [period | YYYY-MM-DD start date]
Writes one parquet of long-format bars (date, ticker, open, high, low, close, volume).
Prices are split/dividend adjusted (auto_adjust=True).
"""
import sys
import time
from pathlib import Path

import pandas as pd
import yfinance as yf

BATCH = 150


def main(csv_path, out_dir, period="10y"):
    span = {"start": period} if period[:1].isdigit() and "-" in period else {"period": period}
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tickers = pd.read_csv(csv_path)["Ticker"].dropna().astype(str).str.strip().unique().tolist()
    frames = []
    for i in range(0, len(tickers), BATCH):
        chunk = tickers[i:i + BATCH]
        for attempt in range(4):
            try:
                raw = yf.download(chunk, **span, interval="1d", auto_adjust=True,
                                  progress=False, threads=True, group_by="column")
                break
            except Exception as e:  # network hiccup, back off and retry
                print(f"batch {i} attempt {attempt} failed: {e}", flush=True)
                time.sleep(2 ** (attempt + 1))
        else:
            continue
        long = raw.stack(level="Ticker", future_stack=True).reset_index()
        long.columns = [c.lower() for c in long.columns]
        long = long.dropna(subset=["close"])
        frames.append(long)
        print(f"{i + len(chunk)}/{len(tickers)} tickers, {long['ticker'].nunique()} with data", flush=True)
    bars = pd.concat(frames, ignore_index=True)
    bars["date"] = pd.to_datetime(bars["date"]).dt.tz_localize(None)
    bars = bars[["date", "ticker", "open", "high", "low", "close", "volume"]]
    bars.to_parquet(out_dir / "bars.parquet", index=False)
    missing = sorted(set(tickers) - set(bars["ticker"].unique()))
    pd.Series(missing, name="ticker").to_csv(out_dir / "missing_tickers.csv", index=False)
    print(f"done: {bars['ticker'].nunique()} tickers, {len(bars):,} rows, {len(missing)} missing")


if __name__ == "__main__":
    main(*sys.argv[1:])
