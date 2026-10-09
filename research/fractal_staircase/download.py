"""Download daily OHLCV from Yahoo Finance for every ticker in universe.csv.

Output: one long-format parquet per batch in DATA_DIR/raw, then a combined
DATA_DIR/ohlcv.parquet with columns [date, ticker, open, high, low, close, volume].
Prices are split/dividend adjusted (auto_adjust=True).
"""
import os
import sys
import time

import pandas as pd
import yfinance as yf

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.environ.get("FS_DATA", os.path.join(HERE, "data"))
RAW = os.path.join(DATA_DIR, "raw")
START = "2004-01-01"
BATCH = 100


def to_long(df, tickers):
    if df is None or df.empty:
        return pd.DataFrame()
    if not isinstance(df.columns, pd.MultiIndex):
        df.columns = pd.MultiIndex.from_product([df.columns, tickers])
    out = df.stack(level=1, future_stack=True).reset_index()
    out.columns = [str(c).lower() for c in out.columns]
    out = out.rename(columns={"level_1": "ticker", "price": "ticker"})
    if "date" not in out.columns:
        out = out.rename(columns={out.columns[0]: "date"})
    keep = ["date", "ticker", "open", "high", "low", "close", "volume"]
    out = out[[c for c in keep if c in out.columns]].dropna(subset=["close"])
    return out


def fetch(tickers):
    for attempt in range(4):
        try:
            df = yf.download(tickers, start=START, auto_adjust=True, actions=False,
                             progress=False, threads=True, group_by="column")
            return to_long(df, tickers)
        except Exception as e:  # network hiccup / rate limit
            print("retry", attempt, e, file=sys.stderr)
            time.sleep(5 * 2 ** attempt)
    return pd.DataFrame()


def main():
    os.makedirs(RAW, exist_ok=True)
    uni = pd.read_csv(os.path.join(HERE, "universe.csv"))
    tickers = uni["Ticker"].astype(str).tolist()
    for i in range(0, len(tickers), BATCH):
        path = os.path.join(RAW, f"batch_{i // BATCH:03d}.parquet")
        if os.path.exists(path):
            continue
        chunk = tickers[i:i + BATCH]
        out = fetch(chunk)
        out.to_parquet(path, index=False)
        print(f"batch {i // BATCH}: {out['ticker'].nunique() if len(out) else 0}/{len(chunk)} tickers, {len(out)} rows", flush=True)
        time.sleep(2)

    allp = pd.concat([pd.read_parquet(os.path.join(RAW, f)) for f in sorted(os.listdir(RAW))], ignore_index=True)
    missing = sorted(set(tickers) - set(allp["ticker"].unique()))
    if missing:
        print("retrying missing individually:", len(missing), flush=True)
        extra = []
        for t in missing:
            out = fetch([t])
            if len(out):
                extra.append(out)
            time.sleep(0.5)
        if extra:
            allp = pd.concat([allp] + extra, ignore_index=True)
    allp["date"] = pd.to_datetime(allp["date"]).dt.tz_localize(None)
    allp = allp.drop_duplicates(["date", "ticker"]).sort_values(["ticker", "date"])
    allp.to_parquet(os.path.join(DATA_DIR, "ohlcv.parquet"), index=False)
    still = sorted(set(tickers) - set(allp["ticker"].unique()))
    print("tickers with data:", allp["ticker"].nunique(), "missing:", still)


if __name__ == "__main__":
    main()
