"""Download adjusted daily close + volume for the finviz universe and benchmarks."""
import sys
import time
from pathlib import Path

import pandas as pd
import yfinance as yf

UNIVERSE_CSV = Path(sys.argv[1])
OUT = Path(sys.argv[2])
START = "1998-06-01"
END = "2026-10-10"
BENCH = ["SPY", "IWM", "RSP", "MDY"]
BATCH = 150

OUT.mkdir(parents=True, exist_ok=True)
tickers = pd.read_csv(UNIVERSE_CSV)["Ticker"].astype(str).str.strip().tolist()
# Yahoo uses '-' for share classes (BRK.B -> BRK-B)
yahoo = [t.replace(".", "-") for t in tickers] + BENCH


def fetch(batch):
    df = yf.download(batch, start=START, end=END, auto_adjust=True, progress=False,
                     threads=True, group_by="column")
    return df["Close"], df["Volume"]


closes, vols, failed = [], [], []
for i in range(0, len(yahoo), BATCH):
    batch = yahoo[i:i + BATCH]
    for attempt in range(3):
        try:
            c, v = fetch(batch)
            break
        except Exception as e:  # network hiccup
            print("retry", i, e, flush=True)
            time.sleep(2 ** (attempt + 1))
    else:
        failed += batch
        continue
    empty = [t for t in batch if t not in c.columns or c[t].dropna().empty]
    failed += empty
    closes.append(c)
    vols.append(v)
    print(f"{i + len(batch)}/{len(yahoo)} done, empty so far: {len(failed)}", flush=True)

# Retry empties one more time in a small batch
if failed:
    retry = list(dict.fromkeys(failed))
    c, v = fetch(retry)
    closes.append(c)
    vols.append(v)

close = pd.concat(closes, axis=1)
vol = pd.concat(vols, axis=1)
close = close.loc[:, ~close.columns.duplicated(keep="last")].dropna(axis=1, how="all")
vol = vol.loc[:, ~vol.columns.duplicated(keep="last")].reindex(columns=close.columns)
close.to_parquet(OUT / "close.parquet")
vol.to_parquet(OUT / "volume.parquet")
missing = sorted(set(yahoo) - set(close.columns))
pd.Series(missing, dtype=str).to_csv(OUT / "missing.csv", index=False)
print("final", close.shape, "missing", len(missing), missing[:50])
