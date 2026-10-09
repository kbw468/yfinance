"""Find the universe names whose last-6-month path matches the reference chart.

Points read off the chart (close, unadjusted) are compared as ratios to the final
value, so dividend adjustment does not matter. Distance = median absolute gap.
"""
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.environ.get("FS_DATA", os.path.join(HERE, "data"))

CHART = {  # date -> close read off the 6M chart
    "2026-04-09": 14.00, "2026-04-20": 14.05, "2026-05-01": 14.05, "2026-05-15": 14.05,
    "2026-06-01": 13.95, "2026-06-10": 14.10, "2026-06-22": 14.40, "2026-07-01": 14.60,
    "2026-07-10": 15.25, "2026-07-20": 15.05, "2026-08-03": 15.50, "2026-08-10": 15.80,
    "2026-08-20": 15.65, "2026-09-01": 15.80, "2026-09-15": 16.05, "2026-09-25": 16.40,
    "2026-10-01": 16.90, "2026-10-08": 17.25,
}


def main():
    src = os.path.join(DATA_DIR, "ohlcv.parquet")
    if os.path.exists(src):
        px = pd.read_parquet(src)
    else:
        raw = os.path.join(DATA_DIR, "raw")
        px = pd.concat([pd.read_parquet(os.path.join(raw, f)) for f in sorted(os.listdir(raw))])
    px["date"] = pd.to_datetime(px["date"]).dt.tz_localize(None)
    wide = px.pivot_table(index="date", columns="ticker", values="close").sort_index()
    wide = wide.loc["2026-03-01":]
    tgt = pd.Series(CHART)
    tgt.index = pd.to_datetime(tgt.index)
    sub = wide.reindex(tgt.index, method="ffill")
    ratio = sub / sub.iloc[-1]
    tr = (tgt / tgt.iloc[-1]).to_numpy()[:, None]
    dist = np.nanmedian(np.abs(ratio.to_numpy() - tr), axis=0)
    res = pd.DataFrame({"ticker": wide.columns, "dist": dist, "last": wide.iloc[-1].to_numpy()})
    res = res.dropna().sort_values("dist")
    print(res.head(int(sys.argv[1]) if len(sys.argv) > 1 else 15).to_string(index=False))


if __name__ == "__main__":
    main()
