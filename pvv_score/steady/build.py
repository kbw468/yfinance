"""Assemble the long research table: every eligible name x session, non-Gaussian features (raw values plus same-day
cross-sectional percentile ranks), trailing steady-state flag, forward path outcomes. Writes CACHE_DIR/steady/table.parquet."""
import sys, time, gc
import numpy as np
import pandas as pd
from ..config import CACHE_DIR
from ..data_io import load_panel, load_universe
from ..clean import clean_panel
from ..features import eligibility
from ..build import sector_close
from . import features, outcomes

OUT = CACHE_DIR / "steady"
START = "2014-06-01"


def main():
    t0 = time.time(); OUT.mkdir(exist_ok=True)
    panel, _ = clean_panel(load_panel(), verbose=False)
    uni = load_universe(); stocks = [t for t in uni.index if t in panel["Close"].columns]
    sb = sector_close(panel["Close"], uni)
    E = eligibility(panel, stocks)
    F = features.compute(panel, sb, stocks); print(f"features {len(F)} {time.time()-t0:.0f}s", file=sys.stderr, flush=True)
    Y = outcomes.compute(panel, stocks); print(f"outcomes {time.time()-t0:.0f}s", file=sys.stderr, flush=True)
    idx = E.loc[START:].stack(); idx = idx[idx].index
    cols = {}
    for k, v in {**F, **Y}.items():
        cols[k] = v.loc[START:].stack(future_stack=True).reindex(idx).values
    T = pd.DataFrame(cols, index=idx); T.index.names = ["date", "ticker"]; T = T.reset_index()
    T["sector"] = T.ticker.map(uni.Sector)
    T["steady"] = ((T.xs_spy_63 > 0) & (T.mdd_63 >= -0.08) & (T.off_high_63 >= -0.05)).astype("int8")
    feat = list(F)
    R = T.groupby("date")[feat].rank(pct=True).astype("float32"); R.columns = [f"cs_{c}" for c in feat]
    T = pd.concat([T, R], axis=1)
    T.to_parquet(OUT / "table.parquet")
    pd.Series(feat).to_csv(OUT / "features.csv", index=False)
    print(f"table {T.shape} {T.date.min().date()}..{T.date.max().date()} {time.time()-t0:.0f}s", file=sys.stderr)


if __name__ == "__main__":
    main()
