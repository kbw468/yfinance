"""RA21 research table: every eligible name x session from 2014-06, the non-Gaussian base features (steady table), the
additional families (extra.py), trade outcomes at 21/42/63 sessions and the superior labels. Writes CACHE_DIR/ra21/table.parquet."""
import sys, time
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from ..config import CACHE_DIR
from ..data_io import load_panel, load_universe
from ..clean import clean_panel
from ..features import eligibility
from ..build import sector_close
from ..volindex import load_vol_indices
from . import outcomes, extra

OUT = CACHE_DIR / "ra21"
START = "2014-06-01"
BASE = CACHE_DIR / "steady" / "table.parquet"


def main():
    t0 = time.time(); OUT.mkdir(exist_ok=True)
    panel, _ = clean_panel(load_panel(), verbose=False)
    uni = load_universe(); stocks = [t for t in uni.index if t in panel["Close"].columns]
    sb = sector_close(panel["Close"], uni)
    E = eligibility(panel, stocks)
    Y = outcomes.compute(panel, stocks); print(f"outcomes {len(Y)} {time.time()-t0:.0f}s", file=sys.stderr, flush=True)
    X = extra.compute(panel, stocks, uni, sb, Y, load_vol_indices()); print(f"extra features {len(X)} {time.time()-t0:.0f}s", file=sys.stderr, flush=True)
    clash = set(X) & set(Y)
    assert not clash, f"feature and outcome names collide: {sorted(clash)}"
    idx = E.loc[START:].stack(); idx = idx[idx].index
    cols = {k: v.loc[START:].stack(future_stack=True).reindex(idx).to_numpy() for k, v in {**X, **Y}.items()}
    T = pd.DataFrame(cols, index=idx); T.index.names = ["date", "ticker"]; T = T.reset_index()
    base_feats = pd.read_csv(CACHE_DIR / "steady" / "features.csv").iloc[:, 0].tolist()
    assert not (set(base_feats) & set(Y)), "base feature and outcome names collide"
    B = pq.read_table(BASE, columns=["date", "ticker", "sector"] + base_feats).to_pandas()
    # the base table must carry trailing features only: recompute path efficiency here and require an exact match
    lr = np.log(panel["Close"][stocks]).diff()
    for w in (21, 63):
        tr = ((np.log(panel["Close"][stocks]) - np.log(panel["Close"][stocks].shift(w))) / lr.abs().rolling(w, min_periods=w).sum()).astype("float32")
        v = tr.loc[START:].stack(future_stack=True).reindex(idx).to_numpy()
        assert np.allclose(B[f"eff_{w}"].to_numpy(), v, equal_nan=True, atol=1e-5), f"base table eff_{w} is not the trailing efficiency (rebuild steady table)"
    assert len(B) == len(T) and (B.date.values == T.date.values).all() and (B.ticker.values == T.ticker.values).all(), "row order differs from the base table"
    T = pd.concat([B, T.drop(columns=["date", "ticker"])], axis=1)
    T["industry"] = T.ticker.map(uni.Industry)
    T = outcomes.label(T)
    T.to_parquet(OUT / "table.parquet")
    pd.Series(list(X)).to_csv(OUT / "extra_features.csv", index=False)
    pd.Series(base_feats).to_csv(OUT / "base_features.csv", index=False)
    print(f"table {T.shape} {T.date.min().date()}..{T.date.max().date()} {time.time()-t0:.0f}s", file=sys.stderr)
    print(T[[f"sup_{h}" for h in outcomes.HORIZONS]].mean().round(4).to_dict(), file=sys.stderr)


if __name__ == "__main__":
    main()
