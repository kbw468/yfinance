"""Second-pass features from the trade outcomes themselves, strictly point-in-time: an entry on day d is known once its exit
close (d+21) has printed, so at the close of t only entries d <= t-22 are used.
  own_sup_rate_252 / _756   share of the name's entries over the last 1 / 3 years that were superior
  own_marpct_med_252        median same-day percentile of its 21-session trade outcome over the last year
  own_stop_rate_252         share of its 21-session trades over the last year that hit the 8% stop
  own_dd_med_252            median drawdown suffered inside its 21-session trades over the last year
  ind_sup_rate_21           industry peers' (leave-one-out; sector if fewer than 3) share of superior entries, last 21 known entry dates
Writes CACHE_DIR/ra21/extra2.parquet aligned to the table's (date, ticker) rows."""
import time
import numpy as np
import pandas as pd
from ..data_io import load_universe
from .common import D
from .extra import _loo_mean

LAG = 22


def main():
    t0 = time.time()
    T = pd.read_parquet(D / "table.parquet", columns=["date", "ticker", "sup_21", "marpct_21", "stopped_21", "dd_21"])
    W = {c: T.pivot(index="date", columns="ticker", values=c) for c in ["sup_21", "marpct_21", "stopped_21", "dd_21"]}
    F = {}
    s = W["sup_21"].shift(LAG)
    F["own_sup_rate_252"] = s.rolling(252, min_periods=126).mean(); F["own_sup_rate_756"] = s.rolling(756, min_periods=378).mean()
    F["own_marpct_med_252"] = W["marpct_21"].shift(LAG).rolling(252, min_periods=126).median()
    F["own_stop_rate_252"] = W["stopped_21"].shift(LAG).rolling(252, min_periods=126).mean()
    F["own_dd_med_252"] = W["dd_21"].shift(LAG).rolling(252, min_periods=126).median()
    uni = load_universe(); cols = list(s.columns)
    ind = uni.Industry.reindex(cols); sec = uni.Sector.reindex(cols)
    gi = {t: list(ind.index[ind == ind[t]]) for t in cols}; gs = {t: list(sec.index[sec == sec[t]]) for t in cols}
    F["ind_sup_rate_21"] = _loo_mean(s.rolling(21, min_periods=10).mean(), gi, gs)
    out = T[["date", "ticker"]].copy()
    for k, v in F.items():
        out[k] = v.stack(future_stack=True).reindex(pd.MultiIndex.from_frame(T[["date", "ticker"]])).to_numpy().astype("float32")
    out.to_parquet(D / "extra2.parquet")
    pd.Series(list(F)).to_csv(D / "extra2_features.csv", index=False)
    print(out.describe().T.round(3).to_string()); print(f"done {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
