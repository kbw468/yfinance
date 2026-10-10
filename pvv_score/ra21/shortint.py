"""Short-interest features from FINRA consolidated short interest (bi-monthly settlements, 2017-12 onward), point-in-time:
a settlement is usable 8 business days after it (FINRA disseminates about 7 business days after settlement).
  si_dtc             days to cover (short position / average daily volume, FINRA's figure), capped at 60
  si_chg_1           change in the short position since the previous settlement
  si_chg_3           change over the last three settlements (about six weeks)
  si_dtc_pctile_own  days to cover vs the name's own last 24 settlements (about one year)
  si_dtc_now         short position / the name's 21-session median volume today
Writes CACHE_DIR/ra21/extra3.parquet aligned to the table rows, and extra3_features.csv."""
import numpy as np
import pandas as pd
from pandas.tseries.offsets import BDay
from ..data_io import load_panel
from .common import D

SRC = D / "finra_si.parquet"


def main():
    si = pd.read_parquet(SRC)
    si["settle"] = pd.to_datetime(si.settlementDate)
    si["ticker"] = si.ticker.str.replace(".", "-", regex=False)
    si = si.sort_values(["ticker", "settle"]).drop_duplicates(["ticker", "settle"], keep="last")
    si["avail"] = (si.settle + BDay(8)).astype("datetime64[ns]")
    g = si.groupby("ticker")
    si["si_dtc"] = si.daysToCoverQuantity.clip(upper=60)
    si["si_chg_1"] = si.currentShortPositionQuantity / g.currentShortPositionQuantity.shift(1) - 1
    si["si_chg_3"] = si.currentShortPositionQuantity / g.currentShortPositionQuantity.shift(3) - 1
    si["si_dtc_pctile_own"] = g.si_dtc.transform(lambda s: s.rolling(24, min_periods=12).rank(pct=True))
    T = pd.read_parquet(D / "table.parquet", columns=["date", "ticker"])
    T["row"] = np.arange(len(T)); T["date"] = T.date.astype("datetime64[ns]")
    cols = ["si_dtc", "si_chg_1", "si_chg_3", "si_dtc_pctile_own", "currentShortPositionQuantity"]
    M = pd.merge_asof(T.sort_values("date"), si[["ticker", "avail"] + cols].sort_values("avail"), left_on="date", right_on="avail", by="ticker",
                      direction="backward", tolerance=pd.Timedelta(days=40)).sort_values("row")
    V = load_panel()["Volume"]
    mv = V.rolling(21, min_periods=15).median().stack(future_stack=True)
    mv.index.names = ["date", "ticker"]
    M["vol21"] = mv.reindex(pd.MultiIndex.from_arrays([M.date, M.ticker])).to_numpy()
    M["si_dtc_now"] = (M.currentShortPositionQuantity / M.vol21).clip(upper=60)
    feats = ["si_dtc", "si_chg_1", "si_chg_3", "si_dtc_pctile_own", "si_dtc_now"]
    out = M[feats].astype("float32").reset_index(drop=True)
    assert len(out) == len(T)
    out.to_parquet(D / "extra3.parquet"); pd.Series(feats).to_csv(D / "extra3_features.csv", index=False)
    print(f"coverage 2018-02+: {M.loc[M.date >= '2018-02-01', 'si_dtc'].notna().mean():.3f}"); print(out.describe().T.round(3).to_string())


if __name__ == "__main__":
    main()
