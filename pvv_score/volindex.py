"""Volatility-index cache: VIX, VXN, MOVE, VVIX, OVX from Yahoo plus IWM 20-session realised vol (RVX is not served by Yahoo).
Stored in the cache as one daily frame; refreshed from the last cached session on each call with refresh=True."""
import numpy as np
import pandas as pd
import yfinance as yf
from .config import CACHE_DIR

YAHOO = {"vix": "^VIX", "vxn": "^VXN", "move": "^MOVE", "vvix": "^VVIX", "ovx": "^OVX", "iwm": "IWM"}   # IWM close feeds the realised-vol proxy
PATH = CACHE_DIR / "volindices.parquet"
COLS = list(YAHOO) + ["iwm_rv20"]


def _iwm_rv20(close: pd.Series) -> pd.Series:
    c = close.dropna()
    return (np.log(c).diff().rolling(20).std() * np.sqrt(252) * 100).rename("iwm_rv20")


def load_vol_indices(refresh: bool = False) -> pd.DataFrame:
    old = pd.read_parquet(PATH) if PATH.exists() else pd.DataFrame(columns=COLS)
    if not refresh and len(old):
        return old[COLS]
    start = (old.index.max() - pd.Timedelta(days=7)).strftime("%Y-%m-%d") if len(old) else "2005-01-01"
    px = yf.download(list(YAHOO.values()), start=start, progress=False, auto_adjust=False)["Close"]
    px = px.rename(columns={v: k for k, v in YAHOO.items()})
    px.index = pd.to_datetime(px.index).tz_localize(None)
    have = [c for c in YAHOO if c in old.columns]
    new = old[have].combine_first(px[list(YAHOO)]) if len(old) else px[list(YAHOO)]
    new.update(px[list(YAHOO)])
    new = new.sort_index()
    if new["iwm"].notna().sum() < 25:                       # first refresh after the IWM column was added: fetch its full history
        full = yf.download("IWM", start="2005-01-01", progress=False, auto_adjust=False)["Close"]
        full = full.iloc[:, 0] if isinstance(full, pd.DataFrame) else full
        full.index = pd.to_datetime(full.index).tz_localize(None)
        new["iwm"] = full.reindex(new.index.union(full.index)).reindex(new.index) if len(new) else full
        new = new.combine_first(full.to_frame("iwm"))
    out = new.assign(iwm_rv20=_iwm_rv20(new["iwm"])).sort_index()[COLS]
    assert out["iwm_rv20"].dropna().index.max() >= out["vix"].dropna().index.max() - pd.Timedelta(days=5), "IWM realised vol is stale"
    out.to_parquet(PATH)
    return out


def today_readings() -> dict:
    v = load_vol_indices().ffill().dropna(how="all").iloc[-1]
    return {k: (None if pd.isna(v[k]) else float(v[k])) for k in COLS} | {"date": v.name.strftime("%Y-%m-%d")}


def readings_at(asof) -> dict:
    """Vol readings as of a session (last available on or before it) plus the MOVE band used as the regime tag."""
    from .regime import move_bucket
    v = load_vol_indices().ffill()
    v = v[v.index <= pd.Timestamp(asof)]
    row = v.iloc[-1]
    out = {k: (None if pd.isna(row[k]) else float(row[k])) for k in COLS} | {"date": row.name.strftime("%Y-%m-%d")}
    out["move_band"] = str(move_bucket(pd.Series([out["move"]])).iloc[0]) if out["move"] is not None else "n/a"
    return out
