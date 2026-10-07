"""Volatility-index cache: VIX, VXN, MOVE, VVIX, OVX from Yahoo plus IWM 20-session realised vol (RVX is not served by Yahoo).
Stored in the cache as one daily frame; refreshed from the last cached session on each call with refresh=True."""
import numpy as np
import pandas as pd
import yfinance as yf
from .config import CACHE_DIR
from .data_io import load_panel

YAHOO = {"vix": "^VIX", "vxn": "^VXN", "move": "^MOVE", "vvix": "^VVIX", "ovx": "^OVX"}
PATH = CACHE_DIR / "volindices.parquet"
COLS = list(YAHOO) + ["iwm_rv20"]


def _iwm_rv20() -> pd.Series:
    panel = load_panel()
    c = panel["Close"]["IWM"].dropna()
    return (np.log(c).diff().rolling(20).std() * np.sqrt(252) * 100).rename("iwm_rv20")


def load_vol_indices(refresh: bool = False) -> pd.DataFrame:
    old = pd.read_parquet(PATH) if PATH.exists() else pd.DataFrame(columns=COLS)
    if not refresh and len(old):
        return old[COLS]
    start = (old.index.max() - pd.Timedelta(days=7)).strftime("%Y-%m-%d") if len(old) else "2005-01-01"
    px = yf.download(list(YAHOO.values()), start=start, progress=False, auto_adjust=False)["Close"]
    px = px.rename(columns={v: k for k, v in YAHOO.items()})
    px.index = pd.to_datetime(px.index).tz_localize(None)
    new = old[list(YAHOO)].combine_first(px[list(YAHOO)]) if len(old) else px[list(YAHOO)]
    new.update(px[list(YAHOO)])
    try:
        rv = _iwm_rv20()
        out = new.join(rv, how="outer")
    except Exception:
        out = new.join(old["iwm_rv20"], how="left") if "iwm_rv20" in old else new.assign(iwm_rv20=np.nan)
    out = out.sort_index()[COLS]
    out.to_parquet(PATH)
    return out


def today_readings() -> dict:
    v = load_vol_indices().ffill().dropna(how="all").iloc[-1]
    return {k: (None if pd.isna(v[k]) else float(v[k])) for k in COLS} | {"date": v.name.strftime("%Y-%m-%d")}
