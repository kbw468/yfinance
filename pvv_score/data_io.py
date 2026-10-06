"""Download / cache daily OHLCV via yfinance (this repo)."""
import sys
import time
import pandas as pd
import yfinance as yf

from .config import CACHE_DIR, UNIVERSE_CSV, START, MARKET, SECTOR_ETF

FIELDS = ["Open", "High", "Low", "Close", "Volume"]


def load_universe() -> pd.DataFrame:
    u = pd.read_csv(UNIVERSE_CSV)[["Ticker", "Company", "Sector", "Industry"]]
    u["Ticker"] = u["Ticker"].str.replace(".", "-", regex=False)  # BRK.B -> BRK-B
    u["SectorETF"] = u["Sector"].map(SECTOR_ETF)
    assert u["SectorETF"].notna().all(), u[u["SectorETF"].isna()]
    return u.set_index("Ticker")


def all_tickers() -> list:
    u = load_universe()
    return sorted(set(u.index) | {MARKET} | set(SECTOR_ETF.values()))


def download(tickers, start=START, batch=100) -> dict:
    """Return dict field -> wide DataFrame (date x ticker). Unadjusted OHLC + Adj Close kept separately."""
    frames = []
    for i in range(0, len(tickers), batch):
        chunk = tickers[i:i + batch]
        for attempt in range(4):
            try:
                df = yf.download(chunk, start=start, auto_adjust=False, actions=False,
                                 progress=False, threads=True, group_by="column")
                break
            except Exception as e:  # noqa
                print("retry", attempt, e, file=sys.stderr)
                time.sleep(2 ** attempt)
        frames.append(df)
        print(f"downloaded {i + len(chunk)}/{len(tickers)}", file=sys.stderr)
    raw = pd.concat(frames, axis=1)
    out = {}
    for f in FIELDS + ["Adj Close"]:
        w = raw[f].copy()
        w.index = pd.to_datetime(w.index).tz_localize(None)
        out[f] = w.sort_index()
    return out


def build_panel(raw: dict) -> dict:
    """Adjust OHLC by the Adj Close / Close ratio so splits and dividends don't create fake gaps/ranges.
    Volume is split-adjusted by yfinance already (auto_adjust=False still adjusts volume for splits)."""
    ratio = raw["Adj Close"] / raw["Close"]
    panel = {f: raw[f] * ratio for f in ["Open", "High", "Low", "Close"]}
    panel["Volume"] = raw["Volume"].astype(float)
    panel["RawClose"] = raw["Close"]
    return panel


def save_panel(panel: dict):
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    for k, v in panel.items():
        v.to_parquet(CACHE_DIR / f"{k.replace(' ', '_')}.parquet")


def load_panel() -> dict:
    return {k: pd.read_parquet(CACHE_DIR / f"{k}.parquet")
            for k in ["Open", "High", "Low", "Close", "Volume", "RawClose"]}


if __name__ == "__main__":
    t = all_tickers()
    print(len(t), "tickers", file=sys.stderr)
    raw = download(t)
    panel = build_panel(raw)
    save_panel(panel)
    c = panel["Close"]
    print(c.shape, c.index.min(), c.index.max())
    print("tickers with <2y history:", c.notna().sum()[c.notna().sum() < 504].to_dict())
