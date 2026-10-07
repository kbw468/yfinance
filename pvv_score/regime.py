"""Market regime tag per date, from the MOVE index (ICE BofA bond-market implied vol).

Tested on 2015 → today (COVID window stripped): VIX, VXN and IWM realised vol buckets do not change the probability that a
50+-signature name posts a top-quartile smooth path (0.33–0.39 in every bucket) nor the forward drawdown of trending names.
MOVE is the one independent reading (corr ≈ 0.27 to the equity vol family) and the one with a confirmed effect: rate-sensitive
low-beta names' top-quartile probability falls from 0.236 (MOVE ≤ 70) to 0.147 (MOVE > 120).

So: rate-sensitive sectors are measured on days in the same MOVE bucket as today; every other sector is measured unconditionally
(conditioning on equity vol only thins the cells). The old SPY trend rule showed no split (0.365 vs 0.362) and is gone."""
import numpy as np
import pandas as pd
from .data_io import load_universe
from .volindex import load_vol_indices

RATE_SENSITIVE = {"Utilities", "Real Estate", "Consumer Defensive", "Financial"}
MOVE_BANDS = [(-np.inf, 70, "move_low"), (70, 120, "move_mid"), (120, np.inf, "move_high")]


def move_bucket(x: pd.Series) -> pd.Series:
    return pd.cut(x, [b[0] for b in MOVE_BANDS] + [np.inf], labels=[b[2] for b in MOVE_BANDS], right=True).astype(object)


def regime_series() -> pd.Series:
    m = load_vol_indices()["move"].ffill()
    return move_bucket(m).rename("regime")


def tag(df: pd.DataFrame) -> pd.DataFrame:
    """regime column: today's-bucket-matchable MOVE band for rate-sensitive names, 'all' for everything else."""
    r = regime_series()
    df = df.copy()
    sec = df["sector"] if "sector" in df.columns else df["ticker"].map(load_universe()["Sector"])
    rs = sec.isin(RATE_SENSITIVE).values
    bucket = pd.to_datetime(df["date"]).map(r).astype(object).values
    df["regime"] = np.where(rs, bucket, "all")
    return df


def today_regime() -> str:
    return regime_series().dropna().iloc[-1]


def same_regime(df: pd.DataFrame, reg: str | None = None) -> np.ndarray:
    """rows comparable to today: unconditional for most sectors, same MOVE band for rate-sensitive ones."""
    reg = reg or today_regime()
    return ((df["regime"] == "all") | (df["regime"] == reg)).values


def today_move() -> float:
    return float(load_vol_indices()["move"].ffill().dropna().iloc[-1])
