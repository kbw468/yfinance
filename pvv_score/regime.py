"""Market regime tag per date from SPY's own tape: risk-on = 126-session return > 0 AND within 10% of the 252-session high.
Everything that measures a probability does so on days in the SAME regime as the date being scored."""
import numpy as np
import pandas as pd
from .data_io import load_panel
from .clean import clean_panel
from .config import MARKET


def regime_series() -> pd.Series:
    p, _ = clean_panel(load_panel(), verbose=False)
    c = p["Close"][MARKET]
    on = (c / c.shift(126) - 1 > 0) & (c / c.rolling(252).max() - 1 > -0.10)
    return pd.Series(np.where(on, "risk_on", "risk_off"), index=c.index, name="regime")


def tag(df: pd.DataFrame) -> pd.DataFrame:
    r = regime_series()
    df = df.copy()
    df["regime"] = df.date.map(r)
    return df


def today_regime() -> str:
    return regime_series().dropna().iloc[-1]
