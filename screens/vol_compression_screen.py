"""Screen a finviz export for the five criteria below using yfinance daily bars.

1. 21d avg volume / 252d avg volume > 1
2. trailing 21-trading-day return > 0
3. trailing 252-trading-day return > 0
4. 21d realized vol < 63d realized vol AND 21d realized vol < 252d realized vol
5. YTD return <= 50%

Usage:
    python screens/vol_compression_screen.py screens/finviz_7.csv [out.csv]
"""
import sys

import numpy as np
import pandas as pd

import yfinance as yf

YTD_CAP = 0.50


def load_tickers(path: str) -> list[str]:
    df = pd.read_csv(path)
    return [t.replace(".", "-") for t in df["Ticker"].dropna().astype(str)]


def realized_vol(logret: pd.Series, n: int) -> float:
    return float(logret.tail(n).std(ddof=1) * np.sqrt(252))


def screen(tickers: list[str]) -> pd.DataFrame:
    px = yf.download(
        tickers,
        period="2y",
        interval="1d",
        auto_adjust=True,
        group_by="column",
        threads=True,
        progress=False,
    )
    close = px["Close"]
    vol = px["Volume"]
    year_start = pd.Timestamp(close.index[-1].year, 1, 1)

    rows = []
    for t in close.columns:
        c = close[t].dropna()
        v = vol[t].reindex(c.index)
        if len(c) < 253:
            continue
        lr = np.log(c).diff().dropna()
        prior_year = c[c.index < year_start]
        if prior_year.empty:
            continue
        rv21, rv63, rv252 = (realized_vol(lr, n) for n in (21, 63, 252))
        rows.append(
            {
                "Ticker": t,
                "Close": c.iloc[-1],
                "RelVol_21_252": v.tail(21).mean() / v.tail(252).mean(),
                "Ret_1M": c.iloc[-1] / c.iloc[-22] - 1,
                "Ret_1Y": c.iloc[-1] / c.iloc[-253] - 1,
                "Ret_YTD": c.iloc[-1] / prior_year.iloc[-1] - 1,
                "RV21": rv21,
                "RV63": rv63,
                "RV252": rv252,
            }
        )
    out = pd.DataFrame(rows).set_index("Ticker")
    mask = (
        (out["RelVol_21_252"] > 1)
        & (out["Ret_1M"] > 0)
        & (out["Ret_1Y"] > 0)
        & (out["RV21"] < out["RV63"])
        & (out["RV21"] < out["RV252"])
        & (out["Ret_YTD"] <= YTD_CAP)
    )
    return out[mask].sort_values("RelVol_21_252", ascending=False)


if __name__ == "__main__":
    src = sys.argv[1] if len(sys.argv) > 1 else "screens/finviz_7.csv"
    dst = sys.argv[2] if len(sys.argv) > 2 else "screens/screen_results.csv"
    res = screen(load_tickers(src))
    res.to_csv(dst)
    pd.set_option("display.width", 200)
    print(res.round(3).to_string())
    print(f"\n{len(res)} tickers pass -> {dst}")
