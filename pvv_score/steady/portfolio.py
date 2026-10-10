"""Book test, non-Gaussian metrics: each month-start, buy the top 20 names by out-of-sample score for horizon h, hold h sessions,
equal weight; sleeves overlap (one new sleeve per month). Compared with the transparent baseline's top 20, all eligible names, SPY.
Reported: CAGR, max drawdown, CAGR / |max drawdown|, monthly gain-to-pain (sum of up months / sum of down months),
share of months beating SPY, median monthly excess, worst month, share of positions hitting an 8% intraday stop, L1 beta, sector concentration."""
import numpy as np
import pandas as pd
from ..config import CACHE_DIR
from .outcomes import HORIZONS, STOP
D = CACHE_DIR / "steady"


def book_returns(O, C, Rt, score, h, n=20, pick="top"):
    dates = np.sort(O.dropna(subset=[score]).date.unique())
    starts = pd.Series(dates).groupby(pd.Series(dates).dt.to_period("M")).min().tolist()
    idx = C.index; sleeves = []
    for d in starts:
        s = O[(O.date == d)].dropna(subset=[score])
        s = s.nlargest(n, score) if pick == "top" else s
        tk = [t for t in s.ticker if t in C.columns]; days = idx[idx > d][:h]
        if not tk or len(days) == 0: continue
        sleeves.append(Rt.loc[days, tk].mean(axis=1))
    R = pd.concat(sleeves, axis=1).mean(axis=1)
    return R


def metrics(r, spy, start, end):
    r = r[(r.index >= start) & (r.index <= end)].dropna(); sp = spy.reindex(r.index)
    eq = (1 + r).cumprod(); yrs = len(r) / 252; cagr = eq.iloc[-1] ** (1 / yrs) - 1; mdd = (eq / eq.cummax() - 1).min()
    mo = (1 + r).resample("ME").prod() - 1; ms = (1 + sp).resample("ME").prod() - 1
    return {"CAGR": cagr, "max drawdown": mdd, "CAGR/|maxDD|": cagr / abs(mdd), "gain-to-pain (months)": mo.clip(lower=0).sum() / -mo.clip(upper=0).sum(),
            "months > SPY": (mo > ms).mean(), "median monthly excess": (mo - ms).median(), "worst month": mo.min(),
            "L1 beta": (r * np.sign(sp)).sum() / sp.abs().sum()}


def main():
    O = pd.read_parquet(D / "oos.parquet")
    C = pd.read_parquet(CACHE_DIR / "Close.parquet").sort_index(); Lo = pd.read_parquet(CACHE_DIR / "Low.parquet").sort_index()
    Rt = C.pct_change(fill_method=None); spy = Rt.SPY
    pd.set_option("display.width", 240)
    for h in HORIZONS:
        books = {"model top 20": book_returns(O, C, Rt, f"raw_{h}", h), "baseline top 20": book_returns(O, C, Rt, "baseline", h),
                 "all eligible names": book_returns(O, C, Rt, f"raw_{h}", h, pick="all")}
        print(f"\n======== {h}-session sleeves, one new sleeve each month, out-of-sample scores")
        for start, end, lab in [("2018-01-01", "2026-12-31", "2018 to date"), ("2018-01-01", "2023-12-31", "2018-2023"), ("2024-01-01", "2026-12-31", "2024 to date")]:
            M = pd.DataFrame({k: metrics(v, spy, start, end) for k, v in books.items()} | {"SPY": metrics(spy, spy, start, end)}).T
            print(f"-- {lab}"); print(M.round(3).to_string())
        # position-level path stats for the model book
        dates = np.sort(O.dropna(subset=[f"raw_{h}"]).date.unique()); starts = pd.Series(dates).groupby(pd.Series(dates).dt.to_period("M")).min().tolist()
        P = pd.concat([O[O.date == d].nlargest(20, f"raw_{h}") for d in starts])
        for lab, X in [("2018+", P), ("2024+", P[P.date >= "2024-01-01"])]:
            X = X.dropna(subset=[f"stop_{h}"])
            print(f"model top-20 positions {lab}: {len(X)}  stopped at -{STOP:.0%} intraday {(X[f'stop_{h}'] <= -STOP).mean():.1%}  beat SPY {(X[f'xs_{h}'] > 0).mean():.1%}  "
                  f"median excess {X[f'xs_{h}'].median():+.2%}  median path drawdown {X[f'ptt_{h}'].median():.2%}  median L1 beta {X.beta_l1_252.median():.2f}  "
                  f"largest sector {X.sector.value_counts(normalize=True).iloc[0]:.0%} {X.sector.value_counts().index[0]}")


if __name__ == "__main__":
    main()
