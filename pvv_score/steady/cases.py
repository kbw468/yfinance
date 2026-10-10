"""Case studies: tape behaviour of named tickers over the last six months, their setup profile at the breakout that started
each run, the traits they share, an out-of-time test of that shared profile on 2015-2025, and today's look-alikes.

Run dating (objective): peak = highest close in the window; launch = lowest close before the peak; breakout = first close
after the launch that sets a new 63-session closing high. Profile = same-day universe percentile of every feature at the
breakout. Names outside the universe are downloaded with the panel's own code and ranked against the universe each day."""
import sys, json
import numpy as np, pandas as pd
from ..config import CACHE_DIR, RESULTS_DIR
from ..data_io import download, build_panel, load_panel
from ..clean import clean_panel
from . import features
D = CACHE_DIR / "steady"; OUT = RESULTS_DIR / "steady" / "cases"


def name_features(names: list) -> pd.DataFrame:
    """Raw features for the names, long format; names missing from the panel are downloaded and added."""
    panel, _ = clean_panel(load_panel(), verbose=False)
    missing = [n for n in names if n not in panel["Close"].columns]
    if missing:
        extra = build_panel(download(missing, start="2012-01-01"))
        for k in panel:
            if k in extra: panel[k] = panel[k].join(extra[k][missing], how="left")
    sb = pd.DataFrame({n: panel["Close"]["SPY"] for n in names}, index=panel["Close"].index)   # placeholder, replaced below
    sec = json.load(open(D / "sector_etf.json")) if (D / "sector_etf.json").exists() else {}
    for n in names:
        etf = sec.get(n)
        if etf and etf in panel["Close"].columns: sb[n] = panel["Close"][etf]
    F = features.compute(panel, sb, names)
    L = pd.concat({k: v.stack(future_stack=True) for k, v in F.items()}, axis=1); L.index.names = ["date", "ticker"]
    return L.reset_index(), panel


def universe_rank(L: pd.DataFrame, T: pd.DataFrame, feats: list) -> pd.DataFrame:
    """Percentile of each name's raw feature value within the universe's values on the same date (names not in the universe)."""
    out = L[["date", "ticker"]].copy()
    U = T[["date"] + feats]
    for f in feats:
        g = dict(tuple(U[["date", f]].dropna().groupby("date")[f]))
        out[f"cs_{f}"] = [np.nan if (pd.isna(x) or d not in g) else (np.searchsorted(np.sort(g[d].values), x, side="right")) / (len(g[d]) + 1) for d, x in zip(L.date, L[f])]
    return out


def runs(C: pd.Series, start) -> dict:
    c = C.loc[start:].dropna()
    peak = c.idxmax(); launch = c.loc[:peak].idxmin()
    hi63 = C.rolling(63, min_periods=63).max()
    after = c.loc[launch:peak]; bo = after[after >= hi63.reindex(after.index)].index
    breakout = bo[0] if len(bo) else peak
    seg = c.loc[launch:peak]
    return {"launch": launch, "breakout": breakout, "peak": peak, "launch_to_peak": c[peak] / c[launch] - 1, "breakout_to_peak": c[peak] / c[breakout] - 1,
            "launch_to_breakout": c[breakout] / c[launch] - 1, "sessions_launch_to_peak": len(seg) - 1, "sessions_breakout_to_peak": len(c.loc[breakout:peak]) - 1,
            "max_dd_in_run": (seg / seg.cummax() - 1).min(), "max_dd_after_breakout": (c.loc[breakout:peak] / c.loc[breakout:peak].cummax() - 1).min(),
            "now_off_peak": c.iloc[-1] / c[peak] - 1}


def main(names: list, start: str):
    OUT.mkdir(parents=True, exist_ok=True)
    feats = pd.read_csv(D / "features.csv").iloc[:, 0].tolist()
    T = pd.read_parquet(D / "table.parquet")
    inside = [n for n in names if n in set(T.ticker.unique())]; outside = [n for n in names if n not in inside]
    P = T[T.ticker.isin(inside)][["date", "ticker"] + feats + [f"cs_{f}" for f in feats]].copy()
    C = pd.read_parquet(CACHE_DIR / "Close.parquet")
    if outside:
        L, panel = name_features(outside)
        R = universe_rank(L[L.date >= "2014-06-01"], T, feats)
        P = pd.concat([P, L[L.date >= "2014-06-01"].merge(R, on=["date", "ticker"])], ignore_index=True)
        for n in outside: C[n] = panel["Close"][n]
    P.to_parquet(OUT / "case_features.parquet")
    R = {n: runs(C[n], start) for n in names}
    RT = pd.DataFrame(R).T; RT.to_csv(OUT / "runs.csv")
    print("RUNS"); print(RT.to_string())
    rows = []
    for n in names:
        for lab in ("launch", "breakout"):
            r = P[(P.ticker == n) & (P.date == R[n][lab])]
            if len(r): rows.append(r.iloc[0].rename(f"{n}|{lab}"))
    S = pd.DataFrame(rows); S.to_csv(OUT / "setups.csv")
    return S, RT
