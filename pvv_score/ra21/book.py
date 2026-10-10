"""Portfolio test of a daily score, as it would be traded. 21 staggered sleeves (Jegadeesh-Titman): sleeve i rebalances on
sessions i, i+21, i+42, ...; at each rebalance it buys the top N names by score at the next open, equal weight, 8% stop
(a gap through the stop fills at the open; stopped money sits in cash), exits at the close 21 sessions later. 10 bp per
round trip. The book is the equal-capital sum of the sleeves, marked daily on closes. Benchmarks on the same rows: every
eligible name (same trade rules) and SPY bought and held.
Statistics, all non-Gaussian: CAGR, deepest daily drawdown, MAR = CAGR / |deepest drawdown|, mean drawdown (pain index),
Omega of daily returns at 0 (sum of gains / sum of losses) and of daily returns over SPY, share of 21-session periods
ahead of SPY, worst 21-session period."""
import numpy as np
import pandas as pd

COST = 0.0010
STOP = 0.08


def _sleeve_paths(dates, O, L, C, picks: dict, hold=21, stop=STOP) -> pd.Series:
    """picks: {entry_signal_date_position: [column indices]} -> daily value path of one sleeve (starts at 1)."""
    n = len(dates); val = np.full(n, np.nan); v = 1.0; last = None
    for p in sorted(picks):
        cols = picks[p]
        if p + 1 + hold > n or not cols:
            continue
        if last is not None and p + 1 < last: continue
        e = O[p + 1, cols]; ok = ~np.isnan(e)
        cols = np.asarray(cols)[ok]; e = e[ok]
        if len(cols) == 0: continue
        sp = e * (1 - stop); alive = np.ones(len(cols), bool); rel = np.ones(len(cols))
        for k in range(1, hold + 1):
            i = p + k
            lo, op, cl = L[i, cols], O[i, cols], C[i, cols]
            hit = alive & (lo <= sp)
            fill = np.where((k > 1) & (op <= sp), op, sp)
            rel = np.where(hit, fill / e, np.where(alive, np.where(np.isnan(cl), rel, cl / e), rel))
            alive &= ~hit
            val[i] = v * np.nanmean(rel)
        v = val[p + hold] * (1 - COST); val[p + hold] = v; last = p + hold
    s = pd.Series(val, index=dates)
    return s.ffill().fillna(1.0)


def run(scores: pd.DataFrame, panel: dict, top_n=20, start=None, end=None, universe_mask: pd.DataFrame | None = None) -> pd.Series:
    """scores: wide (dates x tickers) score, NaN = not eligible. Returns the book's daily value (starts at 1)."""
    S = scores.loc[start:end]
    dates = panel["Close"].loc[S.index[0]:].index
    cols = list(S.columns)
    O = panel["Open"][cols].reindex(dates).to_numpy(); L = panel["Low"][cols].reindex(dates).to_numpy(); C = panel["Close"][cols].reindex(dates).to_numpy()
    Sv = S.reindex(dates).to_numpy()
    sleeves = []
    for off in range(21):
        picks = {}
        for p in range(off, len(S), 21):
            row = Sv[p]; ok = ~np.isnan(row)
            if ok.sum() == 0: continue
            if top_n is None:
                picks[p] = list(np.flatnonzero(ok))
            else:
                idx = np.flatnonzero(ok); picks[p] = list(idx[np.argsort(-row[idx], kind="stable")[:top_n]])
        sleeves.append(_sleeve_paths(dates, O, L, C, picks))
    book = pd.concat(sleeves, axis=1).mean(axis=1)
    return book.loc[:S.index[-1]]


def stats(v: pd.Series, spy: pd.Series) -> dict:
    v = v.dropna(); spy = spy.reindex(v.index).ffill()
    yrs = (v.index[-1] - v.index[0]).days / 365.25
    cagr = (v.iloc[-1] / v.iloc[0]) ** (1 / yrs) - 1
    dd = v / v.cummax() - 1
    r = v.pct_change().dropna(); rs = spy.pct_change().reindex(r.index).fillna(0); x = r - rs
    m = v.iloc[::21]; ms = spy.iloc[::21]; mr = m.pct_change().dropna(); msr = ms.pct_change().reindex(mr.index)
    return {"CAGR": cagr, "deepest drawdown": dd.min(), "MAR (CAGR/|DD|)": cagr / abs(dd.min()) if dd.min() < 0 else np.nan, "pain index (mean DD)": dd.mean(),
            "Omega daily": r[r > 0].sum() / -r[r < 0].sum(), "Omega daily vs SPY": x[x > 0].sum() / -x[x < 0].sum(),
            "21d periods ahead of SPY": float((mr > msr).mean()), "worst 21d period": mr.min(), "median 21d excess": float((mr - msr).median())}
