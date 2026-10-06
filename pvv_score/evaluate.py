"""Single-factor and conditional evaluation.

IC = daily cross-sectional Spearman rank correlation between factor and forward target over eligible names.
Because forward windows overlap (h-day target observed daily), the IC time series is autocorrelated;
t-stats use Newey-West with Bartlett kernel and lag = horizon.
"""
import numpy as np
import pandas as pd
from scipy import stats

from .config import HORIZONS, RECENT_START, BACKTEST_START


def nw_tstat(x: pd.Series, lag: int) -> tuple[float, float, float]:
    """Mean, Newey-West t-stat, and two-sided p-value of the mean of an autocorrelated series."""
    x = x.dropna().values
    n = len(x)
    if n < 20:
        return np.nan, np.nan, np.nan
    m = x.mean()
    e = x - m
    s = np.dot(e, e)
    for k in range(1, min(lag, n - 1) + 1):
        w = 1 - k / (lag + 1)
        s += 2 * w * np.dot(e[k:], e[:-k])
    se = np.sqrt(s) / n
    t = m / se if se > 0 else np.nan
    p = 2 * (1 - stats.norm.cdf(abs(t))) if np.isfinite(t) else np.nan
    return m, t, p


def _demeaned_ranks(df: pd.DataFrame, cols: list, by="date") -> pd.DataFrame:
    rk = df.groupby(by)[cols].rank(pct=True)
    return rk - rk.groupby(df[by]).transform("mean")


def daily_ic(df: pd.DataFrame, factors: list, target: str, by="date", min_n=50) -> pd.DataFrame:
    """Return date x factor matrix of Spearman ICs vs `target` (eligible rows only)."""
    sub = df.loc[df[target].notna(), [by] + factors + [target]]
    dm = _demeaned_ranks(sub, factors + [target], by)
    g = sub[by].values
    ry = dm[target].values[:, None]
    rf = dm[factors]
    num = (rf * ry).groupby(g).sum()
    den = np.sqrt((rf ** 2).groupby(g).sum().mul(((dm[target] ** 2).groupby(g).sum()), axis=0))
    ic = num / den
    n = sub.groupby(by).size()
    ic = ic[n >= min_n]  # need a real cross-section
    ic.index.name = "date"
    return ic


def summarize_ic(ic: pd.DataFrame, lag: int, start=None, end=None) -> pd.DataFrame:
    sl = ic.loc[start:end]
    rows = {}
    for f in sl.columns:
        m, t, p = nw_tstat(sl[f], lag)
        rows[f] = {"ic": m, "t_nw": t, "p": p, "hit": (sl[f] > 0).mean(), "n_days": sl[f].notna().sum(),
                   "ic_ir": sl[f].mean() / sl[f].std() * np.sqrt(252 / lag) if sl[f].std() > 0 else np.nan}
    return pd.DataFrame(rows).T


def decile_spread(df: pd.DataFrame, factor: str, target: str, q=10) -> pd.Series:
    """Per-date mean target of top decile minus bottom decile (raw target units, e.g. excess Sharpe)."""
    sub = df.loc[df[target].notna() & df[factor].notna(), ["date", factor, target]]
    sub["dec"] = sub.groupby("date")[factor].transform(lambda s: pd.qcut(s.rank(method="first"), q, labels=False) if len(s) >= 5 * q else np.nan)
    m = sub.groupby(["date", "dec"])[target].mean().unstack()
    return m[q - 1] - m[0], m


def yearly_ic(ic: pd.DataFrame) -> pd.DataFrame:
    return ic.groupby(ic.index.year).mean()


def factor_autocorr(df: pd.DataFrame, factors: list, lag=21) -> pd.Series:
    """Rank autocorrelation of each factor at `lag` days (turnover proxy)."""
    piv = {f: df.pivot(index="date", columns="ticker", values=f) for f in factors}
    out = {}
    for f, w in piv.items():
        rk = w.rank(axis=1, pct=True)
        out[f] = rk.corrwith(rk.shift(lag), axis=1).mean()
    return pd.Series(out, name=f"rank_autocorr_{lag}")


def bh_fdr(p: pd.Series, alpha=0.05) -> pd.Series:
    """Benjamini-Hochberg: returns boolean 'significant after FDR control'."""
    p = p.dropna().sort_values()
    n = len(p)
    thresh = alpha * np.arange(1, n + 1) / n
    passed = p.values <= thresh
    k = np.where(passed)[0].max() + 1 if passed.any() else 0
    sig = pd.Series(False, index=p.index)
    sig.iloc[:k] = True
    return sig.reindex(p.index)


def evaluate_factors(df: pd.DataFrame, factors: list, targets: dict, recent_start=RECENT_START) -> tuple[pd.DataFrame, dict]:
    """targets: {label: (column, horizon)}. Returns summary table and dict of IC frames."""
    ics = {}
    tabs = []
    for label, (col, h) in targets.items():
        ic = daily_ic(df, factors, col)
        ics[label] = ic
        full = summarize_ic(ic, h, BACKTEST_START).add_prefix(f"{label}|full|")
        rec = summarize_ic(ic, h, recent_start).add_prefix(f"{label}|recent|")
        tabs.append(pd.concat([full, rec], axis=1))
    return pd.concat(tabs, axis=1), ics
