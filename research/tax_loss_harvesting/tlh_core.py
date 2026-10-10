"""Core data prep + portfolio math for the Q4 tax-loss-harvesting study."""
from pathlib import Path

import numpy as np
import pandas as pd

MIN_DV = 1e6           # 63d median dollar volume floor at formation ($)
CLIP = (-0.80, 1.50)   # daily return clip: kills bad ticks, keeps real moves
ANN = 252


class Data:
    def __init__(self, data_dir: Path):
        data_dir = Path(data_dir)
        close = pd.read_parquet(data_dir / "close.parquet")
        vol = pd.read_parquet(data_dir / "volume.parquet")
        irx = pd.read_parquet(data_dir / "irx.parquet").iloc[:, 0]
        # NYSE calendar = days SPY traded
        close = close.loc[close["SPY"].notna()]
        vol = vol.reindex(close.index)
        self.bench = close[["SPY", "IWM", "RSP", "MDY"]]
        self.close = close.drop(columns=self.bench.columns)
        self.vol = vol[self.close.columns]
        self.ret = self.close.pct_change(fill_method=None).clip(*CLIP)
        self.bret = self.bench.pct_change(fill_method=None)
        self.dv63 = (self.close * self.vol).rolling(63, min_periods=40).median()
        # daily risk-free from 13w T-bill (annualized %, discount basis ~ fine here)
        self.rf = (irx.reindex(close.index).ffill() / 100 / ANN).fillna(0)
        self.growth = (1 + self.ret.fillna(0)).cumprod().where(self.close.ffill().notna())
        idx = self.close.index
        s = pd.Series(idx, index=idx)
        self.month_ends = s.groupby([idx.year, idx.month]).last().values

    def me(self, year, month):
        """Last trading day of a month."""
        m = self.month_ends
        sel = m[(pd.DatetimeIndex(m).year == year) & (pd.DatetimeIndex(m).month == month)]
        return pd.Timestamp(sel[0]) if len(sel) else None

    def td_on_or_before(self, date):
        idx = self.close.index
        return idx[idx.searchsorted(pd.Timestamp(date), side="right") - 1]

    def window_ret(self, d0, d1, cols=None):
        """Compounded (clipped) return per stock from close d0 to close d1."""
        g = self.growth if cols is None else self.growth[cols]
        return g.loc[d1] / g.loc[d0] - 1

    def signals(self, t0):
        """Point-in-time signals at formation close t0 (month-end)."""
        y, m = t0.year, t0.month
        t12 = self.me(y - 1, m)
        t1 = self.me(y if m > 1 else y - 1, m - 1 if m > 1 else 12)
        ty = self.me(y - 1, 12)
        if t12 is None or t1 is None:
            return None
        df = pd.DataFrame({
            "R12": self.window_ret(t12, t0),
            "R1": self.window_ret(t1, t0),
            "YTD": self.window_ret(ty, t0) if ty is not None and ty < t0 else np.nan,
            "DV": self.dv63.loc[t0],
        })
        ok = df["R12"].notna() & df["R1"].notna() & (df["DV"] >= MIN_DV)
        return df.loc[ok]

    def bh_path(self, members, d0, d1):
        """Equal-weight buy-and-hold daily return series from close d0 to d1."""
        g = self.growth.loc[d0:d1, members].ffill()
        v = (g / g.iloc[0]).mean(axis=1)
        return v.pct_change().iloc[1:]


def risk_stats(daily: pd.Series, rf: pd.Series, bench: pd.Series):
    """Annualized stats on a (possibly concatenated) daily return series."""
    daily = daily.dropna()
    rf = rf.reindex(daily.index).fillna(0)
    b = bench.reindex(daily.index)
    ex = daily - rf
    dn = ex[ex < 0]
    beta = np.cov(daily, b)[0, 1] / b.var()
    alpha = (daily.mean() - rf.mean() - beta * (b.mean() - rf.mean())) * ANN
    return {
        "ann_ret": daily.mean() * ANN,
        "ann_vol": daily.std() * np.sqrt(ANN),
        "sharpe": ex.mean() / daily.std() * np.sqrt(ANN),
        "sortino": ex.mean() / np.sqrt((dn ** 2).sum() / len(ex)) * np.sqrt(ANN),
        "beta_spy": beta,
        "alpha_spy_ann": alpha,
    }


def max_dd(daily: pd.Series):
    v = (1 + daily.fillna(0)).cumprod()
    return (v / v.cummax() - 1).min()


def tstat(x):
    x = pd.Series(x).dropna()
    return x.mean() / (x.std(ddof=1) / np.sqrt(len(x))) if len(x) > 2 else np.nan
