"""Shared definitions: feature lists, beta buckets (terciles of the L1 capture beta), tie-safe quintile conditions,
fixed rate-of-change conditions, windows."""
import numpy as np
import pandas as pd
from ..config import CACHE_DIR, RESULTS_DIR

D = CACHE_DIR / "steady"
MODEL = RESULTS_DIR / "model"
HORIZONS = (21, 42, 63)
DISCOVER = ("2015-01-01", "2020-12-31")
CONFIRM_START = "2021-01-01"
COVID = ("2020-02-20", "2020-06-30")
TIER_EDGES = [-1, .25, .30, .35, .40, .45, 2]
COMP_Q = [0, .2, .4, .6, .8, 1.0]
MIN_GROUP_N = 1500
# fixed thresholds on non-Gaussian rates of change: name -> (column, op, value)
FIXED = {
    "volume 10d/63d median > 1.5x": ("volume_roc_10_63", ">", 1.5),
    "volume 21d/252d median > 1.25x": ("volume_roc_21_252", ">", 1.25),
    "volume 21d/252d median > 1.5x": ("volume_roc_21_252", ">", 1.5),
    "typical move 21d/252d < 0.8x": ("vol_roc_21_252", "<", 0.8),
    "typical move 21d/252d > 1.25x": ("vol_roc_21_252", ">", 1.25),
    "typical move 5d/63d < 0.8x": ("vol_roc_5_63", "<", 0.8),
    "price 21d > +3%": ("roc_21", ">", 0.03),
    "price 21d > +8%": ("roc_21", ">", 0.08),
    "price 63d > +15%": ("roc_63", ">", 0.15),
    "price 5d < -3%": ("roc_5", "<", -0.03),
}


def features() -> list:
    return pd.read_csv(D / "features.csv").iloc[:, 0].tolist()


def layers() -> dict:
    f = features()
    return {"state": [c for c in f if c.endswith("_pctile_own")], "level": [c for c in f if not c.endswith("_pctile_own")]}


def beta_bucket(df: pd.DataFrame) -> pd.Series:
    """Terciles of the L1 capture beta within each date (low / mid / high)."""
    r = df.groupby("date").beta_l1_252.rank(pct=True)
    return pd.Series(np.select([r <= 1 / 3, r <= 2 / 3, r > 2 / 3], ["low", "mid", "high"], default=None), index=df.index)


def conditions(df: pd.DataFrame, feats: list) -> dict:
    """TOP / BOT quintile conditions per date, tie-safe: a tied block resolves the same way for every name in it.
    BOT = fewer than 20% of the date's names sit strictly below the value; TOP = fewer than 20% sit strictly above it."""
    out = {}
    g = df.groupby("date")
    for f in feats:
        n = g[f].transform("count")
        below = g[f].rank(method="min") - 1
        above = n - g[f].rank(method="max")
        out[f"{f}:BOT"] = ((below / n) < 0.2).fillna(False).values
        out[f"{f}:TOP"] = ((above / n) < 0.2).fillna(False).values
    for name, (col, op, v) in FIXED.items():
        x = df[col].values
        out[name] = np.nan_to_num((x > v) if op == ">" else (x < v), nan=0).astype(bool)
    return out


PLAIN = {
    "roc": "price change", "xs_spy": "excess vs SPY", "xs_sec": "excess vs sector", "off_high": "closeness to high", "off_low": "gain off 52w low",
    "days_since_high": "sessions since high", "mdd": "drawdown shallowness", "mean_dd": "under-water shallowness", "eff": "path efficiency", "up_share": "share of up days",
    "up_pair": "rising pairs", "new_high63_share": "new 63d highs", "new_high252_count": "new 52w highs", "gain_pain": "gain-to-pain", "higher_lows": "higher lows",
    "close_loc": "close location", "gap_share": "overnight share", "mad": "typical daily move", "vol_roc": "typical-move change", "vol_pctile_own": "typical move vs own year",
    "range_comp": "range compression", "range_roc": "range change", "down_up_move": "down/up move size", "worst_day": "worst-day mildness", "volume_roc": "volume change",
    "volume_pctile_own": "volume vs own year", "up_volume_share": "up-volume share", "pullback_volume": "pullback volume", "accum_minus_dist": "accumulation minus distribution",
    "log_dollar_vol": "dollar volume", "up_capture": "up-capture", "down_capture": "down-capture", "beta_l1": "L1 beta", "capture_spread": "capture spread",
    "defend_share": "up on SPY down days", "rs_off_high": "RS line vs its high",
}


def plain(cond: str) -> str:
    if cond in FIXED: return cond
    f, side = cond.rsplit(":", 1)
    own = f.endswith("_pctile_own"); base = f[:-len("_pctile_own")] if own else f
    stem = next((k for k in sorted(PLAIN, key=len, reverse=True) if base.startswith(k)), base)
    suffix = base[len(stem):].strip("_").replace("_", "/")
    name = PLAIN.get(stem, stem) + (f" {suffix}d" if suffix else "")
    return f"{name}{' vs own year' if own else ''} {'top' if side == 'TOP' else 'bottom'} 20%"
