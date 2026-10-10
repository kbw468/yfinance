"""Shared loaders and the per-date AUC (Mann-Whitney on within-date ranks; no distributional assumption)."""
import numpy as np
import pandas as pd
from ..config import CACHE_DIR, RESULTS_DIR

D = CACHE_DIR / "ra21"
OUT = RESULTS_DIR / "ra21"
COVID = ("2020-02-20", "2020-06-30")
EXTRA_FILES = ["extra2", "extra3"]      # side files aligned row-for-row to table.parquet (outcome persistence; short interest)
MARKET = ["mkt_off_high_252", "mkt_roc_21", "mkt_roc_63", "mkt_mad_pctile", "mkt_vix", "mkt_vix_pctile", "mkt_move_pctile",
          "mkt_breadth_63h", "mkt_breadth_up21", "mkt_disp_iqr_21"]


def feature_sets() -> dict:
    base = pd.read_csv(D / "base_features.csv").iloc[:, 0].tolist()
    extra = pd.read_csv(D / "extra_features.csv").iloc[:, 0].tolist()
    for k in EXTRA_FILES:
        if (D / f"{k}_features.csv").exists():
            extra += pd.read_csv(D / f"{k}_features.csv").iloc[:, 0].tolist()
    stock = [f for f in base + extra if f not in MARKET]
    return {"stock": stock, "market": [f for f in extra if f in MARKET], "base": base, "extra": [f for f in extra if f not in MARKET]}


def load(cols: list, start="2015-01-01") -> pd.DataFrame:
    ex = {k: pd.read_csv(D / f"{k}_features.csv").iloc[:, 0].tolist() for k in EXTRA_FILES if (D / f"{k}_features.csv").exists()}
    side = {c for v in ex.values() for c in v}
    T = pd.read_parquet(D / "table.parquet", columns=list(dict.fromkeys(["date", "ticker"] + [c for c in cols if c not in side])))
    for k, v in ex.items():
        want = [c for c in cols if c in v]
        if want: T = pd.concat([T, pd.read_parquet(D / f"{k}.parquet", columns=want)], axis=1)
    T = T[(T.date >= start) & ~((T.date >= COVID[0]) & (T.date <= COVID[1]))]
    return T.sort_values(["date", "ticker"]).reset_index(drop=True)


def date_auc(score: np.ndarray, y: np.ndarray, groups: np.ndarray) -> pd.Series:
    """AUC per group (date, or date x bucket) of score for binary y; rows with NaN score or y are dropped.
    Ties get average ranks (Mann-Whitney). Returns a Series indexed by group code."""
    ok = ~np.isnan(score) & ~np.isnan(y)
    s, yy, g = score[ok], y[ok], groups[ok]
    r = pd.Series(s).groupby(g).rank(method="average").to_numpy()
    n = np.bincount(g, minlength=groups.max() + 1).astype(float)
    n1 = np.bincount(g, weights=yy, minlength=groups.max() + 1)
    s1 = np.bincount(g, weights=r * yy, minlength=groups.max() + 1)
    n0 = n - n1
    with np.errstate(invalid="ignore", divide="ignore"):
        auc = (s1 - n1 * (n1 + 1) / 2) / (n1 * n0)
    auc[(n1 < 5) | (n0 < 5)] = np.nan
    return pd.Series(auc)


def mean_auc_by_date(score, y, date_codes) -> pd.Series:
    return date_auc(np.asarray(score, float), np.asarray(y, float), date_codes)
