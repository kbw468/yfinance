"""Shared definitions: target labels, feature lists, eras, distribution-free stats."""
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.environ.get("FS_DATA", os.path.join(HERE, "data"))
RES_DIR = os.path.join(HERE, "results")
os.makedirs(RES_DIR, exist_ok=True)

ID = ["date", "ticker"]
STATIC = ["sector", "industry", "mcap", "mcap_bucket", "bars"]
LABEL_PREFIX = ("fr", "fdd", "fae", "fer", "fmar")
ERAS = [("2006-01-01", "2012-12-31"), ("2013-01-01", "2019-12-31"), ("2020-01-01", "2026-12-31")]

# ---------------------------------------------------------------- targets
# Staircase = meaningful forward gain whose worst peak-to-trough along the way is
# small relative to that gain (gain-to-pain >= GP). Thresholds are log returns.
TARGETS = {
    "y21": dict(h=21, min_gain=np.log(1.05), gp=3.0),
    "y42": dict(h=42, min_gain=np.log(1.075), gp=3.0),
    "y63": dict(h=63, min_gain=np.log(1.10), gp=3.0),
}


def add_targets(df, targets=TARGETS):
    for name, t in targets.items():
        h = t["h"]
        fr, fdd = df[f"fr{h}"], df[f"fdd{h}"]
        y = (fr >= t["min_gain"]) & (fdd <= fr / t["gp"])
        df[name] = y.astype("float32").where(fr.notna())
    # holds the full 21-63 window: clean first month AND clean quarter
    df["y_hold"] = ((df["y21"] == 1) & (df["y63"] == 1)).astype("float32").where(df["fr63"].notna())
    return df


def is_label(c):
    return (any(c.startswith(p) and c[len(p):].isdigit() for p in LABEL_PREFIX)
            or c.startswith("y") and c[1:].replace("_hold", "").isdigit() or c == "y_hold")


def feature_cols(df, include_ranks=True):
    out = []
    for c in df.columns:
        if c in ID or c in STATIC or is_label(c) or c.startswith("y"):
            continue
        if not include_ranks and c.startswith(("q_", "sq_", "iq_")):
            continue
        if df[c].dtype.kind not in "fiu":
            continue
        out.append(c)
    return out


def load_panel(columns=None):
    df = pd.read_parquet(os.path.join(DATA_DIR, "panel.parquet"), columns=columns)
    return add_targets(df) if columns is None or "fr63" in df.columns else df


# ---------------------------------------------------------------- families
EXPLICIT = {
    # cross-family rate-of-change mixes
    **{k: "roc_mix" for k in ("pv_spread21", "pv_spread63", "pr_spread21", "vv_spread21", "vv_spread63",
                              "pvv21", "pvv63", "pv_ratio63")},
    # how the stock moves with the market
    **{k: "market_link" for k in ("co126", "rel_dn126", "rel_up126", "rel_dn126_mar")},
    # rate of change of ratios / ranks
    "d_vc21_126": "volatility", "d_rc21_252": "volatility", "dq_mar21": "volatility", "vov63": "volatility",
    "tail63": "volatility", "age_quiet": "volatility", "age_comp": "volatility",
    "d_rv10_126": "volume", "d_udv63": "volume", "dq_vlroc21": "volume",
    "d_dn63": "price_roc", "dq_roc63": "price_roc", "dq_rs63": "price_roc", "rs63": "price_roc",
    "rs_roc21": "price_roc", "roc_align": "price_roc", "age_hot": "price_roc",
    "d_mdd63_rel": "drawdown", "d_gp63": "drawdown", "dq_mdd63": "drawdown", "med_uw63": "drawdown",
    "med_uw63_mar": "drawdown", "hi_frac63": "drawdown", "worst63": "drawdown", "best63": "drawdown",
    "jump_asym63": "drawdown",
    "d_er63": "geometry", "d_kt63": "geometry", "dq_er63": "geometry", "dq_kt63": "geometry",
    "kt_min": "geometry", "er_min3": "geometry", "age_trend": "geometry", "age_stair": "geometry",
}
FAMILY_RULES = [
    ("memory", ("mem_",)),
    ("geometry", ("stair_", "bo_", "tmpl", "er", "kt", "hexp", "hl_", "hh_", "nh_", "upfrac", "clv", "ovn",
                  "ses", "gapfreq", "rng", "base_tight", "logp")),
    ("price_roc", ("roc", "acc", "dn")),
    ("volatility", ("mar", "mrg", "vc", "rc", "vroc", "rroc", "vacc", "racc", "asym")),
    ("volume", ("rv", "vlroc", "vlacc", "udv", "ldv", "vnoise")),
    ("drawdown", ("ddh", "mdd", "gp")),
]


def family(col):
    base = col[2:] if col.startswith("q_") else col
    if base in EXPLICIT:
        return EXPLICIT[base]
    if base.startswith(("sq_", "iq_", "sec_", "ind_", "rel_ind", "rel_sec")) or base == "sector_code":
        return "group"
    if base.startswith(("mkt_", "p_mkt_")):
        return "regime"
    for fam, prefixes in FAMILY_RULES:
        if base.startswith(prefixes):
            return fam
    return "other"


# ---------------------------------------------------------------- stats
def add_week_base(df, target):
    """Each row's same-week universe hit rate. Summing it over any subset gives the
    hits that subset would have scored by luck in those same weeks, so
    hits / expected is a lift with the market regime removed."""
    df["_wb"] = df.groupby("date")[target].transform("mean")
    return df


def adj_lift(y, wb):
    e = np.nansum(wb)
    return np.nansum(y) / e if e > 0 else np.nan


def beta_lower(hits, n, q=0.10):
    """Lower quantile of the Jeffreys Beta posterior for a hit rate (no normal approx)."""
    from scipy.stats import beta
    hits = np.asarray(hits, float)
    n = np.asarray(n, float)
    return beta.ppf(q, hits + 0.5, n - hits + 0.5)


def block_bootstrap_rate(dates, y, n_boot=500, block="QS", seed=0):
    """Percentile CI of a hit rate, resampling whole calendar blocks (quarters)."""
    rng = np.random.default_rng(seed)
    d = pd.Series(y.astype(float), index=pd.DatetimeIndex(dates))
    g = d.groupby(d.index.to_period(block[0]))
    s = g.sum().to_numpy()
    c = g.count().to_numpy()
    k = len(s)
    idx = rng.integers(0, k, size=(n_boot, k))
    rates = s[idx].sum(1) / np.maximum(c[idx].sum(1), 1)
    return np.percentile(rates, [5, 50, 95])


MODEL_TAGS = {"binary_xs_y63": "Stock-level inputs only", "binary_y63": "All inputs incl. market regime", "rank": "LambdaRank, all inputs"}


def best_model_tag():
    """Model with the best out-of-sample top-10 lift on the 63-session hit."""
    best, best_lift = None, -1
    for tag in MODEL_TAGS:
        p = os.path.join(RES_DIR, f"model_profile_{tag}.csv")
        if not os.path.exists(p):
            continue
        prof = pd.read_csv(p, index_col=0)
        lift = prof.loc["top10", "p_y63"] / prof.loc["all", "p_y63"]
        if lift > best_lift and os.path.exists(os.path.join(DATA_DIR, f"latest_{tag}.parquet")):
            best, best_lift = tag, lift
    return best
