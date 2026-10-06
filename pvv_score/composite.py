"""Stage 3b: transparent linear composite, estimated walk-forward (no look-ahead in factor selection).

For each test year Y:
  * training = eligible rows with date <= (first day of Y) - 63 trading days (embargo)
  * per factor, IC vs 42d excess Sharpe vs SPY on the training window; blended estimate
        IC_blend = RECENT_WEIGHT * mean(IC, last 3y of training) + (1-RECENT_WEIGHT) * mean(IC, all training)
  * select: |NW t| >= T_MIN on the 3y window AND same sign over the full training window
  * de-duplicate: greedy by |t|, drop any factor with |rank corr| > RHO_MAX to an already selected one
  * weight = sign(IC_blend) * |IC_blend|, normalised to sum |w| = 1
  * score = sum_f w_f * pct-rank_f(date)
Two variants: POOLED (one weight vector) and BETA-BUCKETED (separate weights inside low/mid/high
trailing-beta terciles, ranks computed inside the bucket).
"""
import sys
import numpy as np
import pandas as pd

from .config import CACHE_DIR, RESULTS_DIR, RECENT_WEIGHT, RECENT_START
from .evaluate import daily_ic, summarize_ic, nw_tstat
from .run_eval import load_research, load_registry
from .model import evaluate_prediction, decile_spread

FACTORS = {k: v for k, v in load_registry().items() if v[0] != "roc"}
EMBARGO = 63
T_MIN = 1.5
RHO_MAX = 0.7
RECENT_DAYS = 756
TARGET = "xs_spy_sharpe_42"
H = 42


def tercile_series(df):
    return df.groupby("date")["beta_252"].transform(
        lambda s: pd.qcut(s.rank(method="first"), 3, labels=["low", "mid", "high"]) if s.notna().sum() >= 60 else pd.Series(index=s.index, dtype=object))


def fit_weights(train: pd.DataFrame, factors: list, corr: pd.DataFrame) -> pd.Series:
    ic = daily_ic(train, factors, TARGET)
    if len(ic) < 252:
        return pd.Series(dtype=float)
    recent_start = ic.index[-min(RECENT_DAYS, len(ic))]
    full = summarize_ic(ic, H)
    rec = summarize_ic(ic, H, recent_start)
    blend = RECENT_WEIGHT * rec["ic"] + (1 - RECENT_WEIGHT) * full["ic"]
    ok = (rec["t_nw"].abs() >= T_MIN) & (np.sign(rec["ic"]) == np.sign(full["ic"]))
    cand = rec.loc[ok, "t_nw"].abs().sort_values(ascending=False).index.tolist()
    chosen = []
    for f in cand:
        if all(abs(corr.loc[f, g]) <= RHO_MAX for g in chosen):
            chosen.append(f)
    if not chosen:
        return pd.Series(dtype=float)
    w = blend[chosen]
    return w / w.abs().sum()


def score_rows(sub: pd.DataFrame, w: pd.Series) -> pd.Series:
    if w.empty:
        return pd.Series(np.nan, index=sub.index)
    rk = sub.groupby("date")[list(w.index)].rank(pct=True)
    return (rk * w.values).sum(axis=1, min_count=1)


def walk_forward(df: pd.DataFrame, years=range(2017, 2027)) -> tuple[pd.Series, pd.Series, dict]:
    factors = list(FACTORS)
    bdays = pd.DatetimeIndex(sorted(df.date.unique()))
    pos = pd.Series(np.arange(len(bdays)), index=bdays)
    pooled = pd.Series(np.nan, index=df.index)
    bucketed = pd.Series(np.nan, index=df.index)
    log = {}
    for y in years:
        first = bdays[bdays >= pd.Timestamp(f"{y}-01-01")]
        if len(first) == 0:
            continue
        train_end = bdays[pos[first[0]] - EMBARGO]
        tr = df[(df.date <= train_end) & df[TARGET].notna()]
        te_mask = (df.date >= first[0]) & (df.date <= pd.Timestamp(f"{y}-12-31"))
        # rank correlations on a 1-in-10-day sample of the last 3y of training
        samp = tr[tr.date >= tr.date.max() - pd.Timedelta(days=3 * 365)]
        samp = samp[samp.date.isin(sorted(samp.date.unique())[::10])]
        corr = samp.groupby("date")[factors].rank(pct=True).corr()
        w_pool = fit_weights(tr, factors, corr)
        pooled[te_mask] = score_rows(df[te_mask], w_pool)
        log[(y, "pooled")] = w_pool
        for b in ["low", "mid", "high"]:
            trb = tr[tr.beta_bucket == b]
            w_b = fit_weights(trb, factors, corr)
            m = te_mask & (df.beta_bucket == b)
            bucketed[m] = score_rows(df[m], w_b)
            log[(y, b)] = w_b
        print(f"  {y}: pooled k={len(w_pool)} | " + " ".join(f"{b}:k={len(log[(y,b)])}" for b in ["low", "mid", "high"]), file=sys.stderr)
    return pooled, bucketed, log


SECTOR_GROUP = {
    "Utilities": "defensive", "Consumer Defensive": "defensive", "Healthcare": "defensive", "Real Estate": "defensive",
    "Industrials": "cyclical", "Financial": "cyclical", "Basic Materials": "cyclical", "Energy": "cyclical", "Consumer Cyclical": "cyclical",
    "Technology": "growth", "Communication Services": "growth",
}


def main(state: bool = False, sector: bool = False):
    """sector=True: target is Sharpe vs the name's own sector ETF, factor ranks are taken inside (date, sector),
    and weights are fitted per sector group (defensive / cyclical / growth) instead of per beta tercile."""
    global FACTORS, TARGET
    df = load_research().reset_index(drop=True)
    tag = "state" if state else "level"
    if state:
        FACTORS = {c: ("state", 0, c) for c in df.columns if c.startswith("z_") and c[2:] in load_registry() and load_registry()[c[2:]][0] != "roc"}
    if sector:
        tag += "_sector"
        TARGET = "xs_sec_sharpe_42"
        # within-sector percentile ranks replace raw factor values so the composite compares a name to its sector peers
        fcols = list(FACTORS)
        df[fcols] = df.groupby(["date", "sector"])[fcols].rank(pct=True).astype("float32")
        df["beta_bucket"] = df["sector"].map(SECTOR_GROUP).map({"defensive": "low", "cyclical": "mid", "growth": "high"})
    else:
        df["beta_bucket"] = tercile_series(df)
    pooled, bucketed, log = walk_forward(df)
    df["comp_pooled"] = pooled
    df["comp_bucketed"] = bucketed
    oos = df[df.comp_pooled.notna()].copy()
    oos[["date", "ticker", "comp_pooled", "comp_bucketed"]].to_parquet(CACHE_DIR / f"composite_{tag}_oos_preds.parquet", index=False)

    ev = pd.concat([evaluate_prediction(oos, "comp_pooled", f"composite_pooled_wf_{tag}"),
                    evaluate_prediction(oos[oos.comp_bucketed.notna()], "comp_bucketed", f"composite_betabucket_wf_{tag}")])
    ev.to_csv(RESULTS_DIR / f"composite_{tag}_oos_evaluation.csv")
    rows = []
    for (y, b), w in log.items():
        for f, v in w.items():
            rows.append({"year": y, "bucket": b, "factor": f, "weight": v})
    pd.DataFrame(rows).to_csv(RESULTS_DIR / f"composite_{tag}_weights_by_fold.csv", index=False)

    pd.set_option("display.width", 250, "display.max_rows", 200)
    print(ev.round(3).to_string())
    ic42 = daily_ic(oos, ["comp_pooled", "comp_bucketed"], "xs_spy_sharpe_42")
    print("\nby year (42d IC):\n", ic42.groupby(ic42.index.year).mean().round(3).to_string())
    rec = oos[oos.date >= RECENT_START]
    for c in ["comp_pooled", "comp_bucketed"]:
        sp, dec = decile_spread(rec[rec[c].notna()], c, "xs_spy_sharpe_42")
        print(f"\n{c} decile means recent 42d:", dec.mean().round(2).tolist(), " D10-D1:", round(sp.mean(), 3))


if __name__ == "__main__":
    main(state="--state" in sys.argv, sector="--sector" in sys.argv)
