"""Walk-forward composite, two layers (level = same-day features, state = own-history percentiles), per L1-beta tercile.
Weights = mean per-date rank separation: the feature's average within-bucket percentile among target hits minus its average
among non-hits (equivalent to AUC - 0.5, no variance or correlation terms), over the training window; a feature is kept only if its yearly mean has the same sign in at least 70% of training years and the overall
mean is at least 0.01 in size. No t-statistics. Score = sum of weight x (within-bucket rank - 0.5). avg_score = mean of the
two layers' universe percentiles on the date. Test years 2017..2026, training ends 100 calendar days before the test year."""
import time
import numpy as np, pandas as pd
from .common import layers, beta_bucket, COVID, D

TARGET = "smooth_42"
EMBARGO_DAYS = 100
TEST_YEARS = range(2017, 2027)


def daily_ic(T: pd.DataFrame, feats: list, bucket: str) -> pd.DataFrame:
    """Per-date rank separation of each feature inside one bucket: mean percentile among hits minus mean among non-hits."""
    X = T[T.bucket == bucket]
    X = X[X[TARGET].notna()]
    R = X.groupby("date")[feats].rank(pct=True)
    hit = X[TARGET] == 1
    a = R[hit].groupby(X.date[hit]).mean(); b = R[~hit].groupby(X.date[~hit]).mean()
    return (a - b).dropna(how="all")


def fit_weights(ic: pd.DataFrame, end) -> pd.Series:
    w = ic[ic.index < end]
    if len(w) == 0: return pd.Series(dtype=float)
    m = w.mean(); yearly = w.groupby(w.index.year).mean()
    same = (np.sign(yearly) == np.sign(m)).mean()
    keep = (same >= 0.7) & (m.abs() >= 0.01)
    return m[keep]


def score(df: pd.DataFrame, weights: dict) -> pd.Series:
    """weights: {bucket: Series}; within-bucket per-date ranks x weights."""
    out = pd.Series(np.nan, index=df.index)
    for b, w in weights.items():
        if len(w) == 0: continue
        m = df.bucket == b
        R = df.loc[m].groupby("date")[list(w.index)].rank(pct=True) - 0.5
        out[m] = (R * w.values).sum(axis=1)
    return out


def main():
    t0 = time.time()
    L = layers()
    cols = ["date", "ticker", "sector", "beta_l1_252", TARGET, "smooth_21", "smooth_63"] + L["state"] + L["level"]
    T = pd.read_parquet(D / "table.parquet", columns=cols)
    T = T[~((T.date >= COVID[0]) & (T.date <= COVID[1]))].reset_index(drop=True)
    T["bucket"] = beta_bucket(T)
    print(f"rows {len(T):,} state {len(L['state'])} level {len(L['level'])} {time.time()-t0:.0f}s", flush=True)
    ics = {(lay, b): daily_ic(T, L[lay], b) for lay in L for b in ("low", "mid", "high")}
    print(f"daily ICs {time.time()-t0:.0f}s", flush=True)
    for lay in L:
        T[f"comp_{lay}"] = np.nan
        for Y in TEST_YEARS:
            end = pd.Timestamp(f"{Y}-01-01") - pd.Timedelta(days=EMBARGO_DAYS)
            W = {b: fit_weights(ics[(lay, b)], end) for b in ("low", "mid", "high")}
            te = (T.date >= f"{Y}-01-01") & (T.date <= f"{Y}-12-31")
            T.loc[te, f"comp_{lay}"] = score(T.loc[te], W)
    T["avg_score"] = ((T.groupby("date").comp_state.rank(pct=True) + T.groupby("date").comp_level.rank(pct=True)) / 2).round(8)
    last = T.date.max() - pd.Timedelta(days=EMBARGO_DAYS)
    final = {lay: {b: fit_weights(ics[(lay, b)], last + pd.Timedelta(days=1)) for b in ("low", "mid", "high")} for lay in L}
    T[["date", "ticker", "sector", "bucket", "beta_l1_252", "comp_state", "comp_level", "avg_score", "smooth_21", "smooth_42", "smooth_63"]].to_parquet(D / "ng_composite_oos.parquet")
    pd.concat({f"{lay}|{b}": w for lay, d in final.items() for b, w in d.items()}, axis=1).to_csv(D / "ng_weights.csv")
    for (lay, b), ic in ics.items(): ic.to_parquet(D / f"ng_ic_{lay}_{b}.parquet")
    # quick read: OOS within-date AUC of avg_score for each horizon
    from sklearn.metrics import roc_auc_score
    X = T[T.date >= "2017-01-01"]
    for h in (21, 42, 63):
        a = [roc_auc_score(g[f"smooth_{h}"], g.avg_score) for _, g in X.dropna(subset=[f"smooth_{h}", "avg_score"]).groupby("date") if g[f"smooth_{h}"].nunique() == 2]
        print(f"composite OOS mean within-date AUC smooth_{h}: {np.mean(a):.3f}  (days {len(a)})", flush=True)
    print("final weights, features kept per layer x bucket:", {f"{lay}|{b}": len(w) for lay, d in final.items() for b, w in d.items()})
    print(f"done {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
