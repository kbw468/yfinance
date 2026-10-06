"""Pre-episode event study (identity-neutral).

Episode: a (date, ticker) whose forward blended SPY-excess-Sharpe percentile (21/42/63d average) is >= 0.90,
i.e. the name went on to top-decile risk-adjusted performance over the next 1-3 months. Consecutive episode
days for one ticker are one cluster; the cluster's first day is the episode START, and the state on that day
(all data <= start) is "what preceded the move".

For every state factor z_f (deviation from the name's own 3y norm), ranked cross-sectionally per date:
  * mean rank PRE_LAG sessions before the episode start minus 0.5 (baseline), cluster-robust t-stat
    (the start day itself is selection-contaminated: it is mechanically a local low, z_ret_5 t = -59 at t-0 and
     flat at t-5, so t-0 is not used)
  * lift = P(episode | rank in top quintile) / P(episode) and the same for the bottom quintile
  * the rank path at 20, 10, 5, 0 sessions before the start (sequence: e.g. compression then expansion)
Pooled, by window (full / recent) and by trailing-beta tercile.
"""
import sys
import numpy as np
import pandas as pd

from .config import CACHE_DIR, RESULTS_DIR, RECENT_START
from .run_eval import load_research
from .composite import tercile_series

EPISODE_Q = 0.90
LAGS = (20, 10, 5, 0)
PRE_LAG = 5  # sessions before the cluster start at which the 'preceding state' is measured (t-0 is selection-contaminated)


def cluster_t(x: pd.Series, cl: pd.Series) -> tuple[float, float, int]:
    m = x.groupby(cl).mean()
    n = len(m)
    return x.mean(), x.mean() / (m.std(ddof=1) / np.sqrt(n)) if n > 2 else np.nan, n


def main():
    df = load_research().reset_index(drop=True).sort_values(["ticker", "date"])
    zcols = [c for c in df.columns if c.startswith("z_")]
    df["beta_bucket"] = tercile_series(df)
    df["episode"] = (df["y_blend_spy"] >= EPISODE_Q)
    # cluster ids: runs of episode days per ticker (gap <= 5 sessions joins)
    pos = df.groupby("date").ngroup()
    df["pos"] = pos
    ep = df[df.episode].copy()
    gap = ep.groupby("ticker")["pos"].diff()
    ep["cluster"] = ((gap.isna()) | (gap > 5)).cumsum()
    starts = ep.groupby("cluster").head(1)
    print(f"episodes: {len(ep):,} days in {len(starts):,} clusters; recent clusters: {(starts.date >= RECENT_START).sum():,}", file=sys.stderr)

    # per-date cross-sectional ranks of state factors
    R = df.groupby("date")[zcols].rank(pct=True)
    R.columns = [c for c in zcols]
    R["date"] = df.date.values; R["ticker"] = df.ticker.values; R["episode"] = df.episode.values
    R["beta_bucket"] = df.beta_bucket.values; R["pos"] = df.pos.values
    base = R.episode.mean()

    rows = []
    for win, start in [("full", None), ("recent", RECENT_START)]:
        Rw = R if start is None else R[R.date >= start]
        Sw = starts if start is None else starts[starts.date >= start]
        key = Rw.set_index(["ticker", "pos"])
        for bucket in ["all", "low", "mid", "high"]:
            Rb = Rw if bucket == "all" else Rw[Rw.beta_bucket == bucket]
            Sb = Sw if bucket == "all" else Sw[Sw.beta_bucket == bucket]
            if len(Sb) < 30:
                continue
            idx = pd.MultiIndex.from_arrays([Sb.ticker, Sb.pos - PRE_LAG])
            at_start = key.reindex(idx)
            at_start.index = pd.MultiIndex.from_arrays([Sb.date, Sb.ticker])
            pb = Rb.episode.mean()
            for f in zcols:
                x = at_start[f].dropna()
                cl = Sb.set_index(["date", "ticker"]).loc[x.index, "cluster"]
                m, t, n = cluster_t(x - 0.5, cl)
                top = Rb[Rb[f] >= 0.8].episode.mean() / pb if pb > 0 else np.nan
                bot = Rb[Rb[f] <= 0.2].episode.mean() / pb if pb > 0 else np.nan
                rows.append({"window": win, "bucket": bucket, "factor": f, "mean_rank_excess": m, "t_cluster": t,
                             "n_clusters": n, "lift_top_q": top, "lift_bottom_q": bot})
    prof = pd.DataFrame(rows)
    prof.to_csv(RESULTS_DIR / "event_pre_episode_profile.csv", index=False)

    # rank path before episode start (pooled, recent window): sequence information
    Sw = starts[starts.date >= RECENT_START]
    key = R.set_index(["ticker", "pos"])
    path = {}
    for lag in LAGS:
        idx = pd.MultiIndex.from_arrays([Sw.ticker, Sw.pos - lag])
        path[lag] = key.reindex(idx)[zcols].mean() - 0.5
    path = pd.DataFrame(path)
    path.columns = [f"t-{l}" for l in LAGS]
    path.to_csv(RESULTS_DIR / "event_pre_episode_path_recent.csv")

    pd.set_option("display.width", 250, "display.max_rows", 120)
    v = prof[(prof.window == "recent") & (prof.bucket == "all")].set_index("factor").sort_values("t_cluster", key=np.abs, ascending=False)
    print(f"\nRecent window, pooled: state factors {PRE_LAG} sessions BEFORE episode start (rank excess over 0.5; cluster t; lifts)")
    print(v[["mean_rank_excess", "t_cluster", "n_clusters", "lift_top_q", "lift_bottom_q"]].head(30).round(3).to_string())
    for b in ["low", "mid", "high"]:
        v = prof[(prof.window == "recent") & (prof.bucket == b)].set_index("factor").sort_values("t_cluster", key=np.abs, ascending=False)
        print(f"\nrecent, beta {b}: top 12")
        print(v[["mean_rank_excess", "t_cluster", "n_clusters", "lift_top_q", "lift_bottom_q"]].head(12).round(3).to_string())
    print("\nRank path before episode start (recent, pooled), top 15 by |t-0|:")
    print(path.reindex(path["t-0"].abs().sort_values(ascending=False).index).head(15).round(3).to_string())


if __name__ == "__main__":
    main()
