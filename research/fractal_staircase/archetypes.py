"""Archetypes from user-supplied case studies, tested across the whole universe.

Each case is reduced to the profile it showed before / during its run, then that
profile is scored like everything else in the study: regime-neutral lift on the
63-session staircase hit, split 2006-2016 vs 2017-2026, all names and $50B+.
A profile is only worth adding to the scan if it beats what the scan already uses.

  td_rebase     TD 2025-26: low-vol leader re-basing in a tight range near highs
  vlo_momentum  VLO 2026: mid/high-vol momentum at 52-week highs
  jnj_base      JNJ 2025: tight, very quiet base near highs, 6-month return upper half
  tgt_recovery  TGT 2026: V-recovery from a >=35% drawdown back to the 52-week high
"""
import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view

from common import DATA_DIR, RES_DIR, add_targets, add_week_base, adj_lift

ARCH = {
    "td_rebase": lambda d: (d["q_rng63"] <= 0.10) & (d["q_roc126"] >= 0.8) & (d["ddh252"] > -0.10) & (d["q_mar63"] <= 0.2),
    "vlo_momentum": lambda d: (d["q_roc126"] >= 0.9) & (d["ddh252"] > -0.05) & (d["q_mar63"] >= 0.4),
    "jnj_base": lambda d: (d["q_rng63"] <= 0.10) & (d["q_mar63"] <= 0.10) & (d["q_mdd126"] <= 0.15) & (d["ddh252"] > -0.10) & (d["q_roc126"] >= 0.5),
    "tgt_recovery": lambda d: (d["rec_depth"] >= np.log(1 / 0.65)) & (d["rec_age"] <= 126) & (d["rec_bounce"] >= np.log(1.35)) & (d["ddh252"] > -0.10),
    "quiet_base (reference)": lambda d: (d["q_mar63"] <= 0.2) & (d["q_mdd126"] <= 0.3) & (d["q_roc126"] <= 0.4) & (d["ddh252"] > -0.105),
}


def recovery_features(keys):
    """Depth of the deepest fall inside the last 378 sessions, sessions since its low, and the bounce off that low."""
    px = pd.read_parquet(f"{DATA_DIR}/ohlcv.parquet", columns=["date", "ticker", "close"])
    want = set(zip(keys["date"].to_numpy(), keys["ticker"].to_numpy()))
    out = []
    L = 378
    for t, g in px.groupby("ticker"):
        g = g.sort_values("date")
        x = np.log(g["close"].to_numpy(float))
        if len(x) < L:
            continue
        W = sliding_window_view(x, L)
        am = W.argmin(axis=1)
        low = W[np.arange(len(W)), am]
        prehigh = np.array([w[: i + 1].max() for w, i in zip(W, am)])
        dates = g["date"].to_numpy()[L - 1:]
        keep = np.array([(dd, t) in want for dd in dates])
        if keep.any():
            out.append(pd.DataFrame({"date": dates[keep], "ticker": t, "rec_depth": (prehigh - low)[keep],
                                     "rec_age": (L - 1 - am)[keep], "rec_bounce": (x[L - 1:] - low)[keep]}))
    return pd.concat(out, ignore_index=True)


def main():
    cols = ["date", "ticker", "mcap", "q_rng63", "q_roc126", "ddh252", "q_mar63", "q_mdd126", "fr21", "fdd21", "fr42", "fdd42", "fr63", "fdd63"]
    d = pd.read_parquet(f"{DATA_DIR}/panel.parquet", columns=cols)
    d = add_targets(d)
    d = d[(d["date"] >= "2006-01-01") & d["y63"].notna()].reset_index(drop=True)
    d = add_week_base(d, "y63")
    d = d.merge(recovery_features(d[["date", "ticker"]]), on=["date", "ticker"], how="left")
    big = (d["mcap"] >= 50000).to_numpy()
    first = (d["date"] <= "2016-12-31").to_numpy()
    rows = []
    for name, fn in ARCH.items():
        m = fn(d).fillna(False).to_numpy()
        for cohort, cm in (("all", np.ones(len(d), bool)), ("$50B+", big)):
            mm = m & cm
            s = d[mm]
            rows.append(dict(archetype=name, cohort=cohort, weeks=int(mm.sum()), per_week=mm.sum() / d["date"].nunique(),
                             hit=s["y63"].mean(), lift=adj_lift(s["y63"], s["_wb"]),
                             lift_2006_16=adj_lift(d.loc[mm & first, "y63"], d.loc[mm & first, "_wb"]),
                             lift_2017_26=adj_lift(d.loc[mm & ~first, "y63"], d.loc[mm & ~first, "_wb"]),
                             cohort_lift=adj_lift(d.loc[cm, "y63"], d.loc[cm, "_wb"]),
                             up63=(s["fr63"] > 0).mean(), med_fr63=s["fr63"].median(), med_dd63=s["fdd63"].median()))
    out = pd.DataFrame(rows)
    out.to_csv(f"{RES_DIR}/archetypes.csv", index=False)
    pd.set_option("display.width", 250)
    print(out.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
