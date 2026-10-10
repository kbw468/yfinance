"""Calibration curves, binned isotonic (>= 500 cases per step), no distributional assumption.
Composite: per beta bucket, out-of-sample avg_score (walk-forward, 2017+) -> P(smooth_h) for h in 21/42/63.
Signatures: per beta bucket x composite quintile, confirmation window (2021+), names firing >= 1 confirmed signature:
log(1 + lift sum) and best confirmed probability -> P(smooth_h). A cell with fewer than MIN_GROUP_N rows uses the bucket's
curve across all quintiles; a bucket with too few uses the all-bucket curve. Rate-sensitive sectors on today's MOVE band only."""
import json, time
import numpy as np, pandas as pd
from ..calib import BinnedIsotonic
from ..regime import tag, today_regime, same_regime
from .common import conditions, COMP_Q, MIN_GROUP_N, CONFIRM_START, HORIZONS, D
from .signatures import load_rows


def firing(T, feats, conf):
    C = conditions(T, feats); n = len(T)
    cnt = np.zeros(n, np.int32); ls = np.zeros(n); bp = np.zeros(n)
    for s, l, p in zip(conf.signature, conf.lift_conf, conf.p_conf):
        m = np.ones(n, bool)
        for leg in s.split(" & "): m &= C[leg]
        cnt += m; ls += m * (l - 1.0); bp = np.maximum(bp, m * p)
    return cnt, ls, bp


def main():
    t0 = time.time()
    T, feats = load_rows()
    O = pd.read_parquet(D / "ng_composite_oos.parquet", columns=["date", "ticker", "bucket", "avg_score"])
    T = T.merge(O, on=["date", "ticker"], how="left")
    T = tag(T); reg = today_regime(); same = same_regime(T, reg)
    conf = pd.read_csv(D / "ng_signatures_all.csv"); conf = conf[conf.confirmed].reset_index(drop=True)
    T["n_fire"], T["liftsum"], T["bestp"] = firing(T, feats, conf)
    T["comp_q"] = pd.cut(T.avg_score, COMP_Q, labels=range(len(COMP_Q) - 1), include_lowest=True)
    print(f"firing computed {time.time()-t0:.0f}s", flush=True)
    comp = {}; sig = {}
    for b in ("low", "mid", "high"):
        comp[b] = {}
        for h in HORIZONS:
            m = same & (T.bucket == b).values & (T.date >= "2017-01-01").values & T.avg_score.notna().values & T[f"smooth_{h}"].notna().values
            comp[b][f"iso_q{h}"] = BinnedIsotonic(500).fit(T.avg_score.values[m], T[f"smooth_{h}"].values[m]).to_dict()
    for h in HORIZONS:
        for kind, x in (("ls", np.log1p(T.liftsum.values)), ("bp", T.bestp.values)):
            c = f"{kind}_iso_q{h}"; sig[c] = {}
            base = same & (T.date >= CONFIRM_START).values & (T.n_fire > 0).values & T[f"smooth_{h}"].notna().values
            sig[c]["all|all"] = BinnedIsotonic(500).fit(x[base], T[f"smooth_{h}"].values[base]).to_dict()
            for b in ("low", "mid", "high"):
                mb = base & (T.bucket == b).values
                bucket_curve = BinnedIsotonic(500).fit(x[mb], T[f"smooth_{h}"].values[mb]).to_dict() if mb.sum() >= MIN_GROUP_N else sig[c]["all|all"]
                sig[c][f"{b}|all"] = bucket_curve
                for q in range(len(COMP_Q) - 1):
                    mq = mb & (T.comp_q == q).values
                    sig[c][f"{b}|{q}"] = BinnedIsotonic(500).fit(x[mq], T[f"smooth_{h}"].values[mq]).to_dict() if mq.sum() >= MIN_GROUP_N else bucket_curve
    json.dump(comp, open(D / "ng_calib_composite.json", "w")); json.dump({"comp_q_edges": COMP_Q, "min_group_n": MIN_GROUP_N, "regime": reg, "curves": sig}, open(D / "ng_calib_signature.json", "w"))
    # transparency: realised rate by bucket for names firing, confirmation window
    F = T[(T.date >= CONFIRM_START) & same]
    t = F.groupby(["bucket", pd.cut(F.n_fire, [-1, 0, 5, 20, 50, 10**6], labels=["0", "1-5", "6-20", "21-50", "50+"])], observed=True)[["smooth_21", "smooth_42", "smooth_63"]].mean()
    print(t.round(3).to_string())
    T[["date", "ticker", "bucket", "avg_score", "n_fire", "liftsum", "bestp"]].to_parquet(D / "ng_firing.parquet")
    print(f"done {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
