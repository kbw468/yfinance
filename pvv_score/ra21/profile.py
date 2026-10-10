"""Fingerprint of what precedes a superior 21-session trade, out of sample.
(1) The walk-forward model's top decile vs the universe: median same-day percentile of every feature (0.5 = universe middle).
(2) Actual superior trades vs the rest: the same medians (descriptive; what winners looked like the day before entry).
(3) For the strongest features: superior rate by same-day quintile, 2017-2023 and 2024 onward (shape check, no fitting)."""
import numpy as np
import pandas as pd
from .common import load, feature_sets, D, OUT
from .plain import name


def main(top_k=30):
    fs = feature_sets(); feats = fs["stock"]
    T = load(feats + ["sup_21"], start="2017-01-01")
    O = pd.read_parquet(D / "oos.parquet")
    T = T.merge(O[["date", "ticker", "score"]], on=["date", "ticker"], how="left")
    g = T.groupby("date")
    R = g[feats].rank(pct=True)
    top = (g.score.rank(pct=True) >= 0.9) & T.score.notna()
    prof = pd.DataFrame({"model top decile": R[top].median(), "superior trades": R[T.sup_21 == 1].median(), "other trades": R[T.sup_21 == 0].median()})
    prof["winners minus others"] = prof["superior trades"] - prof["other trades"]
    prof["plain"] = [name(f) for f in prof.index]
    prof = prof.reindex(prof["model top decile"].sub(0.5).abs().sort_values(ascending=False).index)
    prof.round(3).to_csv(OUT / "profile_top_decile.csv")
    imp = pd.read_csv(OUT / "model_importance.csv", index_col=0).iloc[:, 0]
    strongest = [f[2:] for f in imp.index if f.startswith("r_")][:top_k]
    rows = []
    for f in strongest:
        q = (R[f] * 5).clip(upper=4.999).fillna(-1).astype(int) + 1
        for lab, m in (("2017-2023", T.date < "2024-01-01"), ("2024 onward", T.date >= "2024-01-01")):
            r = T[m & (q > 0)].groupby(q[m & (q > 0)]).sup_21.mean()
            rows.append({"feature": f, "plain": name(f), "window": lab, **{f"q{int(k)}": v for k, v in r.items()}})
    Q = pd.DataFrame(rows)
    Q.round(3).to_csv(OUT / "profile_quintile_shapes.csv", index=False)
    print(prof.head(40).round(3).to_string()); print(Q.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
