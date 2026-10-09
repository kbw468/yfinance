"""Stage 4: how the rates of change combine (regime-neutral lifts).

(a) ROC triad cubes: price ROC x volatility ROC x volume ROC, each in per-week
    quintiles -> regime-neutral lift of all 125 cells, with era persistence.
(b) The same triads inside the low-volatility third (where drawdowns are smallest).
(c) Pair scan: every pair among the strongest single features (stage 1), 5x5
    quintile grid, best cell and its lift in each era.
"""
import itertools
import sys

import numpy as np
import pandas as pd

from common import DATA_DIR, ERAS, RES_DIR, add_week_base, adj_lift, load_panel

TRIADS = [
    ("q_roc63", "q_vroc21", "q_vlroc21"),
    ("q_roc21", "q_vroc21", "q_vlroc21"),
    ("q_roc63", "q_vroc63", "q_vlroc63"),
    ("q_acc21", "q_vacc21", "q_vlacc21"),
    ("q_roc63", "q_vc21_126", "q_rv10_126"),
    ("q_roc126", "q_rroc21", "q_rv10_126"),
    ("q_dq_roc63", "q_dq_mar21", "q_dq_vlroc21"),
    ("q_roc21", "q_d_vc21_126", "q_d_rv10_126"),
]


def quint(s):
    return np.floor(np.minimum(s * 5, 4.999))


def era_lifts(d, mask, target):
    out = []
    for a, b in ERAS:
        e = mask & (d["date"] >= a).to_numpy() & (d["date"] <= b).to_numpy()
        out.append(adj_lift(d.loc[e, target], d.loc[e, "_wb"]) if e.sum() >= 150 else np.nan)
    return out


def cube(df, t, target, label):
    d = df[["date", target, "_wb", *t]].dropna()
    qs = [quint(d[c]).astype(int) for c in t]
    key = qs[0] * 100 + qs[1] * 10 + qs[2]
    agg = pd.DataFrame({"k": key, "y": d[target], "wb": d["_wb"]}).groupby("k").agg(
        n=("y", "size"), hits=("y", "sum"), exp=("wb", "sum"))
    rows = []
    for k, r in agg.iterrows():
        if r["n"] < 600:
            continue
        a, b, c = k // 100, (k // 10) % 10, k % 10
        mask = ((qs[0] == a) & (qs[1] == b) & (qs[2] == c)).to_numpy()
        el = era_lifts(d, mask, target)
        rows.append(dict(subset=label, triad=" x ".join(t), cell=f"{t[0]}:Q{a + 1} {t[1]}:Q{b + 1} {t[2]}:Q{c + 1}",
                         n=int(r["n"]), hit=r["hits"] / r["n"], lift=r["hits"] / r["exp"], era_min=np.nanmin(el),
                         **{f"e{i}": v for i, v in enumerate(el)}))
    return rows


def main(target="y63", n_top=20):
    import pyarrow.parquet as pq
    names = pq.read_schema(f"{DATA_DIR}/panel.parquet").names
    uni = pd.read_csv(f"{RES_DIR}/univariate_{target}.csv")
    uni = uni[(uni["era_min"] > 1.02) & uni["feature"].map(lambda f: "q_" + f in names)]
    top = ["q_" + f for f in uni.sort_values("best_lift", ascending=False)["feature"].head(n_top)]
    need = ["date", "ticker", "fr21", "fdd21", "fr42", "fdd42", "fr63", "fdd63", "q_mar63"] + \
        [c for t in TRIADS for c in t] + top
    df = load_panel(columns=list(dict.fromkeys(need)))
    df = df[(df["date"] >= "2006-01-01") & df[target].notna()].reset_index(drop=True)
    df = add_week_base(df, target)

    rows = []
    for t in TRIADS:
        rows += cube(df, t, target, "all")
        rows += cube(df[df["q_mar63"] <= 1 / 3], t, target, "low_vol_third")
    tri = pd.DataFrame(rows).sort_values("lift", ascending=False)
    tri.to_csv(f"{RES_DIR}/triads_{target}.csv", index=False)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_colwidth", 100)
    for sub in ("all", "low_vol_third"):
        s = tri[(tri["subset"] == sub)]
        print(f"\n== triad cells, {sub}: top 15 by lift (era_min = worst era)")
        print(s.head(15).round(3).to_string(index=False))
        print(f"-- era-stable (era_min > 1.1) count: {(s['era_min'] > 1.1).sum()} of {len(s)}")

    pairs = []
    for f1, f2 in itertools.combinations(top, 2):
        d = df[["date", target, "_wb", f1, f2]].dropna()
        q1, q2 = quint(d[f1]).astype(int), quint(d[f2]).astype(int)
        agg = pd.DataFrame({"a": q1, "b": q2, "y": d[target], "wb": d["_wb"]}).groupby(["a", "b"]).agg(
            n=("y", "size"), hits=("y", "sum"), exp=("wb", "sum"))
        agg = agg[agg["n"] >= 2000]
        if agg.empty:
            continue
        agg["lift"] = agg["hits"] / agg["exp"]
        (a, b) = agg["lift"].idxmax()
        el = era_lifts(d, ((q1 == a) & (q2 == b)).to_numpy(), target)
        pairs.append(dict(f1=f1, q1=a + 1, f2=f2, q2=b + 1, n=int(agg.loc[(a, b), "n"]),
                          hit=agg.loc[(a, b), "hits"] / agg.loc[(a, b), "n"], lift=agg.loc[(a, b), "lift"],
                          era_min=np.nanmin(el)))
    pr = pd.DataFrame(pairs).sort_values("lift", ascending=False)
    pr.to_csv(f"{RES_DIR}/pairs_{target}.csv", index=False)
    print("\n== best pair cells")
    print(pr.head(25).round(3).to_string(index=False))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "y63")
