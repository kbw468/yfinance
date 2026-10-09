"""Prototype target: MSI 2024-grade staircases.

MSI's best 126-session stretch (2024-04-18 -> 2024-10-16): +41% with a 3.8% worst
pullback. Targets on the 126-session forward path:
  proto    = gain >= +25% and max drawdown <= 6%
  proto_x  = gain >= +30% and max drawdown <= 5%   (MSI-grade)
Same machinery as the study: weekly rows, regime-neutral lift (hits / sum of each
row's own-week universe hit rate), 2006-2016 vs 2017-2026 halves, all names and $50B+.

Stages: forward path stats -> base rates and MSI's rank -> existing profiles on the new
target -> single-feature scan -> rule search trained 2006-2016, tested 2017-07 onward.
"""
import os
import sys

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from numpy.lib.stride_tricks import sliding_window_view

from common import DATA_DIR, ERAS, RES_DIR, adj_lift, family

H = 126
TARGETS = {"proto": (np.log(1.25), 0.06), "proto_x": (np.log(1.30), 0.05)}


def forward126():
    path = f"{DATA_DIR}/fwd126.parquet"
    if os.path.exists(path):
        return pd.read_parquet(path)
    dates = pd.DatetimeIndex(pd.read_parquet(f"{DATA_DIR}/panel.parquet", columns=["date"])["date"].unique()).astype("datetime64[ns]")
    px = pd.read_parquet(f"{DATA_DIR}/ohlcv.parquet", columns=["date", "ticker", "close"])
    parts = []
    for t, g in px.groupby("ticker"):
        g = g.sort_values("date")
        x = np.log(g["close"].to_numpy(float))
        n = len(x)
        if n <= H:
            continue
        W = sliding_window_view(x, H + 1)
        m = len(W)
        fr = np.full(n, np.nan)
        fdd = np.full(n, np.nan)
        fer = np.full(n, np.nan)
        fr[:m] = W[:, -1] - W[:, 0]
        fdd[:m] = (np.maximum.accumulate(W, axis=1) - W).max(axis=1)
        path_len = np.abs(np.diff(W, axis=1)).sum(axis=1)
        fer[:m] = fr[:m] / np.where(path_len > 0, path_len, np.nan)
        dd = pd.DatetimeIndex(g["date"]).astype("datetime64[ns]")
        keep = dd.isin(dates)
        dd = dd.to_numpy()
        parts.append(pd.DataFrame({"date": dd[keep], "ticker": t, "fr126": fr[keep], "fdd126": fdd[keep], "fer126": fer[keep]}))
    out = pd.concat(parts, ignore_index=True)
    out.to_parquet(path, index=False)
    return out


def load():
    names = pq.read_schema(f"{DATA_DIR}/panel.parquet").names
    base_cols = ["date", "ticker", "sector", "industry", "mcap", "ddh252", "roc126", "stair_b63", "stair_b21", "stair_b126",
                 "stair_scales", "roc_align"]
    q = [c for c in names if c.startswith(("q_", "sq_", "iq_"))]
    d = pd.read_parquet(f"{DATA_DIR}/panel.parquet", columns=base_cols + q)
    d = d.merge(forward126(), on=["date", "ticker"], how="left")
    d = d[d["date"] >= "2006-01-01"].reset_index(drop=True)
    for k, (g, dd) in TARGETS.items():
        d[k] = ((d["fr126"] >= g) & (d["fdd126"] <= dd)).astype("float32").where(d["fr126"].notna())
        d["_wb_" + k] = d.groupby("date")[k].transform("mean")
    return d


PROFILES = {
    "quiet base (scan)": lambda d: (d["q_mar63"] <= 0.2) & (d["q_mdd126"] <= 0.3) & (d["q_roc126"] <= 0.4) & (d["ddh252"] > -0.105),
    "quiet base, any 6m return": lambda d: (d["q_mar63"] <= 0.2) & (d["q_mdd126"] <= 0.3) & (d["ddh252"] > -0.105),
    "quiet base, 6m return upper half": lambda d: (d["q_mar63"] <= 0.2) & (d["q_mdd126"] <= 0.3) & (d["ddh252"] > -0.105) & (d["q_roc126"] >= 0.5),
    "JNJ/MSI base": lambda d: (d["q_rng63"] <= 0.10) & (d["q_mar63"] <= 0.10) & (d["q_mdd126"] <= 0.15) & (d["ddh252"] > -0.10) & (d["q_roc126"] >= 0.5),
    "MSI tight (vol<=5%, 6m DD<=5%, near high)": lambda d: (d["q_mar63"] <= 0.05) & (d["q_mdd126"] <= 0.05) & (d["ddh252"] > -0.06),
    "TD re-base": lambda d: (d["q_rng63"] <= 0.10) & (d["q_roc126"] >= 0.8) & (d["ddh252"] > -0.10) & (d["q_mar63"] <= 0.2),
    "already staircasing 63d": lambda d: d["stair_b63"] == 1,
    "already staircasing 126d": lambda d: d["stair_b126"] == 1,
}


def profile_table(d, target):
    big = (d["mcap"] >= 50000).to_numpy()
    first = (d["date"] <= "2016-12-31").to_numpy()
    lab = d[target].notna().to_numpy()
    wb = "_wb_" + target
    rows = []
    for name, fn in PROFILES.items():
        m = fn(d).fillna(False).to_numpy() & lab
        for cohort, cm in (("all", np.ones(len(d), bool)), ("$50B+", big)):
            mm = m & cm
            s = d[mm]
            rows.append(dict(target=target, profile=name, cohort=cohort, weeks=int(mm.sum()), per_week=mm.sum() / d.loc[lab, "date"].nunique(),
                             hit=s[target].mean(), lift=adj_lift(s[target], s[wb]),
                             lift_2006_16=adj_lift(d.loc[mm & first, target], d.loc[mm & first, wb]),
                             lift_2017_26=adj_lift(d.loc[mm & ~first, target], d.loc[mm & ~first, wb]),
                             cohort_lift=adj_lift(d.loc[cm & lab, target], d.loc[cm & lab, wb]),
                             up126=(s["fr126"] > 0).mean(), med_fr126=np.exp(s["fr126"].median()) - 1, med_dd126=s["fdd126"].median()))
    return pd.DataFrame(rows)


def single_features(d, target):
    wb = "_wb_" + target
    lab = d[d[target].notna()]
    eras = [(lab["date"] >= a) & (lab["date"] <= b) for a, b in ERAS]
    rows = []
    for c in [c for c in d.columns if c.startswith("q_")]:
        b = np.floor(np.minimum(lab[c] * 10, 9.999))
        g = pd.DataFrame({"b": b, "y": lab[target], "w": lab[wb]}).dropna().groupby("b").agg(n=("y", "size"), h=("y", "sum"), e=("w", "sum"))
        g = g[g["n"] >= 500]
        if len(g) < 5:
            continue
        g["lift"] = g["h"] / g["e"]
        best = g["lift"].idxmax()
        el = []
        for em in eras:
            s = lab[em & (b == best)]
            el.append(adj_lift(s[target], s[wb]) if len(s) >= 200 else np.nan)
        rows.append(dict(feature=c, family=family(c), best_decile=int(best) + 1, best_lift=g.loc[best, "lift"],
                         lift_d1=g["lift"].iloc[0], lift_d10=g["lift"].iloc[-1], era_min=np.nanmin(el)))
    return pd.DataFrame(rows).sort_values("best_lift", ascending=False)


def rule_search(d, target):
    sys.argv = sys.argv[:1]
    import rules as R
    R.MIN_N_TRAIN = 1500
    wb = "_wb_" + target
    lab = d[d[target].notna()].reset_index(drop=True)
    cols = [c for c in lab.columns if c.startswith(("q_", "sq_", "iq_")) and c[2:] not in R.DISCRETE]
    tr = (lab["date"] <= "2016-12-31").to_numpy()
    te = (lab["date"] >= "2017-07-01").to_numpy()   # two-quarter embargo for 126-session labels
    C, names = R.build_conditions(lab, cols)
    y = lab[target].to_numpy().astype(np.int8)
    w = lab[wb].to_numpy(np.float32)
    Ctr, Cte = C[:, tr], C[:, te]
    del C
    dtr, dte = lab.loc[tr, "date"].to_numpy(), lab.loc[te, "date"].to_numpy()
    found = R.search(Ctr, y[tr], w[tr], R.MIN_N_TRAIN)
    rows = []
    for key, (s, h, n, e) in found.items():
        m = np.logical_and.reduce(Ctr[list(key)])
        m2 = np.logical_and.reduce(Cte[list(key)])
        nt, ht, et = m2.sum(), (m2 & (y[te] == 1)).sum(), w[te][m2].sum()
        rows.append(dict(rule=" & ".join(names[k] for k in key), depth=len(key), train_n=n, train_lift=h / max(e, 1e-9),
                         test_n=nt, test_hit=ht / max(nt, 1), test_lift=ht / max(et, 1e-9),
                         train_weeks=len(np.unique(dtr[m])), test_weeks=len(np.unique(dte[m2]))))
    out = pd.DataFrame(rows).sort_values("train_lift", ascending=False)
    rng = np.random.default_rng(5)
    yn = y[tr].copy()
    for idx in pd.Series(np.arange(len(yn))).groupby(dtr).indices.values():
        yn[idx] = rng.permutation(yn[idx])
    fn = R.search(Ctr, yn, w[tr], R.MIN_N_TRAIN, beam=20)
    best = max(fn.values(), key=lambda v: v[0])
    out.attrs["null_train_lift"] = best[1] / best[3]
    return out


def main():
    d = load()
    pd.set_option("display.width", 260)
    pd.set_option("display.max_colwidth", 110)
    for k in TARGETS:
        lab = d[d[k].notna()]
        print(f"{k}: base rate {lab[k].mean():.4f} over {len(lab):,} stock-weeks; $50B+ {lab.loc[lab['mcap'] >= 50000, k].mean():.4f}")
    # MSI's rank among all labelled 126-session windows
    lab = d[d["fr126"].notna() & (d["fdd126"] > 0)].copy()
    lab["gp"] = lab["fr126"] / lab["fdd126"]
    msi = lab[(lab["ticker"] == "MSI") & (lab["date"] >= "2024-04-01") & (lab["date"] <= "2024-05-31")]
    best = msi.loc[msi["gp"].idxmax()]
    print(f"MSI best window from {best['date'].date()}: +{np.exp(best['fr126']) - 1:.1%}, max DD {best['fdd126']:.1%}, "
          f"gain/DD {best['gp']:.1f} -> top {(lab['gp'] >= best['gp']).mean():.3%} of all stock-weeks; "
          f"efficiency {best['fer126']:.2f} -> top {(lab['fer126'] >= best['fer126']).mean():.2%}")

    prof = pd.concat([profile_table(d, k) for k in TARGETS], ignore_index=True)
    prof.to_csv(f"{RES_DIR}/prototype_profiles.csv", index=False)
    print(prof.round(3).to_string(index=False))

    sf = single_features(d, "proto")
    sf.to_csv(f"{RES_DIR}/prototype_single_features.csv", index=False)
    print("\nsingle features (proto), top 25 era-stable:")
    print(sf[sf["era_min"] > 1.0].head(25).round(3).to_string(index=False))

    rs = rule_search(d, "proto")
    rs.to_csv(f"{RES_DIR}/prototype_rules.csv", index=False)
    print(f"\nrule search (proto): null training lift {rs.attrs['null_train_lift']:.2f}; median test lift {rs['test_lift'].median():.2f}")
    ok = rs[(rs["test_weeks"] >= 100) & (rs["test_n"] >= 300)].sort_values("test_lift", ascending=False)
    print(ok.head(15).round(3).to_string(index=False))


if __name__ == "__main__":
    main()
