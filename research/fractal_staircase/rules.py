"""Stage 2: exhaustive conjunction-rule search with an honest null (regime-neutral).

Conditions are per-week percentile thresholds on features (q_/sq_/iq_ ranks, plus
group and regime levels): rank >= {0.6,0.7,0.8,0.9} or rank <= {0.1,0.2,0.3,0.4};
flag/count features use their raw values. Beam search grows rules up to MAX_DEPTH
conditions on the TRAIN era. Score = lower 10% quantile of the Jeffreys Beta
posterior of the rule's hit rate divided by the hit rate the same rows would have
had at their own weeks' universe base rate. So a rule gets credit only for picking
better names inside the same weeks, never for landing in bull markets.

Every rule is then re-scored on the untouched TEST era.
Null: labels are shuffled within each week (keeps every week's base rate), the
whole search is re-run, and the best train score is recorded.
"""
import sys

import numpy as np
import pandas as pd

from common import DATA_DIR, RES_DIR, beta_lower, load_panel

TRAIN_END = "2016-12-31"
TEST_START = "2017-04-01"          # one-quarter embargo so 63-bar labels cannot overlap
HI = (0.6, 0.7, 0.8, 0.9)
LO = (0.1, 0.2, 0.3, 0.4)
MAX_DEPTH = 4
BEAM = 60
MIN_N_TRAIN = 1500                 # ~3 names a week across the train era
DISCRETE = {"stair_b21": [1], "stair_b63": [1], "stair_b126": [1], "stair_scales": [1, 2, 3], "roc_align": [3, 4]}


def build_conditions(df, cols):
    conds, names = [], []
    for c in cols:
        v = df[c].to_numpy()
        ok = ~np.isnan(v)
        for t in HI:
            conds.append(ok & (v >= t))
            names.append(f"{c}>={t}")
        for t in LO:
            conds.append(ok & (v <= t))
            names.append(f"{c}<={t}")
    for c, levels in DISCRETE.items():
        v = df[c].to_numpy()
        for lv in levels:
            conds.append(v >= lv)
            names.append(f"{c}>={lv}")
        conds.append(v == 0)
        names.append(f"{c}==0")
    return np.array(conds), names


def wsum(C, w, chunk=128):
    """Row-weighted support of every condition, chunked to keep memory flat."""
    out = np.empty(C.shape[0], dtype=np.float64)
    for i in range(0, C.shape[0], chunk):
        out[i:i + chunk] = C[i:i + chunk].astype(np.float32) @ w
    return out


def _score(h, n, e, min_n):
    rate_lo = beta_lower(h / EFF, np.maximum(n / EFF, 1))
    return np.where(n >= min_n, rate_lo / np.maximum(e / np.maximum(n, 1), 1e-9), -1)


def search(C, y, wb, min_n, max_depth=MAX_DEPTH, beam=BEAM):
    yb = y.astype(bool)
    n = C.sum(1)
    h = C[:, yb].sum(1)
    e = wsum(C, wb)
    score = _score(h, n, e, min_n)
    order = np.argsort(-score)[:beam]
    frontier = [((i,), np.flatnonzero(C[i])) for i in order if score[i] > 0]
    found = {rule: (score[rule[0]], h[rule[0]], n[rule[0]], e[rule[0]]) for rule, _ in frontier}
    for depth in range(2, max_depth + 1):
        cand = {}
        for rule, idx in frontier:
            sub = C[:, idx]
            n2 = sub.sum(1)
            h2 = sub[:, yb[idx]].sum(1)
            e2 = wsum(sub, wb[idx])
            s2 = _score(h2, n2, e2, min_n)
            for j in np.argsort(-s2)[:beam]:
                if s2[j] <= 0 or j in rule:
                    continue
                key = tuple(sorted(rule + (j,)))
                if key not in cand or cand[key][0] < s2[j]:
                    cand[key] = (s2[j], h2[j], n2[j], e2[j])
        best = sorted(cand.items(), key=lambda kv: -kv[1][0])[:beam]
        frontier = []
        for key, val in best:
            found[key] = val
            m = np.logical_and.reduce(C[list(key)])
            frontier.append((key, np.flatnonzero(m)))
    return found


def shuffle_within_date(y, dates, rng):
    ys = y.copy()
    for idx in pd.Series(np.arange(len(y))).groupby(dates).indices.values():
        ys[idx] = rng.permutation(ys[idx])
    return ys


EFF = 6.0   # weekly rows of one name overlap ~13 weeks of forward path; shrink counts before the Beta bound


def main(target="y63", n_null=3, with_regime=False):
    import pyarrow.parquet as pq
    names_all = pq.read_schema(f"{DATA_DIR}/panel.parquet").names
    # regime levels are few independent observations (weeks are autocorrelated); by default
    # rules may only use stock-level and group-level conditions
    lvl = [c for c in names_all if c.startswith(("sec_", "ind_", "rel_") + (("mkt_",) if with_regime else ()))]
    rank_cols = [c for c in names_all if c.startswith(("q_", "sq_", "iq_")) and c[2:] not in DISCRETE]
    need = ["date", "ticker", "fr21", "fdd21", "fr42", "fdd42", "fr63", "fdd63", *DISCRETE] + rank_cols + lvl
    df = load_panel(columns=list(dict.fromkeys(need)))
    df = df[(df["date"] >= "2006-01-01") & df[target].notna()].reset_index(drop=True)
    for c in lvl:  # group/regime levels -> percentile over full history
        df["p_" + c] = df[c].rank(pct=True).astype("float32")
    df = df.drop(columns=lvl)
    cols = rank_cols + ["p_" + c for c in lvl]
    wb_all = df.groupby("date")[target].transform("mean").to_numpy(np.float32)

    tr = (df["date"] <= TRAIN_END).to_numpy()
    te = (df["date"] >= TEST_START).to_numpy()
    y = df[target].to_numpy().astype(np.int8)
    C, names = build_conditions(df, cols)
    dtr = df.loc[tr, "date"].to_numpy()
    dte = df.loc[te, "date"].to_numpy()
    del df
    print("conditions", C.shape, "train rows", tr.sum(), "test rows", te.sum(), flush=True)
    Ctr, Cte = C[:, tr], C[:, te]
    del C
    ytr, yte, wtr, wte = y[tr], y[te], wb_all[tr], wb_all[te]

    found = search(Ctr, ytr, wtr, MIN_N_TRAIN)
    rows, out_keys = [], []
    for key, (s, h, n, e) in found.items():
        out_keys.append(key)
        m = np.logical_and.reduce(Cte[list(key)])
        nt, ht, et = m.sum(), (m & yte.astype(bool)).sum(), wte[m].sum()
        rows.append(dict(rule=" & ".join(names[k] for k in key), depth=len(key),
                         train_n=n, train_hit=h / n, train_lift=h / max(e, 1e-9), train_score=s,
                         test_n=nt, test_hit=ht / max(nt, 1), test_lift=ht / max(et, 1e-9)))
    out = pd.DataFrame(rows)
    # how many distinct weeks each rule fires in (train / test) - concentration check
    wk_tr, wk_te = [], []
    for key in out_keys:
        m = np.logical_and.reduce(Ctr[list(key)])
        wk_tr.append(len(np.unique(dtr[m])))
        m2 = np.logical_and.reduce(Cte[list(key)])
        wk_te.append(len(np.unique(dte[m2])))
    out["train_weeks"] = wk_tr
    out["test_weeks"] = wk_te
    out = out.sort_values("train_score", ascending=False)
    out.to_csv(f"{RES_DIR}/rules_{target}.csv", index=False)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_colwidth", 150)
    print(out.head(40).round(3).to_string(index=False), flush=True)

    rng = np.random.default_rng(7)
    null_best = []
    for i in range(n_null):
        ys = shuffle_within_date(ytr, dtr, rng)
        fnull = search(Ctr, ys, wtr, MIN_N_TRAIN, beam=20)
        best = max(fnull.values(), key=lambda v: v[0])
        null_best.append(dict(score=best[0], hit=best[1] / best[2], lift=best[1] / best[3]))
        print("null", i, {k: round(float(v), 4) for k, v in null_best[-1].items()}, flush=True)
    pd.DataFrame(null_best).to_csv(f"{RES_DIR}/rules_null_{target}.csv", index=False)
    print("null best train score (max over shuffles):", round(max(v["score"] for v in null_best), 3),
          " real best:", round(out["train_score"].max(), 3))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "y63", int(sys.argv[2]) if len(sys.argv) > 2 else 3,
         len(sys.argv) > 3 and sys.argv[3] == "regime")
