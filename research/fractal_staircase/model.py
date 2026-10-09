"""Stage 3: walk-forward gradient-boosted trees (non-parametric).

Modes (all free of Gaussian assumptions):
  binary     - Bernoulli log-loss on one staircase label, all inputs
  binary_xs  - same, stock- and group-level inputs only (no market-state inputs), so
               the model can only separate names inside the same week
  rank       - LambdaRank within each week, graded relevance:
               3 = y_hold, 2 = y63, 1 = positive quarter with max DD <= half the gain
Training data for each test block ends one quarter before the block starts, so no
forward label used in training overlaps the test period.
Outputs out-of-sample top-k outcome profiles, a non-parametric calibration table
(score rank-percentile -> realised hit rates), grouped importance, and today's scores.

Memory: features live in one float32 matrix; blocks are index slices of it.
Usage: python model.py <mode> <target> [final]   ("final" skips the walk-forward)
"""
import os
import sys

import lightgbm as lgb
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from common import DATA_DIR, RES_DIR, add_targets, family, feature_cols

BLOCKS = [(f"{y}-01-01", f"{y + 1}-12-31") for y in range(2010, 2026, 2)]
EMBARGO_DAYS = 100
BASE = dict(learning_rate=0.03, num_leaves=31, min_data_in_leaf=400, feature_fraction=0.5,
            bagging_fraction=0.7, bagging_freq=1, lambda_l2=1.0, verbose=-1, num_threads=4)
ROUNDS = 500
TOPK = (5, 10, 25, 50)
BUCKETS = [0.0, 0.5, 0.8, 0.9, 0.95, 0.98, 0.99, 0.995, 1.0]
OUTCOME_COLS = ["y21", "y63", "y_hold", "fr21", "fr63", "fdd21", "fdd63", "fer63"]
META = ["date", "ticker", "sector", "industry", "mcap_bucket"]


def relevance(meta):
    rel = np.zeros(len(meta), dtype=np.int32)
    smooth = (meta["fr63"] > 0) & (meta["fdd63"] <= meta["fr63"] / 2)
    rel[smooth.to_numpy()] = 1
    rel[(meta["y63"] == 1).to_numpy()] = 2
    rel[(meta["y_hold"] == 1).to_numpy()] = 3
    return rel


def load(xs_only):
    import pyarrow as pa
    schema = pq.read_schema(f"{DATA_DIR}/panel.parquet")
    numeric = {f.name for f in schema if pa.types.is_floating(f.type) or pa.types.is_integer(f.type)}
    stub = pd.DataFrame({c: pd.Series(dtype="float32") for c in schema.names if c in numeric})
    feats = [c for c in feature_cols(stub) if c != "mcap"]
    if xs_only:  # no market-state inputs
        feats = [c for c in feats if not c.startswith(("mkt_", "p_mkt_"))]
    lab_cols = ["fr21", "fdd21", "fer21", "fr42", "fdd42", "fr63", "fdd63", "fer63"]
    meta = pd.read_parquet(f"{DATA_DIR}/panel.parquet", columns=META + lab_cols)
    meta = add_targets(meta)
    keep = (meta["date"] >= "2006-01-01").to_numpy()
    meta = meta[keep].reset_index(drop=True)
    cols = []
    for i in range(0, len(feats), 60):  # read in column chunks to keep the peak low
        part = pd.read_parquet(f"{DATA_DIR}/panel.parquet", columns=feats[i:i + 60])
        cols.append(part[keep].to_numpy(np.float32))
        del part
    X = np.hstack(cols)
    del cols
    sector_code = meta["sector"].astype("category").cat.codes.to_numpy(np.float32)[:, None]
    X = np.hstack([X, sector_code])
    return meta, X, feats + ["sector_code"]


def fit(X, meta, idx, feats, mode, target, seed=11):
    order = idx[np.argsort(meta["date"].to_numpy()[idx], kind="stable")]
    if mode == "rank":
        params = dict(BASE, objective="lambdarank", lambdarank_truncation_level=60, label_gain=[0, 1, 3, 7], seed=seed)
        m = meta.iloc[order]
        grp = m.groupby("date", sort=False).size().to_numpy()
        ds = lgb.Dataset(X[order], relevance(m), group=grp, feature_name=feats, free_raw_data=True)
    else:
        params = dict(BASE, objective="binary", seed=seed)
        ds = lgb.Dataset(X[order], meta[target].to_numpy()[order], feature_name=feats, free_raw_data=True)
    return lgb.train(params, ds, num_boost_round=ROUNDS)


def profile(d):
    return {
        "n": len(d), "p_y63": d["y63"].mean(), "p_y21": d["y21"].mean(), "p_hold": d["y_hold"].mean(),
        "p_up63": (d["fr63"] > 0).mean(), "p_dd63_lt5": (d["fdd63"] < 0.05).mean(),
        "med_fr63": d["fr63"].median(), "q25_fr63": d["fr63"].quantile(0.25), "q75_fr63": d["fr63"].quantile(0.75),
        "med_fdd63": d["fdd63"].median(), "med_fer63": d["fer63"].median(), "med_fr21": d["fr21"].median(),
    }


def evaluate(o):
    o = o.copy()
    o["rk"] = o.groupby("date")["score"].rank(ascending=False, method="first")
    o["pct"] = o.groupby("date")["score"].rank(pct=True)
    rows = {"all": profile(o)}
    for k in TOPK:
        rows[f"top{k}"] = profile(o[o["rk"] <= k])
    prof = pd.DataFrame(rows).T
    o["bucket"] = pd.cut(o["pct"], BUCKETS, include_lowest=True)
    calib = o.groupby("bucket", observed=True).apply(lambda d: pd.Series(profile(d)), include_groups=False)
    return prof, calib


def every_other_week(meta, idx):
    wk = meta["date"].iloc[idx].rank(method="dense").astype(int).to_numpy()
    return idx[wk % 2 == 0]


def main(mode="binary_xs", target="y63", final_only=False):
    tag = f"{mode}_{target}" if mode.startswith("binary") else "rank"
    fit_mode = "rank" if mode == "rank" else "binary"
    meta, X, feats = load(xs_only=mode.endswith("_xs"))
    print(tag, "features", len(feats), "rows", len(meta), "matrix GB", round(X.nbytes / 1e9, 2), flush=True)
    labelled = np.flatnonzero(meta["fr63"].notna().to_numpy())
    dates = meta["date"].to_numpy()

    if not final_only:
        oos, imp = [], pd.Series(0.0, index=feats)
        for a, b in BLOCKS:
            cut = np.datetime64(pd.Timestamp(a) - pd.Timedelta(days=EMBARGO_DAYS))
            tr = labelled[dates[labelled] <= cut]
            te = labelled[(dates[labelled] >= np.datetime64(a)) & (dates[labelled] <= np.datetime64(pd.Timestamp(b)))]
            if len(te) == 0:
                continue
            tr = every_other_week(meta, tr)
            m = fit(X, meta, tr, feats, fit_mode, target)
            imp += pd.Series(m.feature_importance("gain"), index=feats)
            o = meta.iloc[te][["date", "ticker", "sector", "industry", "mcap_bucket"] + OUTCOME_COLS].copy()
            o["score"] = m.predict(X[te])
            oos.append(o)
            prof, _ = evaluate(o)
            print(a[:4], b[:4], "train", len(tr), "test", len(te), "| base y63 %.3f top5 %.3f top10 %.3f top25 %.3f top50 %.3f"
                  % tuple(prof.loc[["all", "top5", "top10", "top25", "top50"], "p_y63"]), flush=True)
        oos = pd.concat(oos, ignore_index=True)
        oos.to_parquet(f"{DATA_DIR}/oos_{tag}.parquet", index=False)
        prof, calib = evaluate(oos)
        prof.to_csv(f"{RES_DIR}/model_profile_{tag}.csv")
        calib.to_csv(f"{RES_DIR}/model_calibration_{tag}.csv")
        pd.set_option("display.width", 250)
        print(prof.round(4).to_string())
        print(calib.round(4).to_string())
        imp = imp / imp.sum()
        fam = imp.groupby([family(c) for c in imp.index]).sum().sort_values(ascending=False)
        imp.sort_values(ascending=False).to_csv(f"{RES_DIR}/model_importance_{tag}.csv")
        fam.to_csv(f"{RES_DIR}/model_family_importance_{tag}.csv")
        print(fam.round(3).to_string())
        print(imp.sort_values(ascending=False).head(30).round(4).to_string(), flush=True)

    # final: 3 seeds on all labelled history, scored on the latest session
    full = every_other_week(meta, labelled)
    last_idx = np.flatnonzero(dates == dates.max())
    scores, contribs = [], []
    for seed in (11, 23, 37):
        m = fit(X, meta, full, feats, fit_mode, target, seed=seed)
        m.save_model(f"{DATA_DIR}/model_{tag}_{seed}.txt")
        scores.append(m.predict(X[last_idx]))
        contribs.append(m.predict(X[last_idx], pred_contrib=True)[:, :-1])
        print("final seed", seed, "done", flush=True)
    last = meta.iloc[last_idx][["date", "ticker", "sector", "industry"]].copy()
    last["score"] = np.mean(scores, axis=0)
    cdf = pd.DataFrame(np.mean(contribs, axis=0), columns=feats, index=last.index)
    famc = cdf.T.groupby([family(c) for c in feats]).sum().T
    famc.columns = ["why_" + c for c in famc.columns]
    last["top_drivers"] = cdf.apply(lambda r: ", ".join(r.sort_values(ascending=False).index[:4]), axis=1)
    last = pd.concat([last, famc], axis=1)
    last["pct"] = last["score"].rank(pct=True)
    last.to_parquet(f"{DATA_DIR}/latest_{tag}.parquet", index=False)
    print("latest", last["date"].iloc[0], "scored", len(last))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "binary_xs", sys.argv[2] if len(sys.argv) > 2 else "y63",
         len(sys.argv) > 3 and sys.argv[3] == "final")
