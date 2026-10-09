"""Shape alphabet: unsupervised k-medians (L1) clustering of normalised price paths.

Every weekly observation's trailing path (63 and 126 bars, each resampled to 21
points and min-max scaled to [0,1]) is assigned to one of K prototype shapes learned
on the train era only. Then: staircase hit rate per shape, train vs test, and the
joint (63-bar shape, 126-bar shape) table to see whether the same geometry at two
scales stacks. L1 / medians throughout (no squared distances).
"""
import os
import sys

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view

from common import DATA_DIR, RES_DIR, add_targets

K = 30
PTS = 21
TRAIN_END = "2016-12-31"
TEST_START = "2017-04-01"


def windows(x, L):
    W = sliding_window_view(x, L)
    idx = np.linspace(0, L - 1, PTS).round().astype(int)
    W = W[:, idx]
    lo, hi = W.min(1, keepdims=True), W.max(1, keepdims=True)
    return (W - lo) / np.where(hi > lo, hi - lo, np.nan)


def kmedians(X, k, iters=25, seed=3):
    rng = np.random.default_rng(seed)
    C = X[rng.choice(len(X), k, replace=False)]
    for _ in range(iters):
        lab = assign(X, C)
        newC = np.array([np.median(X[lab == j], axis=0) if (lab == j).any() else C[j] for j in range(k)])
        if np.allclose(newC, C):
            break
        C = newC
    return C


def assign(X, C, chunk=200_000):
    out = np.empty(len(X), dtype=np.int16)
    for i in range(0, len(X), chunk):
        d = np.abs(X[i:i + chunk, None, :] - C[None, :, :]).sum(-1)
        out[i:i + chunk] = d.argmin(1)
    return out


def main(target="y63"):
    px = pd.read_parquet(os.path.join(DATA_DIR, "ohlcv.parquet"), columns=["date", "ticker", "close"])
    panel = pd.read_parquet(os.path.join(DATA_DIR, "panel.parquet"),
                            columns=["date", "ticker", "fr21", "fdd21", "fr42", "fdd42", "fr63", "fdd63"])
    panel = add_targets(panel)
    keys = set(zip(panel["date"].values, panel["ticker"].values))
    recs = {63: [], 126: []}
    for tkr, g in px.groupby("ticker", sort=False):
        g = g.sort_values("date")
        x = np.log(g["close"].to_numpy(float))
        d = g["date"].to_numpy()
        for L in (63, 126):
            if len(x) < L:
                continue
            W = windows(x, L)
            dd = d[L - 1:]
            keep = np.array([(di, tkr) in keys for di in dd])
            if keep.any():
                recs[L].append(pd.DataFrame(W[keep], columns=[f"p{i}" for i in range(PTS)]).assign(date=dd[keep], ticker=tkr))
    pd.set_option("display.width", 250)
    shape_ids = {}
    for L in (63, 126):
        S = pd.concat(recs[L], ignore_index=True).dropna()
        S = S.merge(panel[["date", "ticker", target]], on=["date", "ticker"], how="left")
        X = S[[f"p{i}" for i in range(PTS)]].to_numpy(np.float32)
        tr = (S["date"] <= TRAIN_END).to_numpy()
        rng = np.random.default_rng(0)
        fit_idx = rng.choice(np.flatnonzero(tr), min(150_000, tr.sum()), replace=False)
        C = kmedians(X[fit_idx], K)
        S["shape"] = assign(X, C)
        lab = S[S[target].notna()]
        base_tr = lab.loc[lab["date"] <= TRAIN_END, target].mean()
        base_te = lab.loc[lab["date"] >= TEST_START, target].mean()
        t = pd.DataFrame({
            "n_train": lab[lab["date"] <= TRAIN_END].groupby("shape").size(),
            "hit_train": lab[lab["date"] <= TRAIN_END].groupby("shape")[target].mean(),
            "n_test": lab[lab["date"] >= TEST_START].groupby("shape").size(),
            "hit_test": lab[lab["date"] >= TEST_START].groupby("shape")[target].mean(),
        })
        t["lift_train"] = t["hit_train"] / base_tr
        t["lift_test"] = t["hit_test"] / base_te
        cen = pd.DataFrame(C, columns=[f"p{i}" for i in range(PTS)])
        t = pd.concat([t, cen], axis=1).sort_values("lift_train", ascending=False)
        t.to_csv(f"{RES_DIR}/shapes{L}_{target}.csv")
        print(f"\n== {L}-bar shapes  base train {base_tr:.4f} test {base_te:.4f}")
        print(t[["n_train", "hit_train", "lift_train", "n_test", "hit_test", "lift_test"]].round(3).to_string())
        shape_ids[L] = S[["date", "ticker", "shape"]].rename(columns={"shape": f"shape{L}"})
        # spell the best shapes as sparklines
        bars = " .:-=+*#%@"
        for sid in t.index[:5]:
            s = "".join(bars[min(int(v * 9.99), 9)] for v in cen.loc[sid])
            print(f"shape {sid:2d} lift_tr {t.loc[sid, 'lift_train']:.2f} lift_te {t.loc[sid, 'lift_test']:.2f}  {s}")
    J = shape_ids[63].merge(shape_ids[126], on=["date", "ticker"]).merge(panel[["date", "ticker", target]], on=["date", "ticker"])
    J = J[J[target].notna()]
    base = J[target].mean()
    jt = J.groupby(["shape63", "shape126"])[target].agg(["mean", "size"])
    jt = jt[jt["size"] >= 1500]
    jt["lift"] = jt["mean"] / base
    jt = jt.sort_values("lift", ascending=False)
    jt.to_csv(f"{RES_DIR}/shapes_joint_{target}.csv")
    print("\n== joint 63x126 shapes (top 15)")
    print(jt.head(15).round(3).to_string())
    pd.concat([shape_ids[63].set_index(["date", "ticker"]), shape_ids[126].set_index(["date", "ticker"])], axis=1) \
        .reset_index().to_parquet(f"{DATA_DIR}/shape_ids.parquet", index=False)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "y63")
