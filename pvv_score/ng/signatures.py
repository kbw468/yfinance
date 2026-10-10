"""Signature discovery and confirmation on non-Gaussian, tie-safe conditions.
Discovery 2015-2020 (COVID out): keep pairs with >= 300 cases and lift >= 1.30 over the window's base rate; triples must add
+0.10 lift over their pair. Confirmation 2021+: >= 150 cases, lift >= 1.20, confirmation probability >= 0.8 x discovery.
Target smooth_42. Rate-sensitive sectors use only sessions in today's MOVE band (regime.same_regime)."""
import time
import numpy as np, pandas as pd
from ..regime import tag, today_regime, same_regime
from .common import features, conditions, FIXED, DISCOVER, CONFIRM_START, COVID, D

MIN_N, MIN_LIFT, TRIPLE_ADD = 300, 1.30, 0.10
MIN_N_CONF, CONF_LIFT, CONF_RATIO = 150, 1.20, 0.8


def load_rows(cols_extra=()):
    feats = features()
    cols = list(dict.fromkeys(["date", "ticker", "sector", "beta_l1_252", "smooth_21", "smooth_42", "smooth_63"] + feats + [v[0] for v in FIXED.values()] + list(cols_extra)))
    T = pd.read_parquet(D / "table.parquet", columns=cols)
    T = T[~((T.date >= COVID[0]) & (T.date <= COVID[1]))].reset_index(drop=True)
    return T, feats


def main():
    t0 = time.time()
    T, feats = load_rows()
    T = tag(T); reg = today_regime(); same = same_regime(T, reg)
    C = conditions(T, feats); names = sorted(C); K = len(names)
    X = np.column_stack([C[k] for k in names]); del C
    print(f"rows {len(T):,} conditions {K} regime {reg} {time.time()-t0:.0f}s", flush=True)
    y = T.smooth_42.values
    disc = (T.date >= DISCOVER[0]).values & (T.date <= DISCOVER[1]).values & ~np.isnan(y) & same
    conf = (T.date >= CONFIRM_START).values & ~np.isnan(y) & same
    def pair_stats(mask):
        n2 = np.zeros((K, K)); h2 = np.zeros((K, K)); idx = np.flatnonzero(mask)
        for s in range(0, len(idx), 200000):
            a = X[idx[s:s + 200000]].astype(np.float32); b = a * y[idx[s:s + 200000], None].astype(np.float32)
            n2 += a.T @ a; h2 += a.T @ b
        return n2, h2
    nd, hd = pair_stats(disc); bd = np.nanmean(y[disc]); bc = np.nanmean(y[conf])
    pdisc = np.divide(hd, nd, out=np.zeros_like(hd), where=nd > 0)
    base = lambda k: names[k].rsplit(":", 1)[0]
    rows = []
    for i in range(K):
        for j in range(i + 1, K):
            if nd[i, j] >= MIN_N and pdisc[i, j] / bd >= MIN_LIFT and base(i) != base(j):
                rows.append(((i, j), nd[i, j], pdisc[i, j]))
    print(f"pairs passing discovery {len(rows)} {time.time()-t0:.0f}s", flush=True)
    Xd = X[disc].astype(np.float32); yd = y[disc].astype(np.float32); pairs = list(rows)
    for (i, j), _, p2 in pairs:
        pm = Xd[:, i] * Xd[:, j]
        n3 = pm @ Xd; h3 = (pm * yd) @ Xd; p3 = np.divide(h3, n3, out=np.zeros_like(h3), where=n3 > 0)
        ok = np.flatnonzero((n3 >= MIN_N) & (p3 / bd >= max(MIN_LIFT, p2 / bd) + TRIPLE_ADD))
        for k in ok:
            if k <= j or base(k) in (base(i), base(j)): continue
            rows.append(((i, j, k), n3[k], p3[k]))
    del Xd
    print(f"candidates (pairs + triples) {len(rows)} {time.time()-t0:.0f}s", flush=True)
    Xc = X[conf]; yc = y[conf]; out = []
    for legs, n_d, p_d in rows:
        m = np.all(Xc[:, list(legs)], axis=1); n_c = int(m.sum()); p_c = float(yc[m].mean()) if n_c else np.nan
        out.append({"signature": " & ".join(names[l] for l in legs), "k": len(legs), "n_disc": int(n_d), "p_disc": float(p_d), "lift_disc": float(p_d / bd),
                    "n_conf": n_c, "p_conf": p_c, "lift_conf": p_c / bc if n_c else np.nan})
    S = pd.DataFrame(out)
    S["confirmed"] = (S.n_conf >= MIN_N_CONF) & (S.lift_conf >= CONF_LIFT) & (S.p_conf >= CONF_RATIO * S.p_disc)
    S = S.sort_values("p_disc", ascending=False).reset_index(drop=True)
    S.to_csv(D / "ng_signatures_all.csv", index=False)
    print(f"discovered {len(S)}, confirmed {int(S.confirmed.sum())}; base disc {bd:.3f} conf {bc:.3f}; regime {reg}; {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
