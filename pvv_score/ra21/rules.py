"""Interpretable multi-factor conjunctions that precede a superior 21-session trade.
Conditions: every stock feature in the top or bottom 20% of the universe that date (tie-safe: a tied block resolves the same
way for every name in it), plus market-state legs (SPY near / off its high, SPY 21-session direction, VIX vs its own year,
breadth terciles fixed on the discovery window). Pairs and triples of distinct features.
Discovery 2015-2020 (COVID out): >= 300 cases, lift >= 1.30 over the window's base rate; a triple must add +0.10 lift over its pair.
Confirmation 2021-2023: >= 150 cases, lift >= 1.20, confirmation rate >= 0.8 x discovery rate.
Hold-out 2024 onward: reported, never used to select. Trade statistics per rule from the hold-out rows."""
import json, time
import numpy as np
import pandas as pd
from .common import load, feature_sets, D, OUT, MARKET

MIN_N, MIN_LIFT, TRIPLE_ADD, MAX_PAIRS_FOR_TRIPLES = 300, 1.30, 0.10, 2000
MIN_N_CONF, CONF_LIFT, CONF_RATIO = 150, 1.20, 0.8
DISC = ("2015-01-01", "2020-12-31"); CONF = ("2021-01-01", "2023-12-31"); HOLD = "2024-01-01"
STATS = ["xs_21", "mar_21", "dd_21", "stopped_21", "sup_42", "sup_63", "xs_63", "beta_l1_252"]


def breadth_cuts(T: pd.DataFrame, disc_mask: np.ndarray) -> list:
    return T.loc[disc_mask].groupby("date").mkt_breadth_63h.first().quantile([1 / 3, 2 / 3]).tolist()


def conditions(T: pd.DataFrame, feats: list, cuts: list) -> dict:
    out = {}
    g = T.groupby("date")
    for f in feats:
        n = g[f].transform("count"); below = g[f].rank(method="min") - 1; above = n - g[f].rank(method="max")
        out[f"{f}:BOT"] = ((below / n) < 0.2).fillna(False).to_numpy()
        out[f"{f}:TOP"] = ((above / n) < 0.2).fillna(False).to_numpy()
    out["mkt:SPY within 5% of high"] = (T.mkt_off_high_252 >= -0.05).to_numpy()
    out["mkt:SPY >5% below high"] = (T.mkt_off_high_252 < -0.05).to_numpy()
    out["mkt:SPY 21d up"] = (T.mkt_roc_21 > 0).to_numpy(); out["mkt:SPY 21d down"] = (T.mkt_roc_21 <= 0).to_numpy()
    out["mkt:VIX top third of its year"] = (T.mkt_vix_pctile >= 2 / 3).to_numpy(); out["mkt:VIX bottom third of its year"] = (T.mkt_vix_pctile <= 1 / 3).to_numpy()
    out["mkt:breadth low"] = (T.mkt_breadth_63h <= cuts[0]).to_numpy(); out["mkt:breadth high"] = (T.mkt_breadth_63h >= cuts[1]).to_numpy()
    return out


def main():
    t0 = time.time()
    fs = feature_sets(); feats = fs["stock"]
    T = load(feats + MARKET + ["sup_21"] + [c for c in STATS if c not in feats])
    y = T.sup_21.to_numpy(float)
    disc = ((T.date >= DISC[0]) & (T.date <= DISC[1])).to_numpy() & ~np.isnan(y)
    conf = ((T.date >= CONF[0]) & (T.date <= CONF[1])).to_numpy() & ~np.isnan(y)
    hold = (T.date >= HOLD).to_numpy() & ~np.isnan(y)
    cuts = breadth_cuts(T, disc); json.dump({"breadth_cuts": cuts}, open(OUT / "rules_meta.json", "w"))
    C = conditions(T, feats, cuts); names = sorted(C); K = len(names)
    X = np.column_stack([C[k] for k in names]); del C
    base = lambda k: names[k].rsplit(":", 1)[0]
    print(f"rows {len(T):,} conditions {K} {time.time()-t0:.0f}s", flush=True)
    idx = np.flatnonzero(disc); nd = np.zeros((K, K)); hd = np.zeros((K, K))
    for s in range(0, len(idx), 150000):
        a = X[idx[s:s + 150000]].astype(np.float32); nd += a.T @ a; hd += a.T @ (a * y[idx[s:s + 150000], None].astype(np.float32))
    bd, bc, bh = y[disc].mean(), y[conf].mean(), y[hold].mean()
    pdisc = np.divide(hd, nd, out=np.zeros_like(hd), where=nd > 0)
    rows = [((i, j), nd[i, j], pdisc[i, j]) for i in range(K) for j in range(i + 1, K)
            if nd[i, j] >= MIN_N and pdisc[i, j] / bd >= MIN_LIFT and base(i) != base(j) and not (names[i].startswith("mkt:") and names[j].startswith("mkt:"))]
    print(f"pairs passing discovery {len(rows)}; base disc {bd:.3f} conf {bc:.3f} hold {bh:.3f} {time.time()-t0:.0f}s", flush=True)
    Xd = X[disc].astype(np.float32); yd = y[disc].astype(np.float32)
    for (i, j), _, p2 in sorted(rows, key=lambda r: -r[2])[:MAX_PAIRS_FOR_TRIPLES]:
        pm = Xd[:, i] * Xd[:, j]
        n3 = pm @ Xd; h3 = (pm * yd) @ Xd; p3 = np.divide(h3, n3, out=np.zeros_like(h3), where=n3 > 0)
        for k in np.flatnonzero((n3 >= MIN_N) & (p3 / bd >= p2 / bd + TRIPLE_ADD)):
            if k <= j or base(k) in (base(i), base(j)) or (names[k].startswith("mkt:") and (names[i].startswith("mkt:") or names[j].startswith("mkt:"))): continue
            rows.append(((i, j, k), n3[k], p3[k]))
    del Xd
    print(f"candidates {len(rows)} {time.time()-t0:.0f}s", flush=True)
    out = []
    Xc, yc = X[conf], y[conf]; Xh, yh = X[hold], y[hold]
    for legs, n_d, p_d in rows:
        mc = np.all(Xc[:, list(legs)], axis=1); n_c = int(mc.sum())
        out.append({"rule": " & ".join(names[l] for l in legs), "k": len(legs), "n_disc": int(n_d), "p_disc": float(p_d), "lift_disc": float(p_d / bd),
                    "n_conf": n_c, "p_conf": float(yc[mc].mean()) if n_c else np.nan})
    S = pd.DataFrame(out); S["lift_conf"] = S.p_conf / bc
    S["confirmed"] = (S.n_conf >= MIN_N_CONF) & (S.lift_conf >= CONF_LIFT) & (S.p_conf >= CONF_RATIO * S.p_disc)
    # hold-out and trade statistics for confirmed rules only
    ok = S[S.confirmed].index; pos = {n: i for i, n in enumerate(names)}
    Th = T.loc[hold, STATS].reset_index(drop=True)
    for r in ok:
        legs = [pos[c] for c in S.at[r, "rule"].split(" & ")]
        m = np.all(Xh[:, legs], axis=1); S.at[r, "n_hold"] = int(m.sum())
        if m.sum():
            S.at[r, "p_hold"] = float(yh[m].mean())
            for c in STATS: S.at[r, f"hold_{'med_' if c in ('xs_21', 'mar_21', 'dd_21', 'xs_63', 'beta_l1_252') else ''}{c}"] = float(Th.loc[m, c].median() if c in ("xs_21", "mar_21", "dd_21", "xs_63", "beta_l1_252") else Th.loc[m, c].mean())
            S.at[r, "hold_mean_xs_21"] = float(Th.loc[m, "xs_21"].mean())
    S["lift_hold"] = S.p_hold / bh
    S = S.sort_values(["confirmed", "lift_conf"], ascending=[False, False]).reset_index(drop=True)
    S.to_csv(D / "rules_all.csv", index=False)
    S[S.confirmed].to_csv(OUT / "rules_confirmed.csv", index=False)
    print(f"discovered {len(S)}, confirmed {int(S.confirmed.sum())}; hold-out base {bh:.3f}; {time.time()-t0:.0f}s")
    c = S[S.confirmed]
    if len(c):
        print(f"confirmed rules in the hold-out: median lift {c.lift_hold.median():.2f}, share with lift > 1: {(c.lift_hold > 1).mean():.2f}, share > 1.2: {(c.lift_hold > 1.2).mean():.2f}")
        print(c.head(40)[["rule", "n_disc", "lift_disc", "n_conf", "lift_conf", "n_hold", "p_hold", "lift_hold"]].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
