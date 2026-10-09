"""Steady-alpha search, low+mid beta names only.
Outcome (per horizon h): next-h-session Sharpe in the universe's top 20% AND max drawdown in the shallowest 30% AND return beats SPY.
Conditions: the production quintile conditions (universe-ranked) plus the same factors ranked WITHIN the low+mid-beta group ("w_" prefix).
Discovery 2015-2020 (COVID excluded), confirmation 2021-2023, holdout 2024+. Same rules as production: disc n>=300 & lift>=1.30, triples must add +0.10 lift,
confirm n>=150 & lift>=1.20 & p_conf>=0.8*p_disc. A within-date shuffled target is run through the identical pipeline as a false-discovery check."""
import pandas as pd, numpy as np, warnings, sys, time; warnings.filterwarnings("ignore")
from pvv_score.signatures import build_conditions, QUINTILE_FACTORS, FIXED
from pvv_score.composite import tercile_series
S = "/tmp/claude-0/-home-user-yfinance/6b0db6a0-0646-53b0-950e-60d46b6034fa/scratchpad/"
t0 = time.time()
fw = [f"fwd_{k}_{h}" for h in (21, 42, 63) for k in ("sharpe", "mdd", "ret")]
cols = list(dict.fromkeys(["date", "ticker", "eligible", "beta_252"] + fw + QUINTILE_FACTORS + [v[0] for v in FIXED.values()]))
R = pd.read_parquet(S + "pvv_cache/research_long.parquet", columns=cols)
R = R[R.eligible.astype(bool) & (R.date >= "2015-01-01")]
R = R[~((R.date >= "2020-02-20") & (R.date <= "2020-06-30"))].reset_index(drop=True)
C = pd.read_parquet("pvv_score/.cache/Close.parquet", columns=["SPY"]).SPY
g = R.groupby("date")
for h in (21, 42, 63):
    spy = (C.shift(-h) / C - 1).reindex(R.date).values
    y = (g[f"fwd_sharpe_{h}"].rank(pct=True) >= 0.8) & (g[f"fwd_mdd_{h}"].rank(pct=True) >= 0.7) & (R[f"fwd_ret_{h}"].values > spy)
    R[f"y{h}"] = np.where(R[f"fwd_sharpe_{h}"].isna() | np.isnan(spy), np.nan, y.astype(float))
R["bucket"] = tercile_series(R)
U = build_conditions(R)                                          # universe-ranked, as production
L = R[R.bucket.isin(["low", "mid"])].reset_index(drop=True)
keep = R.bucket.isin(["low", "mid"]).values
conds = {k: v[keep] for k, v in U.items()}
W = L.groupby("date")[QUINTILE_FACTORS].rank(pct=True)            # within low+mid group
for f in QUINTILE_FACTORS:
    conds[f"w_{f}:TOP"] = (W[f] >= 0.8).fillna(False).values; conds[f"w_{f}:BOT"] = (W[f] <= 0.2).fillna(False).values
names = sorted(conds); K = len(names)
X = np.column_stack([conds[k] for k in names]).astype(np.float32)
per = np.where(L.date <= "2020-12-31", "disc", np.where(L.date <= "2023-12-31", "conf", "hold"))
L.to_parquet(S + "steady_rows.parquet"); np.save(S + "steady_X.npy", X); pd.Series(names).to_csv(S + "steady_names.csv", index=False)
print(f"rows {len(L):,} low+mid, conditions {K}, load {time.time()-t0:.0f}s")
for p in ("disc", "conf", "hold"):
    print(p, {h: round(np.nanmean(L[f"y{h}"].values[per == p]), 4) for h in (21, 42, 63)}, "rows", int((per == p).sum()))
def search(y, label):
    out = []
    ok = ~np.isnan(y)
    stats = {}
    for p in ("disc", "conf", "hold"):
        m = (per == p) & ok; Xp = X[m]; yp = y[m]
        n2 = np.zeros((K, K), np.float64); h2 = np.zeros((K, K), np.float64)
        for s in range(0, len(Xp), 150000):
            a = Xp[s:s+150000]; b = a * yp[s:s+150000, None]
            n2 += a.T @ a; h2 += a.T @ b
        stats[p] = (n2, h2, yp.mean(), m)
    nd, hd, bd, _ = stats["disc"]
    iu = np.triu_indices(K, 1)
    pdisc = np.divide(hd, nd, out=np.zeros_like(hd), where=nd > 0)
    sel = [(i, j) for i, j in zip(*iu) if nd[i, j] >= 300 and pdisc[i, j] / bd >= 1.30 and names[i].split(":")[0].lstrip("w_") != names[j].split(":")[0].lstrip("w_")]
    rows = []
    for i, j in sel:
        rows.append({"legs": (i, j), "n_disc": nd[i, j], "p_disc": pdisc[i, j]})
    # triples: extend each passing pair by a third condition that adds >= +0.10 lift
    md = stats["disc"][3]; Xd = X[md]; yd = y[md]
    for i, j in sel:
        pair = Xd[:, i] * Xd[:, j]
        if pair.sum() < 300: continue
        n3 = pair @ Xd; h3 = (pair * yd) @ Xd
        p3 = np.divide(h3, n3, out=np.zeros_like(h3), where=n3 > 0)
        for k in np.flatnonzero((n3 >= 300) & (p3 / bd >= max(1.40, pdisc[i, j] / bd + 0.10))):
            if k in (i, j) or k < j: continue
            rows.append({"legs": (i, j, k), "n_disc": n3[k], "p_disc": p3[k]})
    D = pd.DataFrame(rows)
    if D.empty: return D
    D["lift_disc"] = D.p_disc / bd
    for p in ("conf", "hold"):
        mp = stats[p][3]; Xp = X[mp]; yp = y[mp]; bp = stats[p][2]
        nn, pp = [], []
        for legs in D["legs"]:
            mm = np.prod(Xp[:, list(legs)], axis=1)
            n_ = mm.sum(); nn.append(n_); pp.append((mm * yp).sum() / n_ if n_ else np.nan)
        D[f"n_{p}"] = nn; D[f"p_{p}"] = pp; D[f"lift_{p}"] = D[f"p_{p}"] / bp
    D["confirmed"] = (D.n_conf >= 150) & (D.lift_conf >= 1.20) & (D.p_conf >= 0.8 * D.p_disc)
    D["signature"] = D["legs"].map(lambda t: " & ".join(names[x] for x in t))
    print(f"[{label}] discovered {len(D)}, confirmed {int(D.confirmed.sum())}")
    return D
y42 = L.y42.values.astype(float)
D = search(y42, "real 42d")
D.drop(columns="legs").to_csv(S + "steady_signatures_42.csv", index=False)
rng = np.random.default_rng(7)
ys = L.groupby("date").y42.transform(lambda s: s.sample(frac=1, random_state=int(rng.integers(1e9))).values).values.astype(float)
Dn = search(ys, "shuffled target 42d")
print(f"done {time.time()-t0:.0f}s")
