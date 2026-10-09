import pandas as pd, numpy as np, warnings; warnings.filterwarnings("ignore")
from pvv_score.signature_rule import plain as p0
S = "/tmp/claude-0/-home-user-yfinance/6b0db6a0-0646-53b0-950e-60d46b6034fa/scratchpad/"
def plain(c):
    if c.startswith("w_"): return p0(c[2:]) + " (within low/mid beta)"
    return p0(c)
D = pd.read_csv(S + "steady_signatures_42.csv"); Cf = D[D.confirmed].copy()
L = pd.read_parquet(S + "steady_rows.parquet"); X = np.load(S + "steady_X.npy"); names = pd.read_csv(S + "steady_names.csv").iloc[:, 0].tolist(); ix = {n: i for i, n in enumerate(names)}
per = np.where(L.date <= "2020-12-31", "disc", np.where(L.date <= "2023-12-31", "conf", "hold"))
print("confirmed:", len(Cf), " holdout lift: median", round(Cf.lift_hold.median(), 2), " share >=1.2:", round((Cf.lift_hold >= 1.2).mean(), 2), " share <1.0:", round((Cf.lift_hold < 1).mean(), 2))
Cf = Cf.sort_values("p_conf", ascending=False)
n = len(L); cnt = np.zeros(n, np.int32); best = np.zeros(n); bestsig = np.full(n, -1)
for k, s in enumerate(Cf.signature):
    m = np.ones(n, bool)
    for leg in s.split(" & "): m &= X[:, ix[leg]] > 0
    cnt += m; upd = m & (Cf.p_conf.iloc[k] > best); best[upd] = Cf.p_conf.iloc[k]; bestsig[upd] = k
L["n_fire"] = cnt; L["best"] = best; L["per"] = per
H = L[L.per == "hold"].dropna(subset=["y42"])
H["nf"] = pd.cut(H.n_fire, [-1, 0, 10, 50, 200, 100000], labels=["0", "1-10", "11-50", "51-200", "200+"])
agg = dict(rows=("y42", "size"), names=("ticker", "nunique"), hit21=("y21", "mean"), hit42=("y42", "mean"), hit63=("y63", "mean"),
           med_ret42=("fwd_ret_42", "median"), med_mdd42=("fwd_mdd_42", "median"), share_dd_worse_8=("fwd_mdd_42", lambda s: (s < -0.08).mean()))
print("\nHOLDOUT 2024+ low/mid beta, by number of confirmed steady signatures firing (target base", round(H.y42.mean(), 3), ")")
print(H.groupby("nf").agg(**agg).round(3).to_string())
H["bp"] = pd.cut(H.best, [-1, 0, .25, .30, .35, 1], labels=["none", "<25", "25-30", "30-35", "35+"])
print("\nHOLDOUT by best confirmed signature p_conf"); print(H.groupby("bp").agg(**agg).round(3).to_string())
# family view: most common leg among confirmed, and the top signatures with deep samples
from collections import Counter
legs = Counter(l for s in Cf.signature for l in s.split(" & "))
print("\nmost common legs across confirmed signatures:"); [print(f"  {v:>4}  {plain(k)}") for k, v in legs.most_common(14)]
T = Cf[(Cf.n_conf >= 500) & (Cf.n_hold >= 300)].sort_values("p_hold", ascending=False).head(15)
print("\ntop by HOLDOUT rate among deep-sample confirmed (n_conf>=500, n_hold>=300):")
for _, r in T.iterrows(): print(f"  disc {r.p_disc:.2f} ({int(r.n_disc)})  conf {r.p_conf:.2f} ({int(r.n_conf)})  hold {r.p_hold:.2f} ({int(r.n_hold)})  | " + " AND ".join(plain(c) for c in r.signature.split(" & ")))
M = L[(L.ticker == "MSI") & (L.date >= "2024-01-01") & (L.date <= "2024-12-31")]
print("\nMSI 2024 (holdout), monthly:"); print(M.groupby(M.date.dt.to_period("M")).agg(days=("n_fire", "size"), mean_fire=("n_fire", "mean"), days_firing=("n_fire", lambda s: (s > 0).sum()), best=("best", "max"), hit42=("y42", "mean")).round(2).to_string())
L[["date", "ticker", "n_fire", "best"]].to_parquet(S + "steady_fire_hist.parquet"); Cf.to_csv(S + "steady_confirmed.csv", index=False)
