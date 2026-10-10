"""Outright long stats for Sep-30-sorted buckets over fixed Q4 sub-windows, by era,
plus average cumulative NEG-POS path (Sep 30 -> Jan 31) by era for charting."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from tlh_core import Data, max_dd, risk_stats, tstat  # noqa: E402

D = Data(Path(sys.argv[1]))
OUT = Path(sys.argv[2])
SPY = D.bret["SPY"]
ERAS = {"1999-2007": range(1999, 2008), "2008-2016": range(2008, 2017),
        "2017-2025": range(2017, 2026), "ALL": range(1999, 2026)}
WIN = {"Oct1_Oct15": ("09-30", "10-15"), "Oct15_Dec15": ("10-15", "12-15"),
       "Oct15_Dec31": ("10-15", "12-31"), "Q4": ("09-30", "12-31")}


def grp(sig):
    r12, r1 = sig.R12, sig.R1
    q = pd.qcut(r12, 5, labels=False) + 1
    return {"NEG_1Y": r12 < 0, "POS_1Y": r12 >= 0, "NEG_1M": r1 < 0, "POS_1M": r1 >= 0,
            "NEG1Y_NEG1M": (r12 < 0) & (r1 < 0), "POS1Y_POS1M": (r12 >= 0) & (r1 >= 0),
            "Q1_1Y": q == 1, "Q5_1Y": q == 5, "ALL": r12.notna()}


paths, yr = {}, []
for y in range(1999, 2026):
    t0 = D.me(y, 9)
    sig = D.signals(t0)
    G = {k: sig.index[v] for k, v in grp(sig).items()}
    for wn, (a, b) in WIN.items():
        d0 = t0 if a == "09-30" else D.td_on_or_before(f"{y}-{a}")
        d1 = D.me(y, 12) if b == "12-31" else D.td_on_or_before(f"{y}-{b}")
        for g, mem in G.items():
            p = D.bh_path(mem, d0, d1)
            paths[(wn, g, y)] = p
            yr.append({"window": wn, "group": g, "year": y, "ret": (1 + p).prod() - 1, "mdd": max_dd(p)})
        sp = SPY.loc[d0:d1].iloc[1:]
        paths[(wn, "SPY", y)] = sp
        yr.append({"window": wn, "group": "SPY", "year": y, "ret": (1 + sp).prod() - 1, "mdd": max_dd(sp)})
yr = pd.DataFrame(yr)
yr.to_csv(OUT / "trade_window_yearly.csv", index=False)

rows = []
for wn in WIN:
    for e, yrs in ERAS.items():
        spy_r = yr[(yr.window == wn) & (yr.group == "SPY")].set_index("year").ret
        for g in ["NEG1Y_NEG1M", "NEG_1Y", "NEG_1M", "Q1_1Y", "Q5_1Y", "POS_1M", "POS_1Y", "POS1Y_POS1M", "ALL", "SPY"]:
            daily = pd.concat([paths[(wn, g, y)] for y in yrs])
            st = risk_stats(daily, D.rf, SPY)
            d = yr[(yr.window == wn) & (yr.group == g) & yr.year.isin(list(yrs))].set_index("year")
            rows.append({"window": wn, "era": e, "group": g, **st, "mean": d.ret.mean(), "median": d.ret.median(),
                         "t": tstat(d.ret), "pct_pos": (d.ret > 0).mean(),
                         "hit_vs_spy": (d.ret > spy_r.reindex(d.index)).mean(),
                         "worst": d.ret.min(), "avg_mdd": d.mdd.mean()})
tw = pd.DataFrame(rows)
tw.to_csv(OUT / "trade_window_risk.csv", index=False)

# spread of NN vs PP and Q1 vs Q5 per window/era
sp = []
for wn in WIN:
    for a, b in [("NEG_1Y", "POS_1Y"), ("NEG1Y_NEG1M", "POS1Y_POS1M"), ("Q1_1Y", "Q5_1Y"), ("NEG_1M", "POS_1M")]:
        ya = yr[(yr.window == wn) & (yr.group == a)].set_index("year").ret
        yb = yr[(yr.window == wn) & (yr.group == b)].set_index("year").ret
        s = ya - yb
        for e, yrs in ERAS.items():
            ss = s.reindex(list(yrs))
            sp.append({"window": wn, "pair": f"{a}-{b}", "era": e, "mean": ss.mean(), "t": tstat(ss),
                       "hit": (ss > 0).mean(), "worst": ss.min(), "best": ss.max()})
pd.DataFrame(sp).to_csv(OUT / "trade_window_spreads.csv", index=False)

# average cumulative raw relative path NEG_1Y / POS_1Y and NN / PP, Sep30 -> Jan31, by trading day
cum = {}
for y in range(1999, 2026):
    t0, t1 = D.me(y, 9), D.me(y + 1, 1)
    sig = D.signals(t0)
    G = {k: sig.index[v] for k, v in grp(sig).items()}
    for a, b in [("NEG_1Y", "POS_1Y"), ("NEG1Y_NEG1M", "POS1Y_POS1M"), ("Q1_1Y", "Q5_1Y")]:
        pa, pb = D.bh_path(G[a], t0, t1), D.bh_path(G[b], t0, t1)
        rel = (1 + pa).cumprod() / (1 + pb).cumprod() - 1
        # map to calendar day-of-season (days since Sep 30) so years align on dates
        cal = (rel.index - pd.Timestamp(f"{y}-09-30")).days
        cum[(f"{a}-{b}", y)] = pd.Series(rel.values, index=cal)
grid = np.arange(1, 124)
recs = []
for (pair, y), s in cum.items():
    s = s[~s.index.duplicated()].reindex(grid, method="ffill").fillna(0)
    for e, yrs in ERAS.items():
        if y in yrs:
            recs.append(pd.DataFrame({"pair": pair, "era": e, "year": y, "day": grid, "rel": s.values}))
cp = pd.concat(recs)
cp.to_csv(OUT / "cum_rel_paths.csv", index=False)
avg = cp.groupby(["pair", "era", "day"]).rel.mean().unstack(["pair", "era"])
avg.to_csv(OUT / "cum_rel_paths_avg.csv")
print(avg.iloc[[14, 30, 45, 60, 76, 92, 107, 122]].round(4).to_string())
