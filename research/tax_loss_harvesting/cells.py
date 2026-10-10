"""Finer cells of the Sep-30 sort (formation years >= START): 1Y quintile x 1M sign, and the 1Y-neg + 1M-neg
bucket split by liquidity, 1Y depth and September depth. Pooled daily risk stats per window.

usage: python cells.py DATA OUT START
"""
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from tlh_core import Data, risk_stats, tstat  # noqa: E402

D = Data(Path(sys.argv[1]))
OUT = Path(sys.argv[2])
START = int(sys.argv[3])
YEARS = list(range(START, 2026))
SPY = D.bret["SPY"]


def cells(sig):
    r12, r1 = sig.R12, sig.R1
    q = pd.qcut(r12, 5, labels=False) + 1
    dvt = pd.qcut(sig.DV, 3, labels=["low", "mid", "high"])
    nn = (r12 < 0) & (r1 < 0)
    c = {}
    for k in range(1, 6):
        c[f"1Y Q{k} + 1M neg"] = (q == k) & (r1 < 0)
        c[f"1Y Q{k} + 1M pos"] = (q == k) & (r1 >= 0)
    for lv in ["low", "mid", "high"]:
        c[f"NN, {lv} liquidity"] = nn & (dvt == lv)
    c["NN, 1Y 0 to -15%"] = nn & (r12 >= -0.15)
    c["NN, 1Y -15 to -35%"] = nn & (r12 < -0.15) & (r12 >= -0.35)
    c["NN, 1Y below -35%"] = nn & (r12 < -0.35)
    c["NN, Sep 0 to -5%"] = nn & (r1 >= -0.05)
    c["NN, Sep -5 to -10%"] = nn & (r1 < -0.05) & (r1 >= -0.10)
    c["NN, Sep below -10%"] = nn & (r1 < -0.10)
    c["1Y Q1-Q2 + Sep below -10%"] = (q <= 2) & (r1 < -0.10)
    c["1Y Q1-Q2 + Sep below -10%, mid/high liquidity"] = (q <= 2) & (r1 < -0.10) & (dvt != "low")
    c["1Y Q2 + 1M neg, mid/high liquidity"] = (q == 2) & (r1 < 0) & (dvt != "low")
    c["NN (all)"] = nn
    c["1M neg (all)"] = r1 < 0
    c["PP (all)"] = (r12 >= 0) & (r1 >= 0)
    return {k: sig.index[v] for k, v in c.items()}


WIN = {"Oct15_Dec15": ("10-15", "12-15"), "Oct15_Dec31": ("10-15", "12-31"), "Q4": ("09-30", "12-31")}
P, Y, N = {}, {}, {}
for y in YEARS:
    t0 = D.me(y, 9)
    C = cells(D.signals(t0))
    for wn, (a, b) in WIN.items():
        d0 = t0 if a == "09-30" else D.td_on_or_before(f"{y}-{a}")
        d1 = D.me(y, 12) if b == "12-31" else D.td_on_or_before(f"{y}-{b}")
        for k, mem in list(C.items()) + [("SPY", None)]:
            if k != "SPY" and len(mem) < 5:
                continue
            p = SPY.loc[d0:d1].iloc[1:] if k == "SPY" else D.bh_path(mem, d0, d1)
            P.setdefault((wn, k), []).append(p)
            Y.setdefault((wn, k), {})[y] = (1 + p).prod() - 1
            N.setdefault((wn, k), []).append(1 if k == "SPY" else len(mem))
rows = []
for (wn, k), ps in P.items():
    st = risk_stats(pd.concat(ps), D.rf, SPY)
    yr = pd.Series(Y[(wn, k)])
    spy = pd.Series(Y[(wn, "SPY")]).reindex(yr.index)
    rows.append({"window": wn, "cell": k, **st, "mean": yr.mean(), "median": yr.median(), "t": tstat(yr),
                 "up": int((yr > 0).sum()), "beat_spy": int((yr > spy).sum()), "years": len(yr),
                 "worst": yr.min(), "worst_vs_spy": (yr - spy).min(), "avg_n": sum(N[(wn, k)]) / len(N[(wn, k)])})
df = pd.DataFrame(rows)
df.to_csv(OUT / f"cells_{START}.csv", index=False)

# current membership (Sep 30, 2026 sort)
sig = D.signals(D.me(2026, 9))
C = cells(sig)
cur = [{"ticker": t, "cell": k} for k, mem in C.items() for t in mem]
pd.DataFrame(cur).to_csv(OUT / "cells_2026_members.csv", index=False)
print(json.dumps({k: len(v) for k, v in C.items()}))
