"""Long-only scan: every basket x market-cap tier x window, scored on yearly excess return vs SPY and on daily
alpha vs SPY (HAC t). Formation years >= START. Market cap at each Sep 30 estimated as in rank_strategies.py.

usage: python long_scan.py DATA UNIVERSE_CSV OUT START
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm

sys.path.insert(0, str(Path(__file__).parent))
from tlh_core import Data, tstat  # noqa: E402

D = Data(Path(sys.argv[1]))
U = pd.read_csv(sys.argv[2])
OUT = Path(sys.argv[3])
START = int(sys.argv[4])
YEARS = list(range(START, 2026))
SPY = D.bret["SPY"]
U["t"] = U.Ticker.str.replace(".", "-", regex=False)
MCAP_NOW = U.set_index("t")["Market Cap"] * 1e6
LAST = D.close.ffill().iloc[-1]
TIERS = ["All", "Large", "Mid", "Small", "Large+Mid"]


def tier_of(idx, t0):
    est = MCAP_NOW.reindex(idx) * D.close.loc[t0, idx] / LAST.reindex(idx)
    return pd.cut(est, [0, 2e9, 10e9, 200e9, np.inf], labels=["Small", "Mid", "Large", "Mega"])


def baskets(sig):
    r12, r1 = sig.R12, sig.R1
    q = pd.qcut(r12, 5, labels=False) + 1
    return {"Sept losers": r1 < 0, "Sept down >10%": r1 < -0.10, "Double losers (1Y & Sept down)": (r12 < 0) & (r1 < 0),
            "Double losers, Sept down >10%": (r12 < 0) & (r1 < -0.10),
            "A: 1Y worst 20-40%, Sept down": (q == 2) & (r1 < 0),
            "B: 1Y worst 40%, Sept down >10%": (q <= 2) & (r1 < -0.10),
            "1Y worst 20%, Sept down": (q == 1) & (r1 < 0), "1Y worst 20-60%, Sept down": (q.between(2, 3)) & (r1 < 0),
            "Double winners (ref)": (r12 >= 0) & (r1 >= 0)}


def win(y):
    t0 = D.me(y, 9)
    o15 = D.td_on_or_before(f"{y}-10-15")
    return {"Oct 15-Dec 15": (o15, D.td_on_or_before(f"{y}-12-15")), "Oct 15-Dec 31": (o15, D.me(y, 12)),
            "Full Q4": (t0, D.me(y, 12))}


Y, P, N = {}, {}, {}
for y in YEARS:
    t0 = D.me(y, 9)
    sig = D.signals(t0)
    tr = tier_of(sig.index, t0)
    Bk = baskets(sig)
    for wn, (a, b) in win(y).items():
        spy = SPY.loc[a:b].iloc[1:]
        for bk, m in Bk.items():
            for tier in TIERS:
                sel = m if tier == "All" else m & (tr.isin(["Large", "Mid"]) if tier == "Large+Mid" else tr == tier)
                mem = sig.index[sel.fillna(False)]
                if len(mem) < 10:
                    continue
                p = D.bh_path(mem, a, b)
                Y.setdefault((bk, tier, wn), {})[y] = (1 + p).prod() - (1 + spy).prod()
                P.setdefault((bk, tier, wn), []).append(p)
                N.setdefault((bk, tier, wn), []).append(len(mem))
    print("year", y, flush=True)
rows = []
for k, d in Y.items():
    s = pd.Series(d)
    if len(s) < 8:
        continue
    daily = pd.concat(P[k])
    X = sm.add_constant(SPY.reindex(daily.index))
    r = sm.OLS(daily - D.rf.reindex(daily.index), X, missing="drop").fit(cov_type="HAC", cov_kwds={"maxlags": 5})
    sharpe = (daily - D.rf.reindex(daily.index)).mean() / daily.std() * np.sqrt(252)
    rows.append({"basket": k[0], "tier": k[1], "window": k[2], "excess_mean": s.mean(), "excess_median": s.median(),
                 "t_excess": tstat(s), "beat": int((s > 0).sum()), "years": len(s), "worst_vs_spy": s.min(),
                 "alpha_ann": r.params["const"] * 252, "t_alpha": r.tvalues["const"], "beta": r.params["SPY"],
                 "sharpe": sharpe, "avg_n": float(np.mean(N[k]))})
R = pd.DataFrame(rows)
R.to_csv(OUT / f"long_scan_{START}.csv", index=False)
print(R.shape)
