"""Rank the Q4 strategies by signal strength within each market-cap tier, formation years >= START.

Market cap at each Sep 30 is estimated as today's market cap (uploaded list) scaled by the stock's price change
since that date, so share-count changes are ignored. Tiers: mega >= $200B, large $10-200B, mid $2-10B, small < $2B.
Long/short strategies are scored on the yearly long-minus-short return; long-only strategies on the yearly basket
return minus SPY. A year counts only when each side holds at least MIN_N names.

usage: python rank_strategies.py DATA UNIVERSE_CSV OUT START
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from tlh_core import ANN, Data, risk_stats, tstat  # noqa: E402

D = Data(Path(sys.argv[1]))
U = pd.read_csv(sys.argv[2])
OUT = Path(sys.argv[3])
START = int(sys.argv[4])
YEARS = list(range(START, 2026))
SPY = D.bret["SPY"]
MIN_N = 3
TIERS = ["All", "Mega", "Large", "Mid", "Small"]
U["t"] = U.Ticker.str.replace(".", "-", regex=False)
MCAP_NOW = U.set_index("t")["Market Cap"] * 1e6
LAST = D.close.ffill().iloc[-1]


def tier_of(idx, t0):
    est = MCAP_NOW.reindex(idx) * D.close.loc[t0, idx] / LAST.reindex(idx)
    return pd.cut(est, [0, 2e9, 10e9, 200e9, np.inf], labels=["Small", "Mid", "Large", "Mega"])


# (key, label, window, long rule, short rule or "SPY")
STRATS = [
    ("sep_ls_dec31", "Long Sept losers / short Sept winners, Oct 15 → Dec 31", "Oct15_Dec31", "M1N", "M1P"),
    ("sep_ls_q4", "Long Sept losers / short Sept winners, full Q4", "Q4", "M1N", "M1P"),
    ("sep_ls_dec15", "Long Sept losers / short Sept winners, Oct 15 → Dec 15", "Oct15_Dec15", "M1N", "M1P"),
    ("nn_pp", "Long double losers / short double winners, Oct 15 → Dec 15", "Oct15_Dec15", "NN", "PP"),
    ("oct_rev", "Long 1Y winners / short 1Y losers, Oct 1–15", "Oct1_15", "Y1P", "Y1N"),
    ("long_a", "Long basket A (1Y worst 20–40%, Sept down), Oct 15 → Dec 15", "Oct15_Dec15", "A", "SPY"),
    ("long_b", "Long basket B (1Y worst 40%, Sept below −10%), Oct 15 → Dec 15", "Oct15_Dec15", "B", "SPY"),
    ("long_sep", "Long Sept losers, Oct 15 → Dec 15", "Oct15_Dec15", "M1N", "SPY"),
    ("long_nn", "Long double losers, Oct 15 → Dec 15", "Oct15_Dec15", "NN", "SPY"),
    ("long_np", "Long 1Y losers with Sept up, Oct 15 → Dec 15", "Oct15_Dec15", "NP", "SPY"),
    ("long_jan", "Long 1Y losers, Dec 15 → Jan 31", "Dec15_Jan31", "J1N", "SPY"),
]


def masks(sig):
    r12, r1 = sig.R12, sig.R1
    q = pd.qcut(r12, 5, labels=False) + 1
    return {"M1N": r1 < 0, "M1P": r1 >= 0, "NN": (r12 < 0) & (r1 < 0), "PP": (r12 >= 0) & (r1 >= 0),
            "NP": (r12 < 0) & (r1 >= 0), "Y1N": r12 < 0, "Y1P": r12 >= 0, "A": (q == 2) & (r1 < 0),
            "B": (q <= 2) & (r1 < -0.10)}


def windows(y):
    t0 = D.me(y, 9)
    o15, d15 = D.td_on_or_before(f"{y}-10-15"), D.td_on_or_before(f"{y}-12-15")
    return {"Oct1_15": (t0, o15), "Oct15_Dec15": (o15, d15), "Oct15_Dec31": (o15, D.me(y, 12)),
            "Q4": (t0, D.me(y, 12)), "Dec15_Jan31": (d15, D.me(y + 1, 1))}


yearly, daily, names = {}, {}, {}
for y in YEARS:
    t0 = D.me(y, 9)
    sig = D.signals(t0)
    tiers = tier_of(sig.index, t0)
    M = masks(sig)
    # Dec-15 re-sort for the January strategy
    d15 = D.td_on_or_before(f"{y}-12-15")
    r12d = D.window_ret(D.td_on_or_before(f"{y - 1}-12-15"), d15)
    jsig = r12d[r12d.notna() & (D.dv63.loc[d15] >= 1e6)]
    jt = tier_of(jsig.index, d15)
    W = windows(y)
    for key, _, wn, lo, sh in STRATS:
        a, b = W[wn]
        spy = SPY.loc[a:b].iloc[1:]
        for tier in TIERS:
            if lo == "J1N":
                pool = jsig.index if tier == "All" else jsig.index[jt == tier]
                long_m = pool[jsig.reindex(pool) < 0]
            else:
                pool = sig.index if tier == "All" else sig.index[tiers == tier]
                long_m = pool[M[lo].reindex(pool)]
            if len(long_m) < MIN_N:
                continue
            pl = D.bh_path(long_m, a, b)
            if sh == "SPY":
                ps, short_n = spy, 1
            else:
                short_m = pool[M[sh].reindex(pool)]
                if len(short_m) < MIN_N:
                    continue
                ps, short_n = D.bh_path(short_m, a, b), len(short_m)
            yearly.setdefault((key, tier), {})[y] = (1 + pl).prod() - (1 + ps).prod()
            daily.setdefault((key, tier), []).append((pl, ps))
            names.setdefault((key, tier), []).append((len(long_m), short_n))
    print("year", y, flush=True)

rows = []
for key, label, wn, lo, sh in STRATS:
    for tier in TIERS:
        if (key, tier) not in yearly:
            continue
        s = pd.Series(yearly[(key, tier)])
        pls = pd.concat([p for p, _ in daily[(key, tier)]])
        diff = pd.concat([p - q for p, q in daily[(key, tier)]])
        long_st = risk_stats(pls, D.rf, SPY)
        n = np.array(names[(key, tier)])
        rows.append({"key": key, "strategy": label, "tier": tier, "kind": "long only" if sh == "SPY" else "long/short",
                     "mean": s.mean(), "median": s.median(), "t": tstat(s), "hit": (s > 0).mean(),
                     "worst": s.min(), "best": s.max(), "years": len(s),
                     "spread_sharpe": diff.mean() / diff.std() * np.sqrt(ANN),
                     "long_sharpe": long_st["sharpe"], "long_alpha": long_st["alpha_spy_ann"],
                     "long_beta": long_st["beta_spy"], "avg_long_n": n[:, 0].mean(),
                     "avg_short_n": n[:, 1].mean() if sh != "SPY" else None,
                     **{f"y{y}": s.get(y) for y in YEARS}})
R = pd.DataFrame(rows)
R.to_csv(OUT / f"strategy_rank_{START}.csv", index=False)
(OUT / f"strategy_rank_{START}.json").write_text(json.dumps(
    {"start": START, "min_n": MIN_N, "rows": json.loads(R.to_json(orient="records"))}))
print("wrote", OUT / f"strategy_rank_{START}.csv")
