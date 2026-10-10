"""Mega-cap cut of the Q4 study, formation years >= START.

1. Point-in-time top-N universe (by 63-day median dollar volume at each Sep 30): bucket stats and pair spreads.
2. Per-name Q4 history for the largest names in the list.
3. Current (Sep 30, 2026) top-100 table with buckets.

usage: python megacap.py DATA UNIVERSE_CSV OUT START
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from tlh_core import Data, risk_stats, tstat  # noqa: E402

D = Data(Path(sys.argv[1]))
U = pd.read_csv(sys.argv[2])
OUT = Path(sys.argv[3])
START = int(sys.argv[4])
YEARS = list(range(START, 2026))
SPY = D.bret["SPY"]
NAMES = ["NVDA", "AAPL", "GOOGL", "MSFT", "AMZN", "META", "AVGO", "TSLA", "MU", "LLY", "AMD", "BRK-B", "JPM",
         "ORCL", "NFLX", "WMT", "V", "XOM", "COST", "UNH"]


def bucket(r12, r1):
    return ("1Y neg" if r12 < 0 else "1Y pos") + (" + 1M neg" if r1 < 0 else " + 1M pos")


def windows(y):
    t0 = D.me(y, 9)
    return {"Q4": (t0, D.me(y, 12)), "Oct1_15": (t0, D.td_on_or_before(f"{y}-10-15")),
            "Oct15_Dec15": (D.td_on_or_before(f"{y}-10-15"), D.td_on_or_before(f"{y}-12-15")),
            "Oct15_Dec31": (D.td_on_or_before(f"{y}-10-15"), D.me(y, 12)), "Jan": (D.me(y, 12), D.me(y + 1, 1))}


B = {"start": START, "top": {}}
for TOPN in (100, 50):
    P, Y, N = {}, {}, {}
    for y in YEARS:
        sig = D.signals(D.me(y, 9)).nlargest(TOPN, "DV")
        r12, r1 = sig.R12, sig.R1
        G = {"NEG_1Y": r12 < 0, "POS_1Y": r12 >= 0, "NEG_1M": r1 < 0, "POS_1M": r1 >= 0,
             "NEG1Y_NEG1M": (r12 < 0) & (r1 < 0), "NEG1Y_POS1M": (r12 < 0) & (r1 >= 0),
             "POS1Y_NEG1M": (r12 >= 0) & (r1 < 0), "POS1Y_POS1M": (r12 >= 0) & (r1 >= 0), "ALL": r12.notna()}
        G = {k: sig.index[v] for k, v in G.items()}
        for wn, (a, b) in windows(y).items():
            for g, mem in list(G.items()) + [("SPY", None)]:
                if g != "SPY" and len(mem) == 0:
                    continue
                pth = SPY.loc[a:b].iloc[1:] if g == "SPY" else D.bh_path(mem, a, b)
                P.setdefault((wn, g), []).append(pth)
                Y.setdefault((wn, g), {})[y] = (1 + pth).prod() - 1
                N.setdefault((wn, g), []).append(1 if g == "SPY" else len(mem))
    rows = []
    for (wn, g), ps in P.items():
        st = risk_stats(pd.concat(ps), D.rf, SPY)
        yr = pd.Series(Y[(wn, g)])
        spy = pd.Series(Y[(wn, "SPY")]).reindex(yr.index)
        st.update({"window": wn, "group": g, "mean": yr.mean(), "median": yr.median(), "t": tstat(yr),
                   "pct_pos": (yr > 0).mean(), "hit_vs_spy": (yr > spy).mean(), "worst": yr.min(),
                   "avg_n": float(np.mean(N[(wn, g)])), "years": len(yr)})
        rows.append(st)
    sp = []
    for wn in windows(YEARS[0]):
        for a, b in [("NEG1Y_NEG1M", "POS1Y_POS1M"), ("NEG_1Y", "POS_1Y"), ("NEG_1M", "POS_1M")]:
            s = (pd.Series(Y.get((wn, a), {})) - pd.Series(Y.get((wn, b), {}))).dropna()
            sp.append({"window": wn, "pair": f"{a}-{b}", "mean": s.mean(), "median": s.median(), "t": tstat(s),
                       "hit": (s > 0).mean(), "worst": s.min(), "best": s.max(), "years": len(s),
                       **{f"y{y}": s.get(y) for y in YEARS}})
    B["top"][TOPN] = {"buckets": rows, "spreads": sp}
    print("top", TOPN, flush=True)

# per-name Q4 history
hist = []
for t in NAMES:
    if t not in D.close.columns:
        continue
    for y in YEARS:
        sig = D.signals(D.me(y, 9))
        if t not in sig.index:
            continue
        rec = {"ticker": t, "year": y, "r12": sig.loc[t, "R12"], "r1": sig.loc[t, "R1"],
               "bucket": bucket(sig.loc[t, "R12"], sig.loc[t, "R1"]),
               "dv_rank": int(sig.DV.rank(ascending=False).loc[t])}
        for wn, (a, b) in windows(y).items():
            r = D.window_ret(a, b, [t]).iloc[0]
            s = D.bench["SPY"].loc[b] / D.bench["SPY"].loc[a] - 1
            rec[wn], rec[f"{wn}_xs"] = r, r - s
        hist.append(rec)
h = pd.DataFrame(hist)
B["names"] = h.to_dict("records")
agg = h.groupby("bucket").agg(n=("ticker", "size"), q4_xs=("Q4_xs", "mean"), q4_xs_med=("Q4_xs", "median"),
                              q4_beat=("Q4_xs", lambda s: (s > 0).mean()), w_xs=("Oct15_Dec15_xs", "mean"),
                              w_beat=("Oct15_Dec15_xs", lambda s: (s > 0).mean()), oct_xs=("Oct1_15_xs", "mean"),
                              jan_xs=("Jan_xs", "mean")).reset_index()
B["names_by_bucket"] = agg.to_dict("records")
neg = h[h.r12 < 0]
B["names_neg1y"] = {"n": len(neg), "q4_xs": neg.Q4_xs.mean(), "q4_beat": (neg.Q4_xs > 0).mean(),
                    "w_xs": neg.Oct15_Dec15_xs.mean(), "w_beat": (neg.Oct15_Dec15_xs > 0).mean()}
pos = h[h.r12 >= 0]
B["names_pos1y"] = {"n": len(pos), "q4_xs": pos.Q4_xs.mean(), "q4_beat": (pos.Q4_xs > 0).mean(),
                    "w_xs": pos.Oct15_Dec15_xs.mean(), "w_beat": (pos.Oct15_Dec15_xs > 0).mean()}

# current top 100 by dollar volume
sc = pd.read_csv(OUT / "screen_2026.csv", index_col=0)
sc = sc.nlargest(100, "DV").copy()
u = U.assign(t=U.Ticker.str.replace(".", "-", regex=False)).set_index("t")
sc = sc.join(u[["Company", "Sector", "Market Cap"]])
sc["bucket"] = [bucket(a, b) for a, b in zip(sc.R12, sc.R1)]
sc.index.name = "ticker"
B["top100_now"] = sc.reset_index()[["ticker", "Company", "Sector", "Market Cap", "DV", "R12", "R1", "QTD",
                                    "bucket"]].to_dict("records")

(OUT / f"megacap_{START}.json").write_text(json.dumps(B, default=float))
print("wrote", OUT / f"megacap_{START}.json")
