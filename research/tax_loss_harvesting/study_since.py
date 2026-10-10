"""Focused study: every table the report needs, restricted to formation years >= START.

usage: python study_since.py DATA OUT START
OUT must already hold the full-run outputs (q4_panel.parquet, monthly_spreads.csv, volume_footprint.csv,
trough_timing.csv, robustness_yearly.csv, misc.json). Writes OUT/since_<START>.json.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm

sys.path.insert(0, str(Path(__file__).parent))
from tlh_core import Data, max_dd, risk_stats, tstat  # noqa: E402

D = Data(Path(sys.argv[1]))
OUT = Path(sys.argv[2])
START = int(sys.argv[3])
LAST_Q4 = 2025
YEARS = list(range(START, LAST_Q4 + 1))
SPY = D.bret["SPY"]
B = {}


def groups(sig):
    r12, r1 = sig["R12"], sig["R1"]
    q = pd.qcut(r12, 5, labels=False) + 1
    dvt = pd.qcut(sig["DV"], 3, labels=False) + 1
    g = {"NEG_1Y": r12 < 0, "POS_1Y": r12 >= 0, "NEG_1M": r1 < 0, "POS_1M": r1 >= 0,
         "NEG1Y_NEG1M": (r12 < 0) & (r1 < 0), "NEG1Y_POS1M": (r12 < 0) & (r1 >= 0),
         "POS1Y_NEG1M": (r12 >= 0) & (r1 < 0), "POS1Y_POS1M": (r12 >= 0) & (r1 >= 0), "ALL": r12.notna()}
    for k in range(1, 6):
        g[f"Q{k}_1Y"] = q == k
    for k, nm in zip((1, 3), ("SMALL", "LARGE")):
        g[f"NEG_1Y_{nm}"] = (r12 < 0) & (dvt == k)
        g[f"POS_1Y_{nm}"] = (r12 >= 0) & (dvt == k)
    return {k: sig.index[v] for k, v in g.items()}


def pooled(paths, yearly, key, rows_extra=None):
    """Pooled daily stats + per-year summary for one (window, group)."""
    daily = pd.concat(paths)
    st = risk_stats(daily, D.rf, SPY)
    yr = pd.Series(yearly)
    st.update({"mean": yr.mean(), "median": yr.median(), "t": tstat(yr), "pct_pos": (yr > 0).mean(),
               "worst": yr.min(), "best": yr.max(), "years": len(yr)})
    if rows_extra:
        st.update(rows_extra)
    return st


# ------------------------------------------------ calendar quarters (Q1-Q3 through 2026, Q4 through 2025)
qrows, qyear = [], []
for qn in ("Q1", "Q2", "Q3", "Q4"):
    yrs = range(START, (LAST_Q4 if qn == "Q4" else 2026) + 1)
    P, Y, N, DD = {}, {}, {}, {}
    for y in yrs:
        t0, t1 = {"Q1": (D.me(y - 1, 12), D.me(y, 3)), "Q2": (D.me(y, 3), D.me(y, 6)),
                  "Q3": (D.me(y, 6), D.me(y, 9)), "Q4": (D.me(y, 9), D.me(y, 12))}[qn]
        G = groups(D.signals(t0))
        G["SPY"] = None
        for g, mem in G.items():
            p = SPY.loc[t0:t1].iloc[1:] if g == "SPY" else D.bh_path(mem, t0, t1)
            P.setdefault(g, []).append(p)
            r = (1 + p).prod() - 1
            Y.setdefault(g, {})[y] = r
            N.setdefault(g, []).append(1 if g == "SPY" else len(mem))
            DD.setdefault(g, []).append(max_dd(p))
            qyear.append({"quarter": qn, "group": g, "year": y, "ret": r})
    spy = pd.Series(Y["SPY"])
    for g in P:
        yr = pd.Series(Y[g])
        qrows.append({"quarter": qn, "group": g, **pooled(P[g], Y[g], g),
                      "hit_vs_spy": (yr > spy).mean(), "avg_mdd": float(np.mean(DD[g])), "avg_n": float(np.mean(N[g])),
                      "first": min(yrs), "last": max(yrs)})
    print("quarter", qn, flush=True)
B["quarters"] = qrows
B["quarter_years"] = qyear

# ------------------------------------------------ Sep-30 sort: sub-windows, Jan, cumulative paths
WIN = {"Oct1_15": ("09-30", "10-15"), "Oct15_Dec15": ("10-15", "12-15"), "Oct15_Dec31": ("10-15", "12-31"),
       "Q4": ("09-30", "12-31"), "Jan": ("12-31", "01-31")}
wP, wY = {}, {}
cum = []
for y in YEARS:
    t0 = D.me(y, 9)
    G = groups(D.signals(t0))
    for wn, (a, b) in WIN.items():
        d0 = t0 if a == "09-30" else D.me(y, 12) if a == "12-31" else D.td_on_or_before(f"{y}-{a}")
        d1 = D.me(y, 12) if b == "12-31" else D.me(y + 1, 1) if b == "01-31" else D.td_on_or_before(f"{y}-{b}")
        for g, mem in list(G.items()) + [("SPY", None)]:
            p = SPY.loc[d0:d1].iloc[1:] if g == "SPY" else D.bh_path(mem, d0, d1)
            wP.setdefault((wn, g), []).append(p)
            wY.setdefault((wn, g), {})[y] = (1 + p).prod() - 1
    t1 = D.me(y + 1, 1)
    for a, b in [("NEG_1Y", "POS_1Y"), ("NEG1Y_NEG1M", "POS1Y_POS1M")]:
        pa, pb = D.bh_path(G[a], t0, t1), D.bh_path(G[b], t0, t1)
        rel = (1 + pa).cumprod() / (1 + pb).cumprod() - 1
        days = (rel.index - pd.Timestamp(f"{y}-09-30")).days
        s = pd.Series(rel.values, index=days)
        s = s[~s.index.duplicated()].reindex(np.arange(1, 124), method="ffill").fillna(0)
        cum += [{"pair": f"{a}-{b}", "year": y, "day": int(d), "rel": float(v)} for d, v in s.items()]
    print("year", y, flush=True)
wrows = []
for (wn, g), ps in wP.items():
    spy = pd.Series(wY[(wn, "SPY")])
    yr = pd.Series(wY[(wn, g)])
    wrows.append({"window": wn, "group": g, **pooled(ps, wY[(wn, g)], g), "hit_vs_spy": (yr > spy).mean()})
B["windows"] = wrows
B["window_years"] = [{"window": wn, "group": g, "year": y, "ret": r} for (wn, g), d in wY.items() for y, r in d.items()]
B["cum_paths"] = cum

PAIRS = [("NEG1Y_NEG1M", "POS1Y_POS1M"), ("NEG_1Y", "POS_1Y"), ("NEG_1M", "POS_1M"), ("Q1_1Y", "Q5_1Y")]
sp = []
for wn in WIN:
    for a, b in PAIRS:
        s = pd.Series(wY[(wn, a)]) - pd.Series(wY[(wn, b)])
        sx = s.drop(2020, errors="ignore")
        sp.append({"window": wn, "pair": f"{a}-{b}", "mean": s.mean(), "median": s.median(), "t": tstat(s),
                   "hit": (s > 0).mean(), "worst": s.min(), "best": s.max(), "mean_ex2020": sx.mean(),
                   "t_ex2020": tstat(sx), **{f"y{y}": s.get(y) for y in YEARS}})
B["spreads"] = sp

# ------------------------------------------------ half-month profile (panel)
p = pd.read_parquet(OUT / "q4_panel.parquet")
p = p[p.year >= START]
HW = ["Oct1_15", "Oct16_31", "Nov1_15", "Nov16_30", "Dec1_15", "Dec16_31", "Jan1_15", "Jan16_31"]
HP = {"NEG1Y-POS1Y": (p.R12 < 0, p.R12 >= 0), "NN-PP": ((p.R12 < 0) & (p.R1 < 0), (p.R12 >= 0) & (p.R1 >= 0)),
      "NEG1M-POS1M": (p.R1 < 0, p.R1 >= 0), "Q1-Q5": (p.R12_q == 1, p.R12_q == 5)}
hm = []
for name, (a, b) in HP.items():
    for w in HW:
        s = p[a].groupby("year")[w].mean() - p[b].groupby("year")[w].mean()
        hm.append({"pair": name, "window": w, "mean": s.mean(), "t": tstat(s), "hit": (s > 0).mean(),
                   **{f"y{y}": s.get(y) for y in YEARS}})
B["halfmonth"] = hm
yw = {}
for w in ["Dec1_15", "Dec16_31", "Jan"]:
    s = p[p.R12 < 0].groupby("year")[w].mean() - p[p.R12 >= 0].groupby("year")[w].mean()
    yw[w] = s.to_dict()
B["panel_years"] = {k: {str(y): v for y, v in d.items()} for k, d in yw.items()}

# ------------------------------------------------ calendar-month alpha, Jan START .. Sep 2026
m = pd.read_csv(OUT / "monthly_spreads.csv")
m = m[(m.fwd_year >= START) & ~((m.fwd_year == 2026) & (m.fwd_month >= 10))].reset_index(drop=True)
ma = []
for col, lab in [("s12", "NEG1Y-POS1Y"), ("s1", "NEG1M-POS1M"), ("s_nn_pp", "NN-PP")]:
    X = pd.concat([pd.get_dummies(m.fwd_month, prefix="m").astype(float), m[["spy"]]], axis=1)
    r = sm.OLS(m[col], X).fit(cov_type="HC1")
    for k in range(12):
        sub = m[m.fwd_month == k + 1][col]
        ma.append({"pair": lab, "month": k + 1, "alpha": r.params.iloc[k], "t": r.tvalues.iloc[k],
                   "raw": sub.mean(), "hit": (sub > 0).mean(), "n": len(sub), "beta": r.params["spy"]})
B["monthly_alpha"] = ma
B["monthly_span"] = [f"{m.fwd_year.min()}-{m.fwd_month.iloc[0]:02d}", f"{m.fwd_year.max()}-{m.fwd_month.iloc[-1]:02d}"]

# ------------------------------------------------ Dec-15 re-sort rebound
rP, rY = {}, {}
for y in YEARS:
    t0 = D.td_on_or_before(f"{y}-12-15")
    sig = pd.DataFrame({"R12": D.window_ret(D.td_on_or_before(f"{y - 1}-12-15"), t0), "DV": D.dv63.loc[t0]})
    sig = sig[sig.R12.notna() & (sig.DV >= 1e6)]
    q = pd.qcut(sig.R12, 5, labels=False) + 1
    jan = D.me(y + 1, 1)
    G = {"DEEP_LT_-30": sig.index[sig.R12 <= -0.30], "Q1_1Y": sig.index[q == 1], "NEG_1Y": sig.index[sig.R12 < 0],
         "ALL": sig.index, "POS_1Y": sig.index[sig.R12 >= 0], "Q5_1Y": sig.index[q == 5]}
    for g, mem in list(G.items()) + [("SPY", None)]:
        pth = SPY.loc[t0:jan].iloc[1:] if g == "SPY" else D.bh_path(mem, t0, jan)
        rP.setdefault(g, []).append(pth)
        rY.setdefault(g, {})[y] = (1 + pth).prod() - 1
B["rebound"] = [{"group": g, **pooled(rP[g], rY[g], g), "hit_vs_spy": (pd.Series(rY[g]) > pd.Series(rY["SPY"])).mean()}
                for g in rP]
B["rebound_years"] = [{"group": g, "year": y, "ret": r} for g, d in rY.items() for y, r in d.items()]

# ------------------------------------------------ volume, trough, robustness
vf = pd.read_csv(OUT / "volume_footprint.csv")
B["volume"] = vf[vf.year >= START].to_dict("records")
B["trough"] = pd.read_csv(OUT / "trough_timing.csv").query("year >= @START").to_dict("records")
rb = pd.read_csv(OUT / "robustness_yearly.csv")
rb = rb[rb.year >= START]
rr = []
for (fl, so, wn), d in rb.groupby(["floor", "sort", "window"]):
    for ex in (False, True):
        dd = d[d.year != 2020] if ex else d
        rec = {"floor": fl, "sort": so, "window": wn, "ex2020": ex}
        for c in ["nn_pp", "n_p", "nn_pp_med", "n_p_med"]:
            rec.update({c: dd[c].mean(), f"{c}_t": tstat(dd[c]), f"{c}_hit": (dd[c] > 0).mean()})
        rr.append(rec)
B["robust"] = rr
B["live"] = json.loads((OUT / "misc.json").read_text())["live_2026"]
B["start"], B["last_q4"] = START, LAST_Q4
B["n_stock_years"] = int(len(p))

(OUT / f"since_{START}.json").write_text(json.dumps(B, default=lambda o: None if o is None else float(o)))
print("wrote", OUT / f"since_{START}.json")
