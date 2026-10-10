"""Full Q4 tax-loss-harvesting study. Writes CSV/JSON results for the report."""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm

sys.path.insert(0, str(Path(__file__).parent))
from tlh_core import ANN, Data, max_dd, risk_stats, tstat  # noqa: E402

D = Data(Path(sys.argv[1]))
OUT = Path(sys.argv[2])
OUT.mkdir(parents=True, exist_ok=True)
YEARS = list(range(1999, 2026))
ERAS = {"1999-2007": range(1999, 2008), "2008-2016": range(2008, 2017), "2017-2025": range(2017, 2026)}
SPY = D.bret["SPY"]
RES = {}


def groups(sig):
    r12, r1 = sig["R12"], sig["R1"]
    q = pd.qcut(r12, 5, labels=False) + 1
    dvt = pd.qcut(sig["DV"], 3, labels=False) + 1
    g = {
        "NEG_1Y": r12 < 0, "POS_1Y": r12 >= 0,
        "NEG_1M": r1 < 0, "POS_1M": r1 >= 0,
        "NEG1Y_NEG1M": (r12 < 0) & (r1 < 0), "NEG1Y_POS1M": (r12 < 0) & (r1 >= 0),
        "POS1Y_NEG1M": (r12 >= 0) & (r1 < 0), "POS1Y_POS1M": (r12 >= 0) & (r1 >= 0),
        "ALL": r12.notna(),
    }
    for k in range(1, 6):
        g[f"Q{k}_1Y"] = q == k
    for k, nm in zip((1, 2, 3), ("SMALL", "MID", "LARGE")):
        g[f"NEG_1Y_{nm}"] = (r12 < 0) & (dvt == k)
        g[f"POS_1Y_{nm}"] = (r12 >= 0) & (dvt == k)
    return {k: sig.index[v] for k, v in g.items()}


# ---------------------------------------------------------------- A. quarter paths
QSPEC = {"Q1": (12, 3), "Q2": (3, 6), "Q3": (6, 9), "Q4": (9, 12)}
paths = {}      # (quarter, group, year) -> daily series
qret = []       # per-year quarter returns
for qn, (fm, em) in QSPEC.items():
    for y in YEARS:
        if qn == "Q1":
            t0, t1 = D.me(y, 12), D.me(y + 1, 3)   # Q1 of y+1, formed Dec y
        else:
            t0, t1 = D.me(y, fm), D.me(y, em)
        sig = D.signals(t0) if t1 is not None else None
        if sig is None:
            continue
        for gname, mem in groups(sig).items():
            if len(mem) < 10:
                continue
            p = D.bh_path(mem, t0, t1)
            paths[(qn, gname, y)] = p
            qret.append({"q": qn, "group": gname, "year": y, "n": len(mem),
                         "ret": (1 + p).prod() - 1, "mdd": max_dd(p)})
        sp = SPY.loc[t0:t1].iloc[1:]
        paths[(qn, "SPY", y)] = sp
        qret.append({"q": qn, "group": "SPY", "year": y, "n": 1,
                     "ret": (1 + sp).prod() - 1, "mdd": max_dd(sp)})
    print("paths", qn, flush=True)
qret = pd.DataFrame(qret)
qret.to_csv(OUT / "quarter_returns_by_year.csv", index=False)


def summarize(qn, gname, years=YEARS):
    ps = [paths[(qn, gname, y)] for y in years if (qn, gname, y) in paths]
    if not ps:
        return None
    daily = pd.concat(ps)
    st = risk_stats(daily, D.rf, SPY)
    yr = qret[(qret.q == qn) & (qret.group == gname) & (qret.year.isin(list(years)))].set_index("year")
    spy = qret[(qret.q == qn) & (qret.group == "SPY")].set_index("year")["ret"].reindex(yr.index)
    st.update({
        "q_mean": yr.ret.mean(), "q_median": yr.ret.median(), "q_t": tstat(yr.ret),
        "q_worst": yr.ret.min(), "q_best": yr.ret.max(),
        "hit_vs_spy": (yr.ret > spy).mean(), "pct_pos": (yr.ret > 0).mean(),
        "avg_mdd": yr.mdd.mean(), "avg_n": yr.n.mean(), "years": len(yr),
    })
    return st


GROUP_LIST = ["NEG_1Y", "POS_1Y", "NEG_1M", "POS_1M", "NEG1Y_NEG1M", "NEG1Y_POS1M",
              "POS1Y_NEG1M", "POS1Y_POS1M", "Q1_1Y", "Q2_1Y", "Q3_1Y", "Q4_1Y", "Q5_1Y",
              "NEG_1Y_SMALL", "POS_1Y_SMALL", "NEG_1Y_MID", "POS_1Y_MID",
              "NEG_1Y_LARGE", "POS_1Y_LARGE", "ALL", "SPY"]
rows = []
for qn in QSPEC:
    for g in GROUP_LIST:
        s = summarize(qn, g)
        if s:
            rows.append({"quarter": qn, "group": g, **s})
risk = pd.DataFrame(rows)
risk.to_csv(OUT / "risk_by_quarter.csv", index=False)

# era breakdown, Q4 and other quarters
rows = []
for era, yrs in ERAS.items():
    for qn in QSPEC:
        for g in ["NEG_1Y", "POS_1Y", "NEG_1M", "POS_1M", "NEG1Y_NEG1M", "POS1Y_POS1M",
                  "Q1_1Y", "Q5_1Y", "ALL", "SPY"]:
            s = summarize(qn, g, yrs)
            if s:
                rows.append({"era": era, "quarter": qn, "group": g, **s})
pd.DataFrame(rows).to_csv(OUT / "risk_by_era.csv", index=False)


# long/short spreads (daily) with beta-hedged alpha, per quarter and era
def ls_daily(qn, a, b, years):
    out = []
    for y in years:
        if (qn, a, y) in paths and (qn, b, y) in paths:
            out.append(paths[(qn, a, y)] - paths[(qn, b, y)])
    return pd.concat(out) if out else pd.Series(dtype=float)


rows = []
for pair in [("NEG_1Y", "POS_1Y"), ("NEG_1M", "POS_1M"), ("NEG1Y_NEG1M", "POS1Y_POS1M"),
             ("Q1_1Y", "Q5_1Y"), ("NEG_1Y_SMALL", "POS_1Y_SMALL"), ("NEG_1Y_LARGE", "POS_1Y_LARGE")]:
    for era, yrs in {"ALL": YEARS, **ERAS}.items():
        for qn in QSPEC:
            ls = ls_daily(qn, *pair, yrs)
            X = sm.add_constant(SPY.reindex(ls.index))
            r = sm.OLS(ls, X, missing="drop").fit(cov_type="HAC", cov_kwds={"maxlags": 5})
            yr = (qret[(qret.q == qn) & (qret.group == pair[0]) & qret.year.isin(list(yrs))].set_index("year").ret
                  - qret[(qret.q == qn) & (qret.group == pair[1]) & qret.year.isin(list(yrs))].set_index("year").ret)
            rows.append({"long": pair[0], "short": pair[1], "era": era, "quarter": qn,
                         "spread_mean": yr.mean(), "spread_t": tstat(yr), "hit": (yr > 0).mean(),
                         "ls_sharpe": ls.mean() / ls.std() * np.sqrt(ANN),
                         "alpha_q": r.params["const"] * 63, "alpha_t": r.tvalues["const"],
                         "beta": r.params["SPY"]})
pd.DataFrame(rows).to_csv(OUT / "long_short.csv", index=False)
print("A done", flush=True)

# ---------------------------------------------------------------- B. Sep-30 sort, daily path to Jan 31
# cumulative relative path NEG_1Y vs POS_1Y from Sep 30 through Jan 31 (beta-hedged)
cum_paths, trough = {}, []
for y in YEARS:
    t0, t1 = D.me(y, 9), D.me(y + 1, 1)
    sig = D.signals(t0)
    g = groups(sig)
    pn, pp = D.bh_path(g["NEG_1Y"], t0, t1), D.bh_path(g["POS_1Y"], t0, t1)
    ls = (pn - pp)
    cum_paths[y] = pd.DataFrame({"ls": ls, "spy": SPY.reindex(ls.index), "neg": pn, "pos": pp})
allls = pd.concat([v for v in cum_paths.values()])
BETA_LS = np.cov(allls.ls, allls.spy)[0, 1] / allls.spy.var()
avg_path = {}
for y, df in cum_paths.items():
    h = df.ls - BETA_LS * df.spy
    cum = (1 + h).cumprod() - 1
    raw = (1 + df.neg).cumprod() / (1 + df.pos).cumprod() - 1
    dates = df.index
    # trough within Oct 1 - Jan 31 on hedged path
    i_min = int(np.argmin(cum.values))
    trough.append({"year": y, "trough_date": dates[i_min].strftime("%Y-%m-%d"),
                   "trough_day": i_min + 1, "trough_val": cum.iloc[i_min],
                   "trough_month_day": dates[i_min].strftime("%m-%d"),
                   "end_val": cum.iloc[-1], "rebound_from_trough": (1 + cum.iloc[-1]) / (1 + cum.iloc[i_min]) - 1,
                   "raw_end": raw.iloc[-1]})
    avg_path[y] = pd.Series(cum.values, index=range(1, len(cum) + 1))
trough = pd.DataFrame(trough)
trough.to_csv(OUT / "trough_timing.csv", index=False)
ap = pd.DataFrame(avg_path)
ap.to_csv(OUT / "hedged_ls_paths_by_year.csv")
RES["beta_ls_sep_jan"] = BETA_LS
print("B done", flush=True)

# ---------------------------------------------------------------- C. per-year window spreads (Sep-30 sort), hedged
panel = pd.read_parquet(OUT / "q4_panel.parquet")
W = ["Oct", "Nov", "Dec", "Dec1_15", "Dec16_31", "Q4", "Jan", "Q4_Jan", "Feb"]
yw = []
for y, df in panel.groupby("year"):
    n, p = df[df.R12 < 0], df[df.R12 >= 0]
    for w in W:
        yw.append({"year": y, "window": w, "neg": n[w].mean(), "pos": p[w].mean(),
                   "neg_med": n[w].median(), "pos_med": p[w].median(), "spy": df["SPY_" + w].iloc[0]})
yw = pd.DataFrame(yw)
yw["spread"] = yw.neg - yw.pos
yw["spread_med"] = yw.neg_med - yw.pos_med
trend = []
for w in W:
    d = yw[yw.window == w].copy()
    b = np.polyfit(d.spy, d.spread, 1)[0]
    d["hedged"] = d.spread - b * d.spy
    yw.loc[d.index, "hedged"] = d.hedged
    X = sm.add_constant(d.year - 2000)
    r = sm.OLS(d.hedged, X).fit(cov_type="HC1")
    eras = {e: d[d.year.isin(list(yrs))].hedged for e, yrs in ERAS.items()}
    trend.append({"window": w, "beta": b, "slope_per_yr": r.params.iloc[1], "slope_t": r.tvalues.iloc[1],
                  **{f"{e}_mean": v.mean() for e, v in eras.items()},
                  **{f"{e}_t": tstat(v) for e, v in eras.items()},
                  **{f"{e}_hit": (v > 0).mean() for e, v in eras.items()},
                  "all_mean": d.hedged.mean(), "all_t": tstat(d.hedged)})
yw.to_csv(OUT / "yearly_window_spreads.csv", index=False)
pd.DataFrame(trend).to_csv(OUT / "trend_tests.csv", index=False)
print("C done", flush=True)

# ---------------------------------------------------------------- D. mid-Dec re-sort: rebound trade
rb = []
rb_paths = {}
for y in YEARS:
    t0 = D.td_on_or_before(f"{y}-12-15")
    t12 = D.td_on_or_before(f"{y - 1}-12-15")
    ty = D.me(y - 1, 12)
    dec, jan = D.me(y, 12), D.me(y + 1, 1)
    sig = pd.DataFrame({"R12": D.window_ret(t12, t0), "YTD": D.window_ret(ty, t0), "DV": D.dv63.loc[t0]})
    sig = sig[sig.R12.notna() & (sig.DV >= 1e6)]
    q = pd.qcut(sig.R12, 5, labels=False) + 1
    grp = {"NEG_1Y": sig.index[sig.R12 < 0], "POS_1Y": sig.index[sig.R12 >= 0],
           "Q1_1Y": sig.index[q == 1], "Q5_1Y": sig.index[q == 5],
           "DEEP_LT_-30": sig.index[sig.R12 <= -0.30], "ALL": sig.index}
    for g, mem in grp.items():
        if len(mem) < 10:
            continue
        a = D.window_ret(t0, dec, mem)
        b = D.window_ret(dec, jan, mem)
        c = D.window_ret(t0, jan, mem)
        p = D.bh_path(mem, t0, jan)
        rb_paths[(g, y)] = p
        rb.append({"year": y, "group": g, "n": len(mem), "Dec15_31": a.mean(), "Jan": b.mean(),
                   "Dec15_Jan31": c.mean(), "Dec15_Jan31_med": c.median()})
    s = D.bench["SPY"]
    rb.append({"year": y, "group": "SPY", "n": 1, "Dec15_31": s[dec] / s[t0] - 1, "Jan": s[jan] / s[dec] - 1,
               "Dec15_Jan31": s[jan] / s[t0] - 1, "Dec15_Jan31_med": s[jan] / s[t0] - 1})
    rb_paths[("SPY", y)] = SPY.loc[t0:jan].iloc[1:]
rb = pd.DataFrame(rb)
rb.to_csv(OUT / "dec15_rebound_by_year.csv", index=False)
rows = []
for era, yrs in {"ALL": YEARS, **ERAS}.items():
    for g in ["NEG_1Y", "POS_1Y", "Q1_1Y", "Q5_1Y", "DEEP_LT_-30", "ALL", "SPY"]:
        ps = [rb_paths[(g, y)] for y in yrs if (g, y) in rb_paths]
        daily = pd.concat(ps)
        st = risk_stats(daily, D.rf, SPY)
        d = rb[(rb.group == g) & rb.year.isin(list(yrs))]
        rows.append({"era": era, "group": g, **st, "mean_Dec15_Jan31": d.Dec15_Jan31.mean(),
                     "med_Dec15_Jan31": d.Dec15_Jan31_med.mean(), "t": tstat(d.Dec15_Jan31),
                     "pct_pos": (d.Dec15_Jan31 > 0).mean(), "mean_Dec15_31": d.Dec15_31.mean(),
                     "mean_Jan": d.Jan.mean(), "avg_n": d.n.mean()})
pd.DataFrame(rows).to_csv(OUT / "dec15_rebound_risk.csv", index=False)
print("D done", flush=True)

# ---------------------------------------------------------------- E. volume footprint
vf = []
V = D.vol
for y in YEARS:
    base0, base1 = D.me(y, 5), D.me(y, 8)            # Jun-Aug baseline
    for label, (form, w0, w1) in {
        "Oct": (D.me(y, 9), D.me(y, 9), D.me(y, 10)),
        "Dec1_20": (D.me(y, 11), D.me(y, 11), D.td_on_or_before(f"{y}-12-20")),
        "Dec21_31": (D.me(y, 11), D.td_on_or_before(f"{y}-12-20"), D.me(y, 12)),
    }.items():
        sig = D.signals(form)
        base = V.loc[base0:base1].iloc[1:].mean()
        win = V.loc[w0:w1].iloc[1:].mean()
        ratio = (win / base).replace([np.inf, -np.inf], np.nan)
        rn, rp = ratio.reindex(sig.index[sig.R12 < 0]).median(), ratio.reindex(sig.index[sig.R12 >= 0]).median()
        q = pd.qcut(sig.R12, 5, labels=False)
        r1, r5 = ratio.reindex(sig.index[q == 0]).median(), ratio.reindex(sig.index[q == 4]).median()
        vf.append({"year": y, "window": label, "neg_abvol": rn, "pos_abvol": rp, "q1_abvol": r1, "q5_abvol": r5,
                   "footprint": rn / rp - 1, "footprint_q1q5": r1 / r5 - 1})
vf = pd.DataFrame(vf)
vf.to_csv(OUT / "volume_footprint.csv", index=False)
print("E done", flush=True)

# ---------------------------------------------------------------- F. live 2026 screen
t0 = D.me(2026, 9)
last = D.close.index[-1]
sig = D.signals(t0)
sig["QTD"] = D.window_ret(t0, last, sig.index)
sig["R12_now"] = D.window_ret(D.td_on_or_before(last - pd.DateOffset(years=1)), last, sig.index)
sig["R1_now"] = D.window_ret(D.td_on_or_before(last - pd.DateOffset(months=1)), last, sig.index)
sig.to_csv(OUT / "screen_2026.csv")
g = groups(sig)
live = {k: {"n": len(v), "qtd_mean": float(sig.loc[v, "QTD"].mean()), "qtd_med": float(sig.loc[v, "QTD"].median())}
        for k, v in g.items()}
live["SPY_qtd"] = float(D.bench["SPY"][last] / D.bench["SPY"][t0] - 1)
live["asof"] = str(last.date())
RES["live_2026"] = live
(OUT / "misc.json").write_text(json.dumps(RES, indent=2, default=float))
print("F done")
