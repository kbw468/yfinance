"""Collapse result CSVs into compact JSON datasets for the dashboard."""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm

OUT = Path(sys.argv[1])
UNIV = Path(sys.argv[2])
DST = Path(sys.argv[3])
DST.mkdir(parents=True, exist_ok=True)


def dump(name, df):
    df = df.replace([np.inf, -np.inf], np.nan)
    recs = json.loads(df.to_json(orient="records", double_precision=5))
    (DST / f"{name}.json").write_text(json.dumps(recs, separators=(",", ":")))
    print(name, len(recs), (DST / f"{name}.json").stat().st_size)


ERA_NAMES = {"ALL": "1999-2025"}
GROUP_LABELS = {
    "NEG_1Y": "1Y negative", "POS_1Y": "1Y positive", "NEG_1M": "1M negative", "POS_1M": "1M positive",
    "NEG1Y_NEG1M": "1Y neg + 1M neg", "NEG1Y_POS1M": "1Y neg + 1M pos",
    "POS1Y_NEG1M": "1Y pos + 1M neg", "POS1Y_POS1M": "1Y pos + 1M pos",
    "Q1_1Y": "1Y quintile 1 (worst)", "Q2_1Y": "1Y quintile 2", "Q3_1Y": "1Y quintile 3",
    "Q4_1Y": "1Y quintile 4", "Q5_1Y": "1Y quintile 5 (best)",
    "NEG_1Y_SMALL": "1Y neg, low liquidity", "POS_1Y_SMALL": "1Y pos, low liquidity",
    "NEG_1Y_MID": "1Y neg, mid liquidity", "POS_1Y_MID": "1Y pos, mid liquidity",
    "NEG_1Y_LARGE": "1Y neg, high liquidity", "POS_1Y_LARGE": "1Y pos, high liquidity",
    "ALL": "All names (equal weight)", "SPY": "SPY", "DEEP_LT_-30": "1Y return -30% or worse",
}
WINDOW_LABELS = {"Oct1_Oct15": "Oct 1-15", "Oct1_15": "Oct 1-15", "Oct15_Dec15": "Oct 15-Dec 15",
                 "Oct15_Dec31": "Oct 15-Dec 31", "Q4": "Q4 (Sep 30-Dec 31)", "Dec15_Jan31": "Dec 15-Jan 31",
                 "Jan": "January"}

# 1. quarter risk (full + eras)
a = pd.read_csv(OUT / "risk_by_quarter.csv").assign(era="1999-2025")
b = pd.read_csv(OUT / "risk_by_era.csv")
qr = pd.concat([a, b], ignore_index=True)
keep = ["quarter", "era", "group", "ann_ret", "ann_vol", "sharpe", "sortino", "beta_spy", "alpha_spy_ann",
        "q_mean", "q_median", "hit_vs_spy", "pct_pos", "avg_mdd", "q_worst", "avg_n", "years"]
qr["key"] = qr.quarter + "|" + qr.era + "|" + qr.group
qr["label"] = qr.group.map(GROUP_LABELS)
dump("quarter_risk", qr[["key", "label"] + keep])

# 2. half-month profile
hm = pd.read_csv(OUT / "halfmonth_profile.csv")
hm["era"] = hm.era.replace(ERA_NAMES)
dump("halfmonth", hm[["pair", "era", "window", "mean", "t", "hit", "median_spread"]])

# 3. calendar-month beta-adjusted alpha by era
m = pd.read_csv(OUT / "monthly_spreads.csv")
m = m[(m.fwd_year >= 2000) | (m.fwd_month >= 10)]
m = m[~((m.fwd_year == 2026) & (m.fwd_month >= 10))].reset_index(drop=True)
m["era"] = np.select([m.fwd_year <= 2007, m.fwd_year <= 2016], ["1999-2007", "2008-2016"], "2017-2026")
rows = []
for col, lab in [("s12", "NEG1Y-POS1Y"), ("s1", "NEG1M-POS1M"), ("s_nn_pp", "NN-PP")]:
    for e, d in [("1999-2026", m)] + list(m.groupby("era")):
        X = pd.concat([pd.get_dummies(d.fwd_month, prefix="m").astype(float), d[["spy"]]], axis=1)
        r = sm.OLS(d[col], X).fit(cov_type="HC1")
        for k in range(12):
            rows.append({"pair": lab, "era": e, "month": k + 1, "alpha": r.params.iloc[k], "t": r.tvalues.iloc[k],
                         "raw": d[d.fwd_month == k + 1][col].mean(), "beta": r.params["spy"]})
dump("monthly_alpha", pd.DataFrame(rows))

# 4. cumulative relative paths (avg by era), every 2nd calendar day to keep it light
cp = pd.read_csv(OUT / "cum_rel_paths_avg.csv", header=[0, 1], index_col=0)
cp = cp.stack([0, 1], future_stack=True).rename("rel").reset_index()
cp.columns = ["day", "pair", "era", "rel"]
cp["era"] = cp.era.replace(ERA_NAMES)
cp = cp[cp.day % 2 == 1]
dump("cum_paths", cp)

# 5. yearly table
tw = pd.read_csv(OUT / "trade_window_yearly.csv")
pv = tw.pivot_table(index="year", columns=["window", "group"], values="ret")
y = pd.DataFrame(index=pv.index)
for w in ["Oct1_Oct15", "Oct15_Dec15", "Q4"]:
    y[f"{w}_np"] = pv[(w, "NEG_1Y")] - pv[(w, "POS_1Y")]
    y[f"{w}_nnpp"] = pv[(w, "NEG1Y_NEG1M")] - pv[(w, "POS1Y_POS1M")]
y["Q4_neg"] = pv[("Q4", "NEG_1Y")]
y["Q4_pos"] = pv[("Q4", "POS_1Y")]
y["Q4_spy"] = pv[("Q4", "SPY")]
yw = pd.read_csv(OUT / "yearly_window_spreads.csv")
y["Jan_np"] = yw[yw.window == "Jan"].set_index("year").spread
y["Dec1_15_np"] = yw[yw.window == "Dec1_15"].set_index("year").spread
rb = pd.read_csv(OUT / "dec15_rebound_by_year.csv").pivot(index="year", columns="group", values="Dec15_Jan31")
y["Dec15_Jan31_deep_vs_spy"] = rb["DEEP_LT_-30"] - rb["SPY"]
tr = pd.read_csv(OUT / "trough_timing.csv").set_index("year")
y["trough_date"] = tr.trough_date
vf = pd.read_csv(OUT / "volume_footprint.csv")
y["vol_dec21_31"] = vf[vf.window == "Dec21_31"].set_index("year").footprint
y["vol_oct"] = vf[vf.window == "Oct"].set_index("year").footprint
dump("yearly", y.reset_index())

# 6. trade windows + Dec-15 rebound risk
t = pd.read_csv(OUT / "trade_window_risk.csv")
t["era"] = t.era.replace(ERA_NAMES)
r = pd.read_csv(OUT / "dec15_rebound_risk.csv").assign(window="Dec15_Jan31")
r["era"] = r.era.replace(ERA_NAMES)
r = r.rename(columns={"mean_Dec15_Jan31": "mean", "med_Dec15_Jan31": "median"})
cols = ["window", "era", "group", "ann_vol", "sharpe", "sortino", "beta_spy", "alpha_spy_ann", "mean", "median",
        "t", "pct_pos"]
wn = pd.concat([t[cols + ["hit_vs_spy", "worst"]], r[cols]], ignore_index=True)
wn.insert(0, "key", wn.window + "|" + wn.era + "|" + wn.group)
wn.insert(1, "label", wn.group.map(GROUP_LABELS))
dump("windows", wn)

# 7. pair spreads per window/era (Sep-30 sort) + robustness
s = pd.read_csv(OUT / "trade_window_spreads.csv")
s["era"] = s.era.replace(ERA_NAMES)
dump("spreads", s)
rob = pd.read_csv(OUT / "robustness_summary.csv")
rob = rob[rob.era.str.startswith("2017")]
rob = rob[["floor", "sort", "window", "era", "metric", "mean", "t", "hit"]].copy()
rob.insert(0, "key", rob.window + "|" + rob["sort"] + "|" + (rob.floor / 1e6).astype(int).astype(str) + "M|"
           + rob.era + "|" + rob.metric)
rob.insert(1, "window_label", rob.window.map(WINDOW_LABELS))
rob.insert(2, "sort_label", rob["sort"].map({"sep30": "Sep 30 sort", "oct15": "Oct 15 re-sort"}))
dump("robust", rob)

# 8. live 2026 + screen
misc = json.loads((OUT / "misc.json").read_text())
live = misc["live_2026"]
lv = [{"group": k, "n": v["n"], "qtd_mean": v["qtd_mean"], "qtd_med": v["qtd_med"]}
      for k, v in live.items() if isinstance(v, dict)]
lv.append({"group": "SPY", "n": 1, "qtd_mean": live["SPY_qtd"], "qtd_med": live["SPY_qtd"]})
lv = pd.DataFrame(lv)
lv.insert(1, "label", lv.group.map(GROUP_LABELS))
dump("live", lv)
sc = pd.read_csv(OUT / "screen_2026.csv", index_col=0)
u = pd.read_csv(UNIV)
u["ticker"] = u.Ticker.str.replace(".", "-", regex=False)
sc = sc.join(u.set_index("ticker")[["Company", "Sector", "Industry", "Market Cap"]])
sc = sc[(sc.R12 < 0) & (sc.R1 < 0)].sort_values("DV", ascending=False)
sc.index.name = "ticker"
sc = sc.reset_index().rename(columns={"Market Cap": "mcap_m", "DV": "dollar_vol"})
sc = sc[["ticker", "Company", "Sector", "Industry", "mcap_m", "dollar_vol", "R12", "R1", "QTD"]]
sc.columns = ["ticker", "company", "sector", "industry", "mcap_m", "dollar_vol", "r12", "r1", "qtd"]
dump("screen", sc)

# 9. headline stats
qa = qr.set_index(["quarter", "era", "group"])
ls = pd.read_csv(OUT / "long_short.csv").set_index(["long", "short", "era", "quarter"])
hmi = hm.set_index(["pair", "era", "window"])
twi = t.set_index(["window", "era", "group"])
si = s.set_index(["window", "pair", "era"])
ri = r.set_index(["era", "group"])
vfe = vf[vf.window == "Dec21_31"].groupby(pd.cut(vf[vf.window == "Dec21_31"].year, [1998, 2007, 2016, 2025]),
                                           observed=True).footprint.mean().tolist()
mo = pd.DataFrame(rows).set_index(["pair", "era", "month"])
robi = rob.set_index(["floor", "sort", "window", "era", "metric"])
H = [
    ("n_tickers", "Tickers in list", 2367, "int"),
    ("n_stock_years", "Stock-years tested", int(pd.read_parquet(OUT / "q4_panel.parquet").shape[0]), "int"),
    ("q4_sharpe_neg", "Q4 Sharpe, 1Y losers", qa.loc[("Q4", "1999-2025", "NEG_1Y"), "sharpe"], "num2"),
    ("q4_sharpe_pos", "Q4 Sharpe, 1Y winners", qa.loc[("Q4", "1999-2025", "POS_1Y"), "sharpe"], "num2"),
    ("q4_ret_neg", "Avg Q4 return, losers", qa.loc[("Q4", "1999-2025", "NEG_1Y"), "q_mean"], "pct1"),
    ("q4_ret_pos", "Avg Q4 return, winners", qa.loc[("Q4", "1999-2025", "POS_1Y"), "q_mean"], "pct1"),
    ("q4_alpha_neg", "Q4 alpha vs SPY, losers", qa.loc[("Q4", "1999-2025", "NEG_1Y"), "alpha_spy_ann"], "pct1"),
    ("q4_alpha_pos", "Q4 alpha vs SPY, winners", qa.loc[("Q4", "1999-2025", "POS_1Y"), "alpha_spy_ann"], "pct1"),
    ("q4_spread", "Q4 loser-winner spread", ls.loc[("NEG_1Y", "POS_1Y", "ALL", "Q4"), "spread_mean"], "spct2"),
    ("q4_spread_t", "t-stat", ls.loc[("NEG_1Y", "POS_1Y", "ALL", "Q4"), "spread_t"], "num1"),
    ("q3_spread", "Q3 loser-winner spread", ls.loc[("NEG_1Y", "POS_1Y", "ALL", "Q3"), "spread_mean"], "spct2"),
    ("q3_spread_mod", "Q3 spread 2017-25", ls.loc[("NEG_1Y", "POS_1Y", "2017-2025", "Q3"), "spread_mean"], "spct2"),
    ("q3_hit_mod", "Q3 hit 2017-25", ls.loc[("NEG_1Y", "POS_1Y", "2017-2025", "Q3"), "hit"], "pct0"),
    ("oct_np_mod", "Oct 1-15 spread 2017-25", hmi.loc[("NEG1Y-POS1Y", "2017-2025", "Oct1_15"), "mean"], "spct2"),
    ("oct_np_mod_t", "t", hmi.loc[("NEG1Y-POS1Y", "2017-2025", "Oct1_15"), "t"], "num1"),
    ("oct_np_old", "Oct 1-15 spread 1999-2007", hmi.loc[("NEG1Y-POS1Y", "1999-2007", "Oct1_15"), "mean"], "spct2"),
    ("oct_q1q5_mod", "Oct 1-15 Q1-Q5 2017-25", hmi.loc[("Q1-Q5", "2017-2025", "Oct1_15"), "mean"], "spct2"),
    ("dec_np_old", "Dec 1-15 spread 1999-2007", hmi.loc[("NEG1Y-POS1Y", "1999-2007", "Dec1_15"), "mean"], "spct2"),
    ("dec_np_mod", "Dec 1-15 spread 2017-25", hmi.loc[("NEG1Y-POS1Y", "2017-2025", "Dec1_15"), "mean"], "spct2"),
    ("dec_q1q5_mod", "Dec 1-15 Q1-Q5 2017-25", hmi.loc[("Q1-Q5", "2017-2025", "Dec1_15"), "mean"], "spct2"),
    ("jan_alpha_old", "Jan loser alpha 1999-2007", mo.loc[("NEG1Y-POS1Y", "1999-2007", 1), "alpha"], "spct2"),
    ("jan_alpha_mid", "Jan loser alpha 2008-16", mo.loc[("NEG1Y-POS1Y", "2008-2016", 1), "alpha"], "spct2"),
    ("jan_alpha_mod", "Jan loser alpha 2017-26", mo.loc[("NEG1Y-POS1Y", "2017-2026", 1), "alpha"], "spct2"),
    ("rb_alpha_old", "Dec15-Jan31 loser alpha 99-07", ri.loc[("1999-2007", "NEG_1Y"), "alpha_spy_ann"], "pct1"),
    ("rb_alpha_mid", "Dec15-Jan31 loser alpha 08-16", ri.loc[("2008-2016", "NEG_1Y"), "alpha_spy_ann"], "pct1"),
    ("rb_alpha_mod", "Dec15-Jan31 loser alpha 17-25", ri.loc[("2017-2025", "NEG_1Y"), "alpha_spy_ann"], "pct1"),
    ("vol_old", "Late-Dec loser volume excess 99-07", vfe[0], "spct1"),
    ("vol_mid", "Late-Dec loser volume excess 08-16", vfe[1], "spct1"),
    ("vol_mod", "Late-Dec loser volume excess 17-25", vfe[2], "spct1"),
    ("nn_long_ret", "NN Oct15-Dec15 avg 2017-25", twi.loc[("Oct15_Dec15", "2017-2025", "NEG1Y_NEG1M"), "mean"], "pct1"),
    ("nn_long_sharpe", "NN Oct15-Dec15 Sharpe", twi.loc[("Oct15_Dec15", "2017-2025", "NEG1Y_NEG1M"), "sharpe"], "num2"),
    ("nn_long_alpha", "NN Oct15-Dec15 alpha", twi.loc[("Oct15_Dec15", "2017-2025", "NEG1Y_NEG1M"), "alpha_spy_ann"], "pct1"),
    ("pp_long_ret", "PP Oct15-Dec15 avg", twi.loc[("Oct15_Dec15", "2017-2025", "POS1Y_POS1M"), "mean"], "pct1"),
    ("pp_long_sharpe", "PP Oct15-Dec15 Sharpe", twi.loc[("Oct15_Dec15", "2017-2025", "POS1Y_POS1M"), "sharpe"], "num2"),
    ("spy_long_ret", "SPY Oct15-Dec15 avg", twi.loc[("Oct15_Dec15", "2017-2025", "SPY"), "mean"], "pct1"),
    ("nnpp_spread", "NN-PP Oct15-Dec15 2017-25", si.loc[("Oct15_Dec15", "NEG1Y_NEG1M-POS1Y_POS1M", "2017-2025"), "mean"], "spct2"),
    ("nnpp_t", "t", si.loc[("Oct15_Dec15", "NEG1Y_NEG1M-POS1Y_POS1M", "2017-2025"), "t"], "num1"),
    ("nnpp_hit", "hit", si.loc[("Oct15_Dec15", "NEG1Y_NEG1M-POS1Y_POS1M", "2017-2025"), "hit"], "pct0"),
    ("nnpp_worst", "worst", si.loc[("Oct15_Dec15", "NEG1Y_NEG1M-POS1Y_POS1M", "2017-2025"), "worst"], "spct2"),
    ("nnpp_ex2020", "ex-2020", robi.loc[(1e6, "sep30", "Oct15_Dec15", "2017-2025 ex2020", "nn_pp"), "mean"], "spct2"),
    ("m1_q4_spread_mod", "NEG1M-POS1M Q4 2017-25", si.loc[("Q4", "NEG_1M-POS_1M", "2017-2025"), "mean"], "spct2"),
    ("m1_q4_t_mod", "t", si.loc[("Q4", "NEG_1M-POS_1M", "2017-2025"), "t"], "num1"),
    ("m1_q4_sharpe_neg", "NEG1M Q4 Sharpe 2017-25", qa.loc[("Q4", "2017-2025", "NEG_1M"), "sharpe"], "num2"),
    ("m1_q4_sharpe_pos", "POS1M Q4 Sharpe 2017-25", qa.loc[("Q4", "2017-2025", "POS_1M"), "sharpe"], "num2"),
    ("live_n_neg", "1Y losers now", live["NEG_1Y"]["n"], "int"),
    ("live_n_nn", "Double losers now", live["NEG1Y_NEG1M"]["n"], "int"),
    ("live_qtd_neg", "QTD losers", live["NEG_1Y"]["qtd_mean"], "spct2"),
    ("live_qtd_pos", "QTD winners", live["POS_1Y"]["qtd_mean"], "spct2"),
    ("live_qtd_nn", "QTD NN", live["NEG1Y_NEG1M"]["qtd_mean"], "spct2"),
    ("live_qtd_pp", "QTD PP", live["POS1Y_POS1M"]["qtd_mean"], "spct2"),
    ("live_qtd_spy", "QTD SPY", live["SPY_qtd"], "spct2"),
]
dump("headline", pd.DataFrame(H, columns=["id", "label", "value", "fmt"]))
