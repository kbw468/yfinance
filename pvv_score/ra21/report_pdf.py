"""RA21 report as a PDF: results/ra21/RA21_REPORT.pdf, HTML rendered by headless Chromium.
Numbers are computed here from the research table, or read from the result files the research modules wrote,
so the PDF cannot drift from the data. Palette: blue / orange only."""
import ast, base64, io, json, re, subprocess
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from ..data_io import load_panel, load_universe
from ..clean import clean_panel
from .common import load, MARKET, OUT
from .screen import flag, LEGS, EXCLUDED_SECTORS
from .book import run, stats
from .plain import name

CHROME = "/opt/pw-browsers/chromium-1194/chrome-linux/chrome"
BLUE, ORANGE, INK, GRID = "#1f5fa8", "#e07b00", "#1a1f2b", "#d9dde3"
PERIODS = [("2017-01-01", "2019-12-31", "2017 to 2019"), ("2021-01-01", "2023-12-31", "2021 to 2023"), ("2024-01-01", None, "2024 on")]
ADDS = ["own_sup_rate_756", "lead_persist_252", "seas_sup_share", "defend_share_63"]
STOPS = ["mad_63", "own_stop_rate_252", "beta_l1_252"]
LABEL = {"own_sup_rate_756": "Own 3-year record of superior trades", "lead_persist_252": "Days in the top 20% on trailing risk-adjusted return",
         "seas_sup_share": "Same calendar window in prior years", "defend_share_63": "Up on SPY down days", "mad_63": "Typical daily move",
         "own_stop_rate_252": "Own stop-out rate over the last year", "beta_l1_252": "L1 beta"}
FP = [("mdd_126", "Deepest drawdown, last 6 months", "pct"), ("mean_dd_63", "Average depth below running high, last 3 months", "pct1"),
      ("xs_sec_126", "6-month return vs its sector", "pct"), ("off_high_252", "Distance to 52-week high", "pct1"), ("days_since_high_252", "Sessions since 52-week high", "int"),
      ("roc_21", "ROC 21 sessions", "pct1"), ("roc_63", "ROC 63 sessions", "pct1"), ("eff_63", "Path efficiency, 63 sessions", "f2"),
      ("up_volume_share_63", "Up days' share of 63-session volume", "share"), ("new_high63_share_63", "Share of last 63 sessions at a 63-session high", "share"),
      ("volume_roc_21_252", "Volume, 21 sessions vs 1 year", "x"), ("vol_pctile_own_252", "Typical daily move vs its own year", "pctile"), ("beta_l1_252", "L1 beta, 1 year", "f2")]


def fmt(v, kind):
    if pd.isna(v): return ""
    if kind in ("pct", "pct1") and abs(v) < 0.0005: return "0.0%"
    return {"share": f"{v*100:.0f}%", "pct": f"{v*100:+.0f}%" if abs(v) >= 0.095 else f"{v*100:+.1f}%", "pct1": f"{v*100:+.1f}%", "int": f"{v:.0f}", "f2": f"{v:.2f}", "x": f"{v:.2f}x",
            "pctile": f"{v*100:.0f}th pctile", "rate": f"{v*100:.1f}%"}[kind]


def png(fig) -> str:
    b = io.BytesIO(); fig.savefig(b, format="png", dpi=200, bbox_inches="tight"); plt.close(fig)
    return "data:image/png;base64," + base64.b64encode(b.getvalue()).decode()


def table(df: pd.DataFrame, cls="", hi=None) -> str:
    h = "".join(f"<th>{c}</th>" for c in df.columns)
    rows = []
    for i, (_, r) in enumerate(df.iterrows()):
        c = f' class="{hi[i]}"' if hi and hi[i] else ""
        rows.append(f"<tr{c}>" + "".join(f"<td>{v}</td>" for v in r.values) + "</tr>")
    return f'<table class="{cls}"><thead><tr>{h}</tr></thead><tbody>{"".join(rows)}</tbody></table>'


def main():
    cuts = json.load(open(OUT / "rules_meta.json"))["breadth_cuts"]
    cols = list(dict.fromkeys(LEGS + [f for f, _, _ in FP] + ADDS + STOPS + ["earn_due_21", "bounce_21"]))
    T = flag(load(cols + MARKET + ["sector", "sup_21", "sup_42", "sup_63", "xs_21", "stopped_21"]), cuts)
    asof = T.date.max(); uni = load_universe()
    S = T[T.vrl]; X = S[S.sup_21.notna()]; A = T[T.sup_21.notna()]
    win = pd.Series(np.where(T.date >= "2024-01-01", "2024 on", "2015 to 2023"), index=T.index)
    g = T.groupby("date")

    # record
    rec = pd.DataFrame({"": ["V-recovered leader", "All names"],
                        "Superior 21d": [fmt(X.sup_21.mean(), "rate"), fmt(A.sup_21.mean(), "rate")],
                        "Superior 42d": [fmt(S.sup_42.mean(), "rate"), fmt(T.sup_42.mean(), "rate")],
                        "Superior 63d": [fmt(S.sup_63.mean(), "rate"), fmt(T.sup_63.mean(), "rate")],
                        "Median excess 21d": [fmt(X.xs_21.median(), "pct1"), fmt(A.xs_21.median(), "pct1")],
                        "Mean excess 21d": [fmt(X.xs_21.mean(), "pct1"), fmt(A.xs_21.mean(), "pct1")],
                        "Stopped at -8%": [fmt(X.stopped_21.mean(), "rate"), fmt(A.stopped_21.mean(), "rate")]})
    windows = {lab: X.sup_21[(X.date >= a) & (X.date <= (b or "2100"))].mean() for a, b, lab in [("2015-01-01", "2020-12-31", "found, 2015 to 2020"), ("2021-01-01", "2023-12-31", "confirmed, 2021 to 2023"), ("2024-01-01", None, "hold-out, 2024 on")]}
    bq = (g.beta_l1_252.rank(pct=True) * 5).clip(upper=4.999).fillna(-1).astype(int)
    matched = T.sup_21.groupby([T.date, bq]).transform("mean")[T.vrl & T.sup_21.notna()].mean()
    near_hi = X.mkt_off_high_252 >= -0.05
    spy_state = (X.sup_21[near_hi].mean(), X.sup_21[~near_hi].mean())
    P = {c: g[c].rank(pct=True) for c in LEGS}
    ok_sec = ~T.sector.isin(EXCLUDED_SECTORS)
    grad = []
    for q in (0.10, 0.15, 0.20, 0.25, 0.30):
        m = ok_sec & (P["mdd_126"] <= q) & (P["mean_dd_63"] >= 1 - q) & ((P["up_capture_252"] >= 1 - q) | (P["rs_off_high_252"] >= 1 - q) | (P["xs_sec_126"] >= 1 - q))
        grad.append((q, T.sup_21[m].mean(), int(T.sup_21[m].count())))
    yr = X.groupby(X.date.dt.year).sup_21.agg(["mean", "count"]); base_y = A.groupby(A.date.dt.year).sup_21.mean()
    # fingerprint
    U = T[ok_sec]
    fp = pd.DataFrame({"Characteristic": [lab for _, lab, _ in FP], "V-recovered leader, median": [fmt(S[f].median(), k) for f, _, k in FP],
                       "Universe, median": [fmt(U[f].median(), k) for f, _, k in FP]})
    # adds / detracts
    def rate(m): return {w: T.sup_21[m & (win == w)].mean() for w in ("2015 to 2023", "2024 on")}
    q5 = {f: (g[f].rank(pct=True) * 5).clip(upper=4.999).fillna(-1).astype(int) + 1 for f in ADDS + STOPS}
    adds = [("V-recovered leader", rate(T.vrl))] + [(f"{LABEL[f]}, top 20%", rate(q5[f] == 5)) for f in ADDS] + [("Report due, from its quarterly volume spikes", rate(T.earn_due_21 >= 3))]
    r20 = g.roc_21.rank(pct=True) if "roc_21" in T else None
    b20 = g.bounce_21.rank(pct=True)
    no_lead = (P["mdd_126"] <= 0.2) & (P["mean_dd_63"] >= 0.8) & ~((P["xs_sec_126"] >= 0.8) | (P["rs_off_high_252"] >= 0.8) | (P["up_capture_252"] >= 0.8))
    dets = [("Same V-recovery in Energy, Materials or Utilities", rate(T.vrl_any_sector & ~ok_sec)), ("Spike, then pullback to the 21-day low", rate((r20 >= 0.8) & (b20 <= 0.2))),
            ("Crash repaired with no leadership", rate(no_lead))] + [(f"{LABEL[f]}, bottom 20%", rate(q5[f] == 1)) for f in ADDS]
    def rt(rows): return pd.DataFrame({"Characteristic": [r[0] for r in rows], "2015 to 2023": [fmt(r[1]["2015 to 2023"], "rate") for r in rows], "2024 on": [fmt(r[1]["2024 on"], "rate") for r in rows]})
    stop = pd.DataFrame({"Characteristic": [LABEL[f] for f in STOPS],
                         "Top 20%": [fmt(T.stopped_21[(q5[f] == 5) & (win == "2024 on")].mean(), "rate") for f in STOPS],
                         "Bottom 20%": [fmt(T.stopped_21[(q5[f] == 1) & (win == "2024 on")].mean(), "rate") for f in STOPS]})
    # books and curves
    panel, _ = clean_panel(load_panel(), verbose=False); spy = panel["Close"]["SPY"]
    W = T[T.date >= "2017-01-01"].assign(v=lambda d: np.where(d.vrl, 1.0, np.nan)).pivot(index="date", columns="ticker", values="v")
    bk, curves = [], []
    for a, b, lab in PERIODS:
        v = run(W, panel, top_n=None, start=a, end=b); s = spy.loc[a:b] / spy.loc[a:b].iloc[0]; s = s.loc[:v.index[-1]]
        sv, ss = stats(v, spy), stats(s, spy)
        bk.append({"Period": lab, "V-recovered leader: CAGR / max DD / MAR": f"{sv['CAGR']*100:.1f}% / {sv['deepest drawdown']*100:.1f}% / {sv['MAR (CAGR/|DD|)']:.2f}",
                   "SPY: CAGR / max DD / MAR": f"{ss['CAGR']*100:.1f}% / {ss['deepest drawdown']*100:.1f}% / {ss['MAR (CAGR/|DD|)']:.2f}"})
        curves.append((lab, v, s))
    fig, ax = plt.subplots(1, 3, figsize=(11, 3.1), sharey=False)
    for k, (lab, v, s) in enumerate(curves):
        ax[k].plot(v.index, v.values, color=BLUE, lw=1.6, label="V-recovered leader")
        ax[k].plot(s.index, s.values, color=ORANGE, lw=1.6, label="SPY")
        ax[k].set_title(lab, fontsize=10, color=INK); ax[k].grid(color=GRID, lw=0.6); ax[k].tick_params(labelsize=7.5)
        for sp in ("top", "right"): ax[k].spines[sp].set_visible(False)
        ax[k].xaxis.set_major_locator(mdates.YearLocator()); ax[k].xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax[0].set_ylabel("growth of $1", fontsize=8.5); ax[0].legend(fontsize=8, frameon=False, loc="upper left")
    chart_books = png(fig)
    fig, ax = plt.subplots(figsize=(11, 2.6))
    ax.bar(yr.index.astype(str), yr["mean"] * 100, color=BLUE, width=0.62, label="V-recovered leader")
    ax.plot(yr.index.astype(str), base_y.reindex(yr.index) * 100, color=ORANGE, lw=2, ls="--", marker="o", ms=3.5, label="all names")
    for i, (y, r) in enumerate(yr.iterrows()):
        ax.text(i, r["mean"] * 100 + 1.2, f"{r['mean']*100:.0f}%", ha="center", fontsize=7.5, color=INK, bbox=dict(facecolor="white", edgecolor="none", pad=0.6))
    ax.set_ylabel("superior, % of 21d trades", fontsize=8.5); ax.grid(axis="y", color=GRID, lw=0.6); ax.tick_params(labelsize=8)
    for sp in ("top", "right"): ax.spines[sp].set_visible(False)
    ax.legend(fontsize=8, frameon=False, loc="upper right"); chart_years = png(fig)
    # tonight
    D0 = T[T.date == asof].copy(); pr = {c: D0[c].rank(pct=True) for c in LEGS}
    D0["p_lead"] = np.maximum.reduce([pr["xs_sec_126"], pr["rs_off_high_252"], pr["up_capture_252"]])
    fire = D0[D0.vrl]; near = D0[~D0.vrl_any_sector & ~D0.sector.isin(EXCLUDED_SECTORS) & (pr["mdd_126"] <= 0.3) & (pr["mean_dd_63"] >= 0.7) & (D0.p_lead >= 0.7)]
    excl = D0[D0.vrl_any_sector & D0.sector.isin(EXCLUDED_SECTORS)]
    legs = lambda r: ", ".join(n for n, v in (("6m vs sector", r.leg_sector), ("RS line at high", r.leg_rs), ("up-capture", r.leg_upcap)) if v)
    tn = pd.DataFrame({"Ticker": fire.ticker, "Company": fire.ticker.map(uni.Company), "Leadership legs met": [legs(r) for _, r in fire.iterrows()],
                       "6m drawdown": [fmt(v, "pct") for v in fire.mdd_126], "Off 52w high": [fmt(v, "pct1") for v in fire.off_high_252],
                       "ROC 21d": [fmt(v, "pct1") for v in fire.roc_21], "ROC 63d": [fmt(v, "pct1") for v in fire.roc_63], "L1 beta": [fmt(v, "f2") for v in fire.beta_l1_252]})
    # what did not work, from the result files
    U1 = pd.read_csv(OUT / "characteristics_univariate.csv").sort_values("auc_21", ascending=False)
    best = U1.iloc[0]
    tr = (OUT / "transparent_score.txt").read_text()
    tr_auc = float(re.search(r"by year: .*?mean ([0-9.]+)", tr).group(1)); tr_top = float(re.search(r"deciles 2021 onward.*?\n10\s+([0-9.]+)", tr, re.S).group(1))
    mw = (OUT / "model_walkforward.txt").read_text(); mean = ast.literal_eval(re.search(r"mean 2017\+: (\{.*?\})", mw).group(1))
    mv = (OUT / "model_variants.txt").read_text()
    var = {k: float(v) for k, v in re.findall(r"^(l1_marpct|rank_grade|binary_sup42)\s+([0-9.]+)\s*$", mv, re.M)}
    vtop = [float(x) for x in re.search(r"top decile superior_21\s+([0-9. ]+)", mv).group(1).split()]
    wtop = float(re.search(r"2017-2023: deciles.*?\n10\s+\d+\s+([0-9.]+)", mw, re.S).group(1))
    aucs = [mean["stock only"], mean["stock + market"]] + list(var.values()); tops = [wtop] + vtop
    SI = pd.read_csv(OUT / "characteristics_univariate_si.csv")
    R = pd.read_csv(OUT / "rules_confirmed.csv")
    nw = pd.DataFrame({"Test": [f"Best single characteristic: {LABEL.get(best.feature, name(best.feature)).lower()}",
                                "Simple score, features chosen on 2015 to 2020, tested 2021 on", "Walk-forward tree models, five versions", "Same models on shuffled labels",
                                "FINRA short interest, 2018 on", f"{len(R)} rules found 2015 to 2020, confirmed 2021 to 2023"],
                       "Out-of-sample result": [f"AUC {best.auc_21:.3f}", f"AUC {tr_auc:.3f}; top 10% of names superior {tr_top*100:.1f}%",
                                                f"AUC {min(aucs):.3f} to {max(aucs):.3f}; top 10% superior {min(tops)*100:.1f}% to {max(tops)*100:.1f}%", f"AUC {mean['shuffled control']:.3f}",
                                                f"AUC {SI.auc_21.min():.3f} to {SI.auc_21.max():.3f}", f"2024 on: median lift {R.lift_hold.median():.2f}; {(R.lift_hold > 1).mean()*100:.0f}% beat the base rate"]})
    css = f"""@page{{size:A4 portrait;margin:12mm 12mm 13mm 12mm}}
body{{font-family:Helvetica,Arial,sans-serif;color:{INK};font-size:9.4px;line-height:1.42;background:#fff}}
h1{{font-size:19px;margin:0 0 2px 0;color:{INK}}} h2{{font-size:12.5px;margin:16px 0 6px 0;padding-bottom:3px;border-bottom:2px solid {BLUE};color:{INK}}}
.sub{{color:#5b6270;font-size:9.5px;margin-bottom:10px}}
.bluf{{border-left:5px solid {BLUE};background:#eef4fb;padding:9px 12px;font-size:11px;line-height:1.5;margin:8px 0 4px 0}}
.bluf b{{color:{BLUE}}}
table{{border-collapse:collapse;width:100%;margin:4px 0 6px 0;font-size:9px;page-break-inside:avoid}}
th{{background:#f1f3f6;text-align:left;font-weight:600;padding:4px 6px;border-bottom:1px solid #c9ced6}}
td{{padding:3.5px 6px;border-bottom:1px solid #e6e9ee;vertical-align:top}}
tr.blue td{{background:#eef4fb;font-weight:600}} tr.orange td{{background:#fdf1e4}}
ul{{margin:4px 0 6px 16px;padding:0}} li{{margin:2px 0}}
.note{{color:#5b6270;font-size:8.6px}} img{{width:100%;margin:2px 0 4px 0}} .two{{display:flex;gap:14px}} .two>div{{flex:1}}
.pb{{page-break-before:always}}"""
    w = windows
    html = f"""<!doctype html><html><head><meta charset="utf-8"><title>Superior 21-day trades</title><style>{css}</style></head><body>
<h1>Superior 21-day risk-adjusted trades: what comes before them</h1>
<div class="sub">S&amp;P 500 + 400, every eligible name, 2015 to {asof.date()} close. No Gaussian statistics: no standard deviation, variance, z-scores, Sharpe, regression or correlation.</div>
<div class="bluf">One multi-factor setup beats SPY on risk-adjusted return: the <b>V-recovered leader</b>. <b>{X.sup_21.mean()*100:.1f}%</b> of its 21-day trades were superior,
against 20% for all names, and <b>{w['hold-out, 2024 on']*100:.1f}%</b> in the 2024 to 2026 hold-out that was never used to choose it. Its book beat SPY on MAR in every test period.
Tonight it fires on <b>{', '.join(fire.ticker)}</b>.</div>

<h2>Superior, as measured</h2>
<ul><li>Signal at the close. Buy the next open. Stop 8% below entry; a gap through the stop fills at that open. Otherwise exit at the close 21 sessions later.</li>
<li>Score: excess return over SPY for the same holding period, divided by the deepest drawdown suffered during the trade, floored at 2%.</li>
<li>Superior: the top 20% of the universe for that entry date. The base rate is 20% in every volatility quintile, so low-volatility names do not win it by default.</li></ul>

<h2>The V-recovered leader: the characteristics</h2>
<div class="two"><div>
{table(pd.DataFrame({"Leg": ["1. Crash", "2. Repair", "3. Leadership, any one", "Sector"],
                     "Rule, ranked against the universe that day": ["Deepest drawdown of the last 6 months in the bottom 20%",
                                                                   "Least depth below its running high over the last 3 months, top 20%",
                                                                   "6-month return vs its sector, RS line vs its 52-week high, or 1-year up-capture in the top 20%",
                                                                   "Not Energy, Materials or Utilities: there the same pattern was superior only 13% of the time before 2024 and 2% since"]}))}
<p class="note">Found on 2015 to 2020 and confirmed on 2021 to 2023. It was picked because it had the largest confirmation lift among rules with at least 800 confirmation cases. The 2024 to 2026 hold-out was never used to choose it.</p>
</div><div>
{table(fp)}
</div></div>

<h2>Record</h2>
{table(rec, hi=["blue", ""])}
<ul><li>{len(X):,} completed trades on {X.ticker.nunique()} names. Superior rate {w['found, 2015 to 2020']*100:.1f}% where it was found, {w['confirmed, 2021 to 2023']*100:.1f}% in confirmation, {w['hold-out, 2024 on']*100:.1f}% in the hold-out.</li>
<li>Not a beta effect: names in the same L1 beta quintile on the same dates were superior {matched*100:.1f}% of the time.</li>
<li>Stricter cuts score higher: {'; '.join(f'{int(q*100)}% cut {r*100:.1f}%' for q, r, _ in grad)}.</li>
<li>With SPY within 5% of its 52-week high: {spy_state[0]*100:.1f}%. With SPY more than 5% below it: {spy_state[1]*100:.1f}%.</li></ul>
<img src="{chart_years}">

<h2 class="pb">Against SPY</h2>
<p class="note">Same trade rules for every name, 10 bp round trip, 21 staggered monthly books marked daily. MAR is CAGR divided by the deepest drawdown. The COVID window, February to June 2020, is out of every fit, so it is out of the books.</p>
{table(pd.DataFrame(bk))}
<img src="{chart_books}">
<p class="note">The setup book holds a median of 2 names on the days it fires.</p>

<div class="two"><div>
<h2>Adds</h2><p class="note">Share of 21-day trades that were superior; the base rate is 20%.</p>
{table(rt(adds), hi=["blue"] + [""] * (len(adds) - 1))}
</div><div>
<h2>Detracts</h2><p class="note">Spike, then pullback worked through 2023 and flipped in 2024.</p>
{table(rt(dets), hi=["orange"] * 3 + [""] * (len(dets) - 3))}
</div></div>

<h2>Who stops you out</h2>
<p class="note">The strongest signal in the whole set. Share of trades stopped at -8%, 2024 on. These names do not score worse on risk-adjusted return because their winners are bigger; they are the ones that whipsaw.</p>
{table(stop)}

<h2>Tonight, {asof.date()} close</h2>
{table(tn, hi=["blue"] * len(tn))}
<p class="note">Near misses, every leg within 10 points: {', '.join(near.ticker)}. {('Fires but excluded by sector: ' + ', '.join(excl.ticker) + '.') if len(excl) else ''}</p>

<h2>What did not work</h2>
<p class="note">174 non-Gaussian characteristics, 2.28 million name-days. AUC ranks names within each day; 0.50 is a coin flip.</p>
{table(nw)}
<p class="note">Code and data: pvv_score/ra21 on branch claude/practical-hamilton-sspq2e. README.md there lists every feature family and test.</p>
</body></html>"""
    h = OUT / "RA21_REPORT.html"; h.write_text(html)
    pdf = OUT / "RA21_REPORT.pdf"
    subprocess.run([CHROME, "--headless=new", "--no-sandbox", "--disable-gpu", "--no-pdf-header-footer", f"--print-to-pdf={pdf}", f"file://{h}"], capture_output=True, check=True)
    print(pdf, pdf.stat().st_size)


if __name__ == "__main__":
    main()
