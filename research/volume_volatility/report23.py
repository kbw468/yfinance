"""Build the 2023-onward findings PDF.

Usage: python report23.py <results23_dir> <sweep_2009_dir> <combo_2009_dir> <out.pdf>
  results23_dir holds: sweep/ (yearly periods), sweep_halves/ (2023-24 vs 2025-26), combo756/, combo300/,
  optional comboW<N>/ (combos on the 2023+ best level window), and backtest runs w20, w50, w60, w120, w180, w20_all.
Reuses the styling and helpers in report.py. Blue / orange only; every colored cell carries its number.
"""
import sys
import tempfile
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.units import inch
from reportlab.platypus import Image, KeepTogether, PageBreak, SimpleDocTemplate, Spacer, Table, TableStyle

import report as rp
from report import BLUE, DIV, H, INK, INK2, MUTED, RULE, P, bullets, cell, day_tables, f3, ft, heat, pick, table

RUN_WINDOWS = [20, 50, 60, 120, 180]


def plabel(p):
    if p == "FULL":
        return "2023–26"
    a, b = p.split("..")
    if a[:4] == b[:4]:
        return a[:4] + (" YTD" if b[5:] != "12" else "")
    return f"{a[:4]}–{b[2:4]}"


def periods_of(df):
    ps = sorted(x for x in df.period.unique() if x != "FULL")
    return ps, ps + ["FULL"]


def heat_chart(mat, rows, cols, title, xlabel, ylabel, cbar, path, size=(6.8, 3.9), vmax=4.0):
    fig, ax = plt.subplots(figsize=size, dpi=220)
    norm = heat(ax, mat, rows, cols, vmax=vmax)
    ax.set_title(title, loc="left")
    if xlabel:
        ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    cb = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=DIV), ax=ax, fraction=0.035, pad=0.02)
    cb.set_label(cbar, color=INK2)
    cb.outline.set_visible(False)
    fig.tight_layout()
    fig.savefig(path, facecolor="white")
    plt.close(fig)


def combo_rank(cb, per):
    q = cb[cb.kind.isin(["LS_Q5", "VV_ONLY_Q5"])].copy()
    q.loc[q.kind == "VV_ONLY_Q5", "roc"] = "alone"
    g = q.groupby(["roc", "period"])[["mean", "t"]].mean().unstack("period")
    return g


def main(res, sweep09_dir, combo09_dir, out_pdf):
    res = Path(res)
    sw = pd.read_csv(res / "sweep" / "sweep.csv")
    swh = pd.read_csv(res / "sweep_halves" / "sweep.csv")
    sw09 = pd.read_csv(Path(sweep09_dir) / "sweep.csv")
    cb09 = pd.read_csv(Path(combo09_dir) / "combo.csv")
    cb = pd.read_csv(res / "combo756" / "combo.csv")
    cb300 = pd.read_csv(res / "combo300" / "combo.csv")
    cbw = sorted(res.glob("comboW*"))
    cbw = (int(cbw[0].name[6:]), pd.read_csv(cbw[0] / "combo.csv")) if cbw else None
    runs = {n: rp.load_run(res, n) for n in [f"w{w}" for w in RUN_WINDOWS] + ["w20_all"]}
    YP, YPF = periods_of(sw)
    HP, _ = periods_of(swh)
    tmp = Path(tempfile.mkdtemp())

    # ----- level sweep -----
    vv = sw[sw.factor == "VV"]
    vsc_t = vv.groupby(["window", "period"]).t.mean().unstack("period")[YPF]
    vsc_s = vv.groupby(["window", "period"]).spread.mean().unstack("period")[YPF]
    best_w = int(vsc_t["FULL"].idxmin())
    heat_chart(vsc_t.values, [str(w) for w in vsc_t.index], [plabel(p) for p in YPF],
               "Volume-volatility level Q5−Q1 by window and year — average t across 8 cells", None,
               "Volume-volatility window (sessions)", "avg t  (orange = high volume volatility underperforms)",
               tmp / "heat.png")
    yrs_neg = (vsc_s[YP] < 0).sum(axis=1)
    h_t = swh[swh.factor == "VV"].groupby(["window", "period"]).t.mean().unstack("period")[HP]
    pick_w = int(h_t[HP[0]].idxmin())
    oos_rank = int(h_t[HP[1]].rank().loc[pick_w])
    w09 = sw09[(sw09.factor == "VV") & (sw09.period == "FULL")].groupby("window").t.mean()

    # ----- ROC -----
    roc = sw[sw.factor == "VV_ROC"]
    rc_t = roc.groupby(["window", "lag", "period"]).ctrl_t.mean().unstack("period")[YPF]
    rc_s = roc.groupby(["window", "lag", "period"]).ctrl_spread.mean().unstack("period")[YPF]
    rmat = rc_t["FULL"].unstack("lag")
    heat_chart(rmat.values, [str(w) for w in rmat.index], [str(c) for c in rmat.columns],
               "ROC with level held fixed — 2023–26 average t", "ROC lag (sessions)",
               "Volume-volatility window (sessions)",
               "avg t  (orange = rising underperforms · blue = rising outperforms)", tmp / "roc.png",
               size=(7.2, 4.15), vmax=3.5)
    fast_keys = [(5, 5), (10, 5), (15, 3), (15, 5)]
    roc_pos_all = int(((rc_s[YP] > 0).all(axis=1)).sum())
    roc_neg_all = int(((rc_s[YP] < 0).all(axis=1)).sum())
    fast_str = ", ".join(f"{k[0]}/{k[1]} {ft(rc_t.loc[k, 'FULL'])}" for k in fast_keys)

    # ----- combos -----
    g = combo_rank(cb, YPF)
    g300 = combo_rank(cb300, YPF)
    base_sp, base_t = g["mean"]["FULL"]["alone"], g["t"]["FULL"]["alone"]
    best_c = g["t"]["FULL"].drop("alone").idxmax()
    best_c_sp = g["mean"]["FULL"][best_c]
    if cbw:
        gw = combo_rank(cbw[1], YPF)
    rp.chart_combo(cb, tmp / "combo.png", cfg=best_c)

    # ----- level detail runs -----
    det_w = min(RUN_WINDOWS, key=lambda w: abs(w - best_w))
    ts_b, ts_20 = runs[f"w{det_w}"]["ts"], runs["w20"]["ts"]
    d_b = [pick(ts_b, day="DOWN", h=h) for h in H]
    u_b = [pick(ts_b, day="UP", h=h) for h in H]
    d_20 = [pick(ts_20, day="DOWN", h=h) for h in H]
    rng = lambda xs: f"{f3(min(x[0] for x in xs))} to {f3(max(x[0] for x in xs))}"
    trng = lambda xs: f"{ft(min(x[1] for x in xs))} to {ft(max(x[1] for x in xs))}"

    story = []
    story += [P("Volume Volatility — 2023 Onward", "title"), Spacer(1, 2),
              P(f"2,367 tickers · signals 2023-01 to 2026-10 · forward 7 / 14 / 21 / 42 sessions · "
                f"scored per year: {', '.join(plabel(p) for p in YP)}", "sub"), Spacer(1, 10)]
    story += [P("Bottom line", "h1")]
    story += bullets([
        f"<b>Best level window since 2023: {best_w} sessions</b> (avg t {ft(vsc_t.loc[best_w, 'FULL'])}). "
        f"Q5−Q1 is negative in {int(yrs_neg.loc[best_w])} of {len(YP)} years. 20 sessions: avg t "
        f"{ft(vsc_t.loc[20, 'FULL'])}, negative in {int(yrs_neg.loc[20])} of {len(YP)}.",
        f"<b>{det_w}-session level, down days:</b> Q5−Q1 {rng(d_b)} (t {trng(d_b)}). <b>Up days:</b> {rng(u_b)} "
        f"(t {trng(u_b)}).",
        f"<b>Selection check:</b> choosing on 2023–24 alone picks {pick_w} sessions. In 2025–26 it ranks "
        f"{oos_rank} of {len(h_t)} (avg t {ft(h_t.loc[pick_w, HP[1]])}).",
        f"<b>Strongest combination since 2023: level + ROC {best_c}.</b> Average long-short {f3(best_c_sp)} vs "
        f"{f3(base_sp)} for the 20-session level alone, t {ft(g['t']['FULL'][best_c])} vs {ft(base_t)}.",
        f"<b>ROC with level held fixed:</b> {roc_pos_all} of {len(rc_s)} window × lag settings positive in every "
        f"year, {roc_neg_all} negative in every year. Fast settings (5–15 / 3–5): avg t {fast_str}.",
    ])
    story += [Spacer(1, 6), P("How to read the numbers", "h2"),
              P("Each figure is the forward return divided by the stock's trailing 20-day volatility scaled to the "
                "horizon (σ·√h), minus that day's universe average. 0 = in line with the universe. Q1–Q5 are fifths "
                "of each ticker's own trailing-252-session distribution. Brackets hold Newey-West t-stats "
                "(lag = horizon). Each year is about 250 signal dates. A 42d forward window needs 42 more sessions, so "
                "2026 YTD runs through August signals at that horizon.", "small")]
    story += [P("Setup", "h1")]
    setup = [
        ["Universe", "Names with 756+ sessions of data (listed by early 2020) — same names for every setting. "
                     "A broader universe (300+ sessions) is checked in Section 5"],
        ["Signals", "Volume volatility = stdev of daily ln(V<sub>t</sub>/V<sub>t−1</sub>) over N sessions, ranked "
                    "vs the ticker's own past 252 sessions. ROC = VV<sub>t</sub>/VV<sub>t−L</sub> − 1"],
        ["Day type / entry", "Up: close > prior close. Down: close < prior close. Entry at signal-day close"],
        ["Filters", "Price ≥ $1, 20d median dollar volume ≥ $500k. Bad volume prints dropped, daily change capped "
                    "at 50×"],
    ]
    story += [Table([[P(f"<b>{a}</b>", "cell"), P(b, "cell")] for a, b in setup], colWidths=[1.25 * inch, 5.9 * inch],
                    style=TableStyle([("LINEBELOW", (0, 0), (-1, -1), 0.25, RULE), ("VALIGN", (0, 0), (-1, -1), "TOP"),
                                      ("TOPPADDING", (0, 0), (-1, -1), 2), ("BOTTOMPADDING", (0, 0), (-1, -1), 2)]))]
    story += [PageBreak()]

    # ----- 1. window -----
    story += [P("1. Level window since 2023", "h1"), Image(str(tmp / "heat.png"), width=6.8 * inch, height=3.9 * inch)]
    rows = [["Window", "2023–24 avg t", "2025–26 avg t", "2023–26 avg t", "Years negative", "2009–26 avg t"]]
    for w in vsc_t.index:
        rows.append([str(w), ft(h_t.loc[w, HP[0]]), ft(h_t.loc[w, HP[1]]), ft(vsc_t.loc[w, "FULL"]),
                     f"{int(yrs_neg.loc[w])} of {len(YP)}", ft(w09.loc[w])])
    story += [Spacer(1, 4), P("Window scores: halves, full period, and the 2009–26 reference", "h2"),
              table(rows, [0.8 * inch, 1.1 * inch, 1.1 * inch, 1.1 * inch, 1.1 * inch, 1.1 * inch])]
    story += bullets([
        f"Strongest since 2023: {best_w} sessions (avg t {ft(vsc_t.loc[best_w, 'FULL'])}). Over 2009–26 the "
        f"strongest was 20 (avg t {ft(w09.loc[20])}).",
        f"Choosing on 2023–24 alone picks {pick_w}. In 2025–26 it ranks {oos_rank} of {len(h_t)}.",
    ])
    story += [PageBreak()]

    # ----- 2. level detail -----
    story += [P(f"2. Level detail since 2023 — {det_w} vs 20 sessions", "h1")]
    story += day_tables([(f"{det_w}-session " + {"Q1": "Q1", "Q5": "Q5", "Q5-Q1": "Q5−Q1", "BASE": "All"}[b],
                          (lambda b: lambda d, h: pick(ts_b, day=d, h=h, bucket=b))(b))
                         for b in ["Q1", "Q5", "BASE", "Q5-Q1"]] +
                        [(f"20-session " + {"Q1": "Q1", "Q5": "Q5", "Q5-Q1": "Q5−Q1"}[b],
                          (lambda b: lambda d, h: pick(ts_20, day=d, h=h, bucket=b))(b))
                         for b in ["Q1", "Q5", "Q5-Q1"]], "Bucket", first_w=1.9 * inch)
    yt = {}
    for w in RUN_WINDOWS:
        yy = runs[f"w{w}"]["yr"]
        yt[w] = yy[yy.factor == "VV"].groupby(["year", "day"]).q5_minus_q1_rax.mean().unstack("day")
    years = [y for y in yt[20].index if y >= 2023]
    rows = [["Window"] + [f"{y}{' YTD' if y == 2026 else ''} {d.title()}" for y in years for d in ("DOWN", "UP")]]
    for w in RUN_WINDOWS:
        rows.append([str(w)] + [f3(yt[w].loc[y, d]) for y in years for d in ("DOWN", "UP")])
    story += [P("Q5−Q1 by year (mean of 4 horizons)", "h2"),
              table(rows, [0.7 * inch] + [0.8 * inch] * (2 * len(years)))]
    story += [PageBreak()]

    # ----- 3. ROC -----
    story += [P("3. Rate of change since 2023 (level held fixed)", "h1"),
              Image(str(tmp / "roc.png"), width=7.0 * inch, height=4.03 * inch)]
    show = fast_keys + [(20, 10), (50, 20), (60, 10), (120, 20)]
    rows = [["Window / lag"] + [plabel(p) for p in YPF]]
    for k in show:
        rows.append([f"{k[0]} / {k[1]}" + (" (fast)" if k[0] <= 15 else "")] + [ft(rc_t.loc[k, p]) for p in YPF])
    story += [Spacer(1, 4), P("Selected settings, avg t by year", "h2"), table(rows, [1.3 * inch] + [0.95 * inch] * len(YPF))]
    story += [PageBreak()]

    # ----- 4. combos -----
    story += [P("4. Combinations since 2023 — level × ROC", "h1"),
              P("Long-short at quintile corners. Level: low is long. Fast ROC (window ≤ 15): high is long. Slower ROC: "
                "low is long. Positive = the combination works."),
              Image(str(tmp / "combo.png"), width=7.0 * inch, height=2.6 * inch)]
    rows = [["Signal", "Avg long−short", "vs level alone", "Avg t"] + [f"{plabel(p)} t" for p in YP]]
    for k in ["alone"] + list(g["t"]["FULL"].drop("alone").sort_values(ascending=False).index):
        lab = "20-session level alone" if k == "alone" else f"Level + ROC {k}" + (" (fast)" if int(k.split("/")[0]) <= 15 else "")
        rows.append([lab, f3(g["mean"]["FULL"][k]), f"{g['mean']['FULL'][k] / base_sp - 1:+.0%}", ft(g["t"]["FULL"][k])] +
                    [ft(g["t"][p][k]) for p in YP])
    story += [P("Ranking, average of 8 day × horizon cells (20-session level)", "h2"),
              table(rows, [1.8 * inch, 0.95 * inch, 0.9 * inch, 0.6 * inch] + [0.65 * inch] * len(YP))]
    f = cb[cb.period == "FULL"]

    def cfn(kind, roc=None, src=f):
        def fn(d, h):
            s = src[(src.kind == kind) & (src.day == d) & (src.h == h)]
            if roc:
                s = s[s.roc == roc]
            r = s.iloc[0]
            return r["mean"], r["t"]
        return fn

    top3 = list(g["t"]["FULL"].drop("alone").sort_values(ascending=False).index[:3])
    story += [Spacer(1, 4), P("Long-short by horizon", "h2")]
    story += day_tables([("Level alone", cfn("VV_ONLY_Q5"))] + [(f"Level + ROC {k}", cfn("LS_Q5", k)) for k in top3],
                        "Signal")
    if cbw:
        wq = cbw[0]
        rows = [["Signal", "Avg long−short", "Avg t"] + [f"{plabel(p)} t" for p in YP]]
        for k in ["alone"] + list(gw["t"]["FULL"].drop("alone").sort_values(ascending=False).index[:4]):
            lab = f"{wq}-session level alone" if k == "alone" else f"{wq}-level + ROC {k}"
            rows.append([lab, f3(gw["mean"]["FULL"][k]), ft(gw["t"]["FULL"][k])] + [ft(gw["t"][p][k]) for p in YP])
        story += [KeepTogether([P(f"Same test on the {wq}-session level", "h2"),
                                table(rows, [1.9 * inch, 1.0 * inch, 0.7 * inch] + [0.75 * inch] * len(YP))])]
    story += [PageBreak()]

    # ----- 5. robustness -----
    story += [P("5. Robustness since 2023", "h1")]
    story += day_tables([("20-session, 756+ sessions", lambda d, h: pick(ts_20, day=d, h=h)),
                         ("20-session, all names", lambda d, h: pick(runs["w20_all"]["ts"], day=d, h=h)),
                         ("20-session, vs universe daily",
                          lambda d, h: pick(runs["w20"]["xs"], day=d, h=h, rank="cross_section"))] +
                        [(f"{w}-session", (lambda w: lambda d, h: pick(runs[f"w{w}"]["ts"], day=d, h=h))(w))
                         for w in RUN_WINDOWS if w != 20], "Level Q5−Q1", first_w=1.9 * inch)
    rows = [["Signal", "756+ sessions avg t", "300+ sessions avg t", "300+ avg long−short"]]
    for k in ["alone"] + top3:
        lab = "20-session level alone" if k == "alone" else f"Level + ROC {k}"
        rows.append([lab, ft(g["t"]["FULL"][k]), ft(g300["t"]["FULL"][k]), f3(g300["mean"]["FULL"][k])])
    story += [P("Combinations on a broader universe (names with 300+ sessions)", "h2"),
              table(rows, [2.0 * inch, 1.4 * inch, 1.4 * inch, 1.4 * inch])]
    pt = runs["w20"]["pt"]
    pt = pt[(pt.factor == "VV") & (pt.n_q5 >= 15) & (pt.n_q1 >= 15)]
    rows = [["Per ticker (20-session)"] + [f"{'Dn' if d == 'DOWN' else 'Up'} {h}d" for d, h in rp.CELLS]]
    rows.append(["% tickers Q5 < Q1"] + [f"{(pt[(pt.day == d) & (pt.h == h)].q5_minus_q1 < 0).mean():.0%}"
                                          for d, h in rp.CELLS])
    story += [Spacer(1, 6), table(rows, [1.6 * inch] + [0.68 * inch] * 8)]

    # ----- 6. vs 2009-26 -----
    g09 = combo_rank(cb09, None)
    story += [PageBreak(), P("6. Since 2023 vs 2009–26", "h1")]
    rows = [["Signal", "2009–26 avg t", "2023–26 avg t"]]
    for w in [20, 50, 60, 120, 180, best_w]:
        if w in vsc_t.index:
            rows.append([f"Level, {w} sessions", ft(w09.loc[w]), ft(vsc_t.loc[w, "FULL"])])
    rows = [rows[0]] + list(dict((r[0], r) for r in rows[1:]).values())
    r09 = sw09[sw09.factor == "VV_ROC"].groupby(["window", "lag", "period"]).ctrl_t.mean().unstack("period")
    for k in fast_keys + [(50, 20), (60, 10)]:
        rows.append([f"ROC {k[0]}/{k[1]}, level held fixed", ft(r09.loc[k, "FULL"]), ft(rc_t.loc[k, "FULL"])])
    for k in ["alone", "5/5", "10/5", "15/5", "20/10", "50/20"]:
        lab = "Combo: level alone" if k == "alone" else f"Combo: level + ROC {k}"
        rows.append([lab, ft(g09["t"]["FULL"][k]), ft(g["t"]["FULL"][k])])
    story += [table(rows, [2.6 * inch, 1.4 * inch, 1.4 * inch])]
    story += [Spacer(1, 6), P("Code and every CSV behind this report: research/volume_volatility/ on branch "
                              "claude/volume-volatility-backtest-sjbtja (results/2023_2026/).", "small")]

    def on_page(c, doc):
        c.saveState()
        c.setFont("Carlito", 7.5)
        c.setFillColor(colors.HexColor(MUTED))
        c.drawString(0.6 * inch, 0.4 * inch, "Volume volatility · 2023 onward")
        c.drawRightString(letter[0] - 0.6 * inch, 0.4 * inch, f"{doc.page}")
        c.restoreState()

    doc = SimpleDocTemplate(out_pdf, pagesize=letter, leftMargin=0.6 * inch, rightMargin=0.6 * inch,
                            topMargin=0.6 * inch, bottomMargin=0.65 * inch, title="Volume Volatility — 2023 Onward",
                            author="Claude Code")
    doc.build(story, onFirstPage=on_page, onLaterPages=on_page)
    print(f"wrote {out_pdf}")


if __name__ == "__main__":
    main(*sys.argv[1:])
