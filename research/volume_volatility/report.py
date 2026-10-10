"""Build the PDF findings report from the 2009-2026 sweep and backtest runs.

Usage: python report.py <sweep_dir> <sweep_raw_dir> <runs_dir> <out.pdf>
  sweep_dir / sweep_raw_dir: sweep.py outputs (clean / --raw-volume)
  runs_dir: backtest.py outputs named w20, w50, w60, w120, w252, w20_alluniverse,
            w20_cv, w60_cv, w20_rank126, w20_rank504, w20_raw
Palette is blue / orange only (deutan-safe); every colored cell also carries its number.
"""
import sys
import tempfile
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
import numpy as np
import pandas as pd
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import inch
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (Image, KeepTogether, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table,
                                TableStyle)

FONT_DIR = "/usr/share/fonts/truetype/crosextra"
BLUE, ORANGE = "#2a78d6", "#eb6834"
INK, INK2, MUTED, RULE, PANEL = "#0b0b0b", "#52514e", "#8a8984", "#d9d8d3", "#f0efec"
H = [7, 14, 21, 42]
CELLS = [(d, h) for d in ("DOWN", "UP") for h in H]
PERIODS = ["2009-01..2016-09", "2016-10..2022-06", "2022-07..2026-10", "FULL"]
PLABEL = {"2009-01..2016-09": "2009–16", "2016-10..2022-06": "2016–22", "2022-07..2026-10": "2022–26",
          "FULL": "2009–26"}
DIV = LinearSegmentedColormap.from_list(
    "orange_gray_blue", ["#a8441a", ORANGE, "#f6c1a6", PANEL, "#9ec5f4", BLUE, "#184f95"])

# ---------- fonts ----------
pdfmetrics.registerFont(TTFont("Carlito", f"{FONT_DIR}/Carlito-Regular.ttf"))
pdfmetrics.registerFont(TTFont("Carlito-Bold", f"{FONT_DIR}/Carlito-Bold.ttf"))
pdfmetrics.registerFont(TTFont("Carlito-Italic", f"{FONT_DIR}/Carlito-Italic.ttf"))
pdfmetrics.registerFontFamily("Carlito", normal="Carlito", bold="Carlito-Bold", italic="Carlito-Italic")
for f in ("Carlito-Regular.ttf", "Carlito-Bold.ttf"):
    font_manager.fontManager.addfont(f"{FONT_DIR}/{f}")
plt.rcParams.update({"font.family": "Carlito", "font.size": 9, "axes.edgecolor": RULE, "axes.labelcolor": INK2,
                     "xtick.color": INK2, "ytick.color": INK2, "axes.titlecolor": INK, "axes.titlesize": 10,
                     "axes.titleweight": "bold", "axes.unicode_minus": True})

ST = {
    "title": ParagraphStyle("title", fontName="Carlito-Bold", fontSize=20, leading=24, textColor=INK),
    "sub": ParagraphStyle("sub", fontName="Carlito", fontSize=10.5, leading=14, textColor=INK2),
    "h1": ParagraphStyle("h1", fontName="Carlito-Bold", fontSize=14, leading=18, textColor=INK, spaceBefore=10,
                         spaceAfter=4),
    "h2": ParagraphStyle("h2", fontName="Carlito-Bold", fontSize=11, leading=14, textColor=INK, spaceBefore=8,
                         spaceAfter=3),
    "body": ParagraphStyle("body", fontName="Carlito", fontSize=9.5, leading=13, textColor=INK, alignment=TA_LEFT),
    "bullet": ParagraphStyle("bullet", fontName="Carlito", fontSize=9.5, leading=13, textColor=INK, leftIndent=12,
                             bulletIndent=2),
    "small": ParagraphStyle("small", fontName="Carlito", fontSize=8, leading=10.5, textColor=INK2),
    "cell": ParagraphStyle("cell", fontName="Carlito", fontSize=8, leading=10, textColor=INK),
}


def P(text, style="body"):
    return Paragraph(text, ST[style])


def bullets(items):
    return [Paragraph(t, ST["bullet"], bulletText="•") for t in items]


def f3(x):
    return f"{x:+.3f}".replace("-", "−")


def ft(x):
    return f"{x:+.1f}".replace("-", "−")


def cell(m, t):
    return f"{f3(m)} ({ft(t)})"


def table(rows, col_widths, header_rows=1, zebra=False, bold_first_col=True):
    t = Table(rows, colWidths=col_widths, repeatRows=header_rows)
    style = [
        ("FONT", (0, 0), (-1, -1), "Carlito", 8),
        ("FONT", (0, 0), (-1, header_rows - 1), "Carlito-Bold", 8),
        ("TEXTCOLOR", (0, 0), (-1, -1), INK),
        ("BACKGROUND", (0, 0), (-1, header_rows - 1), PANEL),
        ("LINEBELOW", (0, header_rows - 1), (-1, header_rows - 1), 0.6, INK2),
        ("LINEBELOW", (0, header_rows), (-1, -1), 0.25, RULE),
        ("ALIGN", (1, 0), (-1, -1), "RIGHT"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 2.5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 2.5),
        ("LEFTPADDING", (0, 0), (-1, -1), 4),
        ("RIGHTPADDING", (0, 0), (-1, -1), 4),
    ]
    if bold_first_col:
        style.append(("FONT", (0, header_rows), (0, -1), "Carlito-Bold", 8))
    t.setStyle(TableStyle(style))
    return t


# ---------- data access ----------
def load_run(runs, name):
    d = Path(runs) / name
    return {"ts": pd.read_csv(d / "buckets_time_series.csv"), "xs": pd.read_csv(d / "buckets_cross_section.csv"),
            "yr": pd.read_csv(d / "yearly_q5_minus_q1.csv"), "pt": pd.read_csv(d / "per_ticker.csv")}


def pick(df, factor="VV", day="DOWN", h=21, bucket="Q5-Q1", metric="RAX", rank=None):
    s = df[(df.factor == factor) & (df.day == day) & (df.h == h) & (df.bucket == bucket) & (df.metric == metric)]
    if rank is not None:
        s = s[s["rank"] == rank]
    r = s.iloc[0]
    return r["mean"], r["t"]


def day_tables(rows, first_label, first_w=1.7 * inch):
    """rows: [(label, fn(day, h) -> (mean, t))]. Returns a Down table stacked over an Up table."""
    out = []
    for day, name in (("DOWN", "Down days"), ("UP", "Up days")):
        data = [[f"{first_label} — {name}"] + [f"{h}d" for h in H]]
        data += [[lab] + [cell(*fn(day, h)) for h in H] for lab, fn in rows]
        out += [table(data, [first_w] + [1.05 * inch] * 4), Spacer(1, 5)]
    return out


def run_fn(df, rank=None):
    return lambda d, h: pick(df, day=d, h=h, rank=rank)


# ---------- charts ----------
def heat(ax, data, row_labels, col_labels, vmax=4.0, fmt=ft):
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)
    ax.imshow(data, cmap=DIV, norm=norm, aspect="auto")
    ax.set_xticks(range(len(col_labels)), col_labels)
    ax.set_yticks(range(len(row_labels)), row_labels)
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_xticks(np.arange(-.5, len(col_labels)), minor=True)
    ax.set_yticks(np.arange(-.5, len(row_labels)), minor=True)
    ax.grid(which="minor", color="white", linewidth=2)
    ax.tick_params(which="minor", length=0)
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            v = data[i, j]
            if np.isnan(v):
                continue
            ax.text(j, i, fmt(v), ha="center", va="center", fontsize=8,
                    color="white" if abs(v) >= 2.4 else INK)
    return norm


def chart_window_heat(sweep, path):
    vv = sweep[sweep.factor == "VV"].groupby(["window", "period"]).t.mean().unstack("period")[PERIODS]
    fig, ax = plt.subplots(figsize=(6.6, 3.9), dpi=220)
    norm = heat(ax, vv.values, [str(w) for w in vv.index], [PLABEL[p] for p in PERIODS])
    ax.set_ylabel("Volume-volatility window (sessions)")
    ax.set_title("Q5−Q1 by window and period — average t across 8 cells", loc="left")
    cb = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=DIV), ax=ax, fraction=0.04, pad=0.02)
    cb.set_label("avg t  (orange = high volume volatility underperforms)", color=INK2)
    cb.outline.set_visible(False)
    fig.tight_layout()
    fig.savefig(path, facecolor="white")
    plt.close(fig)


def chart_roc_heat(sweep, path):
    roc = sweep[(sweep.factor == "VV_ROC") & (sweep.period == "FULL")]
    g = roc.groupby(["window", "lag"]).ctrl_t.mean().unstack("lag")
    fig, ax = plt.subplots(figsize=(7.2, 4.15), dpi=220)
    norm = heat(ax, g.values, [str(w) for w in g.index], [str(c) for c in g.columns], vmax=3.5)
    ax.set_xlabel("ROC lag (sessions)")
    ax.set_ylabel("Volume-volatility window (sessions)")
    ax.set_title("ROC with volume-volatility level held fixed — 2009–26 average t", loc="left")
    cb = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=DIV), ax=ax, fraction=0.035, pad=0.02)
    cb.set_label("avg t  (orange = rising underperforms · blue = rising outperforms)", color=INK2)
    cb.outline.set_visible(False)
    fig.tight_layout()
    fig.savefig(path, facecolor="white")
    plt.close(fig)


def chart_quintiles(ts, path, h=21):
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.6), dpi=220, sharey=True)
    for ax, day, title in zip(axes, ("DOWN", "UP"), ("Down days", "Up days")):
        vals = [pick(ts, day=day, h=h, bucket=f"Q{q}")[0] for q in range(1, 6)]
        ax.bar(range(1, 6), vals, color=BLUE, width=0.62, edgecolor="white", linewidth=1)
        ax.axhline(0, color=INK2, linewidth=0.8)
        for q, v in zip(range(1, 6), vals):
            ax.text(q, v + (0.0012 if v >= 0 else -0.0012), f3(v), ha="center",
                    va="bottom" if v >= 0 else "top", fontsize=8, color=INK)
        ax.set_xticks(range(1, 6), [f"Q{q}" for q in range(1, 6)])
        ax.set_title(f"{title} — {h}d", loc="left")
        ax.grid(axis="y", color=RULE, linewidth=0.5)
        ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    axes[0].set_ylabel("Market-neutral risk-adj. return")
    lim = max(abs(np.array(axes[0].get_ylim()))) * 1.15
    axes[0].set_ylim(-lim, lim)
    fig.text(0.01, 0.005, "Q1 = calmest volume vs the ticker's own past year · Q5 = most erratic", fontsize=7.5,
             color=MUTED)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(path, facecolor="white")
    plt.close(fig)


def chart_yearly(yr, path):
    y = yr[yr.factor == "VV"].groupby(["year", "day"]).q5_minus_q1_rax.mean().unstack("day").dropna(how="all")
    y = y[y.index >= 2009]
    x = np.arange(len(y))
    fig, axes = plt.subplots(2, 1, figsize=(7.0, 3.9), dpi=220, sharex=True, sharey=True)
    for ax, day, name in zip(axes, ("DOWN", "UP"), ("Down days", "Up days")):
        ax.bar(x, y[day], width=0.66, color=BLUE, edgecolor="white", linewidth=0.8)
        ax.axhline(0, color=INK2, linewidth=0.8)
        ax.set_title(f"{name} — 20-session window, Q5−Q1 by year (mean of 4 horizons)", loc="left")
        ax.grid(axis="y", color=RULE, linewidth=0.5)
        ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    axes[1].set_xticks(x, [str(v) if v < 2026 else "2026\nYTD" for v in y.index], fontsize=8)
    fig.tight_layout()
    fig.savefig(path, facecolor="white")
    plt.close(fig)
    return y


# ---------- report ----------
def main(sweep_dir, sweep_raw_dir, runs_dir, out_pdf):
    sweep = pd.read_csv(Path(sweep_dir) / "sweep.csv")
    sweep_raw = pd.read_csv(Path(sweep_raw_dir) / "sweep.csv")
    R = {n: load_run(runs_dir, n) for n in ["w20", "w50", "w60", "w120", "w252", "w20_alluniverse", "w20_cv",
                                            "w60_cv", "w20_rank126", "w20_rank504", "w20_raw"]}
    ts20 = R["w20"]["ts"]
    tmp = Path(tempfile.mkdtemp())
    chart_window_heat(sweep, tmp / "heat.png")
    chart_roc_heat(sweep, tmp / "roc.png")
    chart_quintiles(ts20, tmp / "quint.png")
    yearly = chart_yearly(R["w20"]["yr"], tmp / "yearly.png")

    # ----- numbers used in the text -----
    vv_sc = sweep[sweep.factor == "VV"].groupby(["window", "period"]).t.mean().unstack("period")
    best_full = vv_sc["FULL"].idxmin()
    full_yrs = yearly[yearly.index <= 2025]
    neg_down, neg_up, n_yrs = int((full_yrs["DOWN"] < 0).sum()), int((full_yrs["UP"] < 0).sum()), len(full_yrs)
    d_sp = [pick(ts20, day="DOWN", h=h) for h in H]
    u_sp = [pick(ts20, day="UP", h=h) for h in H]
    q1d = [pick(ts20, day="DOWN", h=h, bucket="Q1") for h in H]
    q5u = [pick(ts20, day="UP", h=h, bucket="Q5") for h in H]
    ret_q1 = pick(ts20, day="DOWN", h=21, bucket="Q1", metric="RET")[0]
    ret_q5 = pick(ts20, day="DOWN", h=21, bucket="Q5", metric="RET")[0]
    corr = lambda a, b: vv_sc[a].corr(vv_sc[b], method="spearman")
    roc = sweep[sweep.factor == "VV_ROC"].groupby(["window", "lag", "period"]).ctrl_t.mean().unstack("period")
    roc_consistent = int((roc[PERIODS[:3]] <= -1.0).all(axis=1).sum())
    roc_top = roc.sort_values("FULL").head(6)
    roc_sp = sweep[sweep.factor == "VV_ROC"].groupby(["window", "lag", "period"]).ctrl_spread.mean().unstack("period")
    short = roc_sp.loc[(roc_sp.index.get_level_values(0) <= 15) & (roc_sp.index.get_level_values(1) <= 15)]
    short_pos = int((short[PERIODS[:3]] > 0).all(axis=1).sum())
    short_keys = [(5, 5), (10, 5), (15, 3), (15, 5)]
    short_t = [roc.loc[k, "FULL"] for k in short_keys]
    clean_vs_raw = (sweep[sweep.factor == "VV"].groupby(["window", "period"]).t.mean()
                    - sweep_raw[sweep_raw.factor == "VV"].groupby(["window", "period"]).t.mean()).abs().max()
    rng = lambda xs: f"{f3(min(x[0] for x in xs))} to {f3(max(x[0] for x in xs))}"
    trng = lambda xs: f"{ft(min(x[1] for x in xs))} to {ft(max(x[1] for x in xs))}"

    story = []
    story += [P("Volume Volatility Backtest", "title"), Spacer(1, 2),
              P("2,367 tickers · daily bars 2005–2026 · signals scored 2009-01 to 2026-10 · forward 7 / 14 / 21 / "
                "42 sessions", "sub"), Spacer(1, 10)]

    story += [P("Bottom line", "h1")]
    story += bullets([
        f"<b>High volume volatility predicts negative forward risk-adjusted returns.</b> 20-session window, ranked "
        f"against each ticker's own past year. Down days: Q5−Q1 {rng(d_sp)} (t {trng(d_sp)}). Up days: "
        f"{rng(u_sp)} (t {trng(u_sp)}).",
        f"<b>Best bucket: down day + calmest volume (Q1)</b>, {rng(q1d)} (t {trng(q1d)}). Raw 21d forward return "
        f"{ret_q1 * 100:.2f}% for Q1 vs {ret_q5 * 100:.2f}% for Q5 on down days.",
        f"<b>Worst bucket: up day + most erratic volume (Q5)</b>, {rng(q5u)} (t {trng(q5u)}).",
        f"<b>Every year.</b> The 20-session down-day spread is negative in {neg_down} of {n_yrs} full years "
        f"(2009–2025); up days {neg_up} of {n_yrs}.",
        f"<b>Window: 20 sessions is the best setting over 2009–26</b> (avg t {ft(vv_sc.loc[20, 'FULL'])}; strongest of "
        f"the 15 tested, 5 to 252). Windows from 10 to 50 all work. The 60–252 windows only worked in 2022–26. 252 is "
        f"the weakest (avg t {ft(vv_sc.loc[252, 'FULL'])}).",
        f"<b>Rate of change, slow version: nothing durable.</b> With the volume-volatility level held fixed, "
        f"{roc_consistent} of 180 window × lag settings stay negative (t ≤ −1) in all three periods. The best ones "
        f"worked in 2009–2022 and went flat in 2022–26.",
        f"<b>Rate of change, fast version: a jump predicts outperformance.</b> At a 5–15 session window and "
        f"3–5 session lag, a short-term rise in volume volatility at a given level is positive in all three "
        f"periods. Full-sample t {ft(min(short_t))} to {ft(max(short_t))}. That runs opposite to the level effect.",
    ])
    story += [Spacer(1, 6), P("How to read the numbers", "h2"),
              P("Each figure is the forward return divided by the stock's trailing 20-day volatility scaled to the "
                "horizon (σ·√h), minus that day's universe average. 0 = in line with the universe; −0.030 = 3% of a "
                "horizon-σ worse. Q1–Q5 are fifths of each ticker's own trailing-252-session distribution of the "
                "factor. Brackets hold Newey-West t-stats (lag = horizon). Each bucket is averaged across names per "
                "date, then across dates.", "small")]

    story += [P("Setup", "h1")]
    setup = [
        ["Universe", "2,367 tickers from the finviz export (today's list); a name is scored once it has 756 sessions "
                     "of history, so every setting is tested on the same names"],
        ["Data", "Yahoo daily OHLCV via yfinance, split/dividend adjusted, 2005-01 to 2026-10. Volume confirmed "
                 "split-adjusted"],
        ["Volume volatility", "Stdev of daily ln(V<sub>t</sub>/V<sub>t−1</sub>) over N sessions. Bad prints (<2% "
                              "of 20d median) dropped; daily change capped at 50×"],
        ["Rate of change", "VV<sub>t</sub> / VV<sub>t−L</sub> − 1"],
        ["Day type", "Up: close > prior close. Down: close < prior close"],
        ["Entry / filter", "Close of signal day. Price ≥ $1, 20d median dollar volume ≥ $500k"],
        ["Periods", "2009-01 to 2016-09 (never used in earlier rounds) · 2016-10 to 2022-06 · 2022-07 to 2026-10"],
    ]
    story += [Table([[P(f"<b>{a}</b>", "cell"), P(b, "cell")] for a, b in setup], colWidths=[1.25 * inch, 5.9 * inch],
                    style=TableStyle([("LINEBELOW", (0, 0), (-1, -1), 0.25, RULE), ("VALIGN", (0, 0), (-1, -1), "TOP"),
                                      ("TOPPADDING", (0, 0), (-1, -1), 2), ("BOTTOMPADDING", (0, 0), (-1, -1), 2)]))]
    story += [PageBreak()]

    # ----- window sweep -----
    story += [P("1. Window length: 5 to 252 sessions", "h1"),
              P(f"Fifteen windows, each scored over three periods on the same dates and names. The rate of change in "
                f"Section 3 uses the same grid."),
              Spacer(1, 4), Image(str(tmp / "heat.png"), width=6.6 * inch, height=3.9 * inch)]
    story += bullets([
        f"2009–16, never examined before this round: 10–20 sessions work (20: avg t "
        f"{ft(vv_sc.loc[20, PERIODS[0]])}). 40 sessions and longer show nothing.",
        f"2016–22: 10–50 work. 2022–26: 50–180 work best, short windows fade.",
        f"The ranking of windows flips between periods: rank correlation of window avg t, 2009–16 vs 2022–26 = "
        f"{corr(PERIODS[0], PERIODS[2]):+.2f}, 2016–22 vs 2022–26 = {corr(PERIODS[1], PERIODS[2]):+.2f}. "
        f"The 40–60 call from the 2018–26 round was a 2022–26 effect.",
        f"Over the full 2009–26 sample the strongest window is {best_full} (avg t {ft(vv_sc.loc[best_full, 'FULL'])}).",
    ])
    story += [Spacer(1, 6), P("Full-sample Q5−Q1 by cell, selected windows (2009–26)", "h2")]
    full = sweep[(sweep.factor == "VV") & (sweep.period == "FULL")]

    def sweep_fn(w):
        g = full[full.window == w]
        return lambda d, h: tuple(g[(g.day == d) & (g.h == h)][["spread", "t"]].iloc[0])

    story += day_tables([(f"{w} sessions", sweep_fn(w)) for w in [5, 10, 20, 40, 60, 120, 252]], "Window")
    story += [PageBreak()]

    # ----- detail at 20 -----
    story += [P("2. The 20-session window in detail (2009–26)", "h1"),
              Image(str(tmp / "quint.png"), width=7.0 * inch, height=2.6 * inch), Spacer(1, 4)]
    for day, name in (("DOWN", "Down days"), ("UP", "Up days")):
        rows = [[name] + [f"{h}d" for h in H]]
        for b in ["Q1", "Q2", "Q3", "Q4", "Q5", "BASE", "Q5-Q1"]:
            lab = {"BASE": "All", "Q5-Q1": "Q5−Q1"}.get(b, b)
            rows.append([lab] + [cell(*pick(ts20, day=day, h=h, bucket=b)) for h in H])
        story += [KeepTogether([table(rows, [0.8 * inch] + [1.1 * inch] * 4)]), Spacer(1, 6)]
    story += bullets([
        f"Down days: Q1 through Q5 step down in order at every horizon. The calmest-volume Q1 bucket is positive "
        f"with t {trng(q1d)}.",
        f"Up days: the spread builds with horizon and is strongest at 42d.",
    ])
    story += [PageBreak(), P("2b. Year by year", "h1"),
              Image(str(tmp / "yearly.png"), width=7.0 * inch, height=3.9 * inch), Spacer(1, 4)]
    story += bullets([f"20-session window: down-day spread negative in {neg_down}/{n_yrs} full years (2009–2025), "
                      f"up-day {neg_up}/{n_yrs}. 2026 YTD is positive."])
    yt = {}
    for n in ("w20", "w60", "w252"):
        yy = R[n]["yr"]
        yt[n] = yy[yy.factor == "VV"].groupby(["year", "day"]).q5_minus_q1_rax.mean().unstack("day")
    rows = [["Year", "20 · Down", "20 · Up", "60 · Down", "60 · Up", "252 · Down", "252 · Up"]]
    for yr in [v for v in yt["w20"].index if v >= 2009]:
        rows.append([str(yr) if yr < 2026 else "2026 YTD"] +
                    [f3(yt[n].loc[yr, d]) for n in ("w20", "w60", "w252") for d in ("DOWN", "UP")])
    cnt = ["Negative (of 17)"] + [f"{int((yt[n].loc[2009:2025, d] < 0).sum())}" for n in ("w20", "w60", "w252")
                                  for d in ("DOWN", "UP")]
    rows.append(cnt)
    story += [Spacer(1, 4), P("Q5−Q1 by year and window (mean of 4 horizons)", "h2"),
              table(rows, [1.15 * inch] + [0.95 * inch] * 6)]
    story += [PageBreak()]

    # ----- ROC -----
    story += [P("3. Rate of change in volume volatility", "h1"),
              P("Without holding the level fixed, ROC mostly re-measures the level: when volume volatility is rising, "
                "it is usually already high. The test that matters is ROC's Q5−Q1 inside each third of "
                "volume-volatility level, averaged. That is ROC's independent signal."),
              Spacer(1, 4), Image(str(tmp / "roc.png"), width=7.0 * inch, height=4.03 * inch)]
    rows = [["Window / lag"] + [PLABEL[p] for p in PERIODS]]
    for (w, l), r in roc_top.iterrows():
        rows.append([f"{w} / {l}"] + [ft(r[p]) for p in PERIODS])
    story += [Spacer(1, 4), P("Strongest settings, level held fixed (avg t by period)", "h2"),
              table(rows, [1.1 * inch] + [1.0 * inch] * 4)]
    story += bullets([
        f"Slow ROC (orange block, window 50–120, lag 10–20): worked in 2009–16 and 2016–22 at t ≈ −2, then went "
        f"flat or positive in 2022–26. Settings with t ≤ −1 in all three periods: {roc_consistent} of 180. "
        f"The 60/10 setting from the last round follows that pattern.",
    ])
    rows = [["Window / lag"] + [PLABEL[p] for p in PERIODS]]
    for k in short_keys:
        rows.append([f"{k[0]} / {k[1]}"] + [ft(roc.loc[k, p]) for p in PERIODS])
    story += [Spacer(1, 4), P("Fast ROC, level held fixed (avg t by period)", "h2"),
              table(rows, [1.1 * inch] + [1.0 * inch] * 4)]
    story += bullets([
        f"Fast ROC (blue block, window 5–15, lag 3–15): positive spread in all three periods for {short_pos} of "
        f"{len(short)} settings. Positive Q5−Q1 means a recent jump in volume volatility outperforms a recent drop "
        f"at the same level.",
        "Weaker than the level effect: per-period t mostly +0.6 to +2.3, full-sample +2.1 to +2.6 on the best "
        "settings. It points the other way from the level, so the two are separate signals.",
    ])
    story += [PageBreak()]

    # ----- robustness -----
    story += [P("4. Robustness — 20-session window, 2009–26, Q5−Q1", "h1")]
    rob = [("Baseline", R["w20"]["ts"]), ("Raw volume (no cleaning)", R["w20_raw"]["ts"]),
           ("CV of volume levels", R["w20_cv"]["ts"]), ("Rank lookback 126", R["w20_rank126"]["ts"]),
           ("Rank lookback 504", R["w20_rank504"]["ts"]), ("No history minimum", R["w20_alluniverse"]["ts"])]
    rob_rows = [(lab, run_fn(df)) for lab, df in rob]
    rob_rows.append(("Ranked vs universe daily", run_fn(R["w20"]["xs"], rank="cross_section")))
    story += day_tables(rob_rows, "Variant")
    story += [P("Other windows, full backtest (own-history rank)", "h2")]
    story += day_tables([(lab, run_fn(R[n]["ts"])) for lab, n in
                         [("20 sessions", "w20"), ("50 sessions", "w50"), ("60 sessions", "w60"),
                          ("120 sessions", "w120"), ("252 sessions", "w252"), ("60 sessions, CV", "w60_cv")]],
                        "Window")
    pt = R["w20"]["pt"]
    pt = pt[(pt.factor == "VV") & (pt.n_q5 >= 30) & (pt.n_q1 >= 30)]
    rows = [["Per ticker"] + [f"{'Dn' if d == 'DOWN' else 'Up'} {h}d" for d, h in CELLS]]
    rows.append(["% tickers Q5 < Q1"] + [f"{(pt[(pt.day == d) & (pt.h == h)].q5_minus_q1 < 0).mean():.0%}"
                                          for d, h in CELLS])
    rows.append(["Median IC"] + [f3(pt[(pt.day == d) & (pt.h == h)].ic.median()) for d, h in CELLS])
    story += [Spacer(1, 6), P("Ticker by ticker (20-session window)", "h2"), table(rows, [1.5 * inch] + [0.7 * inch] * 8)]
    story += bullets([
        f"Data cleaning moves no window's t by more than {clean_vs_raw:.2f}.",
        "Each robustness variant keeps the sign. Ranking each ticker against its own history beats ranking against "
        "the universe each day.",
        f"Ticker by ticker the edge is thin: {int(pt.ticker.nunique()):,} tickers, median IC near zero. "
        "It shows up across the list as a group, not as a timing signal for one name.",
    ])

    story += [PageBreak(), P("5. What changed from earlier rounds", "h1")]
    story += bullets([
        "Round 1 (2016–26, 20/10): volume volatility negative, ROC weak. Holds.",
        "Round 2 (2016–26 sweep to 60): recommended 40–60. Reversed. That result came from 2022–26. Over "
        "2009–26, 20 is the strongest and long windows fail in 2009–16.",
        "Round 3 (2018–26, out-of-sample split): dropped ROC. Refined. Slow ROC with the level held fixed worked "
        "for 2009–2022 and has been flat since mid-2022. Fast ROC (5–15 / 3–5) is a separate, positive signal "
        "that holds in all three periods.",
        "Fixed along the way: the risk-adjustment volatility was tied to the volume-volatility window (now fixed "
        "at 20), and bad volume prints are now cleaned. Neither changed the 20-session results.",
    ])
    story += [Spacer(1, 6), P("Survivorship: the universe is today's list run backward, so names that delisted "
                              "before 2026 are absent. All figures are relative to that universe.", "small"),
              Spacer(1, 4), P("Code and every CSV behind this report: research/volume_volatility/ on branch "
                               "claude/volume-volatility-backtest-sjbtja.", "small")]

    def on_page(c, doc):
        c.saveState()
        c.setFont("Carlito", 7.5)
        c.setFillColor(colors.HexColor(MUTED))
        c.drawString(0.6 * inch, 0.4 * inch, "Volume volatility backtest · 2009–2026")
        c.drawRightString(letter[0] - 0.6 * inch, 0.4 * inch, f"{doc.page}")
        c.restoreState()

    doc = SimpleDocTemplate(out_pdf, pagesize=letter, leftMargin=0.6 * inch, rightMargin=0.6 * inch,
                            topMargin=0.6 * inch, bottomMargin=0.65 * inch, title="Volume Volatility Backtest",
                            author="Claude Code")
    doc.build(story, onFirstPage=on_page, onLaterPages=on_page)
    print(f"wrote {out_pdf}")


if __name__ == "__main__":
    main(*sys.argv[1:])
