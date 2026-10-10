"""Render the Q4 tax-loss study as a PDF report (reportlab + matplotlib).

usage: python make_pdf.py DASH_DIR RESULTS_DIR OUT.pdf
DASH_DIR holds the JSON datasets from export_dashboard.py, RESULTS_DIR the CSVs from the study.
"""
import io
import json
import sys
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import font_manager  # noqa: E402
from reportlab.lib import colors  # noqa: E402
from reportlab.lib.enums import TA_LEFT  # noqa: E402
from reportlab.lib.pagesizes import letter  # noqa: E402
from reportlab.lib.styles import ParagraphStyle  # noqa: E402
from reportlab.lib.units import inch  # noqa: E402
from reportlab.pdfbase import pdfmetrics  # noqa: E402
from reportlab.pdfbase.ttfonts import TTFont  # noqa: E402
from reportlab.platypus import (  # noqa: E402
    CondPageBreak, Image, KeepTogether, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle,
)

DASH, RES, OUT = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
FONT_DIR = Path("/usr/share/fonts/truetype/crosextra")

# ---------------------------------------------------------------- fonts + palette
pdfmetrics.registerFont(TTFont("Sans", str(FONT_DIR / "Carlito-Regular.ttf")))
pdfmetrics.registerFont(TTFont("Sans-Bold", str(FONT_DIR / "Carlito-Bold.ttf")))
pdfmetrics.registerFont(TTFont("Sans-Italic", str(FONT_DIR / "Carlito-Italic.ttf")))
pdfmetrics.registerFont(TTFont("Serif", str(FONT_DIR / "Caladea-Regular.ttf")))
pdfmetrics.registerFont(TTFont("Serif-Bold", str(FONT_DIR / "Caladea-Bold.ttf")))
pdfmetrics.registerFontFamily("Sans", normal="Sans", bold="Sans-Bold", italic="Sans-Italic", boldItalic="Sans-Bold")
for f in ("Carlito-Regular.ttf", "Carlito-Bold.ttf"):
    font_manager.fontManager.addfont(str(FONT_DIR / f))

INK, MUTED, RULE, PANEL = "#1b1b19", "#66655f", "#d6d5cf", "#f3f2ee"
BLUE, ORANGE, GRAY = "#2a78d6", "#eb6834", "#a3a29b"   # deutan-safe: blue / orange / neutral
ERA_COLOR = {"1999-2007": GRAY, "2008-2016": BLUE, "2017-2025": ORANGE, "2017-2026": ORANGE}
ERA3 = ["1999-2007", "2008-2016", "2017-2025"]
DASHES = {"1999-2007": "1999–2007", "2008-2016": "2008–2016", "2017-2025": "2017–2025", "2017-2026": "2017–2026",
          "1999-2025": "1999–2025"}

plt.rcParams.update({
    "font.family": "Carlito", "font.size": 8.5, "axes.edgecolor": RULE, "axes.labelcolor": MUTED,
    "xtick.color": MUTED, "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False,
    "axes.spines.left": False, "axes.grid": True, "axes.grid.axis": "y", "grid.color": "#e6e5df",
    "grid.linewidth": 0.6, "axes.axisbelow": True, "legend.frameon": False, "legend.fontsize": 8,
    "xtick.major.size": 0, "ytick.major.size": 0,
})


def load(name):
    return pd.DataFrame(json.loads((DASH / f"{name}.json").read_text()))


H = load("headline").set_index("id")["value"].to_dict()
QR, HM, MA, CP = load("quarter_risk"), load("halfmonth"), load("monthly_alpha"), load("cum_paths")
YR, WN, SP, RB = load("yearly"), load("windows"), load("spreads"), load("robust")
LV, SC = load("live").set_index("group"), load("screen")
LS = pd.read_csv(RES / "long_short.csv")
OCT_SLOPE_T = pd.read_csv(RES / "trend_tests.csv").set_index("window").loc["Oct", "slope_t"]


def minus(s):
    return s.replace("-", "−")


def pct(v, d=1, sign=False):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "—"
    return minus(f"{v * 100:{'+' if sign else ''}.{d}f}%")


def num(v, d=2, sign=False):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "—"
    return minus(f"{v:{'+' if sign else ''}.{d}f}")


def money(v):
    for div, suf in ((1e12, "T"), (1e9, "B"), (1e6, "M"), (1e3, "K")):
        if abs(v) >= div:
            return f"${v / div:.1f}{suf}"
    return f"${v:.0f}"


# ---------------------------------------------------------------- styles
ST = {
    "title": ParagraphStyle("title", fontName="Serif", fontSize=26, leading=30, textColor=INK, spaceAfter=4),
    "sub": ParagraphStyle("sub", fontName="Sans", fontSize=10, leading=13, textColor=MUTED, spaceAfter=14),
    "h1": ParagraphStyle("h1", fontName="Serif", fontSize=16, leading=20, textColor=INK, spaceBefore=6, spaceAfter=6),
    "h2": ParagraphStyle("h2", fontName="Sans-Bold", fontSize=10.5, leading=13, textColor=INK, spaceBefore=8,
                         spaceAfter=3),
    "body": ParagraphStyle("body", fontName="Sans", fontSize=10, leading=14, textColor=INK, spaceAfter=6),
    "note": ParagraphStyle("note", fontName="Sans", fontSize=8.5, leading=11.5, textColor=MUTED, spaceAfter=6),
    "bullet": ParagraphStyle("bullet", fontName="Sans", fontSize=10, leading=14, textColor=INK, leftIndent=12,
                             bulletIndent=0, spaceAfter=5),
    "bluf": ParagraphStyle("bluf", fontName="Sans", fontSize=11, leading=15.5, textColor=INK, alignment=TA_LEFT),
    "k": ParagraphStyle("k", fontName="Sans", fontSize=8, leading=10, textColor=MUTED),
    "v": ParagraphStyle("v", fontName="Serif", fontSize=20, leading=24, textColor=INK),
    "s": ParagraphStyle("s", fontName="Sans", fontSize=8, leading=10.5, textColor=MUTED),
    "cell": ParagraphStyle("cell", fontName="Sans", fontSize=8, leading=9.5, textColor=INK),
}
W = letter[0] - 1.2 * inch


def P(t, s="body"):
    return Paragraph(t, ST[s])


def fig_image(fig, width=W):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    buf.seek(0)
    img = Image(buf)
    ratio = img.imageHeight / img.imageWidth
    img.drawWidth, img.drawHeight = width, width * ratio
    return img


def table(rows, col_widths, align_left=(0,), bold_rows=(), font=8, zebra=False, header_rows=1):
    t = Table(rows, colWidths=col_widths, repeatRows=header_rows)
    style = [
        ("FONT", (0, 0), (-1, -1), "Sans", font),
        ("FONT", (0, 0), (-1, header_rows - 1), "Sans-Bold", font),
        ("TEXTCOLOR", (0, 0), (-1, header_rows - 1), MUTED),
        ("TEXTCOLOR", (0, header_rows), (-1, -1), INK),
        ("ALIGN", (0, 0), (-1, -1), "RIGHT"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LINEBELOW", (0, header_rows - 1), (-1, header_rows - 1), 0.8, colors.HexColor(INK)),
        ("LINEBELOW", (0, header_rows), (-1, -1), 0.3, colors.HexColor(RULE)),
        ("TOPPADDING", (0, 0), (-1, -1), 2.5), ("BOTTOMPADDING", (0, 0), (-1, -1), 2.5),
        ("LEFTPADDING", (0, 0), (-1, -1), 4), ("RIGHTPADDING", (0, 0), (-1, -1), 4),
    ]
    for c in align_left:
        style.append(("ALIGN", (c, 0), (c, -1), "LEFT"))
    for r in bold_rows:
        style.append(("FONT", (0, r), (-1, r), "Sans-Bold", font))
    if zebra:
        for r in range(header_rows, len(rows), 2):
            style.append(("BACKGROUND", (0, r), (-1, r), colors.HexColor(PANEL)))
    t.setStyle(TableStyle(style))
    return t


def tiles(items, cols=2):
    cells = []
    for k, v, s in items:
        cells.append([P(k, "k"), Spacer(1, 2), P(v, "v"), Spacer(1, 2), P(s, "s")])
    rows = [cells[i:i + cols] for i in range(0, len(cells), cols)]
    gap = 8
    cw = (W - gap * (cols - 1)) / cols
    widths = []
    for i in range(cols):
        widths += [cw] + ([gap] if i < cols - 1 else [])
    grid = []
    for r in rows:
        line = []
        for i, c in enumerate(r):
            line += [c] + ([""] if i < cols - 1 else [])
        grid.append(line)
    t = Table(grid, colWidths=widths)
    st = [("VALIGN", (0, 0), (-1, -1), "TOP"), ("TOPPADDING", (0, 0), (-1, -1), 9),
          ("BOTTOMPADDING", (0, 0), (-1, -1), 9), ("LEFTPADDING", (0, 0), (-1, -1), 10),
          ("RIGHTPADDING", (0, 0), (-1, -1), 10)]
    for ri in range(len(grid)):
        for ci in range(0, len(widths), 2):
            st += [("BACKGROUND", (ci, ri), (ci, ri), colors.HexColor(PANEL)),
                   ("BOX", (ci, ri), (ci, ri), 0.5, colors.HexColor(RULE))]
    if len(grid) > 1:
        st.append(("TOPPADDING", (0, 1), (-1, -1), 9))
    t.setStyle(TableStyle(st))
    return t


def legend_handles(eras):
    from matplotlib.patches import Patch
    return [Patch(color=ERA_COLOR[e], label=DASHES[e]) for e in eras]


def pct_axis(ax, d=1):
    ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: minus(f"{v * 100:+.{d}f}%")))
    ax.axhline(0, color="#8c8b85", lw=0.8, zorder=1)


# ---------------------------------------------------------------- charts
HW = ["Oct1_15", "Oct16_31", "Nov1_15", "Nov16_30", "Dec1_15", "Dec16_31", "Jan1_15", "Jan16_31"]
HW_LAB = ["Oct\n1–15", "Oct\n16–31", "Nov\n1–15", "Nov\n16–30", "Dec\n1–15", "Dec\n16–31", "Jan\n1–15", "Jan\n16–31"]


def chart_halfmonth(pair, title):
    d = HM[(HM.pair == pair) & HM.era.isin(ERA3)]
    fig, ax = plt.subplots(figsize=(7.3, 2.05))
    x = np.arange(len(HW))
    bw = 0.26
    for i, e in enumerate(ERA3):
        vals = [d[(d.window == w) & (d.era == e)]["mean"].iloc[0] for w in HW]
        ax.bar(x + (i - 1) * bw, vals, bw * 0.9, color=ERA_COLOR[e], zorder=2)
    ax.set_xticks(x, HW_LAB)
    pct_axis(ax)
    ax.set_title(title, loc="left", fontsize=9.5, color=INK, fontweight="bold")
    ax.legend(handles=legend_handles(ERA3), ncol=3, loc="upper right", bbox_to_anchor=(1, 1.18))
    return fig_image(fig)


def chart_paths():
    fig, axes = plt.subplots(1, 2, figsize=(7.3, 2.25), sharey=True)
    ticks = [1, 32, 62, 93, 123]
    labs = ["Oct 1", "Nov 1", "Dec 1", "Jan 1", "Jan 31"]
    for ax, (pair, title) in zip(axes, [("NEG_1Y-POS_1Y", "1Y neg − 1Y pos"),
                                        ("NEG1Y_NEG1M-POS1Y_POS1M", "1Y+1M neg − 1Y+1M pos")]):
        for e in ERA3:
            d = CP[(CP.pair == pair) & (CP.era == e)].sort_values("day")
            ax.plot(d.day, d.rel, color=ERA_COLOR[e], lw=2.2 if e == "2017-2025" else 1.6,
                    solid_capstyle="round", label=DASHES[e], zorder=3)
        ax.set_xticks(ticks, labs)
        for t in ticks[1:-1]:
            ax.axvline(t, color="#e6e5df", lw=0.6, zorder=0)
        pct_axis(ax)
        ax.set_title(title, loc="left", fontsize=9.5, color=INK, fontweight="bold")
    axes[1].legend(loc="upper left", ncol=1)
    fig.tight_layout(w_pad=2)
    return fig_image(fig)


def chart_calendar():
    fig, axes = plt.subplots(2, 1, figsize=(7.3, 4.3), sharex=True)
    eras = ["1999-2007", "2008-2016", "2017-2026"]
    x = np.arange(1, 13)
    bw = 0.26
    for ax, (pair, title) in zip(axes, [("NEG1Y-POS1Y", "1Y neg − 1Y pos"), ("NEG1M-POS1M", "1M neg − 1M pos")]):
        ax.axvspan(9.5, 12.5, color="#efeee9", zorder=0)
        for i, e in enumerate(eras):
            d = MA[(MA.pair == pair) & (MA.era == e)].sort_values("month")
            ax.bar(x + (i - 1) * bw, d.alpha, bw * 0.9, color=ERA_COLOR[e], zorder=2)
        pct_axis(ax)
        ax.set_title(title + " · next-month alpha after SPY beta", loc="left", fontsize=9.5, color=INK,
                     fontweight="bold")
    axes[1].set_xticks(x, ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"])
    axes[0].legend(handles=legend_handles(eras), ncol=3, loc="upper right", bbox_to_anchor=(1, 1.22))
    fig.tight_layout(h_pad=1.2)
    return fig_image(fig)


def chart_yearly():
    spec = [("Oct1_Oct15_np", "Oct 1–15: 1Y neg − 1Y pos"),
            ("Oct15_Dec15_nnpp", "Oct 15–Dec 15: 1Y+1M neg − 1Y+1M pos"),
            ("Jan_np", "January: 1Y neg − 1Y pos"),
            ("vol_dec21_31", "Dec 21–31 volume: losers over winners")]
    fig, axes = plt.subplots(2, 2, figsize=(7.3, 4.2))
    for ax, (f, title) in zip(axes.flat, spec):
        d = YR[["year", f]].dropna()
        c = [ERA_COLOR["1999-2007" if y <= 2007 else "2008-2016" if y <= 2016 else "2017-2025"] for y in d.year]
        ax.bar(d.year, d[f], 0.72, color=c, zorder=2)
        for b in (2007.5, 2016.5):
            ax.axvline(b, color="#8c8b85", lw=0.7, ls=(0, (3, 3)), zorder=1)
        pct_axis(ax, 0)
        ax.set_xticks([1999, 2008, 2017, 2025])
        ax.set_title(title, loc="left", fontsize=9, color=INK, fontweight="bold")
    fig.legend(handles=legend_handles(ERA3), ncol=3, loc="upper right", bbox_to_anchor=(0.99, 1.03))
    fig.tight_layout(h_pad=1.6, w_pad=2, rect=(0, 0, 1, 0.97))
    return fig_image(fig)


# ---------------------------------------------------------------- tables
GROUPS_MAIN = ["NEG_1Y", "POS_1Y", "NEG_1M", "POS_1M", "NEG1Y_NEG1M", "NEG1Y_POS1M", "POS1Y_NEG1M", "POS1Y_POS1M",
               "Q1_1Y", "Q2_1Y", "Q3_1Y", "Q4_1Y", "Q5_1Y", "NEG_1Y_SMALL", "POS_1Y_SMALL", "NEG_1Y_LARGE",
               "POS_1Y_LARGE", "ALL", "SPY"]


def risk_table(quarter, era, groups=GROUPS_MAIN):
    d = QR[(QR.quarter == quarter) & (QR.era == era)].set_index("group")
    hdr = ["Bucket", "Names", "Avg qtr", "Median", "Vol", "Sharpe", "Sortino", "Beta", "Alpha", "Beat SPY",
           "Avg DD", "Worst"]
    rows, bold = [hdr], []
    for g in groups:
        if g not in d.index:
            continue
        r = d.loc[g]
        rows.append([r.label, f"{r.avg_n:,.0f}", pct(r.q_mean, 1, True), pct(r.q_median, 1, True), pct(r.ann_vol),
                     num(r.sharpe), num(r.sortino), num(r.beta_spy), pct(r.alpha_spy_ann, 1, True),
                     pct(r.hit_vs_spy, 0), pct(r.avg_mdd), pct(r.q_worst)])
        if g == "SPY":
            bold.append(len(rows) - 1)
    cw = [118] + [37.0] * 11
    return table(rows, cw, bold_rows=bold, zebra=True)


def quarter_sharpe_table(era):
    gs = ["NEG_1Y", "POS_1Y", "NEG_1M", "POS_1M", "NEG1Y_NEG1M", "POS1Y_POS1M", "Q1_1Y", "Q5_1Y", "ALL", "SPY"]
    d = QR[QR.era == era]
    rows = [["Bucket", "Q1 Sharpe", "Q2 Sharpe", "Q3 Sharpe", "Q4 Sharpe", "Q1 alpha", "Q2 alpha", "Q3 alpha",
             "Q4 alpha"]]
    for g in gs:
        dd = d[d.group == g].set_index("quarter")
        rows.append([dd.label.iloc[0]] + [num(dd.loc[q, "sharpe"]) for q in ["Q1", "Q2", "Q3", "Q4"]]
                    + [pct(dd.loc[q, "alpha_spy_ann"], 1, True) for q in ["Q1", "Q2", "Q3", "Q4"]])
    return table(rows, [118] + [50.9] * 8, bold_rows=[len(rows) - 1], zebra=True)


def era_shift_table():
    def hm(pair, w, e):
        return HM[(HM.pair == pair) & (HM.window == w) & (HM.era == e)]["mean"].iloc[0]

    def ma(pair, m, e):
        return MA[(MA.pair == pair) & (MA.month == m) & (MA.era == e)]["alpha"].iloc[0]

    def wn(w, e, g, f):
        return WN[(WN.window == w) & (WN.era == e) & (WN.group == g)][f].iloc[0]

    def ls(q, e, f="spread_mean"):
        return LS[(LS.long == "NEG_1Y") & (LS.short == "POS_1Y") & (LS.era == e) & (LS.quarter == q)][f].iloc[0]

    vol = YR.assign(era=np.select([YR.year <= 2007, YR.year <= 2016], ["1999-2007", "2008-2016"], "2017-2025"))
    vol = vol.groupby("era").vol_dec21_31.mean()
    m_era = {"1999-2007": "1999-2007", "2008-2016": "2008-2016", "2017-2025": "2017-2026"}
    rows = [["Measure (1Y neg − 1Y pos unless noted)", "1999–2007", "2008–2016", "2017–2025"],
            ["Q3 spread (Jul–Sep)"] + [pct(ls("Q3", e), 2, True) for e in ERA3],
            ["Oct 1–15 spread"] + [pct(hm("NEG1Y-POS1Y", "Oct1_15", e), 2, True) for e in ERA3],
            ["Oct 1–15, worst − best quintile"] + [pct(hm("Q1-Q5", "Oct1_15", e), 2, True) for e in ERA3],
            ["Nov 1–15 spread"] + [pct(hm("NEG1Y-POS1Y", "Nov1_15", e), 2, True) for e in ERA3],
            ["Dec 1–15 spread"] + [pct(hm("NEG1Y-POS1Y", "Dec1_15", e), 2, True) for e in ERA3],
            ["Dec 16–31 spread"] + [pct(hm("NEG1Y-POS1Y", "Dec16_31", e), 2, True) for e in ERA3],
            ["Full Q4 spread"] + [pct(ls("Q4", e), 2, True) for e in ERA3],
            ["January alpha after SPY beta (monthly sort)"] + [pct(ma("NEG1Y-POS1Y", 1, m_era[e]), 2, True)
                                                              for e in ERA3],
            ["Dec 15→Jan 31, 1Y losers: alpha (ann.)"] + [pct(wn("Dec15_Jan31", e, "NEG_1Y", "alpha_spy_ann"), 1, True)
                                                         for e in ERA3],
            ["Dec 15→Jan 31 Sharpe: losers / winners"] + [
                f"{num(wn('Dec15_Jan31', e, 'NEG_1Y', 'sharpe'))} / {num(wn('Dec15_Jan31', e, 'POS_1Y', 'sharpe'))}"
                for e in ERA3],
            ["Dec 21–31 volume, losers over winners"] + [pct(vol[e], 1, True) for e in ERA3]]
    return table(rows, [235, 96.7, 96.7, 96.7], zebra=True)


WIN_GROUPS = ["NEG1Y_NEG1M", "NEG_1Y", "NEG_1M", "Q1_1Y", "ALL", "SPY", "POS_1M", "POS_1Y", "POS1Y_POS1M", "Q5_1Y"]


def window_table(window, era, groups=WIN_GROUPS):
    d = WN[(WN.window == window) & (WN.era == era)].set_index("group")
    rows, bold = [["Bucket", "Avg", "Median", "Vol", "Sharpe", "Sortino", "Beta", "Alpha", "Up yrs", "t"]], []
    for g in groups:
        if g not in d.index:
            continue
        r = d.loc[g]
        rows.append([r.label, pct(r["mean"], 1, True), pct(r["median"], 1, True), pct(r.ann_vol), num(r.sharpe),
                     num(r.sortino), num(r.beta_spy), pct(r.alpha_spy_ann, 1, True), pct(r.pct_pos, 0),
                     num(r.t, 1, True)])
        if g == "SPY":
            bold.append(len(rows) - 1)
    return table(rows, [133] + [43.6] * 9, bold_rows=bold, zebra=True)


def spread_table(window):
    pairs = [("NEG1Y_NEG1M-POS1Y_POS1M", "1Y+1M neg − 1Y+1M pos"), ("NEG_1Y-POS_1Y", "1Y neg − 1Y pos"),
             ("NEG_1M-POS_1M", "1M neg − 1M pos"), ("Q1_1Y-Q5_1Y", "Worst − best 1Y quintile")]
    rows = [["Pair", "Era", "Mean", "t", "Ahead", "Worst yr", "Best yr"]]
    for p, lab in pairs:
        for e in ERA3 + ["1999-2025"]:
            r = SP[(SP.window == window) & (SP.pair == p) & (SP.era == e)].iloc[0]
            rows.append([lab if e == ERA3[0] else "", DASHES[e], pct(r["mean"], 2, True), num(r.t, 1, True),
                         pct(r.hit, 0), pct(r.worst, 1, True), pct(r.best, 1, True)])
    t = table(rows, [140, 70, 63, 50, 55, 73.5, 73.5], align_left=(0, 1))
    return t


def robust_table():
    order = ["Oct1_15", "Oct15_Dec15", "Oct15_Dec31", "Jan"]
    d = RB.copy()
    rows = [["Window", "Sort", "Min $ vol", "NN−PP mean", "t", "Ahead", "NN−PP median", "1Y N−P mean", "t",
             "NN−PP ex-2020"]]
    keys = d[d.era == "2017-2025"][["window", "sort", "floor"]].drop_duplicates()
    keys["o"] = keys.window.map(order.index)
    keys = keys.sort_values(["o", "sort", "floor"], ascending=[True, False, True])
    for _, k in keys.iterrows():
        def g(metric, era="2017-2025", f="mean"):
            r = d[(d.window == k.window) & (d["sort"] == k["sort"]) & (d.floor == k.floor) & (d.metric == metric)
                  & (d.era == era)]
            return r[f].iloc[0] if len(r) else np.nan
        lab = d[d.window == k.window].window_label.iloc[0]
        rows.append([lab, "Sep 30" if k["sort"] == "sep30" else "Oct 15", money(k.floor), pct(g("nn_pp"), 2, True),
                     num(g("nn_pp", f="t"), 1, True), pct(g("nn_pp", f="hit"), 0), pct(g("nn_pp_med"), 2, True),
                     pct(g("n_p"), 2, True), num(g("n_p", f="t"), 1, True),
                     pct(g("nn_pp", "2017-2025 ex2020"), 2, True)])
    return table(rows, [70, 42, 48, 58, 34, 40, 63, 58, 34, 78], align_left=(0, 1), zebra=True)


def screen_rows(df, start=1):
    out = []
    for i, r in enumerate(df.itertuples(), start):
        name = r.company if len(str(r.company)) <= 30 else str(r.company)[:29] + "…"
        out.append([str(i), r.ticker, name, r.sector, money(r.mcap_m * 1e6), money(r.dollar_vol), pct(r.r12),
                    pct(r.r1), pct(r.qtd, 1, True)])
    return out


SCREEN_HDR = ["#", "Ticker", "Company", "Sector", "Mkt cap", "$ vol/day", "1Y", "1M (Sep)", "QTD"]
SCREEN_CW = [24, 40, 140, 104, 48, 50, 40, 42, 37]

# ---------------------------------------------------------------- story
h = H
story = []
story += [P("Q4 Tax-Loss Seasonality", "title"),
          P(f"{int(h['n_tickers']):,} tickers · every Q4 from 1999 to 2025 · {int(h['n_stock_years']):,} stock-years · "
            "live read through the Oct 9, 2026 close", "sub")]

bluf = (f"<b>The read.</b> Across every Q4 from 1999 to 2025, a negative trailing 1Y is not a risk-adjusted edge: "
        f"1Y losers and winners tie on Sharpe ({num(h['q4_sharpe_neg'])} vs {num(h['q4_sharpe_pos'])}). The timing "
        f"changed. Since 2017 losers get dumped Oct 1–15 ({pct(h['oct_np_mod'], 2, True)}, t "
        f"{num(h['oct_np_mod_t'], 1, True)}), outrun winners from mid-October into mid-December, and the January "
        f"bounce is gone.<br/><br/>"
        f"<b>Setup, 2017–2025 regime:</b> long the 1Y-negative + 1M-negative basket (Sep 30 sort). "
        f"<b>Entry</b> Oct 15. <b>Exit</b> Dec 15. <b>Invalidation</b>: basket minus the 1Y-positive + 1M-positive "
        f"basket below {pct(h['nnpp_worst'], 1)} by Dec 15, the worst 2017–2025 print.")
box = Table([[P(bluf, "bluf")]], colWidths=[W])
box.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, -1), colors.HexColor(PANEL)),
                         ("LINEBEFORE", (0, 0), (0, -1), 3, colors.HexColor(BLUE)),
                         ("TOPPADDING", (0, 0), (-1, -1), 11), ("BOTTOMPADDING", (0, 0), (-1, -1), 11),
                         ("LEFTPADDING", (0, 0), (-1, -1), 13), ("RIGHTPADDING", (0, 0), (-1, -1), 13)]))
story += [box, Spacer(1, 12)]

story.append(tiles([
    ("Q4 Sharpe, 1Y losers vs winners · 1999–2025", f"{num(h['q4_sharpe_neg'])} <font size=11 color='{MUTED}'>vs"
     f"</font> {num(h['q4_sharpe_pos'])}", f"Avg Q4 return {pct(h['q4_ret_neg'])} vs {pct(h['q4_ret_pos'])}"),
    ("Oct 1–15, losers minus winners · 2017–2025", pct(h["oct_np_mod"], 2, True),
     f"1999–2007: {pct(h['oct_np_old'], 2, True)} · worst vs best quintile {pct(h['oct_q1q5_mod'], 2, True)}"),
    ("Oct 15–Dec 15, 1Y neg + 1M neg basket · 2017–2025",
     f"{pct(h['nn_long_ret'], 1, True)} <font size=11 color='{MUTED}'>Sharpe</font> {num(h['nn_long_sharpe'])}",
     f"1Y pos + 1M pos {pct(h['pp_long_ret'], 1, True)} (Sharpe {num(h['pp_long_sharpe'])}) · SPY "
     f"{pct(h['spy_long_ret'], 1, True)}"),
    ("January loser alpha after SPY beta, then vs now",
     f"{pct(h['jan_alpha_mid'], 2, True)} <font size=11 color='{MUTED}'>→</font> {pct(h['jan_alpha_mod'], 2, True)}",
     "2008–2016 vs 2017–2026, monthly sort, losers minus winners"),
]))
story += [Spacer(1, 10), P("Findings", "h2")]
findings = [
    f"<b>Q4 risk-adjusted is a coin flip on the 1Y sort.</b> Losers earn more (alpha vs SPY "
    f"{pct(h['q4_alpha_neg'])} vs {pct(h['q4_alpha_pos'])}, annualized over Q4 days) but carry more beta and vol, so "
    f"Sharpe lands even. Q4 loser-minus-winner spread {pct(h['q4_spread'], 2, True)}, t "
    f"{num(h['q4_spread_t'], 1, True)}. Q4 is the best quarter for every bucket.",
    f"<b>The bleed happens before Q4.</b> Q3 loser-minus-winner spread {pct(h['q3_spread'], 2, True)} full sample; "
    f"{pct(h['q3_spread_mod'], 2, True)} in 2017–2025, losers ahead in only {pct(h['q3_hit_mod'], 0)} of years.",
    f"<b>Timing moved earlier.</b> 1999–2007 had the classic early-December dip (Dec 1–15 "
    f"{pct(h['dec_np_old'], 2, True)}). 2017–2025 dumps losers Oct 1–15, and Dec 1–15 flipped to "
    f"{pct(h['dec_np_mod'], 2, True)} (worst vs best quintile {pct(h['dec_q1q5_mod'], 2, True)}).",
    f"<b>The January rebound decayed to zero.</b> January loser alpha after SPY beta "
    f"{pct(h['jan_alpha_old'], 2, True)} → {pct(h['jan_alpha_mid'], 2, True)} → {pct(h['jan_alpha_mod'], 2, True)} by era. "
    f"Buying 1Y losers Dec 15, holding to Jan 31: alpha {pct(h['rb_alpha_old'], 1, True)} → "
    f"{pct(h['rb_alpha_mid'], 1, True)} → {pct(h['rb_alpha_mod'], 1, True)} annualized.",
    f"<b>More year-end selling volume, no extra price impact.</b> Dec 21–31 loser volume over winner volume "
    f"{pct(h['vol_old'], 1, True)} → {pct(h['vol_mid'], 1, True)} → {pct(h['vol_mod'], 1, True)}; the Dec 16–31 price "
    f"spread stays flat. Year-by-year slopes are weak (October t {num(OCT_SLOPE_T, 1)}): a regime change, not a smooth trend.",
    f"<b>The modern window is mid-October to mid-December.</b> 1Y+1M neg minus 1Y+1M pos, Oct 15–Dec 15, 2017–2025: "
    f"{pct(h['nnpp_spread'], 2, True)} (t {num(h['nnpp_t'], 1, True)}, ahead {pct(h['nnpp_hit'], 0)} of years, worst "
    f"{pct(h['nnpp_worst'], 2, True)}, ex-2020 {pct(h['nnpp_ex2020'], 2, True)}). Basket alpha "
    f"{pct(h['nn_long_alpha'], 1, True)} annualized.",
    f"<b>The 1M sort carries its own Q4 edge now.</b> 2017–2025, September losers beat September winners in Q4 by "
    f"{pct(h['m1_q4_spread_mod'], 2, True)} (t {num(h['m1_q4_t_mod'], 1, True)}); Q4 Sharpe "
    f"{num(h['m1_q4_sharpe_neg'])} vs {num(h['m1_q4_sharpe_pos'])}.",
]
story += [Paragraph(f, ST["bullet"], bulletText="•") for f in findings]

# Q4 risk
story += [PageBreak(), P("Q4 risk-adjusted returns by bucket", "h1"),
          P("Buckets formed at the last September close, equal weight, buy-and-hold to Dec 31. Annualized stats pool "
            "every Q4 trading day across the era's years. Alpha and beta vs SPY; Sharpe and Sortino over 13-week "
            "T-bills. Avg qtr / Median / Beat SPY / Worst are per-year Q4 figures. Liquidity terciles use 63-day "
            "median dollar volume at formation.", "note"),
          P("1999–2025", "h2"), risk_table("Q4", "1999-2025"),
          P("2017–2025", "h2"), risk_table("Q4", "2017-2025")]
story += [CondPageBreak(3.2 * inch), P("Is Q4 special? Sharpe and alpha by quarter, 1999–2025", "h2"),
          P("Same buckets formed at each quarter's start (Dec 31, Mar 31, Jun 30, Sep 30) and held three months.",
            "note"),
          quarter_sharpe_table("1999-2025")]

# timing
story += [PageBreak(), P("Where in Q4 losers lag and lead", "h1"),
          P("Sep 30 sort held through January. Each bar is the era average of the yearly losers-minus-winners return "
            "in that half-month. Gray 1999–2007, blue 2008–2016, orange 2017–2025.", "note"),
          chart_halfmonth("NEG1Y-POS1Y", "Half-month spread: 1Y neg − 1Y pos"), Spacer(1, 6),
          chart_halfmonth("NN-PP", "Half-month spread: 1Y+1M neg − 1Y+1M pos"), Spacer(1, 6),
          KeepTogether([P("Cumulative relative return since Sep 30 (losers' growth over winners' growth, averaged "
                          "by era)", "h2"), chart_paths()])]

# calendar + yearly
story += [PageBreak(), P("Every month, not just Q4", "h1"),
          P("Same sort at every month-end 1999–2026; next-month spread regressed on SPY with calendar-month dummies, "
            "so bars are alpha after market beta. Q4 shaded.", "note"),
          chart_calendar(), Spacer(1, 8),
          KeepTogether([
              P("Rate of change, year by year", "h1"),
              P("Each bar is one year, colored by era; dashed lines mark the era breaks. Volume panel: median "
                "Dec 21–31 volume vs the Jun–Aug base, losers relative to winners (Nov 30 sort).", "note"),
              chart_yearly()])]

story += [PageBreak(), P("How year-end behavior shifted across eras", "h1"),
          P("Three equal 9-year blocks. Spreads are averages of yearly equal-weight returns; t-stats and hit rates "
            "for each cell are in the dashboard and results tables.", "note"),
          era_shift_table(), Spacer(1, 10),
          P("Oct 15 → Dec 15 window, 2017–2025 (Sep 30 sort)", "h1"), window_table("Oct15_Dec15", "2017-2025"),
          Spacer(1, 6), P("Same window, 1999–2025", "h2"), window_table("Oct15_Dec15", "1999-2025")]

story += [PageBreak(), P("Pair spreads by window and era", "h1"),
          P("Yearly long-minus-short return of equal-weight buckets; t on the yearly series; Ahead = share of years "
            "the long side won.", "note"),
          P("Oct 1–15", "h2"), spread_table("Oct1_Oct15"), Spacer(1, 4),
          P("Oct 15–Dec 15", "h2"), spread_table("Oct15_Dec15")]
story += [PageBreak(), P("Oct 15–Dec 31", "h2"), spread_table("Oct15_Dec31"), Spacer(1, 4),
          P("Full Q4", "h2"), spread_table("Q4"), Spacer(1, 6)]
for i, e in enumerate(ERA3):
    head = [P("Dec 15 → Jan 31 rebound (1Y return re-sorted at Dec 15)", "h1")] if i == 0 else []
    story.append(KeepTogether(head + [P(DASHES[e], "h2"), window_table(
        "Dec15_Jan31", e, ["DEEP_LT_-30", "Q1_1Y", "NEG_1Y", "ALL", "SPY", "POS_1Y", "Q5_1Y"])]))

story += [PageBreak(), P("Robustness, 2017–2025", "h1"),
          P("NN−PP = 1Y+1M negative minus 1Y+1M positive; N−P = 1Y negative minus 1Y positive. Liquidity floor on "
            "63-day median dollar volume. Oct 15 sort recomputes 1Y and 1M as of Oct 15 (point in time). Medians use "
            "the per-year median stock in each bucket.", "note"),
          robust_table()]

# live
lv = LV
story += [Spacer(1, 12), P("Q4 2026 so far", "h1"),
          P("Sep 30, 2026 sort; returns Oct 1 through the Oct 9, 2026 close.", "note"),
          tiles([
              ("1Y negative names", f"{int(lv.loc['NEG_1Y', 'n']):,}",
               f"of which 1M also negative: {int(lv.loc['NEG1Y_NEG1M', 'n']):,}"),
              ("QTD, 1Y neg vs 1Y pos", f"{pct(lv.loc['NEG_1Y', 'qtd_mean'], 2, True)} <font size=11 color='{MUTED}'>"
               f"vs</font> {pct(lv.loc['POS_1Y', 'qtd_mean'], 2, True)}", "Equal-weight mean"),
              ("QTD, 1Y+1M neg vs 1Y+1M pos", f"{pct(lv.loc['NEG1Y_NEG1M', 'qtd_mean'], 2, True)} "
               f"<font size=11 color='{MUTED}'>vs</font> {pct(lv.loc['POS1Y_POS1M', 'qtd_mean'], 2, True)}",
               f"SPY {pct(lv.loc['SPY', 'qtd_mean'], 2, True)}"),
              ("Oct 1–15 flush", "Not yet",
               "Losers are leading winners quarter to date, against the 2017–2025 pattern"),
          ])]

story += [PageBreak(), P("Method", "h1")]
for t in [
    f"<b>Universe.</b> The {int(h['n_tickers']):,} tickers in the uploaded finviz list. Daily dividend- and "
    "split-adjusted closes and volume from Yahoo Finance via yfinance, June 1998 through Oct 9, 2026. "
    f"{int(h['n_stock_years']):,} stock-years pass the Sep 30 screen (about 460 names in 1999, 2,200 by 2025).",
    "<b>Signals.</b> Formed at the last trading day of September: 1Y = trailing 12-month total return, 1M = "
    "September return. Eligibility: a full 12 months of history and 63-day median dollar volume of at least $1M "
    "at formation. Buckets are equal-weighted at formation and held without rebalancing. Daily returns are "
    "clipped to −80% / +150% to remove bad ticks.",
    "<b>Statistics.</b> Annualized return, vol, Sharpe and Sortino pool all daily returns inside the window across "
    "the era's years. Sharpe and Sortino are over 13-week T-bills (^IRX). Alpha and beta are from daily returns vs "
    "SPY. t-stats use one observation per year. Calendar-month alphas regress the monthly spread on SPY with "
    "month dummies (HC1 errors).",
    "<b>Controls.</b> Same buckets in Q1–Q3 and in every calendar month, three 9-year eras, liquidity floors of "
    "$1M / $5M / $20M, an Oct 15 point-in-time re-sort, medians, and with 2020 removed.",
    "<b>Universe bias.</b> The list is today's survivors, so delisted losers are missing. That lifts loser returns, "
    "more in the older years. Comparing Q4 with other quarters and eras with each other on the same universe nets "
    "most of it out.",
    "<b>Reproduce.</b> Code and result tables: research/tax_loss_harvesting on branch "
    "claude/tax-loss-harvesting-analysis-emumq1 of kbw468/yfinance.",
]:
    story.append(P(t))

# appendix: full screen
story += [PageBreak(), P("Appendix: 1Y negative and 1M negative names, Sep 30, 2026", "h1"),
          P(f"All {len(SC):,} names, sorted by 63-day median dollar volume. 1Y and 1M as of Sep 30, 2026; QTD through "
            "Oct 9, 2026. Market cap from the uploaded list ($M converted).", "note")]
story.append(table([SCREEN_HDR] + screen_rows(SC), SCREEN_CW, align_left=(1, 2, 3), font=7, zebra=True))


def on_page(canvas, doc):
    canvas.saveState()
    canvas.setFont("Sans", 7.5)
    canvas.setFillColor(colors.HexColor(MUTED))
    canvas.drawString(0.6 * inch, 0.42 * inch, "Q4 Tax-Loss Seasonality · data through Oct 9, 2026")
    canvas.drawRightString(letter[0] - 0.6 * inch, 0.42 * inch, f"{doc.page}")
    canvas.restoreState()


doc = SimpleDocTemplate(str(OUT), pagesize=letter, leftMargin=0.6 * inch, rightMargin=0.6 * inch,
                        topMargin=0.6 * inch, bottomMargin=0.65 * inch, title="Q4 Tax-Loss Seasonality",
                        author="Claude", subject="Tax-loss harvesting seasonality backtest")
doc.build(story, onFirstPage=on_page, onLaterPages=on_page)
print("wrote", OUT)
